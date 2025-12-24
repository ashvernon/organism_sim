"""
Continuous live simulation: organisms eat, reproduce, and evolve in real time.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List
import pygame

import config
from neural.brain import Brain
from organism.organism import Organism
from organism.nodes import NodeType
from organism.genome import Genome
from organism.growth import GrowthState, try_apply_growth
from world.world import World
from world.physics import (
    apply_actuator_forces,
    solve_edges,
    apply_drag,
    clamp_speed,
    wrap_world,
    separate_organisms,
)
from render.renderer import draw_organism, draw_food, draw_hud
from render import colors

from evolution.reproduction import clone_for_spawn


@dataclass
class LiveAgent:
    organism: Organism
    genome: Genome
    growth: GrowthState
    age: float = 0.0


def wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def make_demo_organism(cx: float, cy: float) -> tuple[Organism, int, int]:
    org = Organism()
    core = org.add_node(NodeType.CORE, cx, cy, radius=12)

    a1 = org.add_node(NodeType.ACTUATOR, cx - 40, cy, angle=math.pi, radius=8)
    a2 = org.add_node(NodeType.ACTUATOR, cx + 40, cy, angle=0.0, radius=8)

    org.add_edge(core.id, a1.id, 40)
    org.add_edge(core.id, a2.id, 40)
    return org, a1.id, a2.id


def ensure_brain_body_io(org: Organism) -> None:
    """
    Add motor neurons for new actuators (and simple sensor stubs when body sensors appear).
    """
    if org.brain is None:
        return

    actuator_ids = org.actuator_ids()
    brain = org.brain
    for act_id in actuator_ids:
        brain.ensure_motor_for_actuator(act_id)

    for sensor_node in [n for n in org.nodes.values() if n.type == NodeType.SENSOR]:
        brain.ensure_sensor(f"sensor_{sensor_node.id}", node_id=sensor_node.id)


def build_agent(x: float, y: float, base_brain: Brain, genome: Genome) -> LiveAgent:
    org, _, _ = make_demo_organism(x, y)
    org.brain = base_brain.clone()
    ensure_brain_body_io(org)
    growth_state = GrowthState(time_since_last_global=genome.grow_interval)
    return LiveAgent(organism=org, genome=genome.clone(), growth=growth_state)


def sense_food(org: Organism, world: World) -> tuple[float, float, float]:
    cx, cy = org.center_of_mass()
    nearest, dist = world.food.nearest_pellet(cx, cy)

    core_node = next(n for n in org.nodes.values() if n.type == NodeType.CORE)
    heading = core_node.angle

    if nearest is None:
        return 0.0, 1.0, 0.0

    dx = nearest.x - cx
    dy = nearest.y - cy
    abs_angle = math.atan2(dy, dx)
    rel = wrap_angle(abs_angle - heading)
    food_sin = math.sin(rel)
    food_cos = math.cos(rel)
    sense_range = 260.0
    food_dist = max(0.0, 1.0 - min(1.0, dist / sense_range))
    return food_sin, food_cos, food_dist


def step_agent(agent: LiveAgent, world: World, dt: float, osc_t: float) -> float:
    org = agent.organism
    org.energy = max(0.0, org.energy - config.ENERGY_DRAIN_PER_SEC * dt)
    energy01 = max(0.0, min(1.0, org.energy / config.MAX_ENERGY))

    grew = try_apply_growth(org, agent.genome, agent.growth, dt)
    if grew:
        ensure_brain_body_io(org)
        solve_edges(org)
        apply_drag(org)
        clamp_speed(org, max_speed=420.0, max_ang=5.0)

    food_sin, food_cos, food_dist = sense_food(org, world)

    if org.brain is None:
        return 0.0

    org.brain.set_sensor("energy", energy01)
    org.brain.set_sensor("osc_sin", math.sin(osc_t * 2.0))
    org.brain.set_sensor("osc_cos", math.cos(osc_t * 2.0))
    org.brain.set_sensor("food_sin", food_sin)
    org.brain.set_sensor("food_cos", food_cos)
    org.brain.set_sensor("food_dist", food_dist)
    org.brain.step()

    actuator_outputs = org.brain.motor_outputs_for_actuators()

    cost = apply_actuator_forces(org, actuator_outputs, dt)
    org.last_actuator_cost = cost
    org.energy = max(0.0, org.energy - cost)
    solve_edges(org)
    apply_drag(org)
    clamp_speed(org, max_speed=420.0, max_ang=5.0)

    org.update_kinematics(dt)
    wrap_world(org, config.SCREEN_W, config.SCREEN_H, margin=60)

    cx, cy = org.center_of_mass()
    gained = world.food.eat_near(cx, cy, reach=config.EAT_REACH)
    if gained > 0:
        org.energy = min(config.MAX_ENERGY, org.energy + gained)

    agent.age += dt
    return gained


def spawn_child(parent: LiveAgent) -> LiveAgent:
    child_spawn = clone_for_spawn(
        parent.organism,
        parent.genome,
        jitter=config.CHILD_SPAWN_JITTER,
        p_weight=config.MUT_P_WEIGHT,
        p_bias=config.MUT_P_BIAS,
        sigma=config.MUT_SIGMA,
    )
    child_spawn.organism.energy = config.REPRO_COST
    ensure_brain_body_io(child_spawn.organism)
    growth = GrowthState(time_since_last_global=child_spawn.genome.grow_interval)
    return LiveAgent(organism=child_spawn.organism, genome=child_spawn.genome, growth=growth)


def cull_excess(population: List[LiveAgent], deaths: int) -> int:
    if len(population) <= config.MAX_POP:
        return deaths
    population.sort(key=lambda a: a.organism.energy)
    overflow = len(population) - config.MAX_POP
    del population[:overflow]
    return deaths + overflow


def main():
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_W, config.SCREEN_H))
    pygame.display.set_caption("organism_sim (Live Evolution)")
    clock = pygame.time.Clock()

    _, a1, a2 = make_demo_organism(config.SCREEN_W / 2, config.SCREEN_H / 2)
    base_brain = Brain.build_starter([a1, a2], seed=1)
    base_genome = Genome.starter()

    world = World.create(config.SCREEN_W, config.SCREEN_H)

    agents: List[LiveAgent] = []
    for _ in range(config.START_POP):
        agents.append(
            build_agent(
                random.uniform(80.0, config.SCREEN_W - 80.0),
                random.uniform(80.0, config.SCREEN_H - 80.0),
                base_brain,
                base_genome,
            )
        )

    births = 0
    deaths = 0
    sim_time = 0.0
    debug = False
    running = True

    while running:
        dt_frame = clock.tick(60) / 1000.0
        dt_frame = min(dt_frame, 1 / 30)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_TAB:
                debug = not debug

        sub_steps = max(1, config.SIM_SPEED)
        dt = dt_frame / sub_steps

        for _ in range(sub_steps):
            sim_time += dt
            world.update(dt)

            for agent in list(agents):
                step_agent(agent, world, dt, sim_time)

            separate_organisms([a.organism for a in agents])

            # Death conditions
            survivors: List[LiveAgent] = []
            for agent in agents:
                if agent.organism.energy <= config.DEATH_ENERGY_FLOOR or agent.age >= config.MAX_AGE_SECONDS:
                    deaths += 1
                else:
                    survivors.append(agent)
            agents = survivors

            # Reproduction pass
            new_agents: List[LiveAgent] = []
            for agent in agents:
                if agent.organism.energy >= config.REPRO_ENERGY_THRESHOLD and len(agents) + len(new_agents) < config.MAX_POP:
                    agent.organism.energy -= config.REPRO_COST
                    new_agents.append(spawn_child(agent))
                    births += 1
            agents.extend(new_agents)

            deaths = cull_excess(agents, deaths)

        # Render
        screen.fill(colors.BG)
        draw_food(screen, world.food.pellets)
        for agent in agents:
            draw_organism(screen, agent.organism, debug=debug)

        avg_energy = sum(a.organism.energy for a in agents) / len(agents) if agents else 0.0
        stats = {
            "population": len(agents),
            "births": births,
            "deaths": deaths,
            "avg_energy": avg_energy,
            "sim_time": sim_time,
        }
        draw_hud(screen, stats)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
