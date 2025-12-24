"""
organism_sim module: main.py

Evolution MVP:
- evaluate N organisms for 20 seconds each (solo)
- fitness = food energy consumed
- select top K elites
- reproduce via mutated clones
- render best individual each generation
"""

from __future__ import annotations
import math
import random
import pygame

from neural.brain import Brain
from organism.organism import Organism
from organism.nodes import NodeType
from world.world import World
from world.physics import apply_actuator_forces, solve_edges, apply_drag, clamp_speed, wrap_world
from render.renderer import draw_organism, draw_food
from render import colors

from evolution.selection import Individual, select_top
from evolution.reproduction import next_generation

SCREEN_W, SCREEN_H = 980, 720


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


def eval_one(brain: Brain, seconds: float = 20.0, seed: int = 0) -> float:
    """
    Headless evaluation (no rendering):
    fitness = total food energy consumed over the episode
    """
    random.seed(seed)
    org, a1, a2 = make_demo_organism(SCREEN_W / 2, SCREEN_H / 2)
    # clone so evaluation can't mutate the population's stored brain
    b = brain.clone()

    world = World.create(SCREEN_W, SCREEN_H)

    dt = 1.0 / 30.0
    steps = int(seconds / dt)

    food_eaten_total = 0.0

    for step in range(steps):
        world.update(dt)

        # energy drain (keeps pressure to eat)
        org.energy = max(0.0, org.energy - 0.002)
        energy01 = max(0.0, min(1.0, org.energy / 10.0))

        cx, cy = org.center_of_mass()
        nearest, dist = world.food.nearest_pellet(cx, cy)

        core_node = next(n for n in org.nodes.values() if n.type == NodeType.CORE)
        heading = core_node.angle

        if nearest is None:
            food_sin = 0.0
            food_cos = 1.0
            food_dist = 0.0
        else:
            dx = nearest.x - cx
            dy = nearest.y - cy
            abs_angle = math.atan2(dy, dx)
            rel = wrap_angle(abs_angle - heading)
            food_sin = math.sin(rel)
            food_cos = math.cos(rel)
            sense_range = 260.0
            food_dist = max(0.0, 1.0 - min(1.0, dist / sense_range))

        now = step * dt
        osc = now * 2.0

        b.set_sensor("energy", energy01)
        b.set_sensor("osc_sin", math.sin(osc))
        b.set_sensor("osc_cos", math.cos(osc))
        b.set_sensor("food_sin", food_sin)
        b.set_sensor("food_cos", food_cos)
        b.set_sensor("food_dist", food_dist)

        b.step()
        actuator_outputs = b.motor_outputs_for_actuators()

        apply_actuator_forces(org, actuator_outputs, dt)
        solve_edges(org)
        apply_drag(org)
        clamp_speed(org, max_speed=420.0, max_ang=5.0)

        org.update_kinematics(dt)
        wrap_world(org, SCREEN_W, SCREEN_H, margin=60)

        gained = world.food.eat_near(cx, cy, reach=14)
        if gained > 0:
            food_eaten_total += gained
            org.energy = min(10.0, org.energy + gained)

    return float(food_eaten_total)


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("organism_sim (Evolution MVP)")
    clock = pygame.time.Clock()

    POP_SIZE = 60           # 30–100
    ELITES = 8
    EPISODE_SECONDS = 20.0

    # Mutation tuning
    MUT_P_WEIGHT = 0.12
    MUT_P_BIAS = 0.10
    MUT_SIGMA = 0.30

    # Build initial population
    # NOTE: assumes your Brain.build_starter creates the food sensors already
    # (energy, osc_sin, osc_cos, food_sin, food_cos, food_dist + motors)
    dummy_org, a1, a2 = make_demo_organism(SCREEN_W / 2, SCREEN_H / 2)
    base_brain = Brain.build_starter([a1, a2], seed=1)

    population = [Individual(brain=base_brain.clone(), fitness=0.0) for _ in range(POP_SIZE)]

    generation = 0
    best_ind = population[0]

    running = True
    while running:
        generation += 1

        # ---- Evaluate population (headless) ----
        for i, ind in enumerate(population):
            # same env seed for fairness across individuals
            ind.fitness = eval_one(ind.brain, seconds=EPISODE_SECONDS, seed=12345)
            # allow quit during long eval
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                    break
            if not running:
                break

        if not running:
            break

        elites = select_top(population, ELITES)
        best_ind = elites[0]

        print(f"Gen {generation:03d} | best fitness={best_ind.fitness:.3f} | avg={sum(i.fitness for i in population)/len(population):.3f}")

        # ---- Produce next generation ----
        population = next_generation(
            elites,
            pop_size=POP_SIZE,
            p_weight=MUT_P_WEIGHT,
            p_bias=MUT_P_BIAS,
            sigma=MUT_SIGMA,
        )

        # ---- Render the best (live preview) ----
        # Run a short visible rollout of the best for ~3 seconds
        org, a1, a2 = make_demo_organism(SCREEN_W / 2, SCREEN_H / 2)
        brain = best_ind.brain.clone()
        world = World.create(SCREEN_W, SCREEN_H)

        preview_secs = 3.0
        t0 = pygame.time.get_ticks() / 1000.0
        debug = False

        while running:
            dt = clock.tick(60) / 1000.0
            dt = min(dt, 1 / 30)

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_TAB:
                        debug = not debug

            world.update(dt)

            org.energy = max(0.0, org.energy - 0.002)
            energy01 = max(0.0, min(1.0, org.energy / 10.0))

            cx, cy = org.center_of_mass()
            nearest, dist = world.food.nearest_pellet(cx, cy)
            core_node = next(n for n in org.nodes.values() if n.type == NodeType.CORE)
            heading = core_node.angle

            if nearest is None:
                food_sin = 0.0
                food_cos = 1.0
                food_dist = 0.0
            else:
                dx = nearest.x - cx
                dy = nearest.y - cy
                abs_angle = math.atan2(dy, dx)
                rel = wrap_angle(abs_angle - heading)
                food_sin = math.sin(rel)
                food_cos = math.cos(rel)
                sense_range = 260.0
                food_dist = max(0.0, 1.0 - min(1.0, dist / sense_range))

            now = pygame.time.get_ticks() / 1000.0
            osc = now * 2.0

            brain.set_sensor("energy", energy01)
            brain.set_sensor("osc_sin", math.sin(osc))
            brain.set_sensor("osc_cos", math.cos(osc))
            brain.set_sensor("food_sin", food_sin)
            brain.set_sensor("food_cos", food_cos)
            brain.set_sensor("food_dist", food_dist)
            brain.step()

            actuator_outputs = brain.motor_outputs_for_actuators()

            apply_actuator_forces(org, actuator_outputs, dt)
            solve_edges(org)
            apply_drag(org)
            clamp_speed(org, max_speed=420.0, max_ang=5.0)

            org.update_kinematics(dt)
            wrap_world(org, SCREEN_W, SCREEN_H, margin=60)

            gained = world.food.eat_near(cx, cy, reach=14)
            if gained > 0:
                org.energy = min(10.0, org.energy + gained)

            screen.fill(colors.BG)
            draw_food(screen, world.food.pellets)
            draw_organism(screen, org, debug=debug)

            # overlay text
            font = pygame.font.Font(None, 26)
            txt = font.render(f"Gen {generation}  Best fitness: {best_ind.fitness:.2f}", True, (235, 235, 235))
            screen.blit(txt, (12, 10))

            pygame.display.flip()

            if (pygame.time.get_ticks() / 1000.0) - t0 >= preview_secs:
                break

    pygame.quit()


if __name__ == "__main__":
    main()
