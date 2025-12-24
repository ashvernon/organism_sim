"""
organism_sim module: world/physics.py

Top-down 2D physics:
- actuators apply thrust along their node angle
- torque emerges from off-center thrust
- edges are solved as simple distance constraints
- optional speed clamp + world wrap keep organisms visible during iteration
- apply_actuator_forces returns an energy/effort cost based on thrust usage
"""

from __future__ import annotations
import math
from typing import Dict

import config
from organism.organism import Organism
from organism.nodes import NodeType

# Tunables (move to config.py later)
ACTUATOR_FORCE = 90.0
TORQUE_SCALE = 0.0015

LINEAR_DRAG = 0.92
ANGULAR_DRAG = 0.86

EDGE_SOLVER_ITERS = 2
EDGE_STIFFNESS = 0.65

# Cost scaling (tune via config for survivability/efficiency)
THRUST_COST = config.ACTUATOR_COST_SCALE


def apply_actuator_forces(org: Organism, actuator_outputs: Dict[int, float], dt: float) -> float:
    """
    actuator_outputs: node_id -> thrust in [-1, 1]
    dt: seconds

    Returns:
        cost (float): effort/energy cost for thrusting this frame
    """
    cx, cy = org.center_of_mass()

    cost_accum = 0.0

    for node_id, thrust in actuator_outputs.items():
        node = org.nodes.get(node_id)
        if node is None or node.type != NodeType.ACTUATOR:
            continue

        thrust = max(-1.0, min(1.0, float(thrust)))
        # Larger actuators should incur higher energetic cost to mimic heavier muscles
        radius_scale = max(node.radius, 1.0)
        cost_accum += abs(thrust) * dt * radius_scale

        fx = math.cos(node.angle) * thrust * ACTUATOR_FORCE
        fy = math.sin(node.angle) * thrust * ACTUATOR_FORCE

        # time-based impulse (prevents runaway acceleration)
        node.vx += fx * dt
        node.vy += fy * dt

        # torque = r x F (2D)
        rx = node.x - cx
        ry = node.y - cy
        torque = rx * fy - ry * fx

        node.ang_v += torque * TORQUE_SCALE * dt

    return cost_accum * THRUST_COST


def solve_edges(org: Organism) -> None:
    # simple position-based constraint solver
    for _ in range(EDGE_SOLVER_ITERS):
        for e in org.edges:
            a = org.nodes[e.a]
            b = org.nodes[e.b]

            dx = b.x - a.x
            dy = b.y - a.y
            dist = math.hypot(dx, dy)
            if dist <= 1e-6:
                continue

            diff = (dist - e.rest_length) / dist
            ox = dx * 0.5 * EDGE_STIFFNESS * diff
            oy = dy * 0.5 * EDGE_STIFFNESS * diff

            a.x += ox
            a.y += oy
            b.x -= ox
            b.y -= oy


def apply_drag(org: Organism) -> None:
    for n in org.nodes.values():
        n.vx *= LINEAR_DRAG
        n.vy *= LINEAR_DRAG
        n.ang_v *= ANGULAR_DRAG


def clamp_speed(org: Organism, max_speed: float = 420.0, max_ang: float = 5.0) -> None:
    """
    Prevents agents from yeeting off due to transient thrust or constraint artifacts.
    """
    max_speed2 = max_speed * max_speed
    for n in org.nodes.values():
        v2 = n.vx * n.vx + n.vy * n.vy
        if v2 > max_speed2:
            v = math.sqrt(v2)
            s = max_speed / max(v, 1e-9)
            n.vx *= s
            n.vy *= s
        n.ang_v = max(-max_ang, min(max_ang, n.ang_v))


def wrap_world(org: Organism, w: int, h: int, margin: int = 60) -> None:
    """
    Arcade-style wrap so the organism stays on-screen during early development.
    """
    for n in org.nodes.values():
        if n.x < -margin:
            n.x = w + margin
        elif n.x > w + margin:
            n.x = -margin

        if n.y < -margin:
            n.y = h + margin
        elif n.y > h + margin:
            n.y = -margin


def separate_organisms(
    organisms,
    radius: float = 18.0,
    strength: float = 0.35,
):
    """
    Apply a soft positional push between organisms that get too close together.
    """

    r2 = radius * radius

    for i in range(len(organisms)):
        a = organisms[i]
        ax, ay = a.center_of_mass()

        for j in range(i + 1, len(organisms)):
            b = organisms[j]
            bx, by = b.center_of_mass()

            dx = bx - ax
            dy = by - ay
            d2 = dx * dx + dy * dy

            if d2 <= 1e-6 or d2 > r2:
                continue

            d = math.sqrt(d2)
            overlap = radius - d
            push = overlap / radius * strength

            nx = dx / d
            ny = dy / d

            for n in a.nodes.values():
                n.x -= nx * push
                n.y -= ny * push

            for n in b.nodes.values():
                n.x += nx * push
                n.y += ny * push
