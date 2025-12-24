"""
organism_sim module: render/renderer.py

Pygame rendering of organisms (top-down).
"""

from __future__ import annotations
import math
import pygame

from organism.organism import Organism
from organism.nodes import NodeType
from render import colors
from world.food import FoodPellet


def _draw_dir_indicator(screen: pygame.Surface, x: float, y: float, angle: float, r: float) -> None:
    dx = math.cos(angle) * r
    dy = math.sin(angle) * r
    pygame.draw.line(screen, colors.DIR, (x, y), (x + dx, y + dy), 2)

def draw_food(screen: pygame.Surface, pellets: list[FoodPellet]) -> None:
    # simple "grass" look: small green dots, brightness by size
    for p in pellets:
        # brightness scales with radius
        v = int(110 + min(120, p.radius * 18))
        col = (40, v, 60)
        pygame.draw.circle(screen, col, (int(p.x), int(p.y)), int(p.radius))


def draw_organism(screen: pygame.Surface, org: Organism, debug: bool = False) -> None:
    debug_font = pygame.font.Font(None, 16) if debug else None

    # edges first
    for e in org.edges:
        a = org.nodes[e.a]
        b = org.nodes[e.b]
        pygame.draw.line(screen, colors.EDGE, (a.x, a.y), (b.x, b.y), 2)

    # nodes
    for n in org.nodes.values():
        if n.type == NodeType.CORE:
            col = colors.CORE
        elif n.type == NodeType.SEGMENT:
            col = colors.SEGMENT
        elif n.type == NodeType.ACTUATOR:
            col = colors.ACTUATOR
        else:
            col = colors.SENSOR

        pygame.draw.circle(screen, col, (int(n.x), int(n.y)), int(n.radius))
        _draw_dir_indicator(screen, n.x, n.y, n.angle, n.radius + 4)

        if debug and debug_font is not None:
            # small id label
            txt = debug_font.render(str(n.id), True, (230, 230, 230))
            screen.blit(txt, (n.x + n.radius + 2, n.y - n.radius - 2))

    if debug and debug_font is not None:
        cx, cy = org.center_of_mass()
        energy_txt = debug_font.render(
            f"E:{org.energy:.2f} cost:{org.last_actuator_cost:.3f}", True, (235, 235, 235)
        )
        screen.blit(energy_txt, (cx + 10, cy - 10))


def draw_hud(screen: pygame.Surface, stats: dict) -> None:
    font = pygame.font.Font(None, 26)

    lines = [
        f"Population: {stats.get('population', 0)}",
        f"Births: {stats.get('births', 0)}  Deaths: {stats.get('deaths', 0)}",
        f"Avg energy: {stats.get('avg_energy', 0.0):.2f}",
        f"Sim time: {stats.get('sim_time', 0.0):.1f}s",
    ]

    y = 10
    for line in lines:
        txt = font.render(line, True, (235, 235, 235))
        screen.blit(txt, (12, y))
        y += 22
