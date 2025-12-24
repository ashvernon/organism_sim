"""
organism_sim module: world/food.py

Food system:
- Spawns as natural clumps ("grass patches") using gaussian scatter around clump centers
- Each pellet has lifespan (ages out)
- Energy is proportional to size (radius)
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import random
from typing import List, Tuple


@dataclass
class FoodPellet:
    x: float
    y: float
    radius: float
    energy: float
    age: float = 0.0
    lifespan: float = 12.0  # seconds

    @property
    def dead(self) -> bool:
        return self.age >= self.lifespan


def radius_to_energy(r: float) -> float:
    # area-ish scaling (feels natural): energy grows faster than radius
    return max(0.1, (r * r) * 0.08)


def spawn_clump(
    cx: float,
    cy: float,
    n: int,
    spread: float,
    r_min: float,
    r_max: float,
    lifespan_range: Tuple[float, float],
) -> List[FoodPellet]:
    """
    Gaussian scatter around (cx, cy) to form natural clumps.
    spread is std-dev in pixels.
    """
    pellets: List[FoodPellet] = []
    for _ in range(n):
        x = random.gauss(cx, spread)
        y = random.gauss(cy, spread)

        r = random.uniform(r_min, r_max)
        life = random.uniform(*lifespan_range)

        e = radius_to_energy(r)
        pellets.append(FoodPellet(x=x, y=y, radius=r, energy=e, lifespan=life))

    return pellets


class FoodField:
    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h
        self.pellets: List[FoodPellet] = []

        # spawn tuning
        self.target_pellets = 320

        self.clump_n_range = (4, 16)
        self.clump_spread_range = (18.0, 60.0)  # pixels
        self.radius_range = (2.0, 6.0)
        self.lifespan_range = (10.0, 200.0)

        self.spawn_accum = 0.0
        self.spawn_rate = 1.3  # clumps per second (approx)

    def update(self, dt: float) -> None:
        # age & cull
        for p in self.pellets:
            p.age += dt
        self.pellets = [p for p in self.pellets if not p.dead]

        # replenish toward target with clumps
        deficit = self.target_pellets - len(self.pellets)
        if deficit <= 0:
            return

        self.spawn_accum += dt * self.spawn_rate
        while self.spawn_accum >= 1.0 and deficit > 0:
            self.spawn_accum -= 1.0
            self._spawn_random_clump()
            deficit = self.target_pellets - len(self.pellets)

    def _spawn_random_clump(self) -> None:
        cx = random.uniform(60, self.w - 60)
        cy = random.uniform(60, self.h - 60)
        n = random.randint(*self.clump_n_range)
        spread = random.uniform(*self.clump_spread_range)

        self.pellets.extend(
            spawn_clump(
                cx, cy,
                n=n,
                spread=spread,
                r_min=self.radius_range[0],
                r_max=self.radius_range[1],
                lifespan_range=self.lifespan_range,
            )
        )

        # keep food within bounds (clip)
        for p in self.pellets[-n:]:
            p.x = max(10, min(self.w - 10, p.x))
            p.y = max(10, min(self.h - 10, p.y))

    def eat_near(self, x: float, y: float, reach: float) -> float:
        """
        Remove pellets within reach and return total energy gained.
        """
        gained = 0.0
        reach2 = reach * reach
        remaining: List[FoodPellet] = []
        for p in self.pellets:
            dx = p.x - x
            dy = p.y - y
            if dx * dx + dy * dy <= reach2:
                gained += p.energy
            else:
                remaining.append(p)
        self.pellets = remaining
        return gained
    
    def nearest_pellet(self, x: float, y: float):
        """
        Returns (pellet, distance) for the nearest pellet, or (None, inf) if none exist.
        """
        best = None
        best_d2 = float("inf")
        for p in self.pellets:
            dx = p.x - x
            dy = p.y - y
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best = p
        if best is None:
            return None, float("inf")
        return best, math.sqrt(best_d2)

