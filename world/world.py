"""
organism_sim module: world/world.py

World state container (food, bounds, later obstacles, etc.)
"""

from __future__ import annotations
from dataclasses import dataclass

from world.food import FoodField


@dataclass
class World:
    w: int
    h: int
    food: FoodField

    @staticmethod
    def create(w: int, h: int) -> "World":
        return World(w=w, h=h, food=FoodField(w, h))

    def update(self, dt: float) -> None:
        self.food.update(dt)
