"""
organism_sim module: evolution/selection.py

Selection helpers.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class Individual:
    brain: object
    fitness: float = 0.0


def select_top(pop: List[Individual], k: int) -> List[Individual]:
    return sorted(pop, key=lambda ind: ind.fitness, reverse=True)[:k]
