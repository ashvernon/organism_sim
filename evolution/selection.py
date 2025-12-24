"""
organism_sim module: evolution/selection.py

Selection helpers.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List

from neural.brain import Brain
from organism.genome import Genome


@dataclass
class Individual:
    brain: Brain
    genome: Genome
    fitness: float = 0.0


def select_top(pop: List[Individual], k: int) -> List[Individual]:
    return sorted(pop, key=lambda ind: ind.fitness, reverse=True)[:k]
