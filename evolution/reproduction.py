"""
organism_sim module: evolution/reproduction.py

Reproduction helpers (elitism + mutated clones).
"""

from __future__ import annotations
import random
from typing import List

from evolution.selection import Individual
from evolution.mutate import mutate_brain_params


def next_generation(
    elites: List[Individual],
    pop_size: int,
    p_weight: float = 0.10,
    p_bias: float = 0.10,
    sigma: float = 0.35,
) -> List[Individual]:
    """
    Keep elites (cloned) and fill the rest with mutated clones of elites.
    """
    new_pop: List[Individual] = []

    # keep exact elite brains (cloned to avoid accidental mutation)
    for e in elites:
        new_pop.append(Individual(brain=e.brain.clone(), fitness=0.0))

    # fill rest
    while len(new_pop) < pop_size:
        parent = random.choice(elites)
        child_brain = parent.brain.clone()
        mutate_brain_params(child_brain, p_weight=p_weight, p_bias=p_bias, sigma=sigma)
        new_pop.append(Individual(brain=child_brain, fitness=0.0))

    return new_pop[:pop_size]
