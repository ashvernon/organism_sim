"""
Live reproduction helpers (elitism no longer drives the main loop).
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List

from evolution.selection import Individual
from evolution.mutate import mutate_brain_params, mutate_genome
from organism.organism import Organism
from organism.genome import Genome


@dataclass
class ChildSpawn:
    organism: Organism
    genome: Genome


def jitter_positions(org: Organism, jitter: float) -> None:
    dx = random.uniform(-jitter, jitter)
    dy = random.uniform(-jitter, jitter)
    for node in org.nodes.values():
        node.x += dx
        node.y += dy


def clone_for_spawn(
    parent: Organism,
    parent_genome: Genome,
    jitter: float,
    p_weight: float = 0.10,
    p_bias: float = 0.10,
    sigma: float = 0.35,
) -> ChildSpawn:
    child = parent.clone_with_brain()
    jitter_positions(child, jitter)

    child_genome = mutate_genome(parent_genome)
    if child.brain is not None:
        mutate_brain_params(child.brain, p_weight=p_weight, p_bias=p_bias, sigma=sigma)

    return ChildSpawn(organism=child, genome=child_genome)


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
        new_pop.append(Individual(brain=e.brain.clone(), genome=e.genome.clone(), fitness=0.0))

    # fill rest
    while len(new_pop) < pop_size:
        parent = random.choice(elites)
        child_brain = parent.brain.clone()
        child_genome = mutate_genome(parent.genome)
        mutate_brain_params(child_brain, p_weight=p_weight, p_bias=p_bias, sigma=sigma)
        new_pop.append(Individual(brain=child_brain, genome=child_genome, fitness=0.0))

    return new_pop[:pop_size]
