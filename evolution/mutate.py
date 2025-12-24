"""
organism_sim module: evolution/mutate.py

Mutation operators for brains (weights + biases) and genomes (growth rules).
"""

from __future__ import annotations
import random
from typing import Iterable

from organism.genome import Genome, GrowthRule


def mutate_brain_params(brain, p_weight: float = 0.10, p_bias: float = 0.10, sigma: float = 0.35) -> None:
    """
    Mutate in-place:
      - With probability p_weight per synapse, add gaussian noise to weight.
      - With probability p_bias per hidden neuron, add gaussian noise to bias.
    """
    synapses, hidden_neurons = brain.get_mutable_param_refs()

    for s in synapses:
        if random.random() < p_weight:
            s.weight += random.gauss(0.0, sigma)

    for n in hidden_neurons:
        if random.random() < p_bias:
            n.bias += random.gauss(0.0, sigma)


def _jitter_rule(rule: GrowthRule, angle_sigma: float, length_sigma: float, radius_sigma: float, cost_sigma: float, cooldown_sigma: float) -> None:
    rule.angle += random.gauss(0.0, angle_sigma)
    rule.length = max(4.0, rule.length + random.gauss(0.0, length_sigma))
    rule.radius = max(1.0, rule.radius + random.gauss(0.0, radius_sigma))
    rule.cost = max(0.05, rule.cost + random.gauss(0.0, cost_sigma))
    rule.cooldown = max(0.05, rule.cooldown + random.gauss(0.0, cooldown_sigma))


def _clone_with_jitter(rule: GrowthRule, anchor_pool: Iterable[str]) -> GrowthRule:
    clone = GrowthRule(**rule.__dict__)
    clone.anchor = random.choice(list(anchor_pool))
    _jitter_rule(clone, angle_sigma=0.25, length_sigma=8.0, radius_sigma=0.8, cost_sigma=0.3, cooldown_sigma=0.25)
    return clone


def mutate_genome(
    genome: Genome,
    p_jitter: float = 0.35,
    p_add_rule: float = 0.12,
    p_remove_rule: float = 0.10,
    angle_sigma: float = 0.18,
    length_sigma: float = 6.0,
    radius_sigma: float = 0.6,
    cost_sigma: float = 0.25,
    cooldown_sigma: float = 0.2,
) -> Genome:
    """
    Return a mutated clone of ``genome``.

    - Jitter rule parameters to vary limb angles/lengths/costs.
    - Occasionally add a cloned rule with anchor jitter.
    - Occasionally drop a rule to encourage pruning.
    - Also perturb global growth gates.
    """

    mutated = genome.clone()

    for r in mutated.rules:
        if random.random() < p_jitter:
            _jitter_rule(r, angle_sigma, length_sigma, radius_sigma, cost_sigma, cooldown_sigma)

    if mutated.rules and random.random() < p_remove_rule:
        idx = random.randrange(len(mutated.rules))
        mutated.rules.pop(idx)

    anchor_pool = ["core", "random_node", "actuator", "sensor", "leaf"]
    if mutated.rules and random.random() < p_add_rule:
        parent = random.choice(mutated.rules)
        mutated.rules.append(_clone_with_jitter(parent, anchor_pool))

    mutated.grow_energy_threshold = max(0.0, mutated.grow_energy_threshold + random.gauss(0.0, 0.4))
    mutated.grow_interval = max(0.05, mutated.grow_interval + random.gauss(0.0, 0.15))

    return mutated
