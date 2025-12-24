"""
organism_sim module: evolution/mutate.py

Mutation operators for brains (weights + biases).
"""

from __future__ import annotations
import random


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
