"""
organism_sim module: neural/neuron.py

Neuron primitives for a Bibites-style, morphology-safe brain.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class NeuronType(Enum):
    SENSOR = 0
    HIDDEN = 1
    MOTOR = 2
    GLOBAL = 3


@dataclass
class Neuron:
    id: int
    type: NeuronType
    bias: float = 0.0
    value: float = 0.0
    node_id: int | None = None  # which body node it's attached to (optional)
    name: str = ""              # debug label
