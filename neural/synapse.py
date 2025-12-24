"""
organism_sim module: neural/synapse.py

Weighted directed connection between neurons.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Synapse:
    src: int
    dst: int
    weight: float
