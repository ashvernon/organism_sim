"""
organism_sim module: organism/edges.py

Edges connect nodes and constrain the body shape.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Edge:
    a: int
    b: int
    rest_length: float
