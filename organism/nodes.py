"""
organism_sim module: organism/nodes.py

Body node primitives for top-down organisms.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class NodeType(Enum):
    CORE = 0
    SEGMENT = 1
    ACTUATOR = 2
    SENSOR = 3


@dataclass
class Node:
    id: int
    type: NodeType
    x: float
    y: float
    angle: float = 0.0
    radius: float = 6.0

    # dynamics
    vx: float = 0.0
    vy: float = 0.0
    ang_v: float = 0.0

    # “bio”
    energy: float = 1.0
    age: int = 0

    @property
    def pos(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def move(self, dt: float) -> None:
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.angle += self.ang_v * dt
