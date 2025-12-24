"""
organism_sim module: organism/genome.py

Genome for morphology growth.

Design goals:
- Small, serializable, mutation-friendly representation
- Encodes a list of growth rules (like a tiny L-system / grammar)
- Rules can be applied when the organism has surplus energy
- Later: inheritance + mutation across generations

This MVP supports "budding" new nodes from existing nodes.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import random


class GrowOp(Enum):
    """Growth operation types."""
    BUD_ACTUATOR = 0
    BUD_SENSOR = 1
    BUD_SEGMENT = 2


@dataclass
class GrowthRule:
    """
    One grammar rule / growth instruction.

    anchor:
      - "core" (default) means attach to the CORE node
      - later you can support "random_node", "actuator", "sensor", etc.
    angle:
      - radians relative to anchor node's angle
    length:
      - edge rest length to new node
    radius:
      - node radius for the new node
    cost:
      - energy cost to perform this growth action
    cooldown:
      - minimum seconds between uses of this rule (per organism)
    """
    op: GrowOp
    anchor: str = "core"
    angle: float = 0.0
    length: float = 40.0
    radius: float = 6.0
    cost: float = 1.5
    cooldown: float = 1.5


@dataclass
class Genome:
    """
    A genome is a list of growth rules plus some global parameters.
    """
    rules: List[GrowthRule] = field(default_factory=list)

    # Global tuning (kept here so evolution can mutate these too later)
    grow_energy_threshold: float = 8.0   # must have at least this energy to grow
    grow_interval: float = 1.0           # seconds between growth attempts (global)

    def pick_rule(self) -> Optional[GrowthRule]:
        if not self.rules:
            return None
        return random.choice(self.rules)

    @staticmethod
    def starter() -> "Genome":
        """
        Starter genome: a small grammar that tends to make tri-/quad-like bodies.
        """
        rules = [
            GrowthRule(op=GrowOp.BUD_ACTUATOR, anchor="core", angle=0.0, length=40.0, radius=8.0, cost=2.0, cooldown=1.0),
            GrowthRule(op=GrowOp.BUD_ACTUATOR, anchor="core", angle=2.094, length=40.0, radius=8.0, cost=2.0, cooldown=1.0),  # +120°
            GrowthRule(op=GrowOp.BUD_ACTUATOR, anchor="core", angle=-2.094, length=40.0, radius=8.0, cost=2.0, cooldown=1.0), # -120°
            GrowthRule(op=GrowOp.BUD_SENSOR, anchor="core", angle=1.047, length=28.0, radius=5.0, cost=1.0, cooldown=0.8),
            GrowthRule(op=GrowOp.BUD_SENSOR, anchor="core", angle=-1.047, length=28.0, radius=5.0, cost=1.0, cooldown=0.8),
        ]
        return Genome(rules=rules, grow_energy_threshold=8.0, grow_interval=1.0)

    def clone(self) -> "Genome":
        # light clone (rules are dataclasses; shallow copy is okay if we copy the list)
        return Genome(
            rules=[GrowthRule(**r.__dict__) for r in self.rules],
            grow_energy_threshold=self.grow_energy_threshold,
            grow_interval=self.grow_interval,
        )
