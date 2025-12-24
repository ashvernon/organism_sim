"""
organism_sim module: organism/growth.py

Growth execution:
- Applies Genome rules to an Organism
- Tracks per-rule cooldowns per organism
- Performs simple "budding" from an anchor node

Important:
- Brain adaptation (adding motor/sensor neurons) is NOT done here.
  Keep growth purely morphological. Your main loop or evolution layer
  should call brain.ensure_* after growth.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
import random
from typing import Dict, List

from organism.organism import Organism
from organism.nodes import NodeType
from organism.genome import Genome, GrowOp, GrowthRule


@dataclass
class GrowthState:
    """
    Per-organism growth bookkeeping.
    Store inside Organism later if you want; for now keep it external.
    """
    time_since_last_global: float = 0.0
    rule_cooldowns: Dict[int, float] = field(default_factory=dict)  # index -> time remaining
    rule_idle_time: Dict[int, float] = field(default_factory=dict)  # index -> seconds since last use


def _find_core_id(org: Organism) -> int:
    for n in org.nodes.values():
        if n.type == NodeType.CORE:
            return n.id
    raise RuntimeError("Organism has no CORE node")


def _anchor_node_id(org: Organism, anchor: str) -> int:
    core_id = _find_core_id(org)

    def _degree_counts() -> Dict[int, int]:
        counts: Dict[int, int] = {nid: 0 for nid in org.nodes.keys()}
        for e in org.edges:
            counts[e.a] = counts.get(e.a, 0) + 1
            counts[e.b] = counts.get(e.b, 0) + 1
        return counts

    anchor = anchor.lower()
    candidates: List[int]

    if anchor == "core":
        candidates = [core_id]
    elif anchor == "random_node":
        candidates = list(org.nodes.keys())
    elif anchor == "actuator":
        candidates = [n.id for n in org.nodes.values() if n.type == NodeType.ACTUATOR]
    elif anchor == "sensor":
        candidates = [n.id for n in org.nodes.values() if n.type == NodeType.SENSOR]
    elif anchor == "leaf":
        deg = _degree_counts()
        candidates = [nid for nid, d in deg.items() if d <= 1]
    else:
        candidates = []

    if not candidates:
        return core_id
    return random.choice(candidates)


def _spawn_bud(org: Organism, anchor_id: int, angle_abs: float, length: float, node_type: NodeType, radius: float) -> int:
    """
    Create a new node at (anchor + length * dir) and connect with an edge.
    Returns new node id.
    """
    a = org.nodes[anchor_id]
    x = a.x + math.cos(angle_abs) * length
    y = a.y + math.sin(angle_abs) * length

    new_node = org.add_node(node_type, x, y, angle=angle_abs, radius=radius)
    org.add_edge(anchor_id, new_node.id, rest_length=length)
    return new_node.id


def try_apply_growth(org: Organism, genome: Genome, state: GrowthState, dt: float) -> bool:
    """
    Attempt to apply one growth rule.
    Returns True if growth happened.
    """
    # Update timers
    state.time_since_last_global += dt
    for k in list(state.rule_cooldowns.keys()):
        state.rule_cooldowns[k] = max(0.0, state.rule_cooldowns[k] - dt)
    for idx in range(len(genome.rules)):
        state.rule_idle_time[idx] = state.rule_idle_time.get(idx, 0.0) + dt

    # Global gate
    if org.energy < genome.grow_energy_threshold:
        return False
    if state.time_since_last_global < genome.grow_interval:
        return False

    # Choose a rule that isn't on cooldown
    if not genome.rules:
        return False

    available: List[int] = []
    weights: List[float] = []
    for idx, rule in enumerate(genome.rules):
        if state.rule_cooldowns.get(idx, 0.0) <= 1e-9 and org.energy >= rule.cost:
            idle = state.rule_idle_time.get(idx, 0.0)
            # prefer rules that have waited longer (avoid starving rarely used rules)
            weights.append(1.0 + idle)
            available.append(idx)

    if not available:
        return False

    chosen_idx = random.choices(available, weights=weights, k=1)[0]

    rule: GrowthRule = genome.rules[chosen_idx]

    anchor_id = _anchor_node_id(org, rule.anchor)
    anchor = org.nodes[anchor_id]

    # absolute angle = anchor.angle + rule.angle
    angle_abs = anchor.angle + rule.angle

    if rule.op == GrowOp.BUD_ACTUATOR:
        _spawn_bud(org, anchor_id, angle_abs, rule.length, NodeType.ACTUATOR, radius=rule.radius)
    elif rule.op == GrowOp.BUD_SENSOR:
        _spawn_bud(org, anchor_id, angle_abs, rule.length, NodeType.SENSOR, radius=rule.radius)
    elif rule.op == GrowOp.BUD_SEGMENT:
        _spawn_bud(org, anchor_id, angle_abs, rule.length, NodeType.SEGMENT, radius=rule.radius)
    else:
        return False

    # Pay energy + set cooldowns
    org.energy = max(0.0, org.energy - rule.cost)
    state.rule_cooldowns[chosen_idx] = rule.cooldown
    state.rule_idle_time[chosen_idx] = 0.0
    state.time_since_last_global = 0.0
    return True
