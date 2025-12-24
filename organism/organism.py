"""
organism_sim module: organism/organism.py

Organism container: nodes + edges + basic update hook.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

from organism.nodes import Node, NodeType
from organism.edges import Edge

if TYPE_CHECKING:
    from neural.brain import Brain


@dataclass
class Organism:
    nodes: Dict[int, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    next_node_id: int = 0

    energy: float = 10.0
    age: int = 0
    brain: Optional["Brain"] = None
    last_actuator_cost: float = 0.0

    def add_node(
        self,
        type: NodeType,
        x: float,
        y: float,
        angle: float = 0.0,
        radius: float = 6.0,
    ) -> Node:
        n = Node(
            id=self.next_node_id,
            type=type,
            x=x,
            y=y,
            angle=angle,
            radius=radius,
        )
        self.nodes[n.id] = n
        self.next_node_id += 1
        return n

    def add_edge(self, a_id: int, b_id: int, rest_length: float) -> None:
        self.edges.append(Edge(a=a_id, b=b_id, rest_length=rest_length))

    def center_of_mass(self) -> tuple[float, float]:
        xs = [n.x for n in self.nodes.values()]
        ys = [n.y for n in self.nodes.values()]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def update_kinematics(self, dt: float) -> None:
        self.age += 1
        for n in self.nodes.values():
            n.age += 1
            n.move(dt)

    def actuator_ids(self) -> List[int]:
        return [n.id for n in self.nodes.values() if n.type == NodeType.ACTUATOR]

    def clone_with_brain(self) -> "Organism":
        clone = Organism()
        for node in self.nodes.values():
            clone.nodes[node.id] = Node(
                id=node.id,
                type=node.type,
                x=node.x,
                y=node.y,
                angle=node.angle,
                radius=node.radius,
                vx=node.vx,
                vy=node.vy,
                ang_v=node.ang_v,
                energy=node.energy,
                age=node.age,
            )
        clone.edges = [Edge(a=e.a, b=e.b, rest_length=e.rest_length) for e in self.edges]
        clone.next_node_id = self.next_node_id
        clone.energy = self.energy
        clone.age = self.age
        clone.last_actuator_cost = self.last_actuator_cost
        if self.brain is not None:
            clone.brain = self.brain.clone()
        return clone
