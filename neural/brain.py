"""
organism_sim module: neural/brain.py

A small evolvable neural network:
- sensors set values externally
- hidden + motor neurons are computed each step
- motor outputs drive actuators in [-1, 1]
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math
import random
import copy


from neural.neuron import Neuron, NeuronType
from neural.synapse import Synapse


def _tanh(x: float) -> float:
    # stable tanh for typical magnitudes
    return math.tanh(max(-20.0, min(20.0, x)))


@dataclass
class Brain:
    neurons: Dict[int, Neuron] = field(default_factory=dict)
    synapses: List[Synapse] = field(default_factory=list)
    next_neuron_id: int = 0

    # convenience: names -> neuron_id
    named: Dict[str, int] = field(default_factory=dict)

    # bookkeeping: actuator node id -> motor neuron id
    actuator_motors: Dict[int, int] = field(default_factory=dict)

    def clone(self) -> "Brain":
        return copy.deepcopy(self)

    def get_mutable_param_refs(self):
        """
        Returns references to mutable params:
          - synapse weights
          - hidden neuron biases (you can include motor biases too, but hidden is enough)
        """
        syn_refs = self.synapses  # list of Synapse objects with .weight
        hidden_bias_refs = [n for n in self.neurons.values() if n.type == NeuronType.HIDDEN]
        return syn_refs, hidden_bias_refs


    def add_neuron(
        self,
        ntype: NeuronType,
        bias: float = 0.0,
        node_id: Optional[int] = None,
        name: str = "",
    ) -> int:
        nid = self.next_neuron_id
        self.next_neuron_id += 1
        self.neurons[nid] = Neuron(id=nid, type=ntype, bias=bias, value=0.0, node_id=node_id, name=name)
        if name:
            self.named[name] = nid
        return nid

    def add_synapse(self, src: int, dst: int, weight: float) -> None:
        self.synapses.append(Synapse(src=src, dst=dst, weight=weight))

    def ensure_sensor(self, name: str, node_id: int | None = None) -> int:
        """
        Ensure a sensor neuron with the given name (and optional body node mapping).
        Returns the neuron id.
        """
        existing = self.named.get(name)
        if existing is not None:
            return existing

        return self.add_neuron(NeuronType.SENSOR, node_id=node_id, name=name)

    def ensure_motor_for_actuator(self, node_id: int) -> int:
        """
        Create a motor neuron (plus minimal wiring) for the given actuator node.
        Returns the motor neuron id.
        """
        existing = self.actuator_motors.get(node_id)
        if existing is not None and existing in self.neurons:
            return existing

        # attempt to reuse any motor already tagged with this node_id
        for n in self.neurons.values():
            if n.type == NeuronType.MOTOR and n.node_id == node_id:
                self.actuator_motors[node_id] = n.id
                return n.id

        idx = len(self.actuator_motors)
        mid = self.add_neuron(NeuronType.MOTOR, bias=0.0, node_id=node_id, name=f"motor_{node_id}")
        self.actuator_motors[node_id] = mid

        # minimal starter wiring: connect to hidden pair if present, otherwise oscillator if present
        h1 = self.named.get("h1")
        h2 = self.named.get("h2")
        osc_sin = self.named.get("osc_sin")
        osc_cos = self.named.get("osc_cos")

        if h1 is not None and h2 is not None:
            if idx % 2 == 0:
                self.add_synapse(h1, mid, 1.0)
                self.add_synapse(h2, mid, -0.8)
            else:
                self.add_synapse(h1, mid, -1.0)
                self.add_synapse(h2, mid, 0.8)
        elif osc_sin is not None and osc_cos is not None:
            phase = 1.0 if idx % 2 == 0 else -1.0
            self.add_synapse(osc_sin, mid, phase)
            self.add_synapse(osc_cos, mid, 0.5)

        return mid

    def set_sensor(self, name: str, value: float) -> None:
        nid = self.named.get(name)
        if nid is None:
            raise KeyError(f"Sensor '{name}' not found")
        self.neurons[nid].value = float(value)

    def get_value(self, name: str) -> float:
        nid = self.named.get(name)
        if nid is None:
            raise KeyError(f"Neuron '{name}' not found")
        return self.neurons[nid].value

    def step(self) -> None:
        # compute new values for hidden/motor/global (sensors keep their externally-set values)
        sums: Dict[int, float] = {nid: 0.0 for nid in self.neurons.keys()}

        # accumulate weighted inputs
        for syn in self.synapses:
            src = self.neurons[syn.src]
            sums[syn.dst] += src.value * syn.weight

        # update non-sensor neurons
        for nid, n in self.neurons.items():
            if n.type == NeuronType.SENSOR:
                continue
            x = sums[nid] + n.bias
            n.value = _tanh(x)

    # ---- helpers to build a decent starter wiring ----

    @staticmethod
    def build_starter(actuator_node_ids: List[int], seed: int | None = None) -> "Brain":
        """
        Starter brain:
        Sensors:
          - energy (0..1)
          - osc_sin, osc_cos (oscillator)
        Motors:
          - one motor per actuator
        Wiring:
          oscillator drives motors in opposite phase, energy gates output.
        """
        if seed is not None:
            random.seed(seed)

        b = Brain()

        # Sensors / Globals
        b.add_neuron(NeuronType.SENSOR, name="energy")
        b.add_neuron(NeuronType.SENSOR, name="osc_sin")
        b.add_neuron(NeuronType.SENSOR, name="osc_cos")

        # Food sensors (relative bearing + closeness)
        b.add_neuron(NeuronType.SENSOR, name="food_sin")   # sin(relative_angle_to_food)
        b.add_neuron(NeuronType.SENSOR, name="food_cos")   # cos(relative_angle_to_food)
        b.add_neuron(NeuronType.SENSOR, name="food_dist")  # 0..1 (1=close, 0=far/none)


        # Hidden "mixer" neurons (optional, gives evolution a place to add complexity later)
        h1 = b.add_neuron(NeuronType.HIDDEN, bias=random.uniform(-0.2, 0.2), name="h1")
        h2 = b.add_neuron(NeuronType.HIDDEN, bias=random.uniform(-0.2, 0.2), name="h2")

        # Connect sensors -> hidden
        # food steering signals
        b.add_synapse(b.named["food_sin"], h1, 1.4)   # steer
        b.add_synapse(b.named["food_sin"], h2, -1.4)  # opposite contribution
        b.add_synapse(b.named["food_dist"], h1, 0.8)  # amplify when close-ish
        b.add_synapse(b.named["food_dist"], h2, 0.8)
        b.add_synapse(b.named["food_cos"], h1, 0.3)   # forward alignment
        b.add_synapse(b.named["food_cos"], h2, 0.3)


        # Motors per actuator
        motor_ids: List[int] = []
        for i, node_id in enumerate(actuator_node_ids):
            mid = b.add_neuron(NeuronType.MOTOR, bias=0.0, node_id=node_id, name=f"motor_{node_id}")
            b.actuator_motors[node_id] = mid
            motor_ids.append(mid)

        # Wire hidden -> motors; alternate phase to create turning motion
        for idx, mid in enumerate(motor_ids):
            if idx % 2 == 0:
                b.add_synapse(h1, mid, 1.2)
                b.add_synapse(h2, mid, -0.8)
            else:
                b.add_synapse(h1, mid, -1.2)
                b.add_synapse(h2, mid, 0.8)

        return b

    def motor_outputs_for_actuators(self) -> Dict[int, float]:
        """
        Returns: {actuator_node_id: thrust [-1,1]}
        """
        out: Dict[int, float] = {}
        for node_id, motor_id in self.actuator_motors.items():
            n = self.neurons.get(motor_id)
            if n is None:
                continue
            out[node_id] = max(-1.0, min(1.0, n.value))

        # fallback: include any motors that aren't tracked in actuator_motors
        for n in self.neurons.values():
            if n.type == NeuronType.MOTOR and n.node_id is not None and n.node_id not in out:
                out[n.node_id] = max(-1.0, min(1.0, n.value))
        return out
