# quantum_composer.py

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ControlFlowOp
from typing import List, Dict, Optional, Union, Sequence, Any, Callable
import warnings
import time
from abc import ABC, abstractmethod

# ----- Base Module -----
class CircuitModule(ABC):
    @abstractmethod
    def get_circuit(self) -> QuantumCircuit:
        pass
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    def get_qubit_roles(self) -> Dict[str, List[qiskit.circuit.Qubit]]:
        return {}
    def get_parameters(self) -> Dict[str, Any]:
        return {}

class QiskitCircuitModule(CircuitModule):
    def __init__(self, circuit: QuantumCircuit, name: Optional[str] = None):
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError("Expected a Qiskit QuantumCircuit")
        self._circuit = circuit.copy()
        self._name = name or circuit.name or "UnnamedCircuit"
    def get_circuit(self) -> QuantumCircuit:
        return self._circuit
    @property
    def name(self) -> str:
        return self._name
    def get_qubit_roles(self) -> Dict[str, List[qiskit.circuit.Qubit]]:
        roles = {}
        for q in self._circuit.qubits:
            reg = self._circuit.find_bit(q).registers
            name = reg[0][0].name if reg else "ungrouped_qubits"
            roles.setdefault(name, []).append(q)
        for c in self._circuit.clbits:
            reg = self._circuit.find_bit(c).registers
            name = reg[0][0].name + "_classical" if reg else "ungrouped_clbits"
            roles.setdefault(name, []).append(c)
        return roles

# ----- Main Composer -----
CombinationRule = Callable[[Dict[str, Any]], Union[QuantumCircuit, Any]]

class QuantumComposer:
    def __init__(self, modules: Sequence[CircuitModule]):
        if not modules:
            raise ValueError("No modules provided")
        self.modules = list(modules)
        self._rules: Dict[str, CombinationRule] = {}
        self._register_rules()

    def _register_rules(self):
        self.register_rule("sequential", self._combine_sequential)
        self.register_rule("merge", self._combine_merge)
        self.register_rule("hardware_aware_sequential", self._combine_hw_aware)
        self.register_rule("slice", self._combine_slice)

    def register_rule(self, name: str, func: CombinationRule):
        if not name or not callable(func):
            raise ValueError("Invalid rule")
        if name in self._rules:
            warnings.warn(f"Overwriting rule: {name}")
        self._rules[name] = func

    def list_rules(self) -> List[str]:
        return list(self._rules.keys())

    def combine(self, rule: str = "sequential", **kwargs) -> Union[QuantumCircuit, Any]:
        if rule not in self._rules:
            raise ValueError(f"Unknown rule '{rule}'. Available: {self.list_rules()}")
        print(f"▶ Combining using rule: {rule}")
        start = time.time()
        out = self._rules[rule](kwargs)
        print(f"✓ Done in {time.time() - start:.2f}s")
        return out

    def _combine_sequential(self, _: Dict[str, Any]) -> QuantumCircuit:
        if not self.modules:
            return QuantumCircuit()
        qubits, clbits = [], []
        qset, cset = set(), set()
        for m in self.modules:
            qc = m.get_circuit()
            for q in qc.qubits:
                if q not in qset:
                    qubits.append(q); qset.add(q)
            for c in qc.clbits:
                if c not in cset:
                    clbits.append(c); cset.add(c)
        combined = QuantumCircuit(qubits, clbits,
                                  name="_then_".join(m.name for m in self.modules))
        for m in self.modules:
            qc = m.get_circuit()
            print(f"  ↪ {m.name}")
            qmap = {q: q for q in qc.qubits}
            cmap = {c: c for c in qc.clbits}
            for inst in qc.data:
                combined.append(inst.operation,
                                [qmap[q] for q in inst.qubits],
                                [cmap[c] for c in inst.clbits])
        return combined

    def _combine_merge(self, kwargs: Dict[str, Any]) -> QuantumCircuit:
        if len(self.modules) < 2:
            raise ValueError("Merge needs ≥2 modules")
        connection_map = kwargs.get("connection_map", {})
        etype = kwargs.get("entangle_type", "cx")

        # 1) Build base circuit using sequential logic
        combined = self._combine_sequential({})

        # 2) Append entanglement on the already-declared qubits
        for src, tgt in connection_map.items():
            q_src = combined.qubits[src]
            q_tgt = combined.qubits[tgt]
            if q_src is q_tgt:
                continue
            if etype == "cx":
                combined.cx(q_src, q_tgt)
            elif etype == "cz":
                combined.cz(q_src, q_tgt)
            elif etype == "swap":
                combined.swap(q_src, q_tgt)
            else:
                raise ValueError(f"Unsupported entangle_type: {etype}")

        return combined

    def _combine_hw_aware(self, kwargs: Dict[str, Any]) -> QuantumCircuit:
        backend = kwargs.get("backend")
        cmap = kwargs.get("coupling_map")
        gates = kwargs.get("basis_gates")
        opt = kwargs.get("optimization_level", 1)
        raw = self._combine_sequential({})
        if raw.num_qubits == 0:
            return raw
        return qiskit.transpile(raw,
                                backend=backend,
                                coupling_map=cmap,
                                basis_gates=gates,
                                optimization_level=opt)

    def _combine_slice(self, _: Dict[str, Any]) -> Any:
        raise NotImplementedError(
            "The 'slice' rule requires circuit-knitting-toolbox,\n"
            "which is not compatible with Qiskit 2.x.\n"
            "Please use Qiskit <2.0 for slicing support."
        )
