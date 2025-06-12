# quantum_composer.py

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ControlFlowOp
from typing import List, Dict, Optional, Union, Sequence, Any, Callable
import warnings
import time
from abc import ABC, abstractmethod
from itertools import zip_longest
import numpy as np # Added for create_sliced_circuit

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
            regs = self._circuit.find_bit(q).registers
            name = regs[0][0].name if regs else "ungrouped_qubits"
            roles.setdefault(name, []).append(q)
        for c in self._circuit.clbits:
            regs = self._circuit.find_bit(c).registers
            name = regs[0][0].name + "_classical" if regs else "ungrouped_clbits"
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

    # --- RULE: sequential ---
    def _combine_sequential(self, _: Dict[str, Any]) -> QuantumCircuit:
        if not self.modules:
            return QuantumCircuit()
        qubits, clbits = [], []
        qset, cset = set(), set()
        for m in self.modules:
            qc = m.get_circuit()
            print(f"  ↪ {m.name}")
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
            qmap = {q: q for q in qc.qubits}
            cmap = {c: c for c in qc.clbits}
            for inst in qc.data:
                combined.append(inst.operation,
                                [qmap[q] for q in inst.qubits],
                                [cmap[c] for c in inst.clbits])
        return combined

    # --- RULE: merge ---
    def _combine_merge(self, kwargs: Dict[str, Any]) -> QuantumCircuit:
        if len(self.modules) < 2:
            raise ValueError("Merge needs ≥2 modules")
        connection_map = kwargs.get("connection_map", {})
        etype = kwargs.get("entangle_type", "cx")
        combined = self._combine_sequential({})
        for src, tgt in connection_map.items():
            q_src = combined.qubits[src]
            q_tgt = combined.qubits[tgt]
            if q_src is q_tgt: continue
            if etype == "cx":
                combined.cx(q_src, q_tgt)
            elif etype == "cz":
                combined.cz(q_src, q_tgt)
            elif etype == "swap":
                combined.swap(q_src, q_tgt)
            else:
                raise ValueError(f"Unsupported entangle_type: {etype}")
        return combined

    # --- RULE: hardware_aware_sequential ---
    def _combine_hw_aware(self, kwargs: Dict[str, Any]) -> QuantumCircuit:
        backend        = kwargs.get("backend")
        cmap         = kwargs.get("coupling_map")
        gates        = kwargs.get("basis_gates")
        opt          = kwargs.get("optimization_level", 1)
        raw = self._combine_sequential({})
        if raw.num_qubits == 0:
            return raw
        return qiskit.transpile(raw,
                                 backend=backend,
                                 coupling_map=cmap,
                                 basis_gates=gates,
                                 optimization_level=opt)

    # --- RULE: slice ---
    def _combine_slice(self, kwargs: Dict[str, Any]) -> QuantumCircuit:
        # Grab parameters
        slice_size = kwargs.get("slice_size", 3)
        connection_rule = kwargs.get("connection_rule", "cx")
        # Updated connection_map to specify module qubits: [(data_q, algo_q)]
        connection_map = kwargs.get("connection_map", [(0, 0)])  # Default connects q0 of each
        if len(self.modules) != 2:
            raise ValueError("Slice rule currently only supports exactly 2 modules")
        data_mod, algo_mod = self.modules
        # Get circuits
        data_qc = data_mod.get_circuit()
        algo_qc = algo_mod.get_circuit()
        # Total qubits = sum of both circuits' qubits
        total_qubits = data_qc.num_qubits + algo_qc.num_qubits
        combined = QuantumCircuit(total_qubits, name=f"Sliced_{data_mod.name}__{algo_mod.name}")
        # Slice into chunks
        def chunk_circuit(qc: QuantumCircuit):
            chunks = []
            for i in range(0, len(qc.data), slice_size):
                sub = QuantumCircuit(qc.num_qubits, qc.num_clbits)
                for inst in qc.data[i : i + slice_size]:
                    sub.append(inst.operation, inst.qubits, inst.clbits)
                chunks.append(sub)
            return chunks
        data_chunks = chunk_circuit(data_qc)
        algo_chunks = chunk_circuit(algo_qc)
        # Interleave with separate qubit ranges
        data_offset = 0
        algo_offset = data_qc.num_qubits
        for d_chunk, a_chunk in zip_longest(data_chunks, algo_chunks, fillvalue=None):
            if d_chunk:
                combined.compose(d_chunk, qubits=range(data_offset, data_offset + d_chunk.num_qubits), inplace=True)
            if a_chunk:
                combined.compose(a_chunk, qubits=range(algo_offset, algo_offset + a_chunk.num_qubits), inplace=True)
            # Apply connections between modules
            for dq, aq in connection_map:
                global_dq = data_offset + dq
                global_aq = algo_offset + aq
                if connection_rule == "cx":
                    combined.cx(global_dq, global_aq)
                elif connection_rule == "cz":
                    combined.cz(global_dq, global_aq)
                elif connection_rule == "swap":
                    combined.swap(global_dq, global_aq)
                else:
                    raise ValueError(f"Unknown connection_rule: {connection_rule}")
        return combined


# Added for create_sliced_circuit
from QPIXL.qiskit.qpixl import cFRQI
from QPIXL.helper import (
    preprocess_image,
    convertToAngles,
    grayPermutation,
    grayCode,
    sfwht,
    ilog2,
    countr_zero,
)
import random

def create_sliced_circuit(
    image_array: np.ndarray,
    compression: float,
    algorithm_circuit: QuantumCircuit,
    slice_size: int = 3,
    insertion_rule: Union[str, Callable[[int, float], bool]] = "interval",
    connection_rule: Union[str, Callable[[int], str]] = "cx",
    connection_map: dict = None,
    name: str = "QPIXL_SlicedComposer",
    interval: int = 4,
    angle_threshold: float = 0.5,
    verbose: bool = True,
) -> QuantumCircuit:
    """
    Interleaves an algorithm into QPIXL encoding using rule-based or custom logic.
    Supports dynamic slice injection and qubit entanglement strategies.

    Args:
        image_array: 1D flattened image data
        compression: Percent compression (0-100)
        algorithm_circuit: Qiskit circuit to inject
        slice_size: Number of gates per slice
        insertion_rule: 'interval', 'angle_threshold', or a callable
        connection_rule: 'cx', 'cz', 'swap', 'random', or a callable
        connection_map: {data_qubit: algo_qubit}
        name: Name for the final circuit
        interval: Used if insertion_rule is 'interval'
        angle_threshold: Used if insertion_rule is 'angle_threshold'
        verbose: Print injection logs
    """
    # -- Preprocess QPIXL angles
    a = convertToAngles(image_array)
    a = preprocess_image(a)
    n = len(a)
    k = ilog2(n)
    a = 2 * a
    a = sfwht(a)
    a = grayPermutation(a)

    if compression > 0:
        cutoff = int((compression / 100.0) * n)
        a[np.argsort(np.abs(a))[:cutoff]] = 0

    storage_qubits = k
    encoding_qubit = 1
    algo_qubits = algorithm_circuit.num_qubits
    total_qubits = storage_qubits + encoding_qubit + algo_qubits
    circuit = QuantumCircuit(total_qubits, name=name)

    # -- Prepare slices
    algo_slices = []
    data = algorithm_circuit.data
    for i in range(0, len(data), slice_size):
        sub = QuantumCircuit(algo_qubits)
        for inst in data[i:i + slice_size]:
            sub.append(inst[0], inst[1], inst[2])
        algo_slices.append(sub)

    slice_idx = 0
    inserted_log = []

    # -- Begin encoding with injections
    circuit.h(range(storage_qubits))
    ctrl, pc, i = 0, 0, 0

    def should_insert(idx, angle):
        if callable(insertion_rule):
            return insertion_rule(idx, angle)
        if insertion_rule == "interval":
            return idx % interval == 0
        if insertion_rule == "angle_threshold":
            return abs(angle) > angle_threshold
        return False

    def pick_connection_type(idx):
        if callable(connection_rule):
            return connection_rule(idx)
        if connection_rule == "random":
            return random.choice(["cx", "cz", "swap"])
        return connection_rule

    while i < 2**k:
        pc = 0
        if a[i] != 0:
            circuit.ry(a[i], storage_qubits)

            if slice_idx < len(algo_slices) and should_insert(i, a[i]):
                slice_circuit = algo_slices[slice_idx]
                circuit.compose(
                    slice_circuit,
                    qubits=range(storage_qubits + 1, total_qubits),
                    inplace=True
                )
                if connection_map:
                    for dq, aq in connection_map.items():
                        q1 = dq
                        q2 = storage_qubits + 1 + aq
                        gate = pick_connection_type(slice_idx)
                        if gate == "cx":
                            circuit.cx(q1, q2)
                        elif gate == "cz":
                            circuit.cz(q1, q2)
                        elif gate == "swap":
                            circuit.swap(q1, q2)
                inserted_log.append((i, slice_idx))
                slice_idx += 1

        if i == 2**k - 1:
            ctrl = 0
        else:
            ctrl = grayCode(i) ^ grayCode(i + 1)
            ctrl = k - countr_zero(ctrl, n_bits=k + 1) - 1
        pc ^= 2**ctrl
        i += 1
        while i < 2**k and a[i] == 0:
            if i == 2**k - 1:
                ctrl = 0
            else:
                ctrl = grayCode(i) ^ grayCode(i + 1)
                ctrl = k - countr_zero(ctrl, n_bits=k + 1) - 1
            pc ^= 2**ctrl
            i += 1
        for j in range(k):
            if (pc >> j) & 1:
                circuit.cx(j, storage_qubits)

    circuit.reverse_bits()

    if verbose:
        print(f"[QPIXL] Interleaved {slice_idx} slices:")
        for i, s in inserted_log:
            print(f"  ↪ At index {i}: inserted slice {s}")

    return circuit
