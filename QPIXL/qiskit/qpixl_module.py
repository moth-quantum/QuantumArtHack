import numpy as np
from qiskit import QuantumCircuit, transpile
from enum import Enum
from QPIXL.qiskit.qpixl import cFRQI

class InjectionPoint(Enum):
    BEFORE_ENCODING = "before"
    DURING_ENCODING = "during"
    AFTER_ENCODING = "after"

class QPIXLModule:
    def __init__(self, image_array, compression=0, name="QPIXLModule", algorithm_qubits=0):
        if not isinstance(image_array, np.ndarray):
            raise TypeError("Expected NumPy array")
        if image_array.ndim != 1:
            raise ValueError("Expected a 1D flattened array")
        if not (0 <= compression <= 100):
            raise ValueError("Compression must be in [0, 100]")

        self.image_array = image_array
        self.compression = compression
        self._name = name
        self._algo_qubits = algorithm_qubits
        self._injections = []

    @property
    def name(self):
        return self._name

    def add_injection(self, when, gate_type, qubits, params=None, condition=None):
        """Injects a standard gate like ry, cx, crx, unitary, etc."""
        self._injections.append({
            "when": InjectionPoint(when),
            "gate": gate_type.lower(),
            "qubits": qubits,
            "params": params or {},
            "cond": condition
        })
        return self

    def add_custom_injection(self, func, when):
        """Injects a custom callable. Signature: func(circuit, idx, angle)"""
        self._injections.append({
            "when": InjectionPoint(when),
            "custom": func
        })
        return self

    def get_circuit(self, optimize=False, verbose=False):
        total_qubits = int(np.log2(len(self.image_array))) + 2 + self._algo_qubits
        circuit = QuantumCircuit(total_qubits, name=self._name)

        # Inject before encoding
        self._apply_injections(circuit, InjectionPoint.BEFORE_ENCODING, total_qubits)

        # Encode angles with QPIXL
        base = cFRQI(self.image_array, self.compression)

        # Inject during encoding (per angle)
        for i, angle in enumerate(self.image_array):
            for inj in self._injections:
                if inj.get("when") != InjectionPoint.DURING_ENCODING:
                    continue
                if "custom" in inj:
                    inj["custom"](circuit, i, angle)
                else:
                    q = self._resolve_qubits(inj["qubits"], total_qubits)
                    if not inj.get("cond") or inj["cond"](i, angle):
                        self._apply_gate(circuit, inj["gate"], q, inj["params"])

        # Combine encoded circuit
        circuit.compose(base, inplace=True)

        # Inject after encoding
        self._apply_injections(circuit, InjectionPoint.AFTER_ENCODING, total_qubits)

        if optimize:
            circuit = transpile(circuit, optimization_level=3)

        if verbose:
            print(f"[QPIXL] qubits={circuit.num_qubits}, depth={circuit.depth()}, compression={self.compression}")

        return circuit.copy()

    def _apply_injections(self, circuit, when, total_qubits):
        for inj in self._injections:
            if inj.get("when") != when:
                continue
            if "custom" in inj:
                inj["custom"](circuit, None, None)
            else:
                q = self._resolve_qubits(inj["qubits"], total_qubits)
                if not inj.get("cond") or inj["cond"](None, None):
                    self._apply_gate(circuit, inj["gate"], q, inj["params"])

    def _resolve_qubits(self, qubit_ids, total):
        return [q if q >= 0 else total + q for q in qubit_ids]

    def _apply_gate(self, circuit, gate, qubits, params):
        theta = params.get("angle", 0)
        if gate == "ry":
            circuit.ry(theta, qubits[0])
        elif gate == "cry":
            circuit.cry(theta, *qubits)
        elif gate == "crx":
            circuit.crx(theta, *qubits)
        elif gate == "cz":
            circuit.cz(*qubits)
        elif gate == "cx":
            circuit.cx(*qubits)
        elif gate == "swap":
            circuit.swap(*qubits)
        elif gate == "unitary":
            circuit.unitary(params["matrix"], qubits, label="U")
        else:
            raise ValueError(f"Unsupported gate: {gate}")

