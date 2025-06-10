import numpy as np
from qiskit import QuantumCircuit, transpile
from QPIXL.qiskit.qpixl import cFRQI
from QPIXL.qiskit.quantum_composer import CircuitModule


class QPIXLModule(CircuitModule):
    """
    Module that wraps QPIXL-encoded image data as a reusable quantum circuit module.
    """

    def __init__(self, image_array: np.ndarray, compression: float = 0, name: str = "QPIXLModule", injected_ops=None):
        if not isinstance(image_array, np.ndarray):
            raise TypeError("image_array must be a NumPy array")
        if len(image_array.shape) != 1:
            raise ValueError("image_array must be a 1D flattened array")
        if not (0 <= compression <= 100):
            raise ValueError("compression must be between 0 and 100")

        self.image_array = image_array
        self.compression = compression
        self._name = name
        self.injected_ops = injected_ops if injected_ops else []

    @property
    def name(self) -> str:
        return self._name

    def get_circuit(self, optimize=False, verbose=False) -> QuantumCircuit:
        """
        Returns a QPIXL-encoded circuit from the image array.
        Can optionally optimize it and include injected gates.
        """
        circuit = cFRQI(self.image_array, self.compression)

        # Inject additional gates
        for op in self.injected_ops:
            gate_type, qubits, *params = op
            if gate_type.lower() == "ry":
                circuit.ry(params[0], qubits[0])
            elif gate_type.lower() == "cz":
                circuit.cz(*qubits)
            elif gate_type.lower() == "cx":
                circuit.cx(*qubits)
            elif gate_type.lower() == "swap":
                circuit.swap(*qubits)
            else:
                raise ValueError(f"Unsupported injected op: {gate_type}")

        # Optimize circuit if requested
        if optimize:
            circuit = transpile(circuit, optimization_level=3)

        # Verbose debug print
        if verbose:
            print(f"[QPIXL] qubits={circuit.num_qubits}, depth={circuit.depth()}, compression={self.compression}")

        return circuit.copy()  # Important: ensure safe usage in Qiskit 2.x strict mode
