"""
QPIXL Circuit Integrator

A flexible module for integrating quantum circuits with QPIXL-encoded data circuits.
"""

import numpy as np
from enum import Enum
from typing import Callable, Dict, List, Optional, Union, Tuple, Any
from qiskit import QuantumCircuit, QuantumRegister
import QPIXL.helper as hlp


class IntegrationMode(Enum):
    """Enum defining different modes of circuit integration."""
    MERGE = "merge"               # Combine circuits side by side
    SEQUENTIAL = "sequential"     # Apply circuits one after another
    ENTANGLE = "entangle"         # Connect circuits with entangling operations
    CUSTOM = "custom"             # User-defined custom integration rule


class CircuitIntegrator:
    """
    A class for integrating quantum circuits with QPIXL-encoded data circuits.
    
    This provides a flexible mechanism to combine arbitrary quantum circuits
    with QPIXL-encoded data circuits using various integration strategies.
    """
    
    def integrate(self, 
                 data_circuit: QuantumCircuit,
                 algorithm_circuit: QuantumCircuit,
                 mode: Union[str, IntegrationMode] = IntegrationMode.MERGE,
                 connection_map: Optional[Dict[int, int]] = None,
                 custom_rule: Optional[Callable] = None,
                 **kwargs) -> QuantumCircuit:
        """
        Integrate a QPIXL data circuit with an algorithm circuit.
        
        Args:
            data_circuit: The QPIXL-encoded data circuit
            algorithm_circuit: An arbitrary quantum circuit
            mode: Integration mode (merge, sequential, entangle, custom)
            connection_map: Dictionary mapping data qubits to algorithm qubits
            custom_rule: Custom function to apply for integration
            **kwargs: Additional arguments for specific integration modes
            
        Returns:
            A quantum circuit combining the data and algorithm circuits
        """
        if isinstance(mode, str):
            try:
                mode = IntegrationMode(mode.lower())
            except ValueError:
                raise ValueError(f"Unknown integration mode: {mode}")
        
        if connection_map is None:
            connection_map = {}
            
        if mode == IntegrationMode.MERGE:
            return self._merge_circuits(data_circuit, algorithm_circuit, connection_map, **kwargs)
        elif mode == IntegrationMode.SEQUENTIAL:
            return self._sequential_circuits(data_circuit, algorithm_circuit, **kwargs)
        elif mode == IntegrationMode.ENTANGLE:
            return self._entangle_circuits(data_circuit, algorithm_circuit, connection_map, **kwargs)
        elif mode == IntegrationMode.CUSTOM:
            if custom_rule is None:
                raise ValueError("For CUSTOM mode, a custom_rule function must be provided")
            return custom_rule(data_circuit, algorithm_circuit, **kwargs)
        else:
            raise ValueError(f"Unsupported integration mode: {mode}")
    
    def _merge_circuits(self, 
                       data_circuit: QuantumCircuit, 
                       algorithm_circuit: QuantumCircuit,
                       connection_map: Dict[int, int],
                       **kwargs) -> QuantumCircuit:
        """
        Merge circuits side by side and optionally add connections between them.
        
        Args:
            data_circuit: QPIXL data circuit
            algorithm_circuit: Algorithm circuit
            connection_map: Dictionary mapping data qubits to algorithm qubits
            
        Returns:
            A merged quantum circuit
        """
        combined = QuantumCircuit(data_circuit.num_qubits + algorithm_circuit.num_qubits)
        
        combined = combined.compose(data_circuit, qubits=range(data_circuit.num_qubits))
        
        algo_qubits = range(data_circuit.num_qubits, 
                          data_circuit.num_qubits + algorithm_circuit.num_qubits)
        combined = combined.compose(algorithm_circuit, qubits=algo_qubits)
        
        for data_qubit, algo_qubit in connection_map.items():
            adjusted_algo_qubit = data_circuit.num_qubits + algo_qubit
            combined.cx(data_qubit, adjusted_algo_qubit)
        
        return combined
    
    def _sequential_circuits(self, 
                           data_circuit: QuantumCircuit, 
                           algorithm_circuit: QuantumCircuit,
                           **kwargs) -> QuantumCircuit:
        """
        Apply circuits sequentially (data circuit followed by algorithm circuit).
        
        Args:
            data_circuit: QPIXL data circuit
            algorithm_circuit: Algorithm circuit
            
        Returns:
            A sequential quantum circuit
        """
        if data_circuit.num_qubits != algorithm_circuit.num_qubits:
            raise ValueError(
                f"Circuits must have same number of qubits for sequential integration: "
                f"{data_circuit.num_qubits} vs {algorithm_circuit.num_qubits}"
            )
            
        combined = data_circuit.copy()
        
        combined.compose(algorithm_circuit, inplace=True)
        
        return combined
    
    def _entangle_circuits(self, 
                          data_circuit: QuantumCircuit,
                          algorithm_circuit: QuantumCircuit,
                          connection_map: Dict[int, int],
                          **kwargs) -> QuantumCircuit:
        """
        Combine circuits and add entanglement operations between them.
        
        Args:
            data_circuit: QPIXL data circuit
            algorithm_circuit: Algorithm circuit
            connection_map: Dictionary mapping data qubits to algorithm qubits
            
        Returns:
            An entangled quantum circuit
        """
        combined = self._merge_circuits(data_circuit, algorithm_circuit, {}, **kwargs)
        
        if not connection_map:
            if data_circuit.num_qubits > 0 and algorithm_circuit.num_qubits > 0:
                connection_map = {0: 0}  
        
        for data_qubit, algo_qubit in connection_map.items():
            adjusted_algo_qubit = data_circuit.num_qubits + algo_qubit
            
            entangle_type = kwargs.get('entangle_type', 'cx')
            
            if entangle_type == 'cx':
                combined.cx(data_qubit, adjusted_algo_qubit)
            elif entangle_type == 'cz':
                combined.cz(data_qubit, adjusted_algo_qubit)
            elif entangle_type == 'swap':
                combined.swap(data_qubit, adjusted_algo_qubit)
            else:
                raise ValueError(f"Unsupported entanglement type: {entangle_type}")
        
        return combined


class QPIXLAlgorithmEncoder:
    """
    A class to create QPIXL circuits with integrated algorithm operations.
    
    This allows algorithm operations to be injected during the QPIXL encoding process,
    similar to the cFRQI_with_alg_demo function in the QPIXL_demo notebook.
    """
    
    def create_circuit(self, 
                     data: np.ndarray, 
                     compression: float = 0,
                     algorithm_ops: Optional[List[Dict[str, Any]]] = None,
                     post_processing: Optional[Callable] = None,
                     algorithm_qubits: int = 1) -> QuantumCircuit:
        """
        Create a QPIXL circuit with integrated algorithm operations.
        
        Args:
            data: Input data array to encode
            compression: Compression ratio (0-100)
            algorithm_ops: List of algorithm operations to apply
                Each operation is a dict with:
                - 'gate': Gate type ('unitary', 'cry', etc.)
                - 'params': Parameters for the gate
                - 'qubits': Qubits to apply the gate to (algorithm and encoding)
            post_processing: Function to apply to the circuit after encoding
            algorithm_qubits: Number of qubits to allocate for algorithm
            
        Returns:
            A QPIXL circuit with integrated algorithm operations
        """
        a = hlp.convertToAngles(data)
        a = hlp.preprocess_image(a)
        n = len(a)
        k = hlp.ilog2(n)
        a = 2 * a
        a = hlp.sfwht(a)
        a = hlp.grayPermutation(a)
        
        if compression > 0:
            a_sort_ind = np.argsort(np.abs(a))
            cutoff = int((compression / 100.0) * n)
            for it in a_sort_ind[:cutoff]:
                a[it] = 0
        
        storage_qubits = QuantumRegister(k, "storage")
        encoding_qubit = QuantumRegister(1, "encoding")
        algo_qubits = QuantumRegister(algorithm_qubits, "algorithm")
        
        circuit = QuantumCircuit(storage_qubits, encoding_qubit, algo_qubits)
        
        circuit.h(storage_qubits)
        
        ctrl, pc, i = 0, 0, 0
        while i < (2**k):
            pc = int(0)  
            
            if a[i] != 0:
                if algorithm_ops:
                    for op in algorithm_ops:
                        gate_type = op.get('gate')
                        params = op.get('params', {})
                        
                        if gate_type == 'unitary':
                            angle = a[i]
                            unitary_matrix = np.array([
                                [np.cos(angle), -1j * np.sin(angle)],
                                [-1j * np.sin(angle), np.cos(angle)]
                            ])
                            circuit.unitary(unitary_matrix, algo_qubits[0], label=f"alg_{i}")
                            
                        elif gate_type == 'cry':
                            circuit.cry(a[i], algo_qubits[0], encoding_qubit[0])
                            
                        elif gate_type == 'crx':
                            circuit.crx(a[i], algo_qubits[0], encoding_qubit[0])
                            
                        elif gate_type == 'custom':
                            custom_func = params.get('func')
                            if custom_func:
                                custom_func(circuit, a[i], i)
                                
                if not any(op.get('gate') == 'cry' for op in algorithm_ops or []):
                    circuit.ry(a[i], encoding_qubit)
            
            if i == ((2**k) - 1):
                ctrl = 0
            else:
                ctrl = hlp.grayCode(i) ^ hlp.grayCode(i + 1)
                ctrl = k - hlp.countr_zero(ctrl, n_bits=k + 1) - 1
            
            pc ^= 2**ctrl  
            i += 1
            
            while i < (2**k) and a[i] == 0:
                if i == ((2**k) - 1):
                    ctrl = 0
                else:
                    ctrl = hlp.grayCode(i) ^ hlp.grayCode(i + 1)
                    ctrl = k - hlp.countr_zero(ctrl, n_bits=k + 1) - 1
                pc ^= 2**ctrl  
                i += 1
            
            for j in range(k):
                if (pc >> j) & 1:
                    circuit.cx(storage_qubits[j], encoding_qubit[0])
        
        circuit.reverse_bits()
        
        if post_processing:
            post_processing(circuit)
        
        return circuit


def create_pattern_function(gate_name: str, **params) -> Tuple[Callable, any]:
    """
    Create a pattern function for use with cFRQIangs.
    
    Args:
        gate_name: Name of the gate to apply ('rx', 'ry', 'crx', 'cry', etc.)
        **params: Parameters for the gate (angle, control, target, etc.)
        
    Returns:
        A tuple of (pattern_function, angle)
    """
    gate_ops = {
        'rx': lambda c: c.rx(params.get('angle', 0), params.get('target', 0)),
        'ry': lambda c: c.ry(params.get('angle', 0), params.get('target', 0)),
        'rz': lambda c: c.rz(params.get('angle', 0), params.get('target', 0)),
        'crx': lambda c: c.crx(params.get('angle', 0), params.get('control', 1), params.get('target', 0)),
        'cry': lambda c: c.cry(params.get('angle', 0), params.get('control', 1), params.get('target', 0)),
        'crz': lambda c: c.crz(params.get('angle', 0), params.get('control', 1), params.get('target', 0)),
        'cx': lambda c: c.cx(params.get('control', 1), params.get('target', 0)),
        'cnot': lambda c: c.cx(params.get('control', 1), params.get('target', 0)),
        'h': lambda c: c.h(params.get('target', 0))
    }
    
    def pattern_func(circ):
        """Pattern function to apply the specified gate to the circuit."""
        gate_name_lower = gate_name.lower()
        if gate_name_lower in gate_ops:
            gate_ops[gate_name_lower](circ)
        else:
            raise ValueError(f"Unsupported gate: {gate_name}")
    
    return pattern_func, params.get('angle', 0)


def create_post_processing(operations: List[Dict]) -> Callable:
    """
    Create a post-processing function to apply operations to a circuit.
    
    Args:
        operations: List of operations to apply
            Each operation is a dict with:
            - 'gate': Gate type ('h', 'x', 'rx', etc.)
            - 'qubits': List of qubits to apply the gate to
            - 'params': Parameters for the gate (if needed)
    
    Returns:
        A post-processing function
    """
    gate_ops = {
        'h': lambda c, q, p: c.h(q),
        'x': lambda c, q, p: c.x(q),
        'y': lambda c, q, p: c.y(q),
        'z': lambda c, q, p: c.z(q),
        'rx': lambda c, q, p: c.rx(p.get('angle', 0), q),
        'ry': lambda c, q, p: c.ry(p.get('angle', 0), q),
        'rz': lambda c, q, p: c.rz(p.get('angle', 0), q),
    }
    
    def post_process(circuit):
        """Apply the specified operations to the circuit."""
        for op in operations:
            gate = op.get('gate', '').lower()
            qubits = op.get('qubits', [0])
            params = op.get('params', {})
            
            for q in qubits:
                if gate in gate_ops:
                    gate_ops[gate](circuit, q, params)
    
    return post_process

# ------------------------------------------------------------
# USAGE EXAMPLES
# ------------------------------------------------------------
# To create circuit integrations:
#
# integrator = CircuitIntegrator()
# combined = integrator.integrate(
#     data_circuit,
#     algorithm_circuit,
#     mode=IntegrationMode.MERGE,
#     connection_map={0: 0, 1: 1}
# )
#
# To create QPIXL circuits with algorithm operations:
#
# encoder = QPIXLAlgorithmEncoder()
# circuit = encoder.create_circuit(
#     data,
#     compression=0,
#     algorithm_ops=[{'gate': 'unitary', 'params': {}}]
# )