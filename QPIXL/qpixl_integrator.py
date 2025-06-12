"""
QPIXL Circuit Integrator

A flexible module for integrating quantum circuits with QPIXL-encoded data circuits.
"""

import numpy as np
from enum import Enum
from typing import Callable, Dict, List, Optional, Union, Tuple, Any
from qiskit import QuantumCircuit, QuantumRegister, transpile
import QPIXL.helper as hlp


class IntegrationMode(Enum):
    """Enum defining different modes of circuit integration."""
    MERGE = "merge"               # Combine circuits side by side
    SEQUENTIAL = "sequential"     # Apply circuits one after another
    ENTANGLE = "entangle"         # Connect circuits with entangling operations
    CUSTOM = "custom"             # User-defined custom integration rule
    SLICED = "sliced"             # Slice and interleave circuits


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
            mode: Integration mode (merge, sequential, entangle, custom, sliced)
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
        elif mode == IntegrationMode.SLICED:
            return self._sliced_circuits(data_circuit, algorithm_circuit, connection_map, **kwargs)
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

    def _slice_circuit(self, 
                    circuit: QuantumCircuit, 
                    slice_size: int) -> List[QuantumCircuit]:
        """
        Slice a quantum circuit into segments of a specified size.
        
        Args:
            circuit: The circuit to slice
            slice_size: Number of gates per slice
            
        Returns:
            List of circuit slices
        """
        circuit_ops = circuit.data
        
        slices = []
        for i in range(0, len(circuit_ops), slice_size):
            end_idx = min(i + slice_size, len(circuit_ops))
            slice_circuit = QuantumCircuit(circuit.num_qubits)
            
            for op in circuit_ops[i:end_idx]:
                instruction = op[0]
                qargs = op[1]
                cargs = op[2] if len(op) > 2 else []
                
                # Map qubits to their indices
                qbit_indices = [circuit.qubits.index(q) for q in qargs]
                
                # Append the instruction to the new circuit
                slice_circuit.append(instruction, qbit_indices, cargs)
                
            slices.append(slice_circuit)
            
        return slices
    
    def _sliced_circuits(self,
                    data_circuit: QuantumCircuit,
                    algorithm_circuit: QuantumCircuit,
                    connection_map: Dict[int, int],
                    **kwargs) -> QuantumCircuit:
        """
        Slice algorithm circuit and interleave slices with data circuit.
        
        Args:
            data_circuit: QPIXL data circuit
            algorithm_circuit: Algorithm circuit to be sliced
            connection_map: Dictionary mapping data qubits to algorithm qubits
            slice_size: Number of gates per slice (default: 3)
            insertion_rule: Rule for where to insert slices ('interval', 'angles', etc.)
            interval: Insert a slice every N operations (if insertion_rule='interval')
            connection_rule: Rule to apply at each insertion point (default: 'cx')
            
        Returns:
            An integrated circuit with interleaved slices
        """
        slice_size = kwargs.get('slice_size', 3)
        connection_rule = kwargs.get('connection_rule', 'cx')
        insertion_rule = kwargs.get('insertion_rule', 'interval')
        interval = kwargs.get('interval', 8)
        
        slices = self._slice_circuit(algorithm_circuit, slice_size)
        
        # Create a new circuit with all qubits from both circuits
        total_qubits = data_circuit.num_qubits + algorithm_circuit.num_qubits
        combined = QuantumCircuit(total_qubits)
        
        # Determine where to insert algorithm slices
        if insertion_rule == 'interval':
            # Create insertion points at regular intervals
            num_ops = len(data_circuit.data)
            insertion_points = list(range(0, num_ops, interval))
            if not insertion_points:
                insertion_points = [0]
        else:
            # Default: evenly distribute slices throughout the circuit
            num_ops = len(data_circuit.data)
            if len(slices) > 1:
                step = max(1, num_ops // len(slices))
                insertion_points = list(range(0, num_ops, step))
                if len(insertion_points) > len(slices):
                    insertion_points = insertion_points[:len(slices)]
            else:
                insertion_points = [num_ops // 2]  # Insert in the middle
        
        # Split the data circuit at insertion points
        segments = []
        last_point = 0
        for point in insertion_points + [len(data_circuit.data)]:
            if point > last_point:
                segment = QuantumCircuit(data_circuit.num_qubits)
                for i in range(last_point, point):
                    if i < len(data_circuit.data):
                        inst, qargs, cargs = data_circuit.data[i]
                        segment.append(inst, qargs, cargs)
                segments.append(segment)
                last_point = point
        
        # Build the combined circuit by alternating segments and slices
        data_qubits = list(range(data_circuit.num_qubits))
        algo_qubits = list(range(data_circuit.num_qubits, total_qubits))
        
        current_slice = 0
        for i, segment in enumerate(segments[:-1]):  # Skip the last segment for now
            # Add the data circuit segment
            combined.compose(segment, data_qubits, inplace=True)
            
            # Add the algorithm slice if available
            if current_slice < len(slices):
                slice_circ = slices[current_slice]
                combined.compose(slice_circ, algo_qubits[:slice_circ.num_qubits], inplace=True)
                
                # Add connections between data and algorithm qubits
                for data_qubit, algo_qubit in connection_map.items():
                    adjusted_algo_qubit = data_circuit.num_qubits + algo_qubit
                    
                    if connection_rule == 'cx':
                        combined.cx(data_qubit, adjusted_algo_qubit)
                    elif connection_rule == 'cz':
                        combined.cz(data_qubit, adjusted_algo_qubit)
                    elif connection_rule == 'swap':
                        combined.swap(data_qubit, adjusted_algo_qubit)
                
                current_slice += 1
        
        # Add the final segment of the data circuit
        if segments:
            combined.compose(segments[-1], data_qubits, inplace=True)
        
        # Add any remaining algorithm slices at the end
        while current_slice < len(slices):
            slice_circ = slices[current_slice]
            combined.compose(slice_circ, algo_qubits[:slice_circ.num_qubits], inplace=True)
            
            # Add connections between data and algorithm qubits
            for data_qubit, algo_qubit in connection_map.items():
                adjusted_algo_qubit = data_circuit.num_qubits + algo_qubit
                
                if connection_rule == 'cx':
                    combined.cx(data_qubit, adjusted_algo_qubit)
                elif connection_rule == 'cz':
                    combined.cz(data_qubit, adjusted_algo_qubit)
                elif connection_rule == 'swap':
                    combined.swap(data_qubit, adjusted_algo_qubit)
            
            current_slice += 1
        
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

    def create_sliced_circuit(self,
                            data: np.ndarray,
                            algorithm_circuit: QuantumCircuit,
                            compression: float = 0,
                            slice_size: int = 3,
                            insertion_rule: str = 'interval',
                            interval: int = 8,
                            connection_rule: str = 'cx',
                            connection_map: Optional[Dict[int, int]] = None,
                            algorithm_qubits: int = 1) -> QuantumCircuit:
        """
        Create a QPIXL circuit with slices of an algorithm circuit integrated at specific points.
        
        Args:
            data: Input data array to encode
            algorithm_circuit: Circuit to slice and integrate
            compression: Compression ratio (0-100)
            slice_size: Number of gates per slice
            insertion_rule: Rule for where to insert slices ('interval', 'angles', 'custom')
            interval: Insert a slice every N loop iterations (if insertion_rule='interval')
            connection_rule: How to connect circuits ('cx', 'cz', 'swap')
            connection_map: Dictionary mapping data qubits to algorithm qubits
            algorithm_qubits: Number of qubits for algorithm
            
        Returns:
            A QPIXL circuit with integrated algorithm slices
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
        
        integrator = CircuitIntegrator()
        slices = integrator._slice_circuit(algorithm_circuit, slice_size)
        current_slice = 0
        
        if connection_map is None:
            connection_map = {0: 0}  
        
        storage_qubits = QuantumRegister(k, "storage")
        encoding_qubit = QuantumRegister(1, "encoding")
        algo_qubits = QuantumRegister(algorithm_qubits, "algorithm")
        
        circuit = QuantumCircuit(storage_qubits, encoding_qubit, algo_qubits)
        
        circuit.h(storage_qubits)
        
        insert_counter = 0
        insertion_points = []
        
        if insertion_rule == 'custom':
            insertion_points = insertion_points or []
        elif insertion_rule == 'angles':
            insertion_points = [i for i, angle in enumerate(a) if angle != 0]
        
        ctrl, pc, i = 0, 0, 0
        while i < (2**k):
            pc = int(0)
            
            insert_slice = False
            if insertion_rule == 'interval' and insert_counter % interval == 0:
                insert_slice = True
            elif insertion_rule == 'angles' and i in insertion_points:
                insert_slice = True
            elif insertion_rule == 'custom' and i in insertion_points:
                insert_slice = True
            
            if insert_slice and current_slice < len(slices):
                slice_circ = slices[current_slice]
                for j in range(min(slice_circ.num_qubits, len(algo_qubits))):
                    for inst, qargs, cargs in slice_circ.data:
                        # Fix the mapping of qubits - use slice_circuit.qubits.index(q) instead of q.index
                        qbit_indices = [slice_circ.qubits.index(q) for q in qargs]
                        mapped_qargs = [algo_qubits[idx] if idx < len(algo_qubits) else qarg 
                                    for idx, qarg in zip(qbit_indices, qargs)]
                        circuit.append(inst, mapped_qargs, cargs)
                
                for data_qubit, algo_qubit in connection_map.items():
                    if connection_rule == 'cx':
                        circuit.cx(storage_qubits[data_qubit], algo_qubits[algo_qubit])
                    elif connection_rule == 'cz':
                        circuit.cz(storage_qubits[data_qubit], algo_qubits[algo_qubit])
                    elif connection_rule == 'swap':
                        circuit.swap(storage_qubits[data_qubit], algo_qubits[algo_qubit])
                
                current_slice += 1
            
            if a[i] != 0:
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
            
            insert_counter += 1
        
        while current_slice < len(slices):
            slice_circ = slices[current_slice]
            for j in range(min(slice_circ.num_qubits, len(algo_qubits))):
                for inst, qargs, cargs in slice_circ.data:
                    # Same fix as above
                    qbit_indices = [slice_circ.qubits.index(q) for q in qargs]
                    mapped_qargs = [algo_qubits[idx] if idx < len(algo_qubits) else qarg 
                                for idx, qarg in zip(qbit_indices, qargs)]
                    circuit.append(inst, mapped_qargs, cargs)
                
            for data_qubit, algo_qubit in connection_map.items():
                if connection_rule == 'cx':
                    circuit.cx(storage_qubits[data_qubit], algo_qubits[algo_qubit])
                elif connection_rule == 'cz':
                    circuit.cz(storage_qubits[data_qubit], algo_qubits[algo_qubit])
                elif connection_rule == 'swap':
                    circuit.swap(storage_qubits[data_qubit], algo_qubits[algo_qubit])
            
            current_slice += 1
        
        circuit.reverse_bits()
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
#
# To create QPIXL circuits with sliced algorithm integration:
#
# encoder = QPIXLAlgorithmEncoder()
# circuit = encoder.create_sliced_circuit(
#     data,
#     algorithm_circuit,
#     compression=0,
#     slice_size=3,
#     insertion_rule='interval',
#     interval=8
# )