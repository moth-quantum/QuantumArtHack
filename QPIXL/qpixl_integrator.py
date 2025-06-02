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
    INTERLEAVED = "interleaved"   # Insert algorithm operations during encoding
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
            mode: Integration mode (merge, sequential, interleaved, entangle, custom)
            connection_map: Dictionary mapping data qubits to algorithm qubits
            custom_rule: Custom function to apply for integration
            **kwargs: Additional arguments for specific integration modes
            
        Returns:
            A quantum circuit combining the data and algorithm circuits
        """
        # Convert string mode to enum if needed
        if isinstance(mode, str):
            try:
                mode = IntegrationMode(mode.lower())
            except ValueError:
                raise ValueError(f"Unknown integration mode: {mode}")
        
        # Default empty connection map if not provided
        if connection_map is None:
            connection_map = {}
            
        # Select integration method based on mode
        if mode == IntegrationMode.MERGE:
            return self._merge_circuits(data_circuit, algorithm_circuit, connection_map, **kwargs)
        elif mode == IntegrationMode.SEQUENTIAL:
            return self._sequential_circuits(data_circuit, algorithm_circuit, **kwargs)
        elif mode == IntegrationMode.INTERLEAVED:
            return self._interleaved_circuits(data_circuit, algorithm_circuit, **kwargs)
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
        # Create a new circuit with combined qubits
        combined = QuantumCircuit(data_circuit.num_qubits + algorithm_circuit.num_qubits)
        
        # Add data circuit to first qubits
        combined = combined.compose(data_circuit, qubits=range(data_circuit.num_qubits))
        
        # Add algorithm circuit to remaining qubits
        algo_qubits = range(data_circuit.num_qubits, 
                          data_circuit.num_qubits + algorithm_circuit.num_qubits)
        combined = combined.compose(algorithm_circuit, qubits=algo_qubits)
        
        # Add connections between circuits if specified
        for data_qubit, algo_qubit in connection_map.items():
            # Adjust algorithm qubit index
            adjusted_algo_qubit = data_circuit.num_qubits + algo_qubit
            # Add CNOT as the basic connection
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
        # Check if circuits have same number of qubits
        if data_circuit.num_qubits != algorithm_circuit.num_qubits:
            raise ValueError(
                f"Circuits must have same number of qubits for sequential integration: "
                f"{data_circuit.num_qubits} vs {algorithm_circuit.num_qubits}"
            )
            
        # Create a copy of the data circuit
        combined = data_circuit.copy()
        
        # Append the algorithm circuit
        combined.compose(algorithm_circuit, inplace=True)
        
        return combined
    
    def _interleaved_circuits(self, 
                             data_circuit: QuantumCircuit,
                             algorithm_circuit: QuantumCircuit,
                             **kwargs) -> QuantumCircuit:
        """
        Create a circuit with algorithm operations interleaved during data preparation.
        
        This is a simplified implementation. For more complex interleaving,
        use QPIXLAlgorithmEncoder instead.
        
        Args:
            data_circuit: QPIXL data circuit
            algorithm_circuit: Algorithm circuit
            
        Returns:
            An interleaved quantum circuit
        """
        # This is a simplified approach - for real interleaving during QPIXL encoding,
        # it's better to use QPIXLAlgorithmEncoder
        
        # Get operations from both circuits
        data_ops = data_circuit.data
        algo_ops = algorithm_circuit.data
        
        # Create a new circuit with the same size as data circuit
        combined = QuantumCircuit(data_circuit.num_qubits)
        
        # Interleave operations with a basic strategy
        data_idx = 0
        algo_idx = 0
        
        # Simple alternating strategy
        while data_idx < len(data_ops) or algo_idx < len(algo_ops):
            # Add a data operation if available
            if data_idx < len(data_ops):
                instr = data_ops[data_idx]
                combined.append(instr.operation, instr.qubits)
                data_idx += 1
            
            # Add an algorithm operation if available
            if algo_idx < len(algo_ops):
                # Map algorithm qubit indices to data circuit qubits
                # This is a simple mapping - may need customization
                instr = algo_ops[algo_idx]
                
                # Skip if algorithm operation uses more qubits than available
                if all(q.index < combined.num_qubits for q in instr.qubits):
                    combined.append(instr.operation, instr.qubits)
                    
                algo_idx += 1
        
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
        # First merge the circuits
        combined = self._merge_circuits(data_circuit, algorithm_circuit, {}, **kwargs)
        
        # If no connections specified, entangle first qubit
        if not connection_map:
            if data_circuit.num_qubits > 0 and algorithm_circuit.num_qubits > 0:
                connection_map = {0: 0}  # Connect first qubit to first algorithm qubit
        
        # Add entanglement operations
        for data_qubit, algo_qubit in connection_map.items():
            # Adjust algorithm qubit index
            adjusted_algo_qubit = data_circuit.num_qubits + algo_qubit
            
            # Get entanglement type from kwargs or use default
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
        # Process the data for QPIXL encoding
        a = hlp.convertToAngles(data)
        a = hlp.preprocess_image(a)
        n = len(a)
        k = hlp.ilog2(n)
        a = 2 * a
        a = hlp.sfwht(a)
        a = hlp.grayPermutation(a)
        
        # Apply compression if requested
        if compression > 0:
            a_sort_ind = np.argsort(np.abs(a))
            cutoff = int((compression / 100.0) * n)
            for it in a_sort_ind[:cutoff]:
                a[it] = 0
        
        # Create quantum registers
        storage_qubits = QuantumRegister(k, "storage")
        encoding_qubit = QuantumRegister(1, "encoding")
        algo_qubits = QuantumRegister(algorithm_qubits, "algorithm")
        
        # Create quantum circuit
        circuit = QuantumCircuit(storage_qubits, encoding_qubit, algo_qubits)
        
        # Apply Hadamard gates to storage qubits
        circuit.h(storage_qubits)
        
        # Perform encoding with algorithm operations
        ctrl, pc, i = 0, 0, 0
        while i < (2**k):
            pc = int(0)  # Reset parity check
            
            # Apply algorithm operations and encoding when angle is non-zero
            if a[i] != 0:
                # Apply algorithm operations if specified
                if algorithm_ops:
                    for op in algorithm_ops:
                        gate_type = op.get('gate')
                        params = op.get('params', {})
                        
                        if gate_type == 'unitary':
                            # Apply unitary gate (like in cFRQI_with_alg_demo)
                            angle = a[i]
                            unitary_matrix = np.array([
                                [np.cos(angle), -1j * np.sin(angle)],
                                [-1j * np.sin(angle), np.cos(angle)]
                            ])
                            circuit.unitary(unitary_matrix, algo_qubits[0], label=f"alg_{i}")
                            
                        elif gate_type == 'cry':
                            # Apply controlled rotation Y gate
                            circuit.cry(a[i], algo_qubits[0], encoding_qubit[0])
                            
                        elif gate_type == 'crx':
                            # Apply controlled rotation X gate
                            circuit.crx(a[i], algo_qubits[0], encoding_qubit[0])
                            
                        elif gate_type == 'custom':
                            # Call custom function with circuit, angle, and index
                            custom_func = params.get('func')
                            if custom_func:
                                custom_func(circuit, a[i], i)
                                
                        # Add more gate types as needed
                
                # Apply the standard encoding rotation if not overridden
                if not any(op.get('gate') == 'cry' for op in algorithm_ops or []):
                    circuit.ry(a[i], encoding_qubit)
            
            # Calculate control qubit
            if i == ((2**k) - 1):
                ctrl = 0
            else:
                ctrl = hlp.grayCode(i) ^ hlp.grayCode(i + 1)
                ctrl = k - hlp.countr_zero(ctrl, n_bits=k + 1) - 1
            
            pc ^= 2**ctrl  # Update parity check
            i += 1
            
            # Skip zero angles
            while i < (2**k) and a[i] == 0:
                if i == ((2**k) - 1):
                    ctrl = 0
                else:
                    ctrl = hlp.grayCode(i) ^ hlp.grayCode(i + 1)
                    ctrl = k - hlp.countr_zero(ctrl, n_bits=k + 1) - 1
                pc ^= 2**ctrl  # Update parity check
                i += 1
            
            # Apply CNOT gates based on parity check
            for j in range(k):
                if (pc >> j) & 1:
                    circuit.cx(storage_qubits[j], encoding_qubit[0])
        
        # Reverse bits
        circuit.reverse_bits()
        
        # Apply post-processing if specified
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
    def pattern_func(circ):
        """Pattern function to apply the specified gate to the circuit."""
        gate_name_lower = gate_name.lower()
        
        if gate_name_lower == 'rx':
            circ.rx(params.get('angle', 0), params.get('target', 0))
        
        elif gate_name_lower == 'ry':
            circ.ry(params.get('angle', 0), params.get('target', 0))
        
        elif gate_name_lower == 'rz':
            circ.rz(params.get('angle', 0), params.get('target', 0))
        
        elif gate_name_lower == 'crx':
            circ.crx(
                params.get('angle', 0), 
                params.get('control', 1), 
                params.get('target', 0)
            )
        
        elif gate_name_lower == 'cry':
            circ.cry(
                params.get('angle', 0), 
                params.get('control', 1), 
                params.get('target', 0)
            )
        
        elif gate_name_lower == 'crz':
            circ.crz(
                params.get('angle', 0), 
                params.get('control', 1), 
                params.get('target', 0)
            )
        
        elif gate_name_lower == 'cx' or gate_name_lower == 'cnot':
            circ.cx(
                params.get('control', 1), 
                params.get('target', 0)
            )
            
        elif gate_name_lower == 'h':
            circ.h(params.get('target', 0))
            
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
    def post_process(circuit):
        """Apply the specified operations to the circuit."""
        for op in operations:
            gate = op.get('gate', '').lower()
            qubits = op.get('qubits', [0])
            params = op.get('params', {})
            
            for q in qubits:
                if gate == 'h':
                    circuit.h(q)
                elif gate == 'x':
                    circuit.x(q)
                elif gate == 'y':
                    circuit.y(q)
                elif gate == 'z':
                    circuit.z(q)
                elif gate == 'rx':
                    circuit.rx(params.get('angle', 0), q)
                elif gate == 'ry':
                    circuit.ry(params.get('angle', 0), q)
                elif gate == 'rz':
                    circuit.rz(params.get('angle', 0), q)
                # Add more gate types as needed
    
    return post_process


# Example usage functions
def example_algorithm_operations():
    """Create example algorithm operations for QPIXLAlgorithmEncoder."""
    return [
        {
            'gate': 'unitary',
            'params': {},  # Angle will be taken from the data
        },
        {
            'gate': 'cry',
            'params': {},  # Angle will be taken from the data
        }
    ]


def example_custom_alg_function(circuit, angle, index):
    """Example custom algorithm function for QPIXLAlgorithmEncoder."""
    # Get the algorithm qubit(s)
    algo_qubits = circuit.qregs[2]  # Assuming algorithm qubits are in the third register
    encoding_qubit = circuit.qregs[1]  # Assuming encoding qubit is in the second register
    
    # Apply some gates based on the angle and index
    circuit.ry(angle / 2, algo_qubits[0])
    circuit.cx(algo_qubits[0], encoding_qubit[0])


if __name__ == "__main__":
    print("QPIXL Circuit Integrator module loaded successfully.")
    print("Use CircuitIntegrator and QPIXLAlgorithmEncoder classes to integrate quantum circuits.")