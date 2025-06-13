"""
Enhanced Audio Processing with QPIXL Circuit Integration

This module provides functions for processing audio data using quantum circuits
with sliced algorithm integration.
"""

import os
import numpy as np
import soundfile
from itertools import chain
from typing import Optional, Callable, Dict, Union, List

import qiskit
from qiskit import QuantumCircuit
from qiskit_aer import Aer

import QPIXL.helper as hlp
from QPIXL.qiskit.qpixl import cFRQI
from QPIXL.qiskit.qpixl_angs import cFRQIangs, decodeAngQPIXL
from QPIXL.qpixl_integrator import CircuitIntegrator, IntegrationMode, QPIXLAlgorithmEncoder


def process_audio_with_slices(
    input_file: str,
    output_dir: str,
    algorithm_circuit: QuantumCircuit,
    slice_size: int = 3,
    insertion_rule: str = 'interval',
    interval: int = 8,
    connection_rule: str = 'cx',
    connection_map: Optional[Dict[int, int]] = None,
    compression: int = 0,
    tag: str = "",
) -> None:
    """
    Process audio with sliced quantum algorithm integration.
    
    Args:
        input_file: Path to the input audio file
        output_dir: Directory to save processed audio
        algorithm_circuit: Algorithm circuit to slice and interleave
        slice_size: Number of gates per slice
        insertion_rule: Rule for where to insert slices ('interval', 'angles', 'custom')
        interval: Insert a slice every N loop iterations (if insertion_rule='interval')
        connection_rule: How to connect circuits ('cx', 'cz', 'swap')
        connection_map: Dictionary mapping data qubits to algorithm qubits
        compression: Compression level (0-100)
        tag: Tag to append to output files
    """
    os.makedirs(output_dir, exist_ok=True)
    file_name, _ = os.path.splitext(os.path.basename(input_file))
    metadata_file_path = os.path.join(
        output_dir, f"{file_name}_metadata_{tag}_c{compression}.txt"
    )
    
    if not os.path.exists(metadata_file_path):
        data, samplerate = soundfile.read(input_file)
        chunk_size = 512
        sections = [
            hlp.pad_0(data[i : i + chunk_size]) for i in range(0, len(data), chunk_size)
        ][:-1]  
        
        decoded = []
        encoder = QPIXLAlgorithmEncoder()
        
        for index, section in enumerate(sections):
            print(
                f"Processing {file_name} chunk: {index + 1}/{len(sections)}", end="\r"
            )
            
            normalized_section = section - np.min(section)
            
            qc = encoder.create_sliced_circuit(
                normalized_section,
                algorithm_circuit,
                compression=compression,
                slice_size=slice_size,
                insertion_rule=insertion_rule,
                interval=interval,
                connection_rule=connection_rule,
                connection_map=connection_map,
                algorithm_qubits=algorithm_circuit.num_qubits
            )
            
            backend = Aer.get_backend('statevector_simulator')
            job = backend.run(qc)
            state_vector = np.real(job.result().get_statevector())
            
            decoded.append(
                decodeAngQPIXL(
                    state=state_vector,
                    qc=qc,
                    trace=algorithm_circuit.num_qubits,  # Trace out algorithm qubits
                    max_pixel_val=section.max(),
                    min_pixel_val=section.min(),
                )
            )
        
        decoded_full = np.array(list(chain.from_iterable(decoded)))
        soundfile.write(
            os.path.join(output_dir, f"{file_name}_output_{tag}_c{compression}.wav"),
            decoded_full,
            samplerate,
        )
        
        with open(metadata_file_path, 'w') as f:
            f.write(f"Original file: {input_file}\n")
            f.write(f"Compression: {compression}%\n")
            f.write(f"Slice size: {slice_size}\n")
            f.write(f"Insertion rule: {insertion_rule}\n")
            f.write(f"Connection rule: {connection_rule}\n")


def create_algorithm_circuit(circuit_type: str, num_qubits: int = 3, **params) -> QuantumCircuit:
    """
    Create a simple algorithm circuit for demonstration.
    
    Args:
        circuit_type: Type of circuit ('hadamard', 'qaoa', 'qft', 'custom')
        num_qubits: Number of qubits
        **params: Additional parameters
        
    Returns:
        A quantum circuit
    """
    circuit = QuantumCircuit(num_qubits)
    
    if circuit_type == 'hadamard':
        for i in range(num_qubits):
            circuit.h(i)
    
    elif circuit_type == 'qaoa':
        angle = params.get('angle', np.pi/4)
        for i in range(num_qubits):
            circuit.h(i)
        for i in range(num_qubits-1):
            circuit.cx(i, i+1)
            circuit.rz(angle, i+1)
            circuit.cx(i, i+1)
        for i in range(num_qubits):
            circuit.rx(angle, i)
    
    elif circuit_type == 'qft':
        for i in range(num_qubits):
            circuit.h(i)
            for j in range(i+1, num_qubits):
                circuit.cp(np.pi/float(2**(j-i)), i, j)
    
    elif circuit_type == 'custom':
        gates = params.get('gates', [])
        for gate in gates:
            name = gate.get('name', '')
            qubits = gate.get('qubits', [0])
            params = gate.get('params', {})
            
            if hasattr(circuit, name):
                if params:
                    getattr(circuit, name)(params.get('angle', 0), *qubits)
                else:
                    getattr(circuit, name)(*qubits)
    
    return circuit