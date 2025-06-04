import QPIXL.helper as hlp
import numpy as np
from qiskit import QuantumCircuit


def permutation(j, perm, total_data_qubits):
    j = (j - perm) % total_data_qubits
    return j


def cFRQI(data, compression, additional_shift=0):
    """Takes a standard image in a numpy array (so that the matrix looks like
    the image you want if you picture the pixels) and returns the QPIXL
    compressed FRQI circuit. The compression ratio determines
    how many gates will be filtered and then cancelled out. Made into code from this paper:
    https://www.nature.com/articles/s41598-022-11024-y

    Args:
        a ([np.array, np.array]): array of flattened numpy arrays, must be flattened and padded with zeros up to a power of two
        compression (float): number between 0 an 100, where 0 is no compression and 100 is no circuit

    Returns:
        QuantumCircuit: qiskit circuit that prepared the encoded image
    """
    import numpy as np
    from qiskit import QuantumCircuit
    import QPIXL.helper as hlp

    # Preprocessing each image: convert to angles, flatten, transform
    for ind, a in enumerate(data):
        data[ind] = hlp.convertToAngles(a)
        data[ind] = hlp.preprocess_image(data[ind])
        n = len(data[ind])
        k = hlp.ilog2(n)
        data[ind] = 2 * data[ind]
        data[ind] = hlp.sfwht(data[ind])
        data[ind] = hlp.grayPermutation(data[ind])
        # Apply compression by zeroing small coefficients
        a_sort_ind = np.argsort(np.abs(data[ind]))
        cutoff = int((compression / 100.0) * n)
        for it in a_sort_ind[:cutoff]:
            data[ind][it] = 0

    # Building the circuit with k address qubits and len(data) amplitude qubits
    circuit = QuantumCircuit(k + len(data))
    circuit.h(range(k))  # Hadamards on address register

    i = 0
    while i < (2**k):
        # Apply RY rotations for all images at index i
        for ind, arr in enumerate(data):
            if arr[i] != 0:
                circuit.ry(arr[i], k + ind)

        # Compute Gray-code transition (control qubit) for moving from i to i+1
        if i == (2**k - 1):
            ctrl = 0
        else:
            g = hlp.grayCode(i) ^ hlp.grayCode(i + 1)
            ctrl = k - hlp.countr_zero(g, n_bits=k+1) - 1

        i += 1  # move to next index

        # Build parity list: one parity mask per image
        pc_list = []
        for ind, arr in enumerate(data):
            pc = 0
            # Include the initial Gray-code transition for this image
            if ctrl:
                pc ^= 2**ctrl
            # Skipping over the  runs of zeros for this image
            while i < (2**k) and arr[i] == 0:
                if i == (2**k - 1):
                    next_ctrl = 0
                else:
                    g = hlp.grayCode(i) ^ hlp.grayCode(i + 1)
                    next_ctrl = k - hlp.countr_zero(g, n_bits=k+1) - 1
                pc ^= 2**next_ctrl
                i += 1
            pc_list.append(pc)

        # Applying  CNOTs: loop over each image, then over address bits
        for ind, pc in enumerate(pc_list):
            for j in range(k):
                if (pc >> j) & 1:
                    circuit.cx(j, k + ind)

    return circuit.reverse_bits()
