import QPIXL.helper as hlp
import numpy as np
from qiskit import QuantumCircuit


def permutation(j, perm, total_data_qubits):
    j = (j - perm) % total_data_qubits
    return j


def cFRQI_old(data, compression, additional_shift=0):
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
    for ind, a in enumerate(data):
        data[ind] = hlp.convertToAngles(data[ind])  # convert grayscale to angles
        data[ind] = hlp.preprocess_image(
            data[ind]
        )  # need to flatten the transpose for easier decoding,
        # only really necessary if you want to recover an image.
        # for classification tasks etc. transpose isn't required.
        n = len(data[ind])
        k = hlp.ilog2(n)

        data[ind] = 2 * data[ind]
        data[ind] = hlp.sfwht(data[ind])
        data[ind] = hlp.grayPermutation(data[ind])
        a_sort_ind = np.argsort(np.abs(data[ind]))

        # set smallest absolute values of a to zero according to compression param
        cutoff = int((compression / 100.0) * n)
        for it in a_sort_ind[:cutoff]:
            data[ind][it] = 0
    # Construct FRQI circuit
    circuit = QuantumCircuit(k + len(data))
    # Hadamard register
    circuit.h(range(k))
    # Compressed uniformly controlled rotation register
    ctrl, pc, i = 0, 0, 0
    while i < (2**k):
        # Reset the parity check
        pc = int(0)
        # Add RY gate
        for ind, arr in enumerate(data):
            if arr[i] != 0:
                circuit.ry(arr[i], k + ind)
        # Loop over sequence of consecutive zero angles to
        # cancel out CNOTS (or rather, to not include them)
        if i == ((2**k) - 1):
            ctrl = 0
        else:
            ctrl = hlp.grayCode(i) ^ hlp.grayCode(i + 1)
            ctrl = k - hlp.countr_zero(ctrl, n_bits=k + 1) - 1
        # Update parity check
        pc ^= 2**ctrl
        i += 1
        pc_list = []
        for ind, a in enumerate(data):
            while i < (2**k) and a[i] == 0:
                # Compute control qubit
                if i == ((2**k) - 1):
                    ctrl = 0
                else:
                    ctrl = hlp.grayCode(i) ^ hlp.grayCode(i + 1)
                    ctrl = k - hlp.countr_zero(ctrl, n_bits=k + 1) - 1

                # Update parity check
                pc ^= 2**ctrl
                print(f"POTENTIAL BUG HERE: This is the special case in which the PC var is calculated as: {pc}")
                i += 1
            pc_list.append(pc)
        for j in range(k):
            for ind in range(len(data)):
                pc = pc_list[ind]
                if (pc >> j) & 1:
                    circuit.cx(permutation(j, ind + additional_shift, k), k + ind)

    return circuit.reverse_bits()


def cFRQI(data, compression, additional_shift=0):
    """Takes a standard image in a numpy array (so that the matrix looks like
    the image you want if you picture the pixels) and returns the QPIXL
    compressed FRQI circuit. The compression ratio determines
    how many gates will be filtered and then cancelled out. Made into code from this paper:
    https://www.nature.com/articles/s41598-022-11024-y

    Args:
        data: ([np.array, np.array]): array of flattened numpy arrays, must be flattened and padded with zeros up to a power of two
        compression (float): number between 0 an 100, where 0 is no compression and 100 is no circuit

    Returns:
        QuantumCircuit: qiskit circuit that prepared the encoded image
    """
    k_max = 0
    for ind, a in enumerate(data):
        data[ind] = hlp.convertToAngles(data[ind])  # convert grayscale to angles
        n = len(data[ind])
        k = hlp.ilog2(n)

        data[ind] = 2 * data[ind]
        data[ind] = hlp.sfwht(data[ind])
        data[ind] = hlp.grayPermutation(data[ind])
        a_sort_ind = np.argsort(np.abs(data[ind]))

        # set smallest absolute values of a to zero according to compression param
        cutoff = int((compression / 100.0) * n)
        for it in a_sort_ind[:cutoff]:
            data[ind][it] = 0

        # remember the largest K
        k_max = k if k > k_max else k_max
    
    k = k_max
    # Construct FRQI circuit
    circuit = QuantumCircuit(k + len(data))
    # Hadamard register
    circuit.h(range(k))
    # Compressed uniformly controlled rotation register
    ctrl, pc, i = 0, 0, 0

    while i < (2**k):
        # Reset the parity check
        pc = int(0)
        # Add RY gate
        for ind, arr in enumerate(data):
            if i < len(arr) and arr[i] != 0:
                circuit.ry(arr[i], k + ind)
        # Loop over sequence of consecutive zero angles to
        # cancel out CNOTS (or rather, to not include them)
        if i == ((2**k) - 1):
            ctrl = 0
        else:
            ctrl = hlp.grayCode(i) ^ hlp.grayCode(i + 1)
            ctrl = k - hlp.countr_zero(ctrl, n_bits=k + 1) - 1

        # Update parity check
        pc ^= 2**ctrl
        i += 1
        for j in range(k):
            for ind in range(len(data)):
                if (pc >> j) & 1:
                    circuit.cx(permutation(j, ind + additional_shift, k), k + ind)
        pc = 0

        for ind, a in enumerate(data):
            if i < len(a):
                while i < (2**k) and a[i] == 0:
                    # Compute control qubit
                    if i == ((2**k) - 1):
                        ctrl = 0
                    else:
                        ctrl = hlp.grayCode(i) ^ hlp.grayCode(i + 1)
                        ctrl = k - hlp.countr_zero(ctrl, n_bits=k + 1) - 1
                    
                    # Update parity check
                    pc ^= 2**ctrl
                    i += 1
                    for j in range(k):
                        for ind in range(len(data)):
                            if (pc >> j) & 1:
                                circuit.cx(permutation(j, ind + additional_shift, k), k + ind)
                    pc = 0


    return circuit.reverse_bits()

# c = cFRQI(np.random.random((3,2**3)),0)
