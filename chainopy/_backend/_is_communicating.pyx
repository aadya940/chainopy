cimport numpy as cnp
import numpy as np
import cython

@cython.wraparound(False)
@cython.boundscheck(False)
def _is_partially_communicating(cnp.ndarray[cnp.float64_t, ndim=2] tpm, list states, str state1, str state2, int threshold):
    cdef Py_ssize_t i, idx1, idx2
    cdef cnp.ndarray[cnp.float64_t, ndim=2] x = tpm
    cdef bint result

    if state1 == state2:
        return True

    idx1 = states.index(state1)
    idx2 = states.index(state2)

    for i in range(threshold):
        if x[idx1, idx2] > 0:
            return True
        
        x = x @ tpm

    return False


@cython.wraparound(False)
@cython.boundscheck(False)
def is_communicating_cython(cnp.ndarray[cnp.float64_t, ndim=2] tpm, list states, str state1, str state2, int threshold):
    if (_is_partially_communicating(tpm, states, state1, state2, threshold) \
        and _is_partially_communicating(tpm, states, state2, state1, threshold)):
        return True
    return False