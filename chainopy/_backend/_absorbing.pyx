import numpy as np
cimport numpy as cnp
import cython
from cython.parallel import prange


@cython.wraparound(False)
@cython.boundscheck(False)
def _absorbing_states_indices(double[:, :] tpm):
    cdef list absorbing_states = []
    cdef int num_states = tpm.shape[0]
    cdef Py_ssize_t state_idx
    with cython.nogil:
        for state_idx in prange(0, num_states):
            if tpm[state_idx, state_idx] == 1:
                with cython.gil:
                    absorbing_states.append(state_idx)
    return absorbing_states