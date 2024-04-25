import numpy as np
cimport numpy as np
from numpy cimport ndarray
cimport cython


@cython.wraparound(False)
@cython.boundscheck(False)
def _simulate_cython(list states, double[:, :] tpm, str initial_state, int n_steps):
    cdef list sims = []
    cdef int init
    cdef double[:] _prod = tpm[init, :]
    cdef int next_state_index = np.argmax(_prod)
    cdef int num_states = len(states)
    
    if initial_state in states:
        init = states.index(initial_state)
        initial_vect = np.zeros((1, len(states)))
        initial_vect[0, init] = 1
    else:
        raise ValueError("Initial state not found in the list of states.")
    
    sims.append(states[next_state_index])
    
    for _ in range(1, n_steps):
        next_state_index = np.random.choice(num_states, p=tpm[next_state_index, :])
        sims.append(states[next_state_index])

    return sims[0: n_steps]
