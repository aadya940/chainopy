"""Chain simulation, pure NumPy.

Also fixes two bugs the Cython version carried: the first step was
chosen by argmax of state 0's row regardless of the initial state
(an uninitialized index), where every step should be a probability
draw from the CURRENT state's row.
"""

import numpy as np


def _simulate_cython(states, tpm, initial_state, n_steps):
    if initial_state not in states:
        raise ValueError("Initial state not found in the list of states.")
    tpm = np.asarray(tpm, dtype=float)
    num_states = len(states)
    current = states.index(initial_state)
    sims = []
    for _ in range(n_steps):
        current = int(np.random.choice(num_states, p=tpm[current, :]))
        sims.append(states[current])
    return sims
