"""State communication, pure NumPy."""

import numpy as np


def _is_partially_communicating(tpm, states, state1, state2, threshold):
    if state1 == state2:
        return True
    idx1, idx2 = states.index(state1), states.index(state2)
    x = np.asarray(tpm, dtype=float)
    tpm = x
    for _ in range(threshold):
        if x[idx1, idx2] > 0:
            return True
        x = x @ tpm
    return False


def is_communicating_cython(tpm, states, state1, state2, threshold):
    return (
        _is_partially_communicating(tpm, states, state1, state2, threshold)
        and _is_partially_communicating(tpm, states, state2, state1, threshold)
    )
