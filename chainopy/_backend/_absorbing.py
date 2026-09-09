"""Absorbing state indices, pure NumPy."""

import numpy as np


def _absorbing_states_indices(tpm):
    tpm = np.asarray(tpm, dtype=float)
    return np.flatnonzero(np.diagonal(tpm) == 1.0).tolist()
