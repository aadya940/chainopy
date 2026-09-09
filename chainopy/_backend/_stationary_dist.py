"""Stationary distribution, pure NumPy.

Fixes the Cython version's LAPACK order bug: dgeev was called on a
C-order array (LAPACK expects Fortran order), which returned wrong
distributions for any chain that is not doubly stochastic.
np.linalg.eig handles memory order correctly.
"""

import numpy as np


def cython_stationary_dist(tpm_T):
    """Left eigenvector of the transition matrix for eigenvalue 1.
    ``tpm_T`` is the TRANSPOSED transition matrix, as the caller
    already passes it.
    """
    w, v = np.linalg.eig(np.asarray(tpm_T, dtype=float))
    close = np.isclose(np.real(w), 1.0) & np.isclose(np.imag(w), 0.0)
    if not np.any(close):
        raise ValueError("This Matrix doesn't have a stationary distribution")
    pi = np.real(v[:, np.argmax(close)])
    total = pi.sum()
    if np.isclose(total, 0.0):
        raise ValueError("This Matrix doesn't have a stationary distribution")
    pi = pi / total
    return pi
