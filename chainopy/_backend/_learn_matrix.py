"""Transition matrix estimation from a sequence, pure NumPy."""

import numpy as np


def learn_matrix_cython(seq, epsilon: float = 1e-16):
    """Kept under its historical name so callers do not change.

    Counts bigram transitions and row-normalizes, with an epsilon
    floor so unseen transitions keep nonzero probability.
    """
    words = seq.split(" ") if isinstance(seq, str) else list(seq)
    unique_words = list(set(words))
    index = {w: k for k, w in enumerate(unique_words)}
    n = len(unique_words)
    counts = np.full((n, n), float(epsilon))
    src = np.fromiter((index[w] for w in words[:-1]), dtype=np.intp,
                      count=max(len(words) - 1, 0))
    dst = np.fromiter((index[w] for w in words[1:]), dtype=np.intp,
                      count=max(len(words) - 1, 0))
    np.add.at(counts, (src, dst), 1.0)
    counts /= counts.sum(axis=1, keepdims=True)
    return counts, unique_words
