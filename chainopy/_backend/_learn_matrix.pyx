cimport numpy as cnp
import numpy as np
import cython

def get_index(list listVar, str element):
    idx =  listVar.index(element)
    return idx

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(False)
def learn_matrix_cython(seq, epsilon: float = 1e-16):
    cdef list unique_words = []
    cdef int num_unique_words
    cdef list words 
    cdef list bigram
    cdef int idx0
    cdef int idx1

    if isinstance(seq, list):
        words = seq
    elif isinstance(seq, str):
        words = seq.split(" ")

    cdef int len_seq = len(words)

    unique_words = list(set(words))
    num_unique_words = len(unique_words)

    cdef cnp.ndarray[cnp.float64_t, ndim=2] transition_matrix = np.ones((num_unique_words, num_unique_words)) * epsilon
    cdef list bigrams = [[words[i], words[i+1]] for i in range(0, len_seq - 1)]
    cdef Py_ssize_t i, j
    cdef double row_sum

    for bigram in bigrams:
        idx0 = get_index(unique_words, bigram[0])
        idx1 = get_index(unique_words, bigram[1])
        transition_matrix[idx0][idx1] += 1

    for i in range(num_unique_words):
        row_sum = transition_matrix[i, :].sum()
        for j in range(num_unique_words):
            transition_matrix[i, j] /= row_sum

    return transition_matrix, unique_words
