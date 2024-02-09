from scipy.linalg.cython_lapack cimport dgeev   # LAPACK Routine
import numpy as np
import cython
cimport numpy as cnp


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cpdef cnp.ndarray[cnp.float64_t, ndim=1] cython_stationary_dist(cnp.ndarray[cnp.float64_t, ndim=2] tpm_T):
    cdef cnp.float64_t* tpm_pointer = <cnp.float64_t*>cnp.PyArray_DATA(tpm_T)
    cdef int n = tpm_T.shape[0]
    cdef char* JOBVL_pointer = <char*>'V'
    cdef char* JOBVR_pointer = <char*>'V'
    cdef cnp.ndarray[cnp.float64_t, ndim=1] WR = np.zeros(n, dtype = np.float64)
    cdef cnp.float64_t* WR_pointer = <cnp.float64_t*>cnp.PyArray_DATA(WR)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] WI = np.zeros(n, dtype = np.float64)
    cdef cnp.float64_t* WI_pointer = <cnp.float64_t*>cnp.PyArray_DATA(WI)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] VL = np.zeros((n, n), dtype = np.float64)
    cdef cnp.float64_t* VL_pointer = <cnp.float64_t*>cnp.PyArray_DATA(VL)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] VR = np.zeros((n, n), dtype = np.float64)
    cdef cnp.float64_t* VR_pointer = <cnp.float64_t*>cnp.PyArray_DATA(VR)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] WORK = np.zeros(4 * n, dtype = np.float64)
    cdef cnp.float64_t* WORK_pointer = <cnp.float64_t*>cnp.PyArray_DATA(WORK)
    cdef int LWORK = len(WORK)
    cdef int INFO = 0

    dgeev(JOBVL_pointer, JOBVR_pointer, &n, tpm_pointer, &n, WR_pointer, \
            WI_pointer, VL_pointer, &n, VR_pointer, &n, WORK_pointer, &LWORK, &INFO)
    
    if INFO != 0:
        raise ValueError("Failed to compute eigenvectors. INFO =", INFO)    
    
    VL = VL.T    
    stationary_distribution = np.squeeze(VL[:, np.isclose(WR, 1.0)])
    stationary_distribution /= np.sum(stationary_distribution)
    
    if np.inf in stationary_distribution:
        raise ValueError("This Matrix doesn't have a stationary distribution")
    
    return stationary_distribution