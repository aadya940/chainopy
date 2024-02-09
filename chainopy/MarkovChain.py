import math
from typing import List, Union, Tuple

import numpy as np
import numba

from .exceptions import handle_exceptions
from .visualizations import _visualize_tpm, _visualize_chain
from .fileio import _save_model_markovchain, _load_model_markovchain
from .caching import cache
from ._backend import (
    _simulate,
    _absorbing,
    _stationary_dist,
    _learn_matrix,
    _is_communicating,
)


class MarkovChain:
    __slots__ = "tpm", "states", "eigendecom", "eigenvalues", "eigenvectors"

    def __init__(
        self, p: Union[np.ndarray, None] = None, states: Union[List[str], None] = None
    ) -> None:
        if p is not None:
            p = np.array(p, dtype=np.float64)

        if states is None:
            if p is not None:
                states = [str(i) for i in range(len(p))]

        self.tpm = p
        self.states = states
        self.eigendecom = None
        self.eigenvalues = None
        self.eigenvectors = None

        self._validate_transition_matrix(p, states)

    def __repr__(self) -> str:
        return (
            f"<Object of type MarkovChain with {self.tpm.shape[0]} x {self.tpm.shape[1]} sized transition matrix  "
            f"\n"
            f"and {len(self.states)} states>"
        )

    def __str__(self) -> str:
        return (
            f"<Object of type MarkovChain with {self.tpm.shape[0]} x {self.tpm.shape[1]} sized transition matrix  "
            f"\n"
            f"and {len(self.states)} states>"
        )

    def _validate_transition_matrix(self, p: np.ndarray, states: List[str]) -> None:
        def elements_range(arr):
            if np.any((arr < 0) | (arr > 1)):
                raise ValueError("All elements in the TPM must be between 0 and 1.")

        if p is not None:
            elements_range(p)
            self.eigendecom = self._is_eigendecomposable()

            if (
                self.eigendecom
                and (self.eigenvectors is None)
                and (self.eigenvalues is None)
            ):
                self.eigenvalues, self.eigenvectors = np.linalg.eig(p)
            else:
                self.eigenvalues = None
                self.eigenvectors = None

            if states is not None:
                if len(states) != len(set(states)):
                    raise ValueError(
                        "Names of the states \
                                            should be Unique"
                    )

                if (
                    p.shape[0] != len(states)
                    or p.shape[1] != len(states)
                    or p.shape[0] != p.shape[1]
                ):
                    raise ValueError(f"Invalid TPM {p}")

                if not np.allclose(np.sum(p, axis=1), 1):
                    raise ValueError(
                        f"Rows of the Transition Probability Matrix \
                                            (TPM) {p} must sum to 1."
                    )

    @handle_exceptions
    def fit(self, data: str) -> np.ndarray:
        """
        Learn Transition Matrix from Sequence of Data

        update self.p = self._learn_matrix(data)
        """
        return self._learn_matrix(data=data)

    def _learn_matrix(self, data: str) -> np.ndarray:
        """
        # Replace existing TPM with new TPM.

        Parameters
        ----------
            data: seq of str objects
                eg: "my name is James"
        Returns
        -------
            np.ndarray:
                transition matrix built from data
        """
        _tpm = _learn_matrix.learn_matrix_cython(data)
        self.tpm = _tpm
        if self.states is None:
            self.states = list(set(data.split(" ")))
        self._validate_transition_matrix(self.tpm, self.states)
        return _tpm

    def simulate(self, initial_state: str, n_steps: int) -> List[str]:
        """
        Simulate the Markov Chain for `n_steps`
        """
        return _simulate._simulate_cython(self.states, self.tpm, initial_state, n_steps)

    @staticmethod
    @numba.jit(nopython=True)
    def _vectorize(states: List[str], initial_state: str) -> np.ndarray:
        """
        Transforms `initial_state` string to OneHot Numpy Vector
        """
        if initial_state in states:
            init = states.index(initial_state)
            initial_vect = np.zeros((1, len(states)))
            initial_vect[0, init] = 1
        else:
            raise ValueError("Initial state not found in the list of states.")
        return initial_vect

    def adjacency_matrix(self) -> np.ndarray:
        """
        Returns adjacency matrix of the chain
        """
        return (self.tpm > 0).astype(int)

    def predict(self, initial_state: str) -> str:
        """
        Return the next most likely states
        """
        initial_vect = self._vectorize(self.states, initial_state)
        return self.states[np.argmax(initial_vect @ self.tpm)]

    @handle_exceptions
    def nstep_distribution(self, n_steps: int) -> np.ndarray:
        """
        Calculates the distribution of the Markov Chain after n-steps
        """
        # Use Efficient Matrix Multiplication
        # if matrix is eigendecomposable

        is_eigendecom = self.eigendecom
        eigvals = self.eigenvalues
        eigvecs = self.eigenvectors

        def diag_power_matmul():
            if is_eigendecom:
                if (eigvals is not None) and (eigvecs is not None):
                    D = np.diag(eigvals)
                    return np.real(eigvecs @ D**n_steps @ np.linalg.inv(eigvecs))

        result = diag_power_matmul()

        return result if is_eigendecom else np.linalg.matrix_power(self.tpm, n_steps)

    @cache(class_method=True)
    def is_ergodic(self) -> bool:
        """
        A Markov chain is called an ergodic Markov chain if it is
        possible to go from every state to every state (not
        necessarily in one move).
        """
        return self.is_irreducible() and self.is_aperiodic()

    @cache(class_method=True)
    def is_symmetric(self) -> bool:
        return np.allclose(self.tpm, self.tpm.transpose())

    @cache(class_method=True)
    def stationary_dist(self) -> np.ndarray:
        """
        Normalized Eigenvector of tpm.transpose() with \
        eigenvalue 1. Raise error if matrix is invalid
        """
        tpm_T = self.tpm.transpose()
        return _stationary_dist.cython_stationary_dist(tpm_T)

    @handle_exceptions
    @cache(class_method=True)
    def is_communicating(self, state1: str, state2: str, threshold: int = 1000) -> bool:
        """
        Checks if two states are communicating or not.

        NOTE:
        =====

        A very small threshold might not let the chain reach convergence
        hence its more prone to errors.
        """
        return _is_communicating.is_communicating_cython(
            self.tpm, self.states, state1, state2, threshold
        )

    @cache(class_method=True)
    def is_irreducible(self) -> bool:
        """
        If all states are communicating, the Markov Chain is irreducible
        """
        return all(
            any(self.is_communicating(state1, state2) for state2 in self.states)
            for state1 in self.states
        )

    def _absorbing_state_indices(self) -> List[int]:
        return _absorbing._absorbing_states_indices(self.tpm)

    @cache(class_method=True)
    def absorbing_states(self) -> List[str]:
        """
        Returns all absorbing states
        """
        indices = self._absorbing_state_indices()
        return [self.states[i] for i in indices]

    @cache(class_method=True)
    def is_absorbing(self) -> bool:
        """
        An absorbing Markov chain is a Markov chain in which every state
        can reach an absorbing state.
        An absorbing state is a state, where once reached, we can't get out.

        # Approach:
        - Check if there are absorbing states:
            - If YES:
                - Check if all other states are transient
            - If NO:
                - Return False
        """
        absorbing_states_ = self.absorbing_states()
        if len(absorbing_states_) == 0:
            return False

        transient_states = self.states - set(absorbing_states_)
        for i in absorbing_states_:
            if all(self.is_communicating(state, i) for state in transient_states):
                return True
        return False

    @cache(class_method=True)
    def is_aperiodic(self) -> bool:
        """
        If any state contains a `self - loop` , the whole chain \
        becomes aperiodic.
        
        A state in a discrete-time Markov chain is periodic if 
        the chain can return to the state only at multiples of some 
        integer larger than 1.

        Formally, period = k, such that `k` is the gcd of `n`, for all \
        TPM^n(i, i) > 0. `k` might not belong to `n`. If `k` = 1, chain
        is aperiodic.
        """
        if self.period() == 1:
            return True
        return False

    @cache(class_method=True)
    def period(self) -> int:
        """
        Returns period of the chain

        Steps:
            If:
                - self-loop, period = 1
            Else:
                - Calculate communicating States
                - Calculate Return Times of the states
                using n, such that TPM^n(i, i) > 0.
                - Calculate GCD of `n`.
                - Return Result
        """
        if np.any(np.diag(self.tpm) > 0):
            return 1

        communicating_states = []

        if self.is_irreducible():
            communicating_states = self.states
        else:
            for i in self.states:
                if all(self.is_communicating(i, s) for s in self.states):
                    communicating_states.append(i)
                else:
                    raise ValueError(
                        "Chain should be Irreducible to \
                                    calculate Period"
                    )

        def return_time(state):
            n = 1
            idx = self.states.index(state)
            while True:
                _dist = self.nstep_distribution(n)[idx, idx]
                if not (np.isclose(_dist, 0)):
                    return n
                n += 1

        return_times = [return_time(cstate) for cstate in communicating_states]
        return math.gcd(*return_times)

    @cache(class_method=True)
    def _is_eigendecomposable(self) -> bool:
        """
        Checks if the matrix is Eigendecomposable,
        That is,
        Is the Matrix Square (YES, TPM is always Square) \
        and Diagonalizable?
        """
        eigenvalues, _ = np.linalg.eig(self.tpm)
        unique_eigenvalues = np.unique(eigenvalues)
        return len(unique_eigenvalues) == self.tpm.shape[0]

    @handle_exceptions
    @cache(class_method=True)
    def is_transient(self, state: str) -> bool:
        """
        If there is a `possibility` of leaving the state
        and never coming back, the state is called Transient.

        => P(Xn = i, X0 = i) < 1, For all `n`.

        To check if a state is transient using the fundamental matrix,
        you need to examine the diagonal element corresponding to that
        state. If N[i, i] < inf, the state is transient; otherwise, it is
        recurrent.

        In summary, for a state i,i:

        If N[i, i] < inf , the state is transient.
        If N[i, i] = inf, the state is recurrent.
        """

        state_idx = self.states.index(state)
        _fundamental_matrix = self.fundamental_matrix()
        if _fundamental_matrix is not None:
            absorbing_idx = self._absorbing_state_indices()
            _fm_state_indices = [
                x for x in list(range(len(self.states))) if x not in absorbing_idx
            ]
            if state_idx in _fm_state_indices:
                if np.isclose(_fundamental_matrix[state_idx, state_idx], np.inf):
                    return False
        else:
            # Use Naive Approach where fundamental Matrix is not
            # defined.

            n_step_tpm = self.tpm
            _truth_vals = []
            while True:
                if n_step_tpm[state_idx][state_idx] < 1:
                    _truth_vals.append(True)
                else:
                    return False
                j = n_step_tpm
                n_step_tpm = n_step_tpm @ self.tpm
                # Check Convergence
                if np.allclose(n_step_tpm, j):
                    break

            if not np.all(np.array(_truth_vals)):
                return False
        return True

    @handle_exceptions
    def is_recurrent(self, state: str) -> bool:
        """
        If fundamental matrix has corresponding diagonal element equal to
        infinity, state is recurrent.
        """
        return not self.is_transient(state)

    @cache(class_method=True)
    def fundamental_matrix(self) -> Union[np.ndarray, None]:
        """
        Gives Information about the expected number of times
        a process is in a certain state before reaching an
        absorbing state.

        It's more relevant in the context of absorbing markov
        chains.

        Fundamental Matrix `N`, is defined for an absorbing Markov
        Chain with `k` absorbing states.
        N = (I - Q)^(-1)
        I = Identity Matrix
        Q = Submatrix of the transition-matrix, obtained by removing
        all the absorbing states.

        If the Markov chain is not absorbing or has no transient
        states, then None is returned.
        """

        absorbing_indices = self._absorbing_state_indices()
        k = len(self.states) - len(absorbing_indices)

        if (not self.is_absorbing()) or (k == 0):
            return None

        I = np.identity(k)

        Q = np.delete(self.tpm, absorbing_indices, axis=0)
        Q = np.delete(Q, absorbing_indices, axis=1)

        return np.linalg.inv(I - Q)

    @cache(class_method=True)
    def absorption_probabilities(self) -> np.ndarray:
        """
        Returns the absorption probabilities matrix for each state.
        """
        fundamental_matrix = self.fundamental_matrix()
        if fundamental_matrix is not None:
            absorbing_indices = self._absorbing_state_indices()
            return fundamental_matrix[:, absorbing_indices]
        else:
            raise ValueError(
                "Cannot compute absorption probabilities \
                            for non-absorbing Markov chains."
            )

    @cache(class_method=True)
    def expected_time_to_absorption(self) -> np.ndarray:
        """
        Returns the expected time to absorption for each state.
        """
        absorption_probs = self.absorption_probabilities()
        return np.sum(absorption_probs, axis=1)

    @cache(class_method=True)
    def expected_number_of_visits(self) -> np.ndarray:
        """
        Returns the expected number of visits

        to each state before absorption.
        """
        absorption_probs = self.absorption_probabilities()
        return np.reciprocal(1 - absorption_probs)

    @cache(class_method=True)
    def expected_hitting_time(self, state: str) -> Union[float, None]:
        """
        Returns the expected hitting time to
        reach the given absorbing state.
        """
        fundamental_matrix = self.fundamental_matrix()
        if fundamental_matrix is not None:
            absorbing_indices = self._absorbing_state_indices()
            state_index = self.states.index(state)
            return fundamental_matrix[state_index, absorbing_indices].sum()
        else:
            raise ValueError(
                "Cannot compute expected hitting time for non-absorbing Markov chains."
            )

    def visualize_transition_matrix(self):
        """
        Visualize the Transition Matrix
        """
        _visualize_tpm(self.tpm, self.states)

    def visualize_chain(self):
        """
        Visualize the Markov Chain
        """
        _visualize_chain(self.tpm, self.states)

    @handle_exceptions
    def save_model(self, filename: str):
        """
        Save Model as a JSON Object
        """
        _save_model_markovchain(self, filename)

    @handle_exceptions
    def load_model(self, path: str):
        """
        Load a ChainoPy Model stored as a JSON Object
        and return as a `MarkovChain` object
        """

        result = _load_model_markovchain(path)
        if len(result) == 3:
            self.tpm, self.states, self.eigendecom = result
        else:
            (
                self.tpm,
                self.states,
                self.eigendecom,
                self.eigenvalues,
                self.eigenvectors,
            ) = result

        self.tpm = np.where(self.tpm == 0, 0.0001, self.tpm)
