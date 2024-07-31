import math
from typing import List, Union

import numpy as np
import numba

from ._exceptions import _handle_exceptions
from ._visualizations import _visualize_tpm, _visualize_chain
from ._fileio import _save_model_markovchain, _load_model_markovchain, _load_text
from ._caching import _cache
from ._backend import (
    _simulate,
    _absorbing,
    _stationary_dist,
    _learn_matrix,
    _is_communicating,
)


class MarkovChain:
    """A class containing Fundamental Functions for Discrete Time Markov Chains.

    This class provides a comprehensive suite of methods for working with
    discrete-time Markov chains (DTMCs), including validation, simulation,
    and analysis functionalities. It supports learning transition probability
    matrices (TPMs) from data, checking various properties of the Markov chain
    (e.g., ergodicity, aperiodicity, symmetry), and computing distributions and
    probabilities related to the chain's behavior over time.

    Attributes
    ----------
    tpm : np.ndarray
        Transition probability matrix (TPM) representing the Markov chain.
    states : List[str]
        List of state names corresponding to the TPM.
    eigendecom : bool
        Flag indicating if the TPM is eigendecomposable.
    eigenvalues : np.ndarray
        Eigenvalues of the TPM.
    eigenvectors : np.ndarray
        Eigenvectors of the TPM.
    epsilon : float
        Small value to avoid numerical issues in calculations.
    """

    __slots__ = "tpm", "states", "eigendecom", "eigenvalues", "eigenvectors", "epsilon"

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
        self.epsilon = 1e-16

        self._validate_transition_matrix(p, states, epsilon=1e-16)

    def __repr__(self) -> str:
        if self.tpm is None:
            return (
                f"<Object of type MarkovChain with uninitialized transition matrix  "
                f"\n"
                f"and unknown states>"
            )
        return (
            f"<Object of type MarkovChain with {self.tpm.shape[0]} x {self.tpm.shape[1]} sized transition matrix  "
            f"\n"
            f"and {len(self.states)} states>"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def _validate_transition_matrix(
        self, p: np.ndarray, states: List[str], epsilon
    ) -> None:
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

                if not (p.shape[0] == len(states) and p.shape[0] == p.shape[1]):
                    raise ValueError(f"Invalid TPM {p}")

                if not np.allclose(np.sum(p, axis=1), 1, atol=epsilon * len(states)):
                    raise ValueError(
                        f"Rows of the Transition Probability Matrix \
                                            (TPM) {p} must sum to 1."
                    )

    @_handle_exceptions
    @_cache(class_method=True)
    def fit(self, data: Union[str, list], epsilon: float = 1e-16) -> np.ndarray:
        """
        Learn Transition Matrix from Sequence (list or str) of Data.
        Each Unique Word is considered a State.
        It will override the current transition-matrix.

        Args
        ----
            data: Union[str, list]
                Data on which the MarkovChain model must
                be fitted.
            epsilon: float
                Small dummy value to avoid zeros in the Transition-Matrix

        Returns
        -------
            ndarray: Transition - Matrix based on `data`

        Usage
        -----
            >>> chainopy.MarkovChain().fit("My name is John.")
        """
        return self._learn_matrix(data=data, epsilon=epsilon)

    def _learn_matrix(self, data: Union[str, list], epsilon: float) -> np.ndarray:
        _tpm, _states = _learn_matrix.learn_matrix_cython(data, epsilon=epsilon)
        self.tpm = _tpm
        self.epsilon = epsilon
        self.states = _states
        self._validate_transition_matrix(self.tpm, self.states, self.epsilon)
        return _tpm

    @_handle_exceptions
    def simulate(self, initial_state: str, n_steps: int) -> List[str]:
        """
        Simulate the Markov Chain for `n_steps` steps.

        Args
        ----
            initial_state: str
                State from which the simulation starts
            n_steps: int
                Number of steps to simulate the chain for

        Returns
        -------
            list: Contains states attained during simulation
        """
        return _simulate._simulate_cython(self.states, self.tpm, initial_state, n_steps)

    @staticmethod
    def _vectorize(states: List[str], initial_state: str) -> np.ndarray:
        if initial_state in states:
            init = states.index(initial_state)
            initial_vect = np.zeros((1, len(states)))
            initial_vect[0, init] = 1
        else:
            raise ValueError("Initial state not found in the list of states.")
        return initial_vect

    def adjacency_matrix(self) -> np.ndarray:
        """
        Returns
        -------
            ndarray: Adjacency matrix of the chain.
        """
        return (self.tpm > self.epsilon).astype(int)

    def predict(self, initial_state: str) -> str:
        """
        Return the next most likely states.

        Args
        ----
            initial_state : str
                Initial state.

        Returns
        -------
            str: Next most likely state.
        """
        initial_vect = self._vectorize(self.states, initial_state)
        return self.states[np.argmax(initial_vect @ self.tpm)]

    @_handle_exceptions
    def nstep_distribution(self, n_steps: int) -> np.ndarray:
        """
        Calculates the distribution of the Markov Chain after n-steps.

        Args
        ----
            n_steps : int
                Number of steps.

        Returns
        -------
            ndarray: Distribution of the Markov Chain.
        """
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

    @_cache(class_method=True)
    def is_ergodic(self) -> bool:
        """
        Checks if the Markov chain is ergodic.

        Returns
        -------
            bool: True if the Markov chain is ergodic, False otherwise.
        """
        return self.is_irreducible() and self.is_aperiodic()

    @_cache(class_method=True)
    def is_symmetric(self) -> bool:
        """
        Checks if the Markov chain is symmetric.

        Returns
        -------
            bool: True if the Markov chain is symmetric, False otherwise.
        """
        return np.allclose(self.tpm, self.tpm.transpose())

    @_cache(class_method=True)
    def stationary_dist(self) -> np.ndarray:
        """
        Returns the stationary distribution of the Markov chain.

        Returns
        -------
            ndarray: Stationary distribution.
        """
        tpm_T = self.tpm.transpose()
        return _stationary_dist.cython_stationary_dist(tpm_T)

    @_handle_exceptions
    @_cache(class_method=True)
    def is_communicating(self, state1: str, state2: str, threshold: int = 1000) -> bool:
        """
        Checks if two states are communicating.

        Args
        ----
            state1 : str
                First state.
            state2 : str
                Second state.
            threshold : int, optional
                Threshold for convergence. Defaults to 1000.

        Returns
        -------
            bool: True if the states are communicating, False otherwise.
        """
        return _is_communicating.is_communicating_cython(
            self.tpm, self.states, state1, state2, threshold
        )

    @_cache(class_method=True)
    def is_irreducible(self) -> bool:
        """
        Checks if the Markov chain is irreducible.

        Returns
        -------
            bool: True if the Markov chain is irreducible, False otherwise.
        """
        return all(
            any(self.is_communicating(state1, state2) for state2 in self.states)
            for state1 in self.states
        )

    def _absorbing_state_indices(self) -> List[int]:
        return _absorbing._absorbing_states_indices(self.tpm)

    @_cache(class_method=True)
    def absorbing_states(self) -> List[str]:
        """
        Returns all absorbing states.

        Returns
        -------
            List[str]: Absorbing states.
        """
        indices = self._absorbing_state_indices()
        return [self.states[i] for i in indices]

    @_cache(class_method=True)
    def is_absorbing(self) -> bool:
        """
        Checks if the Markov chain is absorbing.

        Returns
        -------
            bool: True if the Markov chain is absorbing, False otherwise.
        """
        absorbing_states_ = self.absorbing_states()
        if len(absorbing_states_) == 0:
            return False

        transient_states = set(self.states) - set(absorbing_states_)
        for i in absorbing_states_:
            if all(
                _is_communicating._is_partially_communicating(
                    self.tpm, self.states, state, i, threshold=1000
                )
                for state in transient_states
            ):
                return True
        return False

    @_cache(class_method=True)
    def is_aperiodic(self) -> bool:
        """
        Checks if the Markov chain is aperiodic.

        Returns
        -------
            bool: True if the Markov chain is aperiodic, False otherwise.
        """
        if self.period() == 1:
            return True
        return False

    @_cache(class_method=True)
    def period(self) -> int:
        """
        Returns the period of the Markov chain.

        Returns
        -------
            int: Period of the Markov chain.
        """
        if np.any(np.diag(self.tpm) > self.epsilon):
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

    @_cache(class_method=True)
    def _is_eigendecomposable(self) -> bool:
        eigenvalues, _ = np.linalg.eig(self.tpm)
        unique_eigenvalues = np.unique(eigenvalues)
        return len(unique_eigenvalues) == self.tpm.shape[0]

    @_handle_exceptions
    @_cache(class_method=True)
    def is_transient(self, state: str) -> bool:
        """
        Checks if a state is transient.

        Args
        ----
            state : str
                State to check.

        Returns
        -------
            bool: True if the state is transient, False otherwise.
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
            n_step_tpm = self.tpm
            _truth_vals = []
            while True:
                if n_step_tpm[state_idx][state_idx] < 1:
                    _truth_vals.append(True)
                else:
                    return False
                j = n_step_tpm
                n_step_tpm = n_step_tpm @ self.tpm

                if np.allclose(n_step_tpm, j):
                    break

            if not np.all(np.array(_truth_vals)):
                return False
        return True

    @_handle_exceptions
    def is_recurrent(self, state: str) -> bool:
        """
        Checks if a state is recurrent.

        Args
        ----
            state : str
                State to check.

        Returns
        -------
            bool: True if the state is recurrent, False otherwise.
        """
        return not self.is_transient(state)

    @_cache(class_method=True)
    def fundamental_matrix(self) -> Union[np.ndarray, None]:
        """
        Returns the fundamental matrix.

        Returns
        -------
            Union[ndarray, None]: Fundamental matrix.
        """

        absorbing_indices = self._absorbing_state_indices()
        k = len(self.states) - len(absorbing_indices)

        if not self.is_absorbing():
            return None

        I = np.identity(k)

        Q = np.delete(self.tpm, absorbing_indices, axis=0)
        Q = np.delete(Q, absorbing_indices, axis=1)

        return np.linalg.inv(I - Q)

    @_cache(class_method=True)
    def absorption_probabilities(self) -> np.ndarray:
        """
        Returns the absorption probabilities matrix for each state.

        Returns
        -------
            ndarray: Absorption probabilities matrix.
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

    @_cache(class_method=True)
    def expected_time_to_absorption(self) -> np.ndarray:
        """
        Returns the expected time to absorption for each state.

        Returns
        -------
            ndarray: Expected time to absorption.
        """
        absorption_probs = self.absorption_probabilities()
        return np.sum(absorption_probs, axis=1)

    @_cache(class_method=True)
    def expected_number_of_visits(self) -> np.ndarray:
        """
        Returns the expected number of visits to each state before absorption.

        Returns
        -------
            ndarray: Expected number of visits.
        """
        absorption_probs = self.absorption_probabilities()
        return np.reciprocal(1 - absorption_probs)

    @_cache(class_method=True)
    def expected_hitting_time(self, state: str) -> Union[float, None]:
        """
        Returns the expected hitting time to reach the given absorbing state.

        Args
        ----
            state : str
                Absorbing state.

        Returns
        -------
            Union[float, None]: Expected hitting time.
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
        _visualize_chain(self.tpm, self.states, self.epsilon)

    @_handle_exceptions
    def save_model(self, filename: str, epsilon: float = 1e-16):
        """
        Save Model as a JSON Object.
        If tpm is sparsifyable, it stores tpm
        as a sparse matrix.

        Args
        ----
            filename : str
                Name of the file to save.
            epsilon: float
                Small dummy value to avoid zeros in the Transition-Matrix
        """
        _save_model_markovchain(self, filename, epsilon=epsilon)

    @_handle_exceptions
    def load_model(self, path: str):
        """
        Load a ChainoPy Model stored as a JSON Object
        and return as a `MarkovChain` object.

        Args
        ----
            path : str
                Path to the file.

        Raises:
        ------
            ValueError: If the file cannot be loaded.
        """

        result = _load_model_markovchain(path)
        if len(result) == 3:
            self.tpm, self.states, self.eigendecom, self.epsilon = result
        elif len(result) == 4:
            self.tpm, self.states, self.eigendecom, self.epsilon = result
        elif len(result) == 5:
            (
                self.tpm,
                self.states,
                self.eigendecom,
                self.eigenvalues,
                self.eigenvectors,
            ) = result
        elif len(result) == 6:
            (
                self.tpm,
                self.states,
                self.eigendecom,
                self.eigenvalues,
                self.eigenvectors,
                self.epsilon,
            ) = result

        self.tpm = np.where(self.tpm == 0, self.epsilon, self.tpm)

    @_cache(class_method=True)
    def marginal_dist(self, state: str):
        """
        Args
        ----
        state: str
            State for which to calculate the marginal distribution

        Returns
        -------
        float:
            marginal distribution of a state
        """
        _idx = self.states.index(state)
        return np.sum(self.tpm[:, _idx])

    def fit_from_file(self, path: str, epsilon: float = 1e-16):
        """
        Args
        ----
        path: str
            path to the text file
        epsilon: float
            small value to avoid zero division

        Returns
        -------
        ndarray:
            Transition Matrix trained from the text file.
            If `self.tpm` is None. Then this sets `self.tpm`
            to the new transition-matrix.
        """
        _data_list = _load_text(path)
        if (_data_list is None) or (len(_data_list) == 0):
            raise ValueError("Invalid contents of the text file.")

        return _learn_matrix.learn_matrix_cython(_data_list, epsilon)
