import numpy as np
from typing import List
from statsmodels.tsa.ar_model import AutoReg

from .markov_chain import MarkovChain


class MarkovSwitchingModel:
    """
    This Model estimates the 1D target values given the Current Regime.
    The Regimes follow a First Order Markov Process.
    """

    def __init__(self):
        self.regimes = None
        self.num_regimes = None
        self.models = {}
        self.num_models = None
        self._markov_chain = MarkovChain()

    @staticmethod
    def _validate_data(ts_data, regime_sequence):
        if not (len(ts_data.shape) == 1):
            raise ValueError("Time Series should be a 1 Dimensional Array.")

        if not (ts_data.shape[0] == len(regime_sequence)):
            raise ValueError(
                "Regime Sequence and Time Series Data \
                            must be of the same length"
            )

        if not (isinstance(ts_data, np.ndarray)):
            raise ValueError("Time Series data must be a Numpy Array")

        if not (isinstance(regime_sequence, list)):
            raise ValueError("Regime Sequence must be a list")

        if np.nan in ts_data:
            raise ValueError("Time Series Data must not contain NaN values")

        if None in regime_sequence:
            raise ValueError("Regime Sequence must not contain None values")

    def fit(self, ts_data, regime_sequence, lags=1):
        """
        Parameters
        ----------
        ts_data: ndarray,
            1D Target values at different timepoints
        regime_sequence: list
                    Training data consisting of Regimes in chronological
                    Order.
        """
        MarkovSwitchingModel._validate_data(
            ts_data=ts_data, regime_sequence=regime_sequence
        )
        self._learn_regime_proba(regime_sequence)
        self._learn_models(ts_data, regime_sequence, lags=lags)

    def _learn_regime_proba(self, regime_sequence: List[str]) -> np.ndarray:
        """
        Parameters
        ----------
        regime_sequence: list,
                    Training data consisting of Regimes in chronological
                    Order.

        Returns
        -------
        ndarray: Markov Transition Matrix calculated based on `regime_sequence`
        """
        if (self._markov_chain.tpm is None) and (self.regimes is None):
            self._markov_chain.fit(regime_sequence)
            self.regimes = self._markov_chain.states
            self.num_regimes = len(self.regimes)

    def _learn_models(
        self,
        ts_data: np.ndarray,
        regime_sequence: List[str],
        lags,
    ) -> dict:
        """
        Parameters
        ----------
        ts_data: ndarray,
            1D Target values at different timepoints
        regime_sequence: list
            regimes corresponding to target values at each timepoint

        Returns
        -------
        dict: Returns Autoregressive models for each Regime
        """
        _regime_sequence = np.array(regime_sequence)
        for i in range(self.num_regimes):
            X = ts_data[_regime_sequence == self.regimes[i]]
            self.models[self.regimes[i]] = AutoReg(X, lags=lags).fit()

    def predict(self, start_regime: str, steps: int = 1) -> np.ndarray:
        """
        Predicts the expected mean for the given number of steps into the future.

        Parameters
        ----------
        regime_sequence : List[str]
            Regime sequence indicating the current regime at each time step.
        steps : int, optional
            Number of steps into the future to predict, by default 1.

        Returns
        -------
        np.ndarray
            Array of predicted Target Values for each feature for each step.
        np.ndarray
            Array of predicted Regimes for each feature for each step.
        """
        predictions = np.zeros(steps, dtype=np.float32)
        regime_predictions = []
        current_regime = start_regime
        for i in range(steps):
            _model = self.models[current_regime]
            prediction = _model.model.predict(_model.params, start=steps, end=steps)
            predictions[i] = prediction
            current_regime = self._markov_chain.predict(current_regime)
            regime_predictions.append(current_regime)
        return (predictions, np.array(regime_predictions))
