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
        Trains and sets the models `self.models` and `self._markov_chain`
        attributes

        Parameters
        ----------
        ts_data: ndarray,
            1D Target values at different timepoints
        regime_sequence: list
                    Training data consisting of Regimes in chronological
                    Order.
        lags: int,
            Time lags to consider during autoregression
        """
        MarkovSwitchingModel._validate_data(
            ts_data=ts_data, regime_sequence=regime_sequence
        )
        self._learn_regime_proba(regime_sequence)
        self._learn_models(ts_data, regime_sequence, lags=lags)

    def _learn_regime_proba(self, regime_sequence: List[str]) -> np.ndarray:
        """
        Learns transition probabilities for regimes. Overrides, if these
        probabilities already exist.

        Parameters
        ----------
        regime_sequence: list,
                    Training data consisting of Regimes in chronological
                    Order.
        """
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
        """
        _regime_sequence = np.array(regime_sequence)
        for i in range(self.num_regimes):
            X = ts_data[_regime_sequence == self.regimes[i]]
            self.models[self.regimes[i]] = AutoReg(X, lags=lags).fit()

    def predict(self, start_regime: str, steps: int = 1) -> np.ndarray:
        """
        Predicts the target values for given number of steps into the future.

        Parameters
        ----------
        start_regime : str
            Regime at the start of the prediction
        steps : int, optional
            Number of steps into the future to predict, by default 1.

        Returns
        -------
        np.ndarray
            Array of predicted Target Values for each feature for each step.
        """
        predictions = np.zeros(steps, dtype=np.float32)
        regime_predictions = []
        current_regime = start_regime
        regime_predictions = self._markov_chain.simulate(current_regime, steps)
        for i, regime in enumerate(regime_predictions):
            _model = self.models[regime]
            prediction = _model.model.predict(_model.params)[-1]
            predictions[i] = prediction

        return predictions, np.array(regime_predictions)

    def evaluate(self, ts_test, ts_pred):
        """
        Evaluates the accuracy of the model using the
        mean squared error metric.

        Parameters
        ----------
        ts_test: ndarray
            Real target values of the Time Series
        ts_pred: ndarray
            Predicted target values of the Time Series

        Returns
        -------
        float: mean square error between `ts_test` and `ts_pred`
        """
        return np.mean(np.square(ts_test - ts_pred))
