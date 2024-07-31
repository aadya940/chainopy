import numpy as np
from typing import List, Tuple
from statsmodels.tsa.ar_model import AutoReg

from .markov_chain import MarkovChain


class MarkovSwitchingModel:
    """MarkovSwitchingModel estimates 1D target values given the current regime.
    The regimes follow a first-order Markov process.

    Attributes
    ----------
    regimes : list
        List of regimes identified in the training data.
    num_regimes : int
        Number of unique regimes.
    models : dict
        Dictionary mapping regimes to their respective AutoReg models.
    num_models : int
        Number of models, corresponding to the number of unique regimes.
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

        Args
        ----
        ts_data: ndarray
            1D Target values at different timepoints
        regime_sequence: list
            Training data consisting of Regimes in chronological Order.
        lags: int,
            Time lags to consider during autoregression
        """
        MarkovSwitchingModel._validate_data(
            ts_data=ts_data, regime_sequence=regime_sequence
        )
        self._learn_regime_proba(regime_sequence)
        self._learn_models(ts_data, regime_sequence, lags=lags)

    def _learn_regime_proba(self, regime_sequence: str) -> np.ndarray:
        """
        Learns transition probabilities for regimes. Overrides, if these
        probabilities already exist.

        Args
        ----
        regime_sequence: list,
            Training data consisting of Regimes in chronological Order.
        """
        __regime_str = ""
        for i, regime in enumerate(regime_sequence):
            if i == len(regime_sequence) - 1:
                __regime_str += regime
                continue

            __regime_str += regime + " "

        self._markov_chain.fit(__regime_str)
        self.regimes = self._markov_chain.states
        self.num_regimes = len(self.regimes)

    def _learn_models(
        self,
        ts_data: np.ndarray,
        regime_sequence: List[str],
        lags,
    ) -> dict:
        """
        Args
        ----
        ts_data: ndarray,
            1D Target values at different timepoints
        regime_sequence: list
            regimes corresponding to target values at each timepoint
        """
        _regime_sequence = np.array(regime_sequence)
        for i in range(self.num_regimes):
            X = ts_data[_regime_sequence == self.regimes[i]]
            self.models[self.regimes[i]] = AutoReg(X, lags=lags).fit()

    def predict(
        self, start_regime: str, steps: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts the target values for a given number of steps into the future.

        Args
        ----
        start_regime: str
            Regime at the start of the prediction.
        steps: int, optional
            Number of steps into the future to predict, by default 1.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]: Tuple containing the array of predicted
        target values and the predicted regime sequence.
        """
        predictions = np.zeros(steps, dtype=np.float32)
        regime_predictions = self._markov_chain.simulate(start_regime, steps)

        # Initialize current values for each regime
        current_values = {
            regime: list(
                self.models[regime].model.endog[
                    -len(self.models[regime].model.ar_lags) :
                ]
            )
            for regime in self.regimes
        }

        for i, regime in enumerate(regime_predictions):
            model = self.models[regime]
            available_lags = len(current_values[regime])

            if available_lags < len(model.model.ar_lags):
                # Use all available data if there are fewer data points than lags
                start_index = -available_lags
                prediction = model.model.predict(
                    model.params, start=start_index, end=-1
                )[0]
            else:
                # Use the full lag window if enough data points are available
                prediction = model.predict(
                    start=len(model.model.endog), end=len(model.model.endog)
                )[0]

            predictions[i] = prediction

            # Update current values for the regime with the new prediction
            current_values[regime].append(prediction)
            if len(current_values[regime]) > len(model.model.ar_lags):
                current_values[regime].pop(0)

        return predictions, np.array(regime_predictions)

    def evaluate(self, ts_test, ts_pred):
        """
        Evaluates the accuracy of the model using the
        mean squared error metric.

        Args
        ----
        ts_test: ndarray
            Real target values of the Time Series
        ts_pred: ndarray
            Predicted target values of the Time Series

        Returns
        -------
        float: mean square error between `ts_test` and `ts_pred`
        """
        return np.mean(np.square(ts_test - ts_pred))
