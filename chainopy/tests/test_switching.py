import numpy as np
import random

from ..markov_switching import MarkovSwitchingModel


def test_model_fit():
    data = np.random.normal(0, 1, 1000)
    regime_col = [random.choice(["High", "Low", "Stagnant"]) for _ in range(1000)]

    mod = MarkovSwitchingModel()
    mod.fit(data, regime_col)

    assert mod._markov_chain.tpm.shape == (3, 3)
    assert len(mod.models) == 3


def test_model_predict():
    data = np.random.normal(0, 1, 1000)
    regime_col = [random.choice(["High", "Low", "Stagnant"]) for _ in range(1000)]

    mod = MarkovSwitchingModel()
    mod.fit(data, regime_col)

    ans, ans_regimes = mod.predict("High", 5)
    assert len(ans) == 5
    assert len(ans_regimes) == 5
