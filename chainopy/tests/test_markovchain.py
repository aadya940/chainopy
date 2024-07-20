import pytest
import pydtmc
import numpy as np
import os

from ..markov_chain import MarkovChain


def test_init():
    x = MarkovChain([[0, 1], [1, 0]], states=["Rain", "No-Rain"])
    assert x is not None


def test_validate_matrix():
    with pytest.raises(ValueError):
        x = MarkovChain([[0, 5], [1, 0.0000000001]])

    with pytest.raises(ValueError):
        x = MarkovChain(
            [[0.00000000000000000001, 0.5], [0.01, 0.99]], states=["Rain", "No-Rain"]
        )


def test_simulate():
    x = MarkovChain([[0, 1], [1, 0]], states=["Rain", "No-Rain"])

    sims = x.simulate("Rain", 1000)
    assert len(sims) == 1000
    assert "Rain" in sims
    assert "No-Rain" in sims


def test_nstep_distribution():
    x = MarkovChain([[0, 1], [1, 0]], states=["Rain", "No-Rain"])
    sims = x.nstep_distribution(20)
    assert sims.shape == x.tpm.shape
    P = np.array([[1, 0], [0.1, 0.9]])
    x = MarkovChain(P, states=["State" + str(i) for i in range(P.shape[0])])
    assert x._is_eigendecomposable() is True
    assert np.allclose(x.nstep_distribution(20), np.linalg.matrix_power(P, 20))


def test_is_absorbing():
    x = MarkovChain([[0, 1], [1, 0]], states=["Rain", "No-Rain"])
    y = pydtmc.MarkovChain([[0, 1], [1, 0]], states=["Rain", "No-Rain"])
    assert y.is_absorbing == x.is_absorbing()
    assert x.is_absorbing() is False


def test_stationary_dist():
    x = MarkovChain([[0, 1], [1, 0]], states=["Rain", "No-Rain"])
    stationary_dist = x.stationary_dist()
    if stationary_dist is not None:
        assert np.allclose(stationary_dist.sum(), 1)
        assert np.isclose(
            np.logical_and(stationary_dist, [0.5, 0.5]), [True, True]
        ).all()


def test_fit():
    x = MarkovChain()
    _tpm = x.fit("My name is Aadya")
    assert _tpm is not None
    assert x.tpm is not None


_tpm_inputs = [
    [[0, 1], [1, 0]],
    [[0.1, 0.3, 0.4, 0.2], [0, 0.1, 0.2, 0.7], [0, 0, 1, 0], [0, 0, 0, 1]],
]


@pytest.mark.parametrize("_tpm", _tpm_inputs)
def test_fundamental_matrix(_tpm):
    x = MarkovChain(_tpm)
    y = pydtmc.MarkovChain(_tpm)

    fm_x = x.fundamental_matrix()
    fm_y = y.fundamental_matrix

    if (fm_x is None) and (fm_y is None):
        assert True
    else:
        assert np.allclose(fm_x, fm_y, atol=1e-4, equal_nan=True)


def test_is_transient():
    x = MarkovChain([[0, 1], [1, 0]], ["Rain", "No-Rain"])
    y = pydtmc.MarkovChain([[0, 1], [1, 0]], ["Rain", "No-Rain"])
    assert x.is_transient("Rain") == y.is_transient_state("Rain")


def test_is_recurrent():
    x = MarkovChain([[0, 1], [1, 0]], ["Rain", "No-Rain"])
    y = pydtmc.MarkovChain([[0, 1], [1, 0]], ["Rain", "No-Rain"])
    assert x.is_recurrent("Rain") == y.is_recurrent_state("Rain")


def test_adjacency_matrix():
    x = MarkovChain([[0, 1], [1, 0]])

    assert np.allclose(x.adjacency_matrix(), np.array([[0, 1], [1, 0]]))


def test_handle_exceptions():
    x = MarkovChain([[0, 1], [1, 0]])

    with pytest.raises(ValueError):
        x.nstep_distribution(-10)
        x.is_transient(100)
        x.is_recurrent(100)
        x.fit("")


def test_is_ergodic():
    x = MarkovChain(
        [[0.1, 0.8, 0.1], [0.2, 0.6, 0.2], [1, 0, 0]], ["Sunny", "Rainy", "Cloudy"]
    )

    y = pydtmc.MarkovChain(
        [[0.1, 0.8, 0.1], [0.2, 0.6, 0.2], [1, 0, 0]], ["Sunny", "Rainy", "Cloudy"]
    )

    assert x.is_ergodic() == y.is_ergodic

    x = MarkovChain([[0, 1], [1, 0]], ["Rain", "No-Rain"])
    y = pydtmc.MarkovChain([[0, 1], [1, 0]], ["Rain", "No-Rain"])

    assert x.is_ergodic() == y.is_ergodic


def test_is_irreducible():
    x = MarkovChain(
        [[0.1, 0.8, 0.1], [0.2, 0.6, 0.2], [1, 0, 0]], ["Sunny", "Rainy", "Cloudy"]
    )

    y = pydtmc.MarkovChain(
        [[0.1, 0.8, 0.1], [0.2, 0.6, 0.2], [1, 0, 0]], ["Sunny", "Rainy", "Cloudy"]
    )

    assert x.is_irreducible() == y.is_irreducible

    x = MarkovChain([[0, 1], [1, 0]], ["Rain", "No-Rain"])
    y = pydtmc.MarkovChain([[0, 1], [1, 0]], ["Rain", "No-Rain"])

    assert x.is_irreducible() == y.is_irreducible


def test_is_aperiodic():
    x = MarkovChain(
        [[0.1, 0.8, 0.1], [0.2, 0.6, 0.2], [1, 0, 0]], ["Sunny", "Rainy", "Cloudy"]
    )

    y = pydtmc.MarkovChain(
        [[0.1, 0.8, 0.1], [0.2, 0.6, 0.2], [1, 0, 0]], ["Sunny", "Rainy", "Cloudy"]
    )

    assert x.is_aperiodic() == y.is_aperiodic

    x = MarkovChain([[0, 1], [1, 0]], ["Rain", "No-Rain"])
    y = pydtmc.MarkovChain([[0, 1], [1, 0]], ["Rain", "No-Rain"])
    assert x.is_aperiodic() == y.is_aperiodic


def test_is_communicating():
    x = MarkovChain(
        [[0.1, 0.8, 0.1], [0.2, 0.6, 0.2], [1, 0, 0]], ["Sunny", "Rainy", "Cloudy"]
    )

    y = pydtmc.MarkovChain(
        [[0.1, 0.8, 0.1], [0.2, 0.6, 0.2], [1, 0, 0]], ["Sunny", "Rainy", "Cloudy"]
    )

    for i in x.states:
        for j in y.states:
            assert x.is_communicating(i, j) == y.are_communicating(i, j)

    x = MarkovChain([[0, 1], [1, 0]], ["Sunny", "Rainy"])

    y = pydtmc.MarkovChain([[0, 1], [1, 0]], ["Sunny", "Rainy"])

    for i in x.states:
        for j in y.states:
            assert x.is_communicating(i, j) == y.are_communicating(i, j)


def test_absorbing_states():
    x = MarkovChain(
        [[0.1, 0.8, 0.1], [0.2, 0.6, 0.2], [1, 0, 0]], ["Sunny", "Rainy", "Cloudy"]
    )

    y = pydtmc.MarkovChain(
        [[0.1, 0.8, 0.1], [0.2, 0.6, 0.2], [1, 0, 0]], ["Sunny", "Rainy", "Cloudy"]
    )

    assert x.absorbing_states() == y.absorbing_states

    x = MarkovChain([[0, 1], [1, 0]], ["Sunny", "Rainy"])

    y = pydtmc.MarkovChain([[0, 1], [1, 0]], ["Sunny", "Rainy"])

    assert x.absorbing_states() == y.absorbing_states


def test_period():
    x = MarkovChain([[0, 1], [1, 0]], ["Sunny", "Rainy"])

    assert x.period() == 2


def test_predict():
    x = MarkovChain(
        [[0.1, 0.8, 0.1], [0.2, 0.6, 0.2], [1, 0, 0]], ["Sunny", "Rainy", "Cloudy"]
    )

    y = pydtmc.MarkovChain(
        [[0.1, 0.8, 0.1], [0.2, 0.6, 0.2], [1, 0, 0]], ["Sunny", "Rainy", "Cloudy"]
    )

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert x.predict("Sunny") == y.predict(1, "Sunny")[-1]


def test_save_and_load_model():
    x = MarkovChain([[0.7, 0.3, 0.0], [0.4, 0.5, 0.1], [0.0, 0.0, 1.0]])
    _tpm_x = x.tpm
    x.save_model("file_x.json")
    assert "file_x.json" in os.listdir(".")
    y = MarkovChain()
    y.load_model(path="file_x.json")
    assert np.allclose(_tpm_x, y.tpm)
    os.remove("file_x.json")
