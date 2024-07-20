import pytest
import numpy as np
import torch

from ..nn import MarkovChainNeuralNetwork, divergance_analysis
from ..markov_chain import MarkovChain


@pytest.fixture
def mock_markov_chain():
    tpm = np.array([[0.5, 0.5], [0.3, 0.7]])
    states = ["Rain", "No-Rain"]
    mc = MarkovChain(tpm, states)
    return mc


def test_markov_chain_neural_network_init(mock_markov_chain):
    with pytest.raises(ValueError):
        MarkovChainNeuralNetwork("invalid_type", 2)

    with pytest.raises(ValueError):
        mc = MarkovChain(None, None)
        MarkovChainNeuralNetwork(mc, 2)


def test_markov_chain_neural_network_forward(mock_markov_chain):
    mc_nn = MarkovChainNeuralNetwork(mock_markov_chain, 2)
    input_data = torch.tensor([[torch.rand(1), 0.1, 0.9]])
    output = mc_nn(input_data)
    assert output.shape == (1, 2)


def test_markov_chain_neural_network_training(mock_markov_chain):
    mc_nn = MarkovChainNeuralNetwork(mock_markov_chain, 2)
    mc_nn.train_model(1000, 10, 0.01, verbose=False)
    assert mc_nn.optimizer is not None
    assert mc_nn.scheduler is not None
    assert mc_nn.loss_function is not None
    assert mc_nn.input_data is not None
    assert mc_nn.output_data is not None


def test_divergance_analysis(mock_markov_chain):
    mc_nn = MarkovChainNeuralNetwork(mock_markov_chain, 2)
    mc_nn.train_model(1000, 10, 0.01, verbose=False)
    kl_divergence = divergance_analysis(mock_markov_chain, mc_nn)
    assert isinstance(kl_divergence, float)
