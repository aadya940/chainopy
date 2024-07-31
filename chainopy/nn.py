import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import warnings
import random

from .markov_chain import MarkovChain
from ._backend import _learn_matrix


class MarkovChainNeuralNetwork(nn.Module):
    """
    Neural network for simulating Markov chain behavior.

    Args
    ----
        markov_chain : chainopy.MarkovChain
            Markov chain object.
        num_layers : int
            Number of layers in the neural network.

    Attributes
    ----------
        input_dim : tuple
            Input shape of the neural network. Calculated
            using the transition matrix.
        output_dim : tuple
            Output shape of the neural network. Calculated
            using the transition matrix.

    Raises
    ------
        ValueError: If markov_chain is not of type MarkovChain.
    """

    def __init__(self, markov_chain, num_layers):
        super().__init__()

        if not isinstance(markov_chain, MarkovChain):
            raise ValueError("Object of type Markov Chain required.")

        if markov_chain.tpm is None:
            raise ValueError(
                f"Can't pass a MarkovChain object with type of type {type(markov_chain.tpm)}"
            )

        self._mc = markov_chain
        self._tpm = torch.from_numpy(self._mc.tpm)
        _shape = markov_chain.tpm.shape[0]

        self.input_dim = _shape + 1
        self.output_dim = _shape

        self._layers = []
        self._layers.append(nn.Linear(self.input_dim, self.output_dim))

        for _ in range(num_layers - 1):
            self._layers.append(nn.Linear(self.output_dim, self.output_dim))

        self._layers = nn.ModuleList(self._layers)

        self.input_data = None
        self.output_data = None

    def forward(self, x):
        """
        Forward pass of the neural network.

        Args
        ----
            x : torch.tensor
                Input data.

        Returns
        -------
            torch.Tensor: Output data after passing through the network.

        """

        for layer in self._layers[:-1]:
            x = F.relu(layer(x))

        return F.softmax(self._layers[-1](x), dim=1)

    def _generate_training_data(self, num_samples):
        """
        Generates training data for the model.

        Args
        ----
            num_samples : int
                Number of samples to generate.
                In reality, number of samples are multiple
                of num_states, nearest to `num_samples` for
                equally distribution of samples.

        Returns
        -------
            torch.Tensor: Input data.
            torch.Tensor: Output data.

        References
        ----------
            Makov-Chain-Neural-Networks<https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w42/Awiszus_Markov_Chain_Neural_CVPR_2018_paper.pdf>_.
        """

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            states = [
                MarkovChain._vectorize(self._mc.states, i) for i in self._mc.states
            ]

            states = torch.from_numpy(np.array(states))

            _cf = torch.cumsum(self._tpm, dim=1)

            input_data = []
            output_data = []

            _num_states = len(self._mc.states)

            j = 0

            for state in states:
                step = num_samples // len(self._mc.states)
                _random_values = torch.rand(step)

                for val in _random_values:
                    _input = torch.cat((val.view(1), state.squeeze()))
                    decision_idx = torch.searchsorted(_cf[j], val.item())
                    _one_hot_array = np.zeros(_num_states)
                    _one_hot_array[decision_idx] = 1
                    _output = torch.from_numpy(_one_hot_array)
                    input_data.append(_input)
                    output_data.append(_output)

                j += 1

            input_data = torch.stack(input_data).float()
            output_data = torch.stack(output_data).float()

            return input_data, output_data.squeeze()

    def train_model(
        self,
        num_samples,
        epochs,
        learning_rate,
        momentum=0.9,
        verbose=True,
        patience=500,
        factor=0.5,
    ):
        """

        Trains the neural network model.

        Args
        ----
            num_samples : int
                Number of training samples.
            epochs : int
                Number of training epochs.
            learning_rate : float
                Learning rate for optimization.
            momentum : float
                Momentum factor (default is 0.9).
            verbose : bool, optional
                If True, prints training progress (default is True).
            patience : int, optional
                Patience parameter for learning rate scheduler (default is 500).
            factor : float, optional
                Factor by which the learning rate will be reduced (default is 0.5).
        """

        self.optimizer = optim.SGD(
            self.parameters(), lr=learning_rate, momentum=momentum
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            threshold=1,
        )

        self.loss_function = nn.CrossEntropyLoss()

        self.input_data, self.output_data = self._generate_training_data(num_samples)

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self(self.input_data)
            loss = self.loss_function(outputs, self.output_data)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss)

            if verbose:
                print(f"Epoch: {epoch}/{epochs}, Loss: {loss.item():.4f}")
            else:
                if epoch == epochs - 1:
                    print(f"Epoch: {epoch}/{epochs}, Loss: {loss.item():.4f}")

    def get_weights(self):
        """
        Returns the weights of the model.

        Returns
        -------
            dict: Dictionary containing layer names and corresponding weights.

        """
        weights_dict = self.state_dict()

        return {
            layer_name: param
            for layer_name, param in weights_dict.items()
            if "weight" in layer_name
        }

    def simulate_random_walk(self, start_state, steps):
        """
        Simulates a random walk based on the trained model.

        Args
        ----
            start_state: int
                Starting state for the random walk.
            steps: int
                Number of steps to simulate.

        Returns
        -------
            list: List of states representing the random walk.

        """

        current_state = torch.from_numpy(
            MarkovChain._vectorize(self._mc.states, start_state)
        )

        random_values = torch.rand(steps)
        markov_walk = [start_state]

        with torch.no_grad():
            for rand_value in random_values:
                input_vector = torch.cat(
                    (rand_value.view(1), current_state.squeeze()), dim=0
                ).float()
                next_state_probabilities = self(input_vector.unsqueeze(0))
                next_state = torch.argmax(next_state_probabilities.squeeze())
                markov_walk.append(self._mc.states[next_state])
                current_state = F.one_hot(
                    torch.tensor([next_state]), num_classes=self.output_dim
                ).squeeze()

        return markov_walk


def divergance_analysis(mc: MarkovChain, nn: MarkovChainNeuralNetwork) -> float:
    """
    KL Divergance between `MarkovChain.tpm` and
    `MarkovChain().fit(MarkovChainNeuralNetwork.simulate_random_walk).tpm`.

    Args
    ----
        mc: MarkovChain
            Original Markov Chain that is used to fit the `MarkovChainNeuralNetwork`.
        nn: MarkovChainNeuralNetwork
            The fitted `MarkovChainNeuralNetwork`.

    Returns
    -------
        float: KL-Divergance
            Lower the KL-Divergance, better the fit.

    NOTES
    -----
        KL-Divergance<https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>_.
    """
    _real_tpm = mc.tpm.flatten()
    _epsilon = mc.epsilon

    def _generate_estimated_tpm():
        # len(mc.states) * 200 steps so there are enough states for efficient estimation
        _observed_seq_list = nn.simulate_random_walk(
            random.choice(mc.states), len(mc.states) * 200
        )
        _estimated_tpm, _ = _learn_matrix.learn_matrix_cython(
            _observed_seq_list, epsilon=_epsilon
        )
        return _estimated_tpm

    _est_tpm = _generate_estimated_tpm().flatten()
    _kl_divergance = np.sum(_real_tpm * np.log(_real_tpm / _est_tpm))
    return _kl_divergance
