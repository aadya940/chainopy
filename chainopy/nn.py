import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import warnings

from .MarkovChain import MarkovChain


class MarkovChainNeuralNetwork(nn.Module):
    def __init__(self, markov_chain, num_layers):
        super().__init__()

        if not isinstance(markov_chain, MarkovChain):
            raise ValueError("Object of type Markov Chain required.")

        if markov_chain.tpm is None:
            raise ValueError(
                f"Can't pass a MarkovChain object with type of type {type(markov_chain.tpm)}"
            )

        self._mc = markov_chain
        self._tpm = torch.Tensor(self._mc.tpm)
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
        for layer in self._layers[:-1]:
            x = F.relu(layer(x))

        return F.softmax(self._layers[-1](x), dim=1)

    def _generate_training_data(self, num_samples):
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
        weights_dict = self.state_dict()

        return {
            layer_name: param
            for layer_name, param in weights_dict.items()
            if "weight" in layer_name
        }

    def simulate_random_walk(self, start_state, steps):
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
