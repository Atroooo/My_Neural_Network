import numpy as np


class LayerDense:
    """Class to represent a layer in a neural network."""

    def __init__(self, n_inputs, n_neurons):
        """Initializes the layer with random weights and biases.

        Args:
            n_inputs (int): number of inputs
            n_neurons (int): number of neurons
        """
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        """Calculates the outputs of the layer.

        Args:
            inputs (np.array): inputs to the layer
        """
        self.output = np.dot(inputs, self.weights) + self.biases


class ActivationReLU:
    """Class to represent the ReLU activation function."""

    def forward(self, inputs):
        """Calculates the output of the ReLU activation function.

        Args:
            inputs (np.array): inputs to the activation function
        """
        self.output = np.maximum(0, inputs)


class ActivationSoftmax:
    """Class to represent the softmax activation function."""

    def forward(self, inputs):
        """Calculates the output of the softmax activation function.

        Args:
            inputs (np.array): inputs to the activation function
        """
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
