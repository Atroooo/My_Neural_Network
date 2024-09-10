import numpy as np


class LayerDense:
    """Class to represent a layer in a neural network."""

    def __init__(self, n_inputs, n_neurons):
        """Initializes the layer with random weights and biases.

        Args:
            n_inputs (int): Number of inputs to the layer.
            n_neurons (int): Number of neurons in the layer.
        """
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        """Calculates the outputs of the layer.

        Args:
            inputs (np.array): Inputs to the layer.
        """
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        """Backpropagates the gradient of the loss function.

        Args:
            dvalues (np.array): Gradient of the loss function with respect to
                the layer's outputs.
        """
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
