import numpy as np

X = [[1.0, 2.0, 3.0, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


class Layer_Dense:
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
        """Calculates the output of the layer.

        Args:
            inputs (np.array): inputs to the layer
        """
        self.output = np.dot(inputs, self.weights) + self.biases


layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
print(layer1.output, '\n')
layer2.forward(layer1.output)
print(layer2.output)
