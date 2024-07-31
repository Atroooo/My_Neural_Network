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


class Loss:
    """Class to represent the loss function."""

    def calculate(self, output, y):
        """Calculates the loss between the predicted and true values.

        Args:
            output (np.array): predicted values
            y (np.array): true values
        """
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class LossCategoricalCrossentropy(Loss):
    """Class to represent the categorical crossentropy loss function."""

    def forward(self, y_pred, y_true):
        """Calculates the loss between the predicted and true values.

        Args:
            y_pred (np.array): predicted values
            y_true (np.array): true values
        """
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        """Calculates the gradient of the loss function.

        Args:
            dvalues (np.array): array of predictions
            y_true (np.array): array of true values
        """
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples
