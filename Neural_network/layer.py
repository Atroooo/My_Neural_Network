import numpy as np


class LayerDense:
    """Class to represent a layer in a neural network."""

    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        """Initializes the layer with random weights and biases.

        Args:
            n_inputs (int): Number of inputs to the layer.
            n_neurons (int): Number of neurons in the layer.
            weight_regularizer_l1 (int, optional): Weight L1 regularization.
                Defaults to 0.
            weight_regularizer_l2 (int, optional): Weight L2 regularization.
                Defaults to 0.
            bias_regularizer_l1 (int, optional): Bias L1 regularization.
                Defaults to 0.
            bias_regularizer_l2 (int, optional): Bias L2 regularization.
                Defaults to 0.
        """
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs, training):
        """Calculates the outputs of the layer.

        Args:
            inputs (np.array): Inputs to the layer.
            training (bool): Flag indicating whether the model is in
                training
        """
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
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

        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1

        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * \
                             self.weights

        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * \
                            self.biases

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

    def get_parameters(self):
        """Returns the weights and biases of the layer."""
        return self.weights, self.biases

    def set_parameters(self, weights, biases):
        """Sets the weights and biases of the layer.

        Args:
            weights (np.array): Weights to set.
            biases (np.array): Biases to set.
        """
        self.weights = weights
        self.biases = biases


class LayerDropout:
    """Class to represent a dropout layer in a neural network.
    Regularization technique to prevent overfitting by randomly setting"""

    def __init__(self, rate):
        """Initializes the dropout layer with a given rate.

        Args:
            rate (float): Fraction of the input units to drop.
        """
        # Store rate, we invert it as for example for dropout
        # of 0.1 we need success rate of 0.9
        self.rate = 1 - rate

    def forward(self, inputs, training):
        """Applies dropout to the inputs.

        Args:
            inputs (np.array): Inputs to the dropout layer.
            training (bool): Flag indicating whether the model is in
                training mode.
        """
        # Save input values
        self.inputs = inputs

        # If not in the training mode - return values
        if not training:
            self.output = inputs.copy()
            return

        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate,
                                              size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        """Backpropagates the gradient of the loss function.

        Args:
            dvalues (_type_): Gradient of the loss function
                with respect to the layer's outputs.
        """
        # Gradient on values
        self.dinputs = dvalues * self.binary_mask


class LayerInput:
    """Class to represent an input layer in a neural network.
    This is considered a layer in a neural network but doesn’t
    have weights and biases associated with it. The input layer only
    contains the training data, and we’ll only use it as a “previous”
    layer to the first layer during the iteration of the layers in a loop"""

    def forward(self, inputs, training):
        """Passes the inputs forward.

        Args:
            inputs (np.array): Inputs to the layer.
            training (bool): Flag indicating whether the model is in
                training
        """
        self.output = inputs
