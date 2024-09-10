import numpy as np


class ActivationReLU:
    """Class to represent the ReLU activation function."""

    def forward(self, inputs):
        """Calculates the output of the ReLU activation function.

        Args:
            inputs (np.array): Inputs to the activation function.
        """
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        """Backpropagates the gradient of the loss function.

        Args:
            dvalues (np.array): Gradient of the loss function with respect to
                the activation's output.
        """
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0


class ActivationSoftmax:
    """Class to represent the softmax activation function."""

    def forward(self, inputs):
        """Calculates the output of the softmax activation function.

        Args:
            inputs (np.array): Inputs to the activation function.
        """
        self.inputs = inputs

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        """Calculates the gradient of the softmax activation function.

        Args:
            dvalues (np.array): Gradient of the loss function with respect to
                the activation's output.
        """
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)

            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - \
                np.dot(single_output, single_output.T)

            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,
                                         single_dvalues)
