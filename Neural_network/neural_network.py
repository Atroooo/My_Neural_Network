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


class Loss:
    """Class to represent the loss function."""

    def calculate(self, output, y):
        """Calculates the loss between the predicted and true values.

        Args:
            output (np.array): Predicted values.
            y (np.array): True values.
        """
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class LossCategoricalCrossentropy(Loss):
    """Class to represent the categorical crossentropy loss function."""

    def forward(self, y_pred, y_true):
        """Calculates the categorical crossentropy loss between the predicted
            and true values.

        Args:
            y_pred (np.array): Predicted values.
            y_true (np.array): True values.
        """
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values, only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        """Calculates the gradient of the categorical crossentropy
            loss function.

        Args:
            dvalues (np.array): Gradient of the loss function with respect to
                the predictions.
            y_true (np.array): True values.
        """
        samples = len(dvalues)
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues

        # Normalize gradient
        self.dinputs = self.dinputs / samples


class ActivationSoftmaxLossCategoricalCrossentropy():
    """Softmax classifier - combined Softmax activation
        and cross-entropy loss for faster backward step."""

    def __init__(self):
        """Creates a combined activation and loss function object."""
        self.activation = ActivationSoftmax()
        self.loss = LossCategoricalCrossentropy()

    def forward(self, inputs, y_true):
        """Performs a forward pass of the combined activation and
            loss function.

        Args:
            inputs (np.array): Inputs to the model.
            y_true (np.array): True labels.

        Returns:
            float: Loss value.
        """
        # Output layer's activation function
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        """Performs a backward pass of the combined activation and loss
            function.

        Args:
            dvalues (np.array): Gradient of the loss function with respect to
                the model's output.
            y_true (np.array): True labels.
        """
        samples = len(dvalues)
        # If labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


class OptimizerSGD:
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, learning_rate=1.0):
        """Initializes the optimizer with the given learning rate.

        Args:
            learning_rate (float, optional): Learning rate of the optimizer.
                Defaults to 1.0.
        """
        self.learning_rate = learning_rate

    def update_params(self, layer):
        """Updates the weights and biases of a layer.

        Args:
            layer (np.array): Layer to update.
        """
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

#249