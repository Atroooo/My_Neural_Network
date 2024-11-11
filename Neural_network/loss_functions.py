import numpy as np
from activation_functions import ActivationSoftmax


class Loss:
    """Class to represent the loss function."""

    def remember_trainable_layers(self, trainable_layers):
        """Sets/remembers the trainable layers for the loss function.

        Args:
            trainable_layers (array): Trainable layers.
        """
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, *, include_regularization=False):
        """Calculates the loss between the predicted and true values.

        Args:
            output (np.array): Predicted values.
            y (np.array): True values.
            include_regularization (bool, optional): Whether to include

        Returns:
            np.array: Loss values.
        """
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        # Add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        # If just data loss - return it
        if not include_regularization:
            return data_loss

        return data_loss, self.regularization_loss()

    def calculate_accumulated(self, *, include_regularization=False):
        """Calculates the accumulated loss.

        Args:
            include_regularization (bool, optional): Whether to include

        Returns:
            float: Accumulated loss.
        """
        data_loss = self.accumulated_sum / self.accumulated_count
        if not include_regularization:
            return data_loss

        return data_loss, self.regularization_loss()

    def regularization_loss(self):
        """Calculates the regularization loss for the layer.

        Args:
            layer (LayerDense): Layer to calculate the regularization loss for.

        Returns:
            float: Regularization loss.
        """
        # 0 by default
        regularization_loss = 0

        for layer in self.trainable_layers:

            # L1 regularization - weights
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * \
                            np.sum(np.abs(layer.weights))

            # L2 regularization - weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * \
                            np.sum(layer.weights *
                                   layer.weights)
            # L1 regularization - biases
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * \
                            np.sum(np.abs(layer.biases))

            # L2 regularization - biases
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * \
                            np.sum(layer.biases *
                                   layer.biases)

        return regularization_loss

    def new_pass(self):
        """Resets the accumulated loss and count for the next pass."""
        self.accumulated_sum = 0
        self.accumulated_count = 0


class LossCategoricalCrossentropy(Loss):
    """Class to represent the categorical crossentropy loss function."""

    def forward(self, y_pred, y_true):
        """Calculates the categorical crossentropy loss between the predicted
            and true values.

        Args:
            y_pred (np.array): Predicted values.
            y_true (np.array): True values.

        Returns:
            np.array: Loss values.
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
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
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

    def __init__(self):  # Not needed for the model class
        """Creates a combined activation and loss function object."""
        self.activation = ActivationSoftmax()
        self.loss = LossCategoricalCrossentropy()

    def forward(self, inputs, y_true):  # Not needed for the model class
        """Performs a forward pass of the combined activation and
            loss function.

        Args:
            inputs (np.array): Inputs to the model.
            y_true (np.array): True labels.

        Returns:
            np.array: Loss value.
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
        # Normalize gradient
        self.dinputs = self.dinputs / samples


class LossBinaryCrossentropy(Loss):
    """Class to represent the binary crossentropy loss function."""

    def forward(self, y_pred, y_true):
        """Calculates the binary crossentropy loss between the predicted

        Args:
            y_pred (np.array): Predicted values.
            y_true (np.array): True values.

        Returns:
            np.array: Loss values.
        """
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        sample_loss = -(y_true * np.log(y_pred_clipped) +
                        (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_loss = np.mean(sample_loss, axis=-1)

        return sample_loss

    def backward(self, dvalues, y_true):
        """Backpropagates the gradient of the loss function.

        Args:
            dvalues (np.array): Gradient of the loss function with respect to
                the model's output.
            y_true (np.array): True values.
        """
        samples = len(dvalues)
        outputs = len(dvalues[0])

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # Calculate gradient
        self.dinputs = -(y_true / clipped_dvalues -
                         (1 - y_true) / (1 - clipped_dvalues)) / outputs

        # Normalize gradient
        self.dinputs = self.dinputs / samples


class LossMeanSquaredError(Loss):
    """Class to represent the mean squared error loss function. (L2 loss)"""

    def forward(self, y_pred, y_true):
        """Calculates the mean squared error loss between the predicted
            and true values.

        Args:
            y_pred (np.array): Predicted values.
            y_true (np.array): True values.

        Returns:
            np.array: Loss values.
        """
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        """Backpropagates the gradient of the loss function.

        Args:
            dvalues (np.array): Gradient of the loss function with respect to
                the model's output.
            y_true (np.array): True values.
        """
        samples = len(dvalues)
        outputs = len(dvalues[0])

        # Gradient on values
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples


class Loss_MeanAbsoluteError(Loss):
    """Class to represent the mean absolute error loss function. (L1 loss)"""

    def forward(self, y_pred, y_true):
        """Calculates the mean absolute error loss between the predicted

        Args:
            y_pred (np.array): Predicted values.
            y_true (np.array): True values.

        Returns:
            np.array: Loss values.
        """
        # Calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        # Return losses
        return sample_losses

    def backward(self, dvalues, y_true):
        """Backpropagates the gradient of the loss function.

        Args:
            dvalues (np.array): Gradient of the loss function with respect to
                the model's output.
            y_true (np.array): True values.
        """
        samples = len(dvalues)
        outputs = len(dvalues[0])

        # Calculate gradient
        self.dinputs = np.sign(y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples
