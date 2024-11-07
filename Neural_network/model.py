from layer import LayerInput
from activation_functions import ActivationSoftmax
from loss_functions import LossCategoricalCrossentropy, \
    ActivationSoftmaxLossCategoricalCrossentropy


class Model:
    """Class to represent a neural network model."""

    def __init__(self):
        """Initializes the model with an empty list of layers."""
        self.layers = []

        # Softmax classifier's output object
        self.softmax_classifier_output = None

    def add(self, layer):
        """Adds a layer to the model.

        Args:
            layer (object): Layer to add to the model.
        """
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):
        """Sets the loss function, optimizer and accuracy for the model.

        Args:
            loss (Loss): Loss function to use.
            optimizer (Optimizer): Optimizer to use.
            accuracy (Accuracy): Accuracy object to use.
        """
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def finalize(self):
        """Finalizes the model by setting up all the connections
        between the layers."""
        # Create and set the input layer
        self.input_layer = LayerInput()

        # Count all the objects
        layer_count = len(self.layers)

        # Initialize a list containing trainable layers:
        self.trainable_layers = []

        # Iterate the objects
        for i in range(layer_count):
            # If it's the first layer,
            # the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]

            # All layers except for the first and the last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            # The last layer - the next object is the loss
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            # If layer contains an attribute called "weights",
            # it's a trainable layer -
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        # Update loss object with trainable layers
        self.loss.remember_trainable_layers(self.trainable_layers)

        # If output activation is Softmax and
        # loss function is Categorical Cross-Entropy
        # create an object of combined activation
        # and loss function containing faster gradient calculation
        if isinstance(self.layers[-1], ActivationSoftmax) and \
           isinstance(self.loss, LossCategoricalCrossentropy):
            self.softmax_classifier_output = \
                ActivationSoftmaxLossCategoricalCrossentropy()

    def forward(self, X, training):
        """Performs forward pass of the model.

        Args:
            X (np.array): input data.
            training (bool): whether the model is in training mode.

        Returns:
            np.array: output of the model.
        """
        # Call forward method on the input layer
        # this will set the output property that
        # the first layer in "prev" object is expecting
        self.input_layer.forward(X, training)

        # Call forward method of every object in a chain
        # Pass output of the previous object as a parameter
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        # "layer" is now the last object from the list,
        # return its output
        return layer.output

    def backward(self, output, y):
        """Performs backward pass of the model.

        Args:
            output (np.array): Output of the model.
            y (np.array): Target data.
        """
        # If softmax classifier
        if self.softmax_classifier_output is not None:
            # First call backward method
            # on the combined activation/loss
            # this will set dinputs property
            self.softmax_classifier_output.backward(output, y)
            # Since we'll not call backward method of the last layer
            # which is Softmax activation
            # as we used combined activation/loss
            # object, let's set dinputs in this object
            self.layers[-1].dinputs = \
                self.softmax_classifier_output.dinputs
            # Call backward method going through
            # all the objects but last
            # in reversed order passing dinputs as a parameter
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return

        # First call backward method on the loss
        # this will set dinputs property that the last
        # layer will try to access shortly
        self.loss.backward(output, y)

        # Call backward method going through all the objects
        # in reversed order passing dinputs as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):
        """Trains the model on the given data.

        Args:
            X (ndarray): Input data.
            y (ndarray): Target data.
            epochs (int): Number of epochs to train for.
            print_every (int): How often to print the training progress.
            validation_data (tuple): Data to use for validation.
        """
        # Initialize accuracy object
        self.accuracy.init(y)

        # Main training loop
        for epoch in range(1, epochs+1):
            # Perform the forward pass
            output = self.forward(X, training=True)

            # Calculate loss
            data_loss, regularization_loss = \
                self.loss.calculate(output, y, include_regularization=True)
            loss = data_loss + regularization_loss

            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(
                    output)
            accuracy = self.accuracy.calculate(predictions, y)

            # Perform the backward pass
            self.backward(output, y)

            # Optimize (update parameters)
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            # Print a summary
            if not epoch % print_every:
                print(f'epoch: {epoch}, ' +
                      f'acc: {accuracy:.3f}, ' +
                      f'loss: {loss:.3f} (' +
                      f'data_loss: {data_loss:.3f}, ' +
                      f'reg_loss: {regularization_loss:.3f}), ' +
                      f'lr: {self.optimizer.current_learning_rate}')

        # If there is the validation data
        if validation_data is not None:

            # For better readability
            X_val, y_val = validation_data

            # Perform the forward pass
            output = self.forward(X_val, training=False)

            # Calculate the loss
            loss = self.loss.calculate(output, y_val)

            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(
                    output)
            accuracy = self.accuracy.calculate(predictions, y_val)

            # Print a summary
            print('validation, ' +
                  f'acc: {accuracy:.3f}, ' +
                  f'loss: {loss:.3f}')
