import pickle
import copy
import numpy as np
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

    def set(self, *, loss=None, optimizer=None, accuracy=None):
        """Sets the loss function, optimizer and accuracy for the model.

        Args:
            loss (Loss): Loss function to use. Defaults to None.
            optimizer (Optimizer): Optimizer to use. Defaults to None.
            accuracy (Accuracy): Accuracy object to use. Defaults to None.
        """
        if loss is not None:
            self.loss = loss

        if optimizer is not None:
            self.optimizer = optimizer

        if accuracy is not None:
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
        if self.loss is not None:
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

    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1,
              validation_data=None,):
        """Trains the model on the given data.

        Args:
            X (ndarray): Input data.
            y (ndarray): Target data.
            epochs (int): Number of epochs to train for.
            batch_size (int): Number of samples in a batch.
            print_every (int): How often to print the training progress.
            validation_data (tuple): Data to use for validation.
        """
        # Initialize accuracy object
        self.accuracy.init(y)

        # Default value if batch size is not being set
        train_steps = 1

        # Calculate number of steps
        if batch_size is not None:
            # Dividing rounds down.
            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                # Add `1` to include this not full batch
                train_steps += 1

        # Main training loop
        for epoch in range(1, epochs+1):

            print(f'epoch: {epoch}')

            # Reset accumulated values in loss and accuracy objects
            self.loss.new_pass()
            self.accuracy.new_pass()

            # Iterate over steps
            for step in range(train_steps):
                # If batch size is not set train using full dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                # Otherwise slice a batch
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                # Perform the forward pass
                output = self.forward(batch_X, training=True)

                # Calculate loss
                data_loss, regularization_loss = \
                    self.loss.calculate(output, batch_y,
                                        include_regularization=True)
                loss = data_loss + regularization_loss

                # Get predictions and calculate an accuracy
                predictions = self.output_layer_activation.predictions(
                                                output)
                accuracy = self.accuracy.calculate(predictions,
                                                   batch_y)

                # Perform backward pass
                self.backward(output, batch_y)

                # Optimize (update parameters)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                # Print a summary
                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, ' +
                          f'acc: {accuracy:.3f}, ' +
                          f'loss: {loss:.3f} (' +
                          f'data_loss: {data_loss:.3f}, ' +
                          f'reg_loss: {regularization_loss:.3f}), ' +
                          f'lr: {self.optimizer.current_learning_rate}')

        # Get and print epoch loss and accuracy
        epoch_data_loss, epoch_regularization_loss = \
            self.loss.calculate_accumulated(
                    include_regularization=True)
        epoch_loss = epoch_data_loss + epoch_regularization_loss
        epoch_accuracy = self.accuracy.calculate_accumulated()

        print('training, ' +
              f'acc: {epoch_accuracy:.3f}, ' +
              f'loss: {epoch_loss:.3f} (' +
              f'data_loss: {epoch_data_loss:.3f}, ' +
              f'reg_loss: {epoch_regularization_loss:.3f}), ' +
              f'lr: {self.optimizer.current_learning_rate}')

        # If there is the validation data
        if validation_data is not None:
            self.evaluate(*validation_data, batch_size=batch_size)

    def evaluate(self, X_val, y_val, *, batch_size=None):
        """Evaluates the model on the given data.

        Args:
            X_val (ndarray): Input data.
            y_val (ndarray): Target data.
            batch_size (int): Number of samples in a batch.
        """
        validation_steps = 1

        # Calculate number of steps
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        # Reset accumulated values in loss
        # and accuracy objects
        self.loss.new_pass()
        self.accuracy.new_pass()

        # Iterate over steps
        for step in range(validation_steps):

            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[step*batch_size:(step+1)*batch_size]
                batch_y = y_val[step*batch_size:(step+1)*batch_size]

            # Perform the forward pass
            output = self.forward(batch_X, training=False)

            # Calculate the loss
            self.loss.calculate(output, batch_y)

            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(
                            output)
            self.accuracy.calculate(predictions, batch_y)

        # Get and print validation loss and accuracy
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        # Print a summary
        print('validation, ' +
              f'acc: {validation_accuracy:.3f}, ' +
              f'loss: {validation_loss:.3f}')

    def predict(self, X, *, batch_size=None):
        """Predicts the output of the model on the given data.

        Args:
            X (np.array): Input data.
            batch_size (int): Number of samples in a batch.

        Returns:
            np.vstack: Predictions.
        """
        # Default value if batch size is not being set
        prediction_steps = 1

        # Calculate number of steps
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1

        output = []

        # Iterate over steps
        for step in range(prediction_steps):

            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]

            # Perform the forward pass
            batch_output = self.forward(batch_X, training=False)

            # Append batch prediction to the list of predictions
            output.append(batch_output)

        return np.vstack(output)

    def get_parameters(self):
        """Get the model parameters.

        Returns:
            dict: Dictionary containing the parameters.
        """
        # Create a list for parameters
        parameters = []

        # Iterate trainable layers and get their parameters
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())

        # Return a dictionary
        return parameters

    def set_parameters(self, parameters):
        """Set the model parameters.

        Args:
            parameters (dict): Dictionary containing the parameters.
        """
        # Iterate over the parameters and layers
        for parameter_set, layer in zip(parameters,
                                        self.trainable_layers):
            layer.set_parameters(*parameter_set)

    def save_parameters(self, path):
        """Save the model parameters to a file.

        Args:
            path (str): Path to the file.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)

    def load_parameters(self, path):
        """Load the model parameters from a file.

        Args:
            path (str): Path to the file.
        """
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    def save(self, path):
        """Save the model to a file.

        Args:
            path (str): Path to the file.
        """
        # Make a deep copy of current model instance
        model = copy.deepcopy(self)

        # Reset accumulated values in loss and accuracy objects
        model.loss.new_pass()
        model.accuracy.new_pass()

        # Remove data from input layer
        # and gradients from the loss object
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        # For each layer remove inputs, output and dinputs properties
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs',
                             'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)

        # Open a file in the binary-write mode and save the model
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path):
        """Load the model from a file.

        Args:
            path (str): Path to the file.

        Returns:
            Model: Model instance.
        """
        # Open file in the binary-read mode, load the model and return it
        with open(path, 'rb') as f:
            model = pickle.load(f)

        return model
