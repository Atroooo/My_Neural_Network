import numpy as np


class OptimizerSGD:
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        """Initializes the optimizer with the given learning rate, decay and
        momentum.

        Args:
            learning_rate (float, optional): Learning rate of the optimizer.
                Defaults to 1.0.
            decay (float, optional): Decay rate of the learning rate.
            momentum (float, optional): Momentum value. Defaults to 0.
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        """Call once before any parameter updates.
        Updates the learning rate based on the decay rate."""
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        """Updates the weights and biases of the given layer.

        Args:
            layer (np.array): Layer to update.
        """
        if self.momentum:
            # If layer does not contain momentum arrays, create them
            # filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                # If there is no momentum array for weights,
                # The array doesn't exist for biases yet either.
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            # Vanilla SGD updates
            weight_updates = -self.current_learning_rate * \
                layer.dweights
            bias_updates = -self.current_learning_rate * \
                layer.dbiases
        # Update weights and biases using either vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        """Call once after any parameter updates.
        Increments the number of iterations."""
        self.iterations += 1


class OptimizerAdagrad:
    """Adaptive Gradient optimizer. Built on the SGD optimizer."""
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        """Initializes the optimizer with the given learning rate, decay and
        epsilon.

        Args:
            learning_rate (float, optional): Learning rate of the optimizer.
                Defaults to 1.0.
            decay (float, optional): Decay rate of the learning rate.
                Defaults to 0.0.
            epsilon (float, optional): Small value to avoid division by zero.
                Defaults to 1e-7.
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self):
        """Call once before any parameter updates.
        Updates the learning rate based on the decay rate."""
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        """Updates the weights and biases of the given layer.

        Args:
            layer (np.array): Layer to update.
        """
        # If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * \
            layer.dweights / \
            (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
            layer.dbiases / \
            (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        """Call once after any parameter updates.
        Increments the number of iterations."""
        self.iterations += 1


class OptimizerRMSprop:
    """RMSprop optimizer. Built on the Adagrad optimizer."""

    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 rho=0.9):
        """Initializes the optimizer with the given learning rate, epsilon
            and rho.

        Args:
            learning_rate (float, optional): Learning rate of the optimizer.
                Defaults to 1.0.
            decay (float, optional): Decay rate of the learning rate.
                Defaults to 0.0.
            epsilon (float, optional): Small value to avoid division by zero.
                Defaults to 1e-7.
            rho (float, optional): Decay rate of the cache.
                Defaults to 0.9.
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):
        """Call once before any parameter updates.
        Updates the learning rate based on the decay rate."""
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        """Updates the weights and biases of the given layer.

        Args:
            layer (np.array): Layer to update.
        """
        # If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + \
            (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + \
            (1 - self.rho) * layer.dbiases**2

        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * \
            layer.dweights / \
            (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
            layer.dbiases / \
            (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        """Call once after any parameter updates.
        Increments the number of iterations."""
        self.iterations += 1


class OptimizerAdam:
    """Adam optimizer. Built on RMSprop and Momentum optimizers."""
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        """Initializes the optimizer with the given learning rate, dacay,
        epsilon, beta_1 and beta_2.

        Args:
            learning_rate (float, optional): Learning rate of the optimizer.
                Defaults to 0.001.
            decay (float, optional): Decay rate of the learning rate.
                Defaults to 0.0.
            epsilon (float, optional): Small value to avoid division by zero.
                Defaults to 1e-7.
            beta_1 (float, optional): Divider for the momentum to correct it.
                Defaults to 0.9.
            beta_2 (float, optional): Divider for the cache to correct it.
                Defaults to 0.999.
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        """Call once before any parameter updates.
        Updates the learning rate based on the decay rate."""
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        """Updates the weights and biases of the given layer.

        Args:
            layer (np.array): Layer to update.
        """
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum with current gradients
        layer.weight_momentums = self.beta_1 * \
            layer.weight_momentums + \
            (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * \
            layer.bias_momentums + \
            (1 - self.beta_1) * layer.dbiases

        # Get corrected momentum self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))

        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases**2

        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * \
            weight_momentums_corrected / \
            (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
            bias_momentums_corrected / \
            (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        """Call once after any parameter updates.
        Increments the number of iterations."""
        self.iterations += 1
