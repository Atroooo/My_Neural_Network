import numpy as np
from nnfs.datasets import sine_data
from layer import LayerDense
from activation_functions import ActivationReLU, ActivationLinear
from loss_functions import LossMeanSquaredError
from optimizers import OptimizerAdam

# Create dataset
X, y = sine_data()

dense1 = LayerDense(1, 64)
activation1 = ActivationReLU()
dense2 = LayerDense(64, 64)
activation2 = ActivationReLU()
dense3 = LayerDense(64, 1)
activation3 = ActivationLinear()
loss_function = LossMeanSquaredError()
# optimizer = OptimizerAdam()
# optimizer = OptimizerAdam(learning_rate=0.01, decay=1e-3)
# optimizer = OptimizerAdam(learning_rate=0.05, decay=1e-3)
optimizer = OptimizerAdam(learning_rate=0.005, decay=1e-3)

# Accuracy precision for accuracy calculation
# There are no really accuracy factor for regression problem,
# but we can simulate/approximate it. We'll calculate it by checking
# how many values have a difference to their ground truth equivalent
# less than given precision
# We'll calculate this precision as a fraction of standard deviation
# of al the ground truth values
accuracy_precision = np.std(y) / 250

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)

    # Calculate loss
    data_loss = loss_function.calculate(activation3.output, y)
    regularization_loss = \
        loss_function.regularization_loss(dense1) + \
        loss_function.regularization_loss(dense2) + \
        loss_function.regularization_loss(dense3)
    loss = data_loss + regularization_loss

    # Calculate accuracy from output of activation2 and targets
    # To calculate it we're taking absolute difference between
    # predictions and ground truth values and compare if differences
    # are lower than given precision value
    predictions = activation3.output
    accuracy = np.mean(np.absolute(predictions - y) <
                       accuracy_precision)
    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f} (' +
              f'data_loss: {data_loss:.3f}, ' +
              f'reg_loss: {regularization_loss:.3f}), ' +
              f'lr: {optimizer.current_learning_rate}')

    # Backward pass
    loss_function.backward(activation3.output, y)
    activation3.backward(loss_function.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()
