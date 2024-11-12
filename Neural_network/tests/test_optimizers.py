import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from layer import LayerDense
from activation_functions import ActivationReLU
from loss_functions import ActivationSoftmaxLossCategoricalCrossentropy
from optimizers import OptimizerSGD, OptimizerAdagrad, OptimizerRMSprop, \
    OptimizerAdam


nnfs.init()

print("################## Test 1 ##################\n")

X, y = spiral_data(samples=100, classes=3)
dense1 = LayerDense(2, 64)
activation1 = ActivationReLU()
dense2 = LayerDense(64, 3)
loss_activation = ActivationSoftmaxLossCategoricalCrossentropy()

# Create optimizer
# optimizer = OptimizerSGD()
# optimizer = OptimizerSGD(0.85)
# optimizer = OptimizerSGD(decay=1e-2)
# optimizer = OptimizerSGD(decay=1e-3)
# optimizer = OptimizerSGD(decay=1e-3, momentum=0.5)
optimizer = OptimizerSGD(decay=1e-3, momentum=0.9)

# Train in loop
for epoch in range(10001):
    # Perform a forward pass of our training data through this layer
    dense1.forward(X)

    # Perform a forward pass through activation function
    activation1.forward(dense1.output)

    # Perform a forward pass through second Dense layer
    dense2.forward(activation1.output)

    # Perform a forward pass through the activation/loss function
    loss = loss_activation.forward(dense2.output, y)

    # Calculate accuracy from output of activation2 and targets
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f} ' +
              f'lr: {optimizer.current_learning_rate}')

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

print("\n################## Test 2 ##################\n")

nnfs.init()

X, y = spiral_data(samples=100, classes=3)
dense1 = LayerDense(2, 64)
activation1 = ActivationReLU()
dense2 = LayerDense(64, 3)
loss_activation = ActivationSoftmaxLossCategoricalCrossentropy()

# Create optimizer
optimizer = OptimizerAdagrad(decay=1e-4)

# Train in loop
for epoch in range(10001):
    # Perform a forward pass of our training data through this layer
    dense1.forward(X)

    # Perform a forward pass through activation function
    activation1.forward(dense1.output)

    # Perform a forward pass through second Dense layer
    dense2.forward(activation1.output)

    # Perform a forward pass through the activation/loss function
    loss = loss_activation.forward(dense2.output, y)

    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, ' +
              f'lr: {optimizer.current_learning_rate}')

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

print("\n################## Test 3 ##################\n")

X, y = spiral_data(samples=100, classes=3)
dense1 = LayerDense(2, 64)
activation1 = ActivationReLU()
dense2 = LayerDense(64, 3)
loss_activation = ActivationSoftmaxLossCategoricalCrossentropy()

# Create optimizer
# optimizer = OptimizerRMSprop(decay=1e-4)
optimizer = OptimizerRMSprop(learning_rate=0.02, decay=1e-5,
                             rho=0.999)

# Train in loop
for epoch in range(10001):
    # Perform a forward pass of our training data through this layer
    dense1.forward(X)

    # Perform a forward pass through activation function
    activation1.forward(dense1.output)

    # Perform a forward pass through second Dense layer
    dense2.forward(activation1.output)

    # Perform a forward pass through the activation/loss function
    loss = loss_activation.forward(dense2.output, y)

    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, ' +
              f'lr: {optimizer.current_learning_rate}')

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

print("\n################## Test 4 ##################\n")

X, y = spiral_data(samples=100, classes=3)
dense1 = LayerDense(2, 64)
activation1 = ActivationReLU()
dense2 = LayerDense(64, 3)
loss_activation = ActivationSoftmaxLossCategoricalCrossentropy()

# Create optimizer
# optimizer = OptimizerAdam(learning_rate=0.02, decay=1e-5)
optimizer = OptimizerAdam(learning_rate=0.05, decay=5e-7)

# Train in loop
for epoch in range(10001):
    # Perform a forward pass of our training data through this layer
    dense1.forward(X)

    # Perform a forward pass through activation function
    activation1.forward(dense1.output)

    # Perform a forward pass through second Dense layer
    dense2.forward(activation1.output)

    # Perform a forward pass through the activation/loss function
    loss = loss_activation.forward(dense2.output, y)

    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, ' +
              f'lr: {optimizer.current_learning_rate}')

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
