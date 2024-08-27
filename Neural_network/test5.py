import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from neural_network import LayerDense, ActivationReLU, \
     ActivationSoftmaxLossCategoricalCrossentropy, OptimizerSGD


nnfs.init()

X, y = spiral_data(samples=100, classes=3)
dense1 = LayerDense(2, 64)
activation1 = ActivationReLU()
dense2 = LayerDense(64, 3)
loss_activation = ActivationSoftmaxLossCategoricalCrossentropy()

# Create optimizer
optimizer = OptimizerSGD()

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
              f'loss: {loss:.3f}')

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights and biases
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
