import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from layer import LayerDense, LayerDropout
from activation_functions import ActivationReLU
from loss_functions import ActivationSoftmaxLossCategoricalCrossentropy
from optimizers import OptimizerAdam


nnfs.init()

print("################## Test 1 ##################\n")

X, y = spiral_data(samples=1000, classes=3)

# dense1 = LayerDense(2, 64, weight_regularizer_l2=5e-4,
# bias_regularizer_l2=5e-4)
dense1 = LayerDense(2, 512, weight_regularizer_l2=5e-4,
                    bias_regularizer_l2=5e-4)
activation1 = ActivationReLU()

# Create dropout layer
dropout1 = LayerDropout(0.1)

# dense2 = LayerDense(64, 3)
dense2 = LayerDense(512, 3)
loss_activation = ActivationSoftmaxLossCategoricalCrossentropy()
optimizer = OptimizerAdam(learning_rate=0.05, decay=5e-5)

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    # Perform a forward pass through Dropout layer
    dropout1.forward(activation1.output)

    dense2.forward(dropout1.output)

    data_loss = loss_activation.forward(dense2.output, y)
    regularization_loss = \
        loss_activation.loss.regularization_loss(dense1) + \
        loss_activation.loss.regularization_loss(dense2)
    loss = data_loss + regularization_loss
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)

    accuracy = np.mean(predictions == y)
    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f} (' +
              f'data_loss: {data_loss:.3f}, ' +
              f'reg_loss: {regularization_loss:.3f}), ' +
              f'lr: {optimizer.current_learning_rate}')

    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    dropout1.backward(dense2.dinputs)
    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

# Validate the model
X_test, y_test = spiral_data(samples=100, classes=3)
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)

loss = loss_activation.forward(dense2.output, y_test)
predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == y_test)

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')
