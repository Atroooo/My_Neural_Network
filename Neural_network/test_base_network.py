import nnfs
from layer import LayerDense
from activation_functions import ActivationReLU, ActivationSoftmax
from loss_functions import LossCategoricalCrossentropy
from nnfs.datasets import spiral_data

# Test layer, activation function and loss function

print("################## Test 1 ##################\n")

X = [[1.0, 2.0, 3.0, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

layer1 = LayerDense(4, 5)
layer2 = LayerDense(5, 2)

layer1.forward(X)
print(layer1.output, '\n')
layer2.forward(layer1.output)
print(layer2.output, '\n')

print("################## Test 2 ##################\n")

nnfs.init()
X, y = spiral_data(100, 3)

layer1 = LayerDense(2, 5)
activation1 = ActivationReLU()

layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output, '\n')

print("################## Test 3 ##################\n")

X, y = spiral_data(samples=100, classes=3)
dense1 = LayerDense(2, 3)
activation1 = ActivationReLU()

dense2 = LayerDense(3, 3)
activation2 = ActivationSoftmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5], '\n')

print("################## Test 4 ##################\n")
X, y = spiral_data(samples=100, classes=3)

dense1 = LayerDense(2, 3)
activation1 = ActivationReLU()

dense2 = LayerDense(3, 3)
activation2 = ActivationSoftmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss_function = LossCategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)

print("Loss:", loss)
