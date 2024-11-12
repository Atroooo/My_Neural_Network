from nnfs.datasets import sine_data, spiral_data
from model import Model
from layer import LayerDense, LayerDropout
from activation_functions import ActivationReLU, ActivationLinear, \
    ActivationSigmoid, ActivationSoftmax
from loss_functions import LossMeanSquaredError, LossBinaryCrossentropy, \
    LossCategoricalCrossentropy
from optimizers import OptimizerAdam
from accuracy import AccuracyRegression, AccuracyCategorical

print("################## Test 1 ##################\n")

X, y = sine_data()

# Instantiate the model
model = Model()

# Add layers
model.add(LayerDense(1, 64))
model.add(ActivationReLU())
model.add(LayerDense(64, 64))
model.add(ActivationReLU())
model.add(LayerDense(64, 1))
model.add(ActivationLinear())

# Set loss, optimizer and accuracy objects
model.set(
    loss=LossMeanSquaredError(),
    optimizer=OptimizerAdam(learning_rate=0.005, decay=1e-3),
    accuracy=AccuracyRegression())

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, epochs=10000, print_every=100)

print("\n################## Test 2 ##################\n")

# Create train and test dataset
X, y = spiral_data(samples=100, classes=2)
X_test, y_test = spiral_data(samples=100, classes=2)

# Reshape labels to be a list of lists
# Inner list contains one output (either 0 or 1)
# per each output neuron, 1 in this case
y = y.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Instantiate the model
model = Model()

# Add layers
model.add(LayerDense(2, 64, weight_regularizer_l2=5e-4,
                     bias_regularizer_l2=5e-4))
model.add(ActivationReLU())
model.add(LayerDense(64, 1))
model.add(ActivationSigmoid())

# Set loss, optimizer and accuracy objects
model.set(
    loss=LossBinaryCrossentropy(),
    optimizer=OptimizerAdam(decay=5e-7),
    accuracy=AccuracyCategorical(binary=True)
)

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, validation_data=(X_test, y_test),
            epochs=10000, print_every=100)

print("\n################## Test 3 ##################\n")

# Create dataset
X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

# Instantiate the model
model = Model()

# Add layers
model.add(LayerDense(2, 512, weight_regularizer_l2=5e-4,
                     bias_regularizer_l2=5e-4))
model.add(ActivationReLU())
model.add(LayerDropout(0.1))
model.add(LayerDense(512, 3))
model.add(ActivationSoftmax())

# Set loss, optimizer and accuracy objects
model.set(
    loss=LossCategoricalCrossentropy(),
    optimizer=OptimizerAdam(learning_rate=0.05, decay=5e-5),
    accuracy=AccuracyCategorical()
)

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, validation_data=(X_test, y_test),
            epochs=10000, print_every=100)
