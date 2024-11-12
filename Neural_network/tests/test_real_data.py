import os
import cv2
import numpy as np
from model import Model
from layer import LayerDense
from activation_functions import ActivationReLU, ActivationSoftmax
from loss_functions import LossCategoricalCrossentropy
from optimizers import OptimizerAdam
from accuracy import AccuracyCategorical


def load_mnist_dataset(dataset, path):
    """Load MNIST dataset

    Args:
        dataset (np.array): Dataset to load
        path (str): Path to the dataset

    Returns:
        np.array: Data and labels
    """
    # Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))

    # Create lists for samples and labels
    X = []
    y = []

    # For each label folder
    for label in labels:
        # And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            # Read the image
            image = cv2.imread(os.path.join(
                                path, dataset, label, file
                                ), cv2.IMREAD_UNCHANGED)
            # And append it and a label to the lists
            X.append(image)
            y.append(label)
            # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')


def create_data_mnist(path):
    """Create data from MNIST dataset

    Args:
        path (str): Path to the MNIST dataset

    Returns:
        np.array: Training data and testing data
    """
    # Load both sets separately
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)
    # And return all the data
    return X, y, X_test, y_test


print("Loading data...")

X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

print("Data loaded. Starting training...")

# Shuffle the training dataset
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

# Scale and reshape samples
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) -
          127.5) / 127.5

print("################## Test 1 ##################\n")
# Instantiate the model
model = Model()

# Add layers
model.add(LayerDense(X.shape[1], 64))
model.add(ActivationReLU())
model.add(LayerDense(64, 64))
model.add(ActivationReLU())
model.add(LayerDense(64, 10))
model.add(ActivationSoftmax())

# Set loss, optimizer and accuracy objects
model.set(
    loss=LossCategoricalCrossentropy(),
    optimizer=OptimizerAdam(decay=5e-5),
    accuracy=AccuracyCategorical()
)

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, validation_data=(X_test, y_test),
            epochs=5, batch_size=128, print_every=100)


print("\n################## Test 2 ##################\n")
# Instantiate the model
model = Model()

model.add(LayerDense(X.shape[1], 128))
model.add(ActivationReLU())
model.add(LayerDense(128, 128))
model.add(ActivationReLU())
model.add(LayerDense(128, 10))
model.add(ActivationSoftmax())

# Set loss, optimizer and accuracy objects
model.set(
    loss=LossCategoricalCrossentropy(),
    optimizer=OptimizerAdam(decay=1e-3),
    accuracy=AccuracyCategorical()
)
# Finalize the model
model.finalize()

# Train the model
model.train(X, y, validation_data=(X_test, y_test),
            epochs=10, batch_size=128, print_every=100)

print("Evaluation :")
model.evaluate(X, y)

print("\nCreating a new model using the same parameters...")
parameters = model.get_parameters()

model = Model()
model.add(LayerDense(X.shape[1], 128))
model.add(ActivationReLU())
model.add(LayerDense(128, 128))
model.add(ActivationReLU())
model.add(LayerDense(128, 10))
model.add(ActivationSoftmax())
model.set(
    loss=LossCategoricalCrossentropy(),
    accuracy=AccuracyCategorical())
model.finalize()

# Set model with parameters instead of training it
model.set_parameters(parameters)

model.evaluate(X_test, y_test)

model.save_parameters('fashion_mnist.parms')

print("\n################## Test 4 ##################\n")

model = Model()
model.add(LayerDense(X.shape[1], 128))
model.add(ActivationReLU())
model.add(LayerDense(128, 128))
model.add(ActivationReLU())
model.add(LayerDense(128, 10))
model.add(ActivationSoftmax())
model.set(
    loss=LossCategoricalCrossentropy(),
    accuracy=AccuracyCategorical())
model.finalize()

# Use the parameters from the file
print("Loading the parameters from the file...")
model.load_parameters('fashion_mnist.parms')
model.evaluate(X_test, y_test)

model.save('fashion_mnist.model')

print("\n################## Test 5 ##################\n")

# Load the model
model = Model.load('fashion_mnist.model')
print("Model loaded from the file.")
model.evaluate(X_test, y_test)

confidences = model.predict(X_test[:5])
predictions = model.output_layer_activation.predictions(confidences)
print(predictions)
print(y_test[:5])
