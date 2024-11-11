import numpy as np


class Accuracy:
    """Class that calculates accuracy."""
    def calculate(self, predictions, y):
        """Calculates accuracy based on the predictions and
        the ground truth values.

        Args:
            predictions (np.array): Predictions made by the model.
            y (np.array): Ground truth values.

        Returns:
            float: Accuracy value.
        """
        # Get comparison results
        comparisons = self.compare(predictions, y)

        # Calculate an accuracy
        accuracy = np.mean(comparisons)

        # Add accumulated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        return accuracy

    def calculate_accumulated(self):
        """Calculates accumulated accuracy based on the accumulated sum and
        count of the samples.

        Returns:
            float: Accumulated accuracy value.
        """
        # Calculate an accuracy
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy

    def new_pass(self):
        """Resets the accumulated sum and count of the samples."""
        self.accumulated_sum = 0
        self.accumulated_count = 0


class AccuracyRegression(Accuracy):
    """Class that calculates accuracy for regression problems."""

    def __init__(self):
        """Initializes the accuracy object."""
        self.precision = None

    def init(self, y, reinit=False):
        """Initializes the precision value based on the passed-in ground truth.
        Initialization from inside the model object.

        Args:
            y (np.array): Ground truth values.
            reinit (bool, optional): Allows to recalculate precision by force.
            Defaults to False.
        """
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def compare(self, predictions, y):
        """Compares predictions to the ground truth values.

        Args:
            predictions (np.array): Predictions made by the model.
            y (np.array): Ground truth values.

        Returns:
            np.array: Array of booleans.
        """
        return np.absolute(predictions - y) < self.precision


class AccuracyCategorical(Accuracy):
    """Class that calculates accuracy for categorical problems."""

    def __init__(self, *, binary=False):
        """_summary_

        Args:
            binary (bool, optional): _description_. Defaults to False.
        """
        self.binary = binary

    # No initialization is needed
    def init(self, y):
        pass

    def compare(self, predictions, y):
        """Compares predictions to the ground truth values

        Args:
            predictions (np.array): Predictions made by the model.
            y (np.array): Ground truth values.

        Returns:
            np.array: Array of booleans.
        """
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y
