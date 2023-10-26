import numpy as np


def MSE(predictions: np.ndarray, targets: np.ndarray) -> float:
    """ Compute the Mean Squared Error (MSE) between predictions and targets.

    The Mean Squared Error is a measure of the average squared difference
    between predicted and actual values. It's a popular metric for regression tasks.

    Formula:
    MSE = (1/n) * Σ (predictions - targets)^2

    where:
    - n is the number of samples
    - Σ denotes the sum
    - predictions are the predicted values by the model
    - targets are the true values
    TODO implement this function. This function is expected to be implemented without the use of loops.

    """
    return np.mean((predictions - targets) ** 2)


def TruePositives(predictions: np.ndarray, targets: np.ndarray):
    tp = (predictions == 1) & (targets == 1)
    return np.sum(tp)


def TrueNegatives(predictions: np.ndarray, targets: np.ndarray):
    tn = (predictions == 0) & (targets == 0)
    return np.sum(tn)


def FalsePositives(predictions: np.ndarray, targets: np.ndarray):
    fp = (predictions == 1) & (targets == 0)
    return np.sum(fp)


def FalseNegatives(predictions: np.ndarray, targets: np.ndarray):
    fn = (predictions == 0) & (targets == 1)
    return np.sum(fn)


def accuracy(predictions: np.ndarray, targets: np.ndarray):
    return np.mean(predictions == targets)


def precision(predictions: np.ndarray, targets: np.ndarray):
    tp = TruePositives(predictions, targets)
    fp = FalsePositives(predictions, targets)
    return tp / (tp + fp)


def recall(predictions: np.ndarray, targets: np.ndarray):
    tp = TruePositives(predictions, targets)
    fn = FalseNegatives(predictions, targets)
    return tp / (tp + fn)


def f1_score(predictions: np.ndarray, targets: np.ndarray):
    pr = precision(predictions, targets)
    rc = recall(predictions, targets)
    if pr == 0. or rc == 0.:
        return 0.
    return 2 * pr * rc / (pr + rc)


