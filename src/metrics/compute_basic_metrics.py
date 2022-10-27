import numpy as np
from sklearn import metrics


def compute_basic_metrics(predictions, targets):
    """

    Takes two arrays: predictions for some messages and actual labels. Return dict with metric names and their values
    on this dataset.

    :param predictions: Two dimensional array (n x 3) with predicted labels. :param targets: Two dimensional array (n x 3) with real labels.
    :return: Dict with 8 metrics and their values: accuracy, f1_score_micro,
    f1_score_macro, precision_micro, precision_macro, recall_micro, recall_macro, hamming_loss.
    """
    predictions = np.array(predictions) >= 0.5
    return {
        "accuracy": metrics.accuracy_score(targets, predictions),
        "f1_score_micro": metrics.f1_score(targets, predictions, average="micro"),
        "f1_score_macro": metrics.f1_score(targets, predictions, average="macro"),
        "precision_micro": metrics.precision_score(targets, predictions, average="micro"),
        "precision_macro": metrics.precision_score(targets, predictions, average="macro"),
        "recall_micro": metrics.recall_score(targets, predictions, average="micro"),
        "recall_macro": metrics.recall_score(targets, predictions, average="macro"),
        "hamming_loss": metrics.hamming_loss(targets, predictions),
    }
