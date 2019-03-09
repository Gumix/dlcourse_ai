import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''

    tp = np.sum(np.logical_and(prediction, ground_truth))
    tn = np.sum(np.logical_and(np.logical_not(prediction), np.logical_not(ground_truth)))
    fp = np.sum(np.logical_and(prediction, np.logical_not(ground_truth)))
    fn = np.sum(np.logical_and(np.logical_not(prediction), ground_truth))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''

    return np.sum(prediction == ground_truth) / prediction.shape
