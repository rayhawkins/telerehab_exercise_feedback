# Some packages that will be useful
import numpy as np  # For matrix algebra
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  # For visualizations
import torch
from torchmetrics import Accuracy, F1Score
from sklearn.metrics import confusion_matrix

# Some pseudocode for some functions that may be useful
def classification_accuracy(prediction, ground_truth):
    # Compute accuracy from predicted labels to ground_truth labels

    # Instantiate the Accuracy metric
    accuracy_metric = Accuracy(task ="binary")
    print ("accuracy_metric", accuracy_metric)

    # Compute accuracy
    accuracy = accuracy_metric(prediction, ground_truth)

    print(f"Accuracy: {accuracy.item() * 100:.2f}%")

    return classification_accuracy




def classification_f1score(prediction, ground_truth):
    # Compute F1_score from predicted labels to ground_truth labels
    task = 'binary'

    # Instantiate the F1 metric with the task argument
    f1_metric = F1Score(num_classes=1, task=task)  # For binary classification, num_classes is 1

    # Compute F1 score
    f1_score = f1_metric(prediction, ground_truth)

    print(f"F1 Score: {f1_score.item() * 100:.2f}%")

    return classification_f1score


def prediction_pixelwise_accuracy(prediction, ground_truth):
    # Compute pixelwise accuracy between generated frames of video and real frames of video
    # Question: Do we need this part for measuring the accuracy of VideoGPT reconstructed frames?
    if prediction.shape != ground_truth.shape:
        raise ValueError("Frames must have the same shape.")

    pixel_difference = torch.abs(prediction - ground_truth)
    pixelwise_accuracy = torch.mean(pixel_difference).item()
    # return mean_difference
    return pixelwise_accuracy


# def create_confusion_matrix(metrics, groups):
#
#     # Create a confusion matrix for classification accuracies of different groups of data
#     # Question: Why do we want to include groups in the confusion matrix? Shouldn't we find confusion matrix in each group seperately?
#     return matrix

# def create_confusion_matrix_classification():
#     """
#     for assessing gesture classification
#     """
#
#     return matrix

def create_confusion_matrix(ground_truth, prediction):
    """
    for assessing Binary classifications
    To Use 9 times for correct/incorrect gestures
    """
    cf_matrix = confusion_matrix(ground_truth, prediction)
    return cf_matrix

    # cf_matrix = confusion_matrix(y_true, y_pred)
    # df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
    #                      columns=[i for i in classes])
    # plt.figure(figsize=(12, 7))
    # sn.heatmap(df_cm, annot=True)
    # plt.savefig('output.png')


def plot_confusion_matrix(cf_matrix, classes):
    # Create a confusion matrix plot from the matrix of confusion values
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes], columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sns.heatmap(df_cm, annot=True)
    plt.savefig('output.png')
    # plt.plot(cf_matrix)
    return True

# # Example usage
# predictions = torch.tensor([1, 0, 1, 1, 0, 1, 0, 1])
# ground_truths = torch.tensor([1, 0, 0, 1, 1, 1, 0, 1])
#
# classification_accuracy(predictions, ground_truths)
# classification_f1score(predictions, ground_truths)
#
# frame1 = torch.tensor([[1.0, 0.5, 0.2], [0.8, 0.3, 0.9]], dtype=torch.float32)
# frame2 = torch.tensor([[0.9, 0.6, 0.1], [0.7, 0.2, 0.8]], dtype=torch.float32)
#
# # Compute mean pixel-wise difference
# difference = prediction_pixelwise_accuracy(frame1, frame2)
# print(f"Mean Pixel-wise Difference: {difference*100:.3f}")