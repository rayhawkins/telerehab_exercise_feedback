import numpy as np

random_seed = 42  # for reproducibility
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import concatenate, Flatten, Dropout, Dense, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
from sklearn.model_selection import train_test_split
# from IPython.core.debugger import set_trace
# import matplotlib.pyplot as plt
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from GCN.data_processing import Data_Loader
from GCN.graph import Graph
from GCN.sgcn_lstm import Sgcn_Lstm
from sklearn.metrics import mean_squared_error, mean_absolute_error

import argparse

data_dir = r"C:\Users\rfgla\Documents\Ray\telerehab_exercise_feedback\data\skeleton_data_gestures_combined_correct_sorted\fold_0\EF"
ckpt = r"C:\Users\rfgla\Documents\Ray\telerehab_exercise_feedback\STGCN-rehab-main\gestures_combined_networks\fold_0\model_EF.hdf5"

"""import the whole dataset"""
test_data_loader = Data_Loader(data_dir, False)

"""import the graph data structure"""
graph = Graph(len(test_data_loader.body_part))

"""Split the data into training and validation sets while preserving the distribution"""
test_x, test_y = test_data_loader.scaled_x, test_data_loader.scaled_y

"""Train the algorithm"""
algorithm = Sgcn_Lstm(test_x, test_y, graph.AD, graph.AD2, graph.bias_mat_1, graph.bias_mat_2)
algorithm.create_model()
algorithm.model.load_weights(ckpt)

"""Test the model"""
y_pred = algorithm.prediction(test_x)
y_pred = np.argmax(y_pred, axis=1)
test_y = np.squeeze(test_y)
test_y = np.argmax(test_y, axis=1)
print(y_pred)
print(test_y)

"""Performance matric"""
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def F1_score(y_true, y_pred):
    TP = 0
    FP = 0
    FN = 0
    for t, this_label in enumerate(y_true):
        if this_label == y_pred[t]:
            TP += 1
        else:
            if y_pred[t] == 1:
                FP += 1
            else:
                FN == 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2. * (precision * recall) / (precision + recall)
    return F1, precision, recall

acc = accuracy(test_y, y_pred)
F1, precision, recall = F1_score(test_y, y_pred)

print(acc)
print(F1)
print(precision)
print(recall)
