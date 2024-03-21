import numpy as np
random_seed = 42  # for reproducibility
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import concatenate, Flatten, Dropout, Dense, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
from sklearn.model_selection import train_test_split
# from IPython.core.debugger import set_trace
#import matplotlib.pyplot as plt
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from GCN.data_processing import Data_Loader
from GCN.graph import Graph
from GCN.sgcn_lstm import Sgcn_Lstm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

import argparse

# Add the arguments
parent_dir = r"C:\Users\rfgla\Documents\Ray\telerehab_exercise_feedback\data\skeleton_data_gestures_combined_correct_sorted_LMSO"
save_dir = r"C:\Users\rfgla\Documents\Ray\telerehab_exercise_feedback\STGCN-rehab-main\gestures_combined_networks_LMSO"
use_class_weights = True
lr = 0.0001
epoch = 1000
batch_size = 32
ckpt = r"C:\Users\rfgla\Documents\Ray\telerehab_exercise_feedback\STGCN-rehab-main\pretrain model\best_model.hdf5"
# gestures = ["EFL", "EFR", "SFL", "SFR", "SAL", "SAR", "SFE", "STL", "STR"]
gestures = ["EF", "SF", "SA", "SFE", "ST"]

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

for this_patient in os.listdir(parent_dir):
    print(this_patient)
    if not os.path.isdir(os.path.join(parent_dir, this_patient)):
        continue
    for this_gesture in gestures:
        dir = os.path.join(parent_dir, this_patient, this_gesture)
        if not os.path.exists(dir):
            continue

        """import the whole dataset"""
        train_data_loader = Data_Loader(dir, True)
        test_data_loader = Data_Loader(dir, False)

        """import the graph data structure"""
        graph = Graph(len(train_data_loader.body_part))

        """Split the data into training and validation sets while preserving the distribution"""
        train_x, train_y = train_data_loader.scaled_x, train_data_loader.scaled_y
        n_classes = len(np.unique(train_y))
        if use_class_weights:
            class_weights = {0: (1. / np.sum(train_y[:, 0] == 0)) * (len(train_y) / 2.),
                             1: (1. / np.sum(train_y[:, 0] == 1)) * (len(train_y) / 2.)}
        else:
            class_weights = None

        save_name = os.path.join(save_dir, this_patient, f"model_{this_gesture}.hdf5")

        """Train the algorithm"""
        algorithm = Sgcn_Lstm(train_x, train_y, graph.AD, graph.AD2, graph.bias_mat_1, graph.bias_mat_2, lr=lr,
                              epoach=epoch, batch_size=batch_size, n_classes=n_classes, save_name=save_name,
                              class_weight=class_weights)
        algorithm.create_model()
        if ckpt is not None:
            algorithm.model.load_weights(ckpt, by_name=True)
        history = algorithm.train()
