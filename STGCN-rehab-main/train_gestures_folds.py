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
parent_dir = r"C:\Users\rfgla\Documents\Ray\telerehab_exercise_feedback\data\skeleton_data_gestures_combined_correct_sorted"
save_dir = r"C:\Users\rfgla\Documents\Ray\telerehab_exercise_feedback\STGCN-rehab-main\gestures_combined_networks"
lr = 0.0001
epoch = 1000
batch_size = 32
ckpt = r"C:\Users\rfgla\Documents\Ray\telerehab_exercise_feedback\STGCN-rehab-main\pretrain model\best_model.hdf5"
k = 10  # number of folds
# gestures = ["EFL", "EFR", "SFL", "SFR", "SAL", "SAR", "SFE", "STL", "STR"]
gestures = ["EF", "SF", "SA", "SFE", "ST"]

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

for this_fold in range(k):
    print(this_fold)
    for this_gesture in gestures:
        dir = os.path.join(parent_dir, f"fold_{this_fold}", this_gesture)

        """import the whole dataset"""
        train_data_loader = Data_Loader(dir, True)
        test_data_loader = Data_Loader(dir, False)

        """import the graph data structure"""
        graph = Graph(len(train_data_loader.body_part))

        """Split the data into training and validation sets while preserving the distribution"""
        train_x, train_y = train_data_loader.scaled_x, train_data_loader.scaled_y
        n_classes = len(np.unique(train_y))

        save_name = os.path.join(save_dir, f"fold_{this_fold}", f"model_{this_gesture}.hdf5")
        """Train the algorithm"""
        algorithm = Sgcn_Lstm(train_x, train_y, graph.AD, graph.AD2, graph.bias_mat_1, graph.bias_mat_2, lr=lr,
                              epoach=epoch, batch_size=batch_size, n_classes=n_classes, save_name=save_name)
        algorithm.create_model()
        if ckpt is not None:
            algorithm.model.load_weights(ckpt, by_name=True)
        history = algorithm.train()