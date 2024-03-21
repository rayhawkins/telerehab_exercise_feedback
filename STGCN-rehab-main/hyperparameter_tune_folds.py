import numpy as np
random_seed = 42  # for reproducibility
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import concatenate, Flatten, Dropout, Dense, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
# from IPython.core.debugger import set_trace
#import matplotlib.pyplot as plt
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from GCN.data_processing import Data_Loader
from GCN.graph import Graph
from tunable_GCN import Sgcn_Lstm

from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

import argparse

# Add the arguments
parent_dir = r"C:\Users\rfgla\Documents\Ray\telerehab_exercise_feedback\data\skeleton_data_gestures_combined"
batch_size = 32
ckpt = r"C:\Users\rfgla\Documents\Ray\telerehab_exercise_feedback\STGCN-rehab-main\pretrain model\best_model.hdf5"

exercises = ["EF", "SF", "SA", "SFE", "ST"]
for this_exercise in exercises:
    print(this_exercise)
    dir = os.path.join(parent_dir, this_exercise)

    """import the whole dataset"""
    train_data_loader = Data_Loader(dir, True)

    """import the graph data structure"""
    graph = Graph(len(train_data_loader.body_part))

    """Split the data into training and validation sets while preserving the distribution"""
    train_x, train_y = train_data_loader.scaled_x, train_data_loader.scaled_y
    n_classes = len(np.unique(train_y))
    class_weights = {'0': (1. / np.sum(train_y == 0)) * (len(train_y) / 2.),
                     '1': (1. / np.sum(train_y == 1)) * (len(train_y) / 2.)}

    """Train the algorithm"""
    algorithm = Sgcn_Lstm(train_x, train_y, graph.AD, graph.AD2, graph.bias_mat_1, graph.bias_mat_2,
                          n_classes=n_classes, class_weight=class_weights, ckpt=ckpt)

    tuner = kt.Hyperband(algorithm.create_model,
                         objective='val_loss',
                         max_epochs=50,
                         factor=3,
                         directory='tuning',
                         project_name=f'exercise_tuning_{this_exercise}')

    earlystopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    tuner.search(train_x, train_y, epochs=50, validation_split=0.2, callbacks=[earlystopping])

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=10)[0]

    print(f"""
    The hyperparameter search is complete. The optimal learning rate for the {this_exercise} optimizer
    is {best_hps.get('learning_rate')}.""")
