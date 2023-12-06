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

import argparse

# Create the parser
my_parser = argparse.ArgumentParser(description='List of argument')

# Add the arguments
my_parser.add_argument('--dir', type=str, default='Kimore_ex5',
                       help='the name of exercise.', required=True)

my_parser.add_argument('--lr', type=int, default= 0.0001,
                       help='initial learning rate for optimizer.')

my_parser.add_argument('--epoch', type=int, default= 1000,
                       help='number of epochs to train.')

my_parser.add_argument('--batch_size', type=int, default= 10,
                       help='training batch size.')

my_parser.add_argument('--ckpt', type=str, default=None,
                       help='checkpoint to load initial weights')
#my_parser.add_argument('Path',
#                       type=str,
#                       help='the path to list')

# Execute the parse_args() method
args = my_parser.parse_args()

"""import the whole dataset"""
train_data_loader = Data_Loader(args.dir, True)
test_data_loader = Data_Loader(args.dir, False)

"""import the graph data structure"""
graph = Graph(len(train_data_loader.body_part))

"""Split the data into training and validation sets while preserving the distribution"""
train_x, train_y = train_data_loader.scaled_x, train_data_loader.scaled_y
n_classes = len(np.unique(train_y))
print(train_y.shape)
"""Train the algorithm"""
algorithm = Sgcn_Lstm(train_x, train_y, graph.AD, graph.AD2, graph.bias_mat_1, graph.bias_mat_2, lr=args.lr,
                      epoach=args.epoch, batch_size=args.batch_size, n_classes=n_classes)
algorithm.create_model()
if args.ckpt is not None:
    algorithm.model.load_weights(args.ckpt, by_name=True)
history = algorithm.train()


