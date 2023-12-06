import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import signal
from IPython.core.debugger import set_trace
import tensorflow as tf

index_Spine_Base=0
index_Spine_Mid=3
index_Neck=6
index_Head=9
index_Shoulder_Left=12
index_Elbow_Left=15
index_Wrist_Left=18
index_Hand_Left=21
index_Shoulder_Right=24
index_Elbow_Right=27
index_Wrist_Right=30
index_Hand_Right=33
index_Hip_Left=36
index_Knee_Left=39
index_Ankle_Left=42
index_Foot_Left=45
index_Hip_Right=48
index_Knee_Right=51
index_Ankle_Right=54
index_Foot_Right=57
index_Spine_Shoulder=60
index_Tip_Left=63
index_Thumb_Left=66
index_Tip_Right=69
index_Thumb_Right=72


class Data_Loader():
    def __init__(self, dir, train=True):
        self.train = train
        self.num_channel = 3
        self.num_timestep = 100
        self.dir = dir
        self.body_part = self.body_parts()       
        self.dataset = []
        self.sequence_length = []
        self.new_label = []
        self.x, self.info, self.y = self.import_dataset()
        self.num_samples = self.y.shape[0]
        self.num_joints = len(self.body_part)
        self.sc1 = StandardScaler()
        self.sc2 = StandardScaler()
        self.num_classes = len(np.unique(self.y))
        self.scaled_x, self.scaled_y = self.preprocessing()

    @staticmethod
    def body_parts():
        body_parts = [index_Spine_Base, index_Spine_Mid, index_Neck, index_Head, index_Shoulder_Left, index_Elbow_Left, index_Wrist_Left, index_Hand_Left, index_Shoulder_Right, index_Elbow_Right, index_Wrist_Right, index_Hand_Right, index_Hip_Left, index_Knee_Left, index_Ankle_Left, index_Foot_Left, index_Hip_Right, index_Knee_Right, index_Ankle_Right, index_Ankle_Right, index_Spine_Shoulder, index_Tip_Left, index_Thumb_Left, index_Tip_Right, index_Thumb_Right
]
        return body_parts
    
    def import_dataset(self):
        if self.train:
            x = pd.read_csv(self.dir+"/train_x.csv", header = None).iloc[:,:].values
            y = pd.read_csv(self.dir+"/train_y.csv", header = None).iloc[:,:].values
        else:
            x = pd.read_csv(self.dir + "/test_x.csv", header=None).iloc[:, :].values
            y = pd.read_csv(self.dir + "/test_y.csv", header=None).iloc[:, :].values

        return x[:, :-7], x[:, -7:], y

    def preprocessing(self):
        X = np.zeros((self.x.shape[0], self.num_joints * self.num_channel)).astype('float32')
        for row in range(self.x.shape[0]):
            counter = 0
            for parts in self.body_part:
                for i in range(self.num_channel):
                    X[row, counter + i] = self.x[row, parts + i]
                counter += self.num_channel

        y = np.reshape(self.y, (-1, 1))
        y = tf.one_hot(y, self.num_classes)
        y = np.squeeze(y, axis=1)
        X = self.sc1.fit_transform(X)

        X_ = np.zeros((self.num_samples, self.num_timestep, self.num_joints, self.num_channel))
        for batch in range(X_.shape[0]):
            for timestep in range(X_.shape[1]):
                for node in range(X_.shape[2]):
                    for channel in range(X_.shape[3]):
                        X_[batch, timestep, node, channel] = X[
                            timestep + (batch * self.num_timestep), channel + (node * self.num_channel)]

        X = X_
        return X, y

