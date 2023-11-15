import torch
import torch.nn as nn
from torch.nn import flatten
import torch.optim as optim

import sys
sys.path.append('/Users/owner/Documents/BME Fall2023/BME1570/Assignment 3/telerehab_exercise_feedback/VideoGPT-master')
from videogpt.vqvae import VQVAE
import argparse

class classification_network(nn.Module):
    def __init__(self, args):

        # Load VQ-VAE and set all parameters to no grad
        self.vqvae = VQVAE.load_from_checkpoint(classes.vqvae)
        for p in self.vqvae.parameters():
            p.requires_grad = False
        self.vqvae.codebook._need_init = False
        self.vqvae.eval()

        super(classification_network, self).__init__(in_channels, classes)


        # Transformer classifier
        self.in_channels = self.vqvae.in_channels

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.args.out_channels, kernel_size=self.args.kernel_size)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=self.args.kernel_size)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.args.out_channels, kernel_size=self.args.kernel_size)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=self.args.kernel_size)
        # Linear classifier
        self.linear1 = nn.Linear(in_features=self.args.in_features, out_features=self.classes)
        self.relu3 = nn.ReLU()

        # Softmax classifier
        self.linear2 = nn.Linear(in_features=self.args.in_features, out_features=self.classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)

        self.save_hyperparameters()

    def forward(self, z):
        # pass the input through the first layer
        z = self.conv1(z)
        z = self.relu1(z)
        z = self.maxpool1(z)

        # pass the output through previous layer through the second layer
        z = self.conv2(z)
        z = self.relu2(z)
        z = self.maxpool2(z)

        # flatten the output from the previous layer and pass it
        z = flatten(z, self.args.flatten)
        z = self.Linear1(z)
        z = self.relu3(z)

        # Pass output through the softmax classifier

        z = self.Linear2(z)
        z = self.logSoftmax(z)

        return z

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--vqvae', type=str, default='kinetics_stride4x4x4',
                            help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--in_channels', type=int, default=2)
        parser.add_argument('--classes', type=int, default=2)
        parser.add_argument('--out_channels', type=int, default=8)
        parser.add_argument('--in_features', type=int, default=2)
        parser.add_argument('--out_features', type=int, default=2)
        parser.add_argument('--kernel_size', type=int, default=576)
        parser.add_argument('--lr', type=float, default=3e-4)

# Define loss function and optimizer
def configure_optimizer(self):
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.999))
    return loss_function, optimizer
