import torch
import torch.nn as nn
from torch import flatten
import torch.optim as optim
import argparse
import sys
sys.path.append(r'C:\Users\rfgla\Documents\Ray\telerehab_exercise_feedback\VideoGPT-master')
from videogpt.vqvae import VQVAE


class Classifier(nn.Module):
    def __init__(self, args):
        args.kernel_size = (1, args.kernel_size, args.kernel_size)
        super().__init__()
        self.args = args

        # Load VQ-VAE and set all parameters to no grad
        self.vqvae = VQVAE.load_from_checkpoint(args.vqvae)
        for p in self.vqvae.parameters():
            p.requires_grad = False
        self.vqvae.codebook._need_init = False
        self.vqvae.eval()

        self.in_channels = self.vqvae.latent_shape

        # First convolutional layer
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=self.args.out_channels, kernel_size=self.args.kernel_size)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool3d(kernel_size=self.args.kernel_size)

        # Second convolutional layer
        self.conv2 = nn.Conv3d(in_channels=self.args.out_channels, out_channels=self.args.out_channels, kernel_size=self.args.kernel_size)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool3d(kernel_size=self.args.kernel_size)

        # Linear classifier
        self.FC1 = nn.Linear(in_features=1728, out_features=self.args.n_classes)
        self.relu3 = nn.ReLU()

        # Softmax classifier
        self.FC2 = nn.Linear(in_features=9, out_features=self.args.n_classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, z):
        # Pass the input through the first layer
        z = self.conv1(z)
        z = self.relu1(z)
        z = self.maxpool1(z)

        # Pass the output through previous layer through the second layer
        z = self.conv2(z)
        z = self.relu2(z)
        z = self.maxpool2(z)

        # Flatten the output from the previous layer and pass it
        z = flatten(z, 1)
        z = self.FC1(z)

        return z

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--vqvae', type=str, default='kinetics_stride4x4x4',
                            help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--n_classes', type=int, default=2)
        parser.add_argument('--out_channels', type=int, default=8)
        parser.add_argument('--kernel_size', type=int, default=3)
        parser.add_argument('--lr', type=float, default=3e-4)

        return parser

# Define loss function and optimizer
    def loss_optimizer(self):
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.999))
        return loss_function, optimizer
