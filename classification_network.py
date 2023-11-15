import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append('/Users/owner/Documents/BME Fall2023/BME1570/Assignment 3/telerehab_exercise_feedback/VideoGPT-master')
from videogpt.vqvae import VQVAE
import argparse

class classification_network(nn.Module):
    def __init__(self, input_channels, n_classes, patch_size, h_size, nhead, n_layers, mlp_dim, dropout):
        super(classification_network, self).__init__(input_channels, n_classes)
        self.input_channels=input_channels
        self.n_classes=n_classes
        self.patch_size=patch_size
        self.h_size=h_size
        self.nhead=nhead
        self.n_layers=n_layers
        self.mpl_dim=mlp_dim
        self.dropout=dropout

        # Load VQ-VAE and set all parameters to no grad
        self.vqvae = VQVAE.load_from_checkpoint(n_classes.vqvae)
        for p in self.vqvae.parameters():
            p.requires_grad = False
        self.vqvae.codebook._need_init = False
        self.vqvae.eval()

        # Transformer classifier
        self.input_channels = self.vqvae.input_channels

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_channels,
            nhead=self.vqvae.nhead,
            dim_feedforward=self.vqvae.mlp_dim,
            dropout=self.vqvae.dropout,
        )

        self.transfomer_encoder = nn.TransfomerEncoder(
            encoder_layer,
            num_layers=self.vqvae.n_layers,
        )
        self.classifier = nn.Linear(patch_size * patch_size * input_channels, self.vqvae.n_classes)

        self.save_hyperparameters()

    def forward(self, z):
        with torch.no_grad():
            z = self.vqvae.encoder(z)
        z = self.transformer_encoder(z)
        z = z.mean(dim=1) # Global average pooling
        z = self.classifier(z)
        return z

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--vqvae', type=str, default='kinetics_stride4x4x4',
                            help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--n_classes', type=int, default=2)

        parser.add_argument('--dim_feedforward', type=int, default=576)
        parser.add_argument('--n_heads', type=int, default=4)
        parser.add_argument('--n_layers', type=int, default=8)
        parser.add_argument('--lr', type=float, default=3e-4)
        parser.add_argument('--dropout', type=float, default=0.2)

# Define loss function and optimizer
def configure_optimizer(self,learning_rate=0.001):
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    return loss_function, optimizer
