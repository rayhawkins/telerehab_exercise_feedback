import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import sys
sys.path.append(r'C:\Users\rfgla\Documents\Ray\telerehab_exercise_feedback\VideoGPT-master')
from videogpt.attention import AttentionStack, LayerNorm, AddBroadcastPosEmbed
from videogpt.utils import shift_dim
from videogpt.vqvae import VQVAE
import argparse


class Classifier(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        # Load VQ-VAE and set all parameters to no grad
        self.vqvae = VQVAE.load_from_checkpoint(args.vqvae)
        for p in self.vqvae.parameters():
            p.requires_grad = False
        self.vqvae.codebook._need_init = False
        self.vqvae.eval()

        # Transformer classifier
        self.shape = self.vqvae.latent_shape

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.shape,
            nhead=self.args.n_heads,
            dim_feedforward=self.args.dim_feedforward,
            dropout=self.args.dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.args.n_layers,
        )

        self.classifier == nn.Linear(self.shape, self.args.n_classes)

        self.save_hyperparameters()

    def forward(self, x):
        with torch.no_grad():
            x = self.vqvae.encoder(x)
        x - self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.classifier(x)

        return x

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

        return parser

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.999))






