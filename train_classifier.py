import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F
import pytorch_lightning as pl
import os

import sys
sys.path.append(r'C:\Users\rfgla\Documents\Ray\telerehab_exercise_feedback\VideoGPT-master')
from tunable_data import VideoData
from transformer_classifier import Classifier as TransformerClassifier
from convolutional_classifier import Classifier as ConvolutionalClassifier


def train_classifier(num_epochs, config, data_dir=None, model_type='convolutional', ckpt=None):
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--data_path', type=str, default='/home/wilson/data/datasets/bair.hdf5')
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)

    if model_type == "convolutional":
        parser = ConvolutionalClassifier.add_model_specific_args(parser)
    else:
        parser = TransformerClassifier.add_model_specific_args(parser)

    config_args = []
    for key in config.keys():
        config_args.append(key)
        config_args.append(str(config[key]))
    config_args.append("--data_path")
    config_args.append(data_dir)
    config_args.append("--gpus")
    config_args.append(str(1))
    args = parser.parse_args(config_args)
    args.data_path = data_dir

    print("Creating data loaders ...")
    data = VideoData(args)
    # pre-make relevant cached files if necessary
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    args.n_classes = data.n_classes

    if model_type == 'convolutional':
        model = ConvolutionalClassifier(args)
    else:
        model = TransformerClassifier(args)
    criterion = nn.CrossEntropyLoss()

    # Set training to GPU
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    model.to(device)

    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    start_epoch = 0
    if ckpt is not None:
        checkpoint = torch.load(ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        model.train()

    print("Starting train ...")
    losses = []
    for epoch in range(start_epoch, num_epochs):
        print(f"{epoch=}")
        epoch_loss = 0
        epoch_correct = 0
        epoch_count = 0
        epoch_steps = 0
        for idx, batch in enumerate(train_loader):
            # Send batch data to the gpu
            video, label = batch["video"], batch["label"]
            video, label = video.to(device), label.to(device)

            predictions = model(video)

            loss = criterion(predictions, label)

            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)

            optimizer.step()
            losses.append(loss)

            correct = predictions.argmax(axis=1) == label
            epoch_correct += correct.sum().item()
            epoch_count += correct.size(0)
            epoch_loss += loss.item()

            # print statistics
            epoch_steps += 1
            if epoch_steps % 100 == 99:  # print every 100 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, epoch_steps + 1, epoch_loss / epoch_steps)
                )
                epoch_loss = 0.0

        val_epoch_loss = 0
        val_epoch_correct = 0
        val_epoch_count = 0
        for idx, batch in enumerate(val_loader):
            # Send batch to the gpu
            video, label = batch["video"], batch["label"]
            video, label = video.to(device), label.to(device)
            with torch.no_grad():
                predictions = model(video)
                val_loss = criterion(predictions, label)
                correct = predictions.argmax(axis=1) == label

                val_epoch_correct += correct.sum().item()
                val_epoch_count += correct.size(0)
                val_epoch_loss += val_loss.item()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_epoch_loss,
        }, os.path.join(data_dir, f"checkpoint_{epoch}.ckpt"))

        print(f"{epoch_loss=}")
        print(f"epoch accuracy: {epoch_correct / epoch_count}")
        print(f"{val_epoch_loss=}")
        print(f"test epoch accuracy: {val_epoch_correct / val_epoch_count}")

    return model, optimizer, val_epoch_loss


def main(num_epochs, args, data_dir, model_type, ckpt=None):
    torch.manual_seed(4321)
    final_model, optimizer, final_loss = train_classifier(num_epochs, args, data_dir, model_type, ckpt)
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': final_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': final_loss,
    }, os.path.join(data_dir, f"final_model.ckpt"))


if __name__ == '__main__':
    data_folder = r"C:\Users\rfgla\Documents\Ray\telerehab_exercise_feedback\data\gesture_sorted_data"
    vqvae_path = r"C:\Users\rfgla\Documents\Ray\telerehab_exercise_feedback\VideoGPT-master\lightning_logs\version_23\checkpoints\epoch=60-step=188489.ckpt"
    model_type = "convolutional"
    checkpoint = None

    # Config dict for convolutional classifier
    config = {
        "--batch_size": 32,
        "--vqvae": vqvae_path,
        "--kernel_size": 3,
        "--out_channels": 3,
        "--n_classes": 9,
        "--lr": 8e-4
    }

    # Config dict for transformer classifier
    """
    config = {
        "--batch_size": 32,
        "--vqvae": vqvae_path,
        "--n_heads": 2,
        "--dim_feedforward": 512,
        "--dropout": 0.2,
        "--n_layers": 2,
        "--lr": 7e-4
    }
    """

    main(num_epochs=100, args=config, data_dir=data_folder, model_type=model_type, ckpt=checkpoint)