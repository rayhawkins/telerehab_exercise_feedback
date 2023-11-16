import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F
import pytorch_lightning as pl

import sys
sys.path.append(r'C:\Users\rfgla\Documents\Ray\telerehab_exercise_feedback\VideoGPT-master')
from tunable_data import VideoData
from transformer_classifier import Classifier as TransformerClassifier
from convolutional_classifier import Classifier as ConvolutionalClassifier


def train_classifier(num_epochs, config, data_dir=None, model_type='convolutional'):
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--data_path', type=str, default='/home/wilson/data/datasets/bair.hdf5')
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)

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

    print("Starting train ...")
    for epoch in range(num_epochs):
        print(f"{epoch=}")
        epoch_loss = 0
        epoch_correct = 0
        epoch_count = 0
        for idx, batch in enumerate(train_loader):
            # Send batch data to the gpu
            video, label = batch["video"], batch["label"]
            video, label = video.to(device), label.to(device)

            predictions = model(video)

            loss = criterion(predictions, label)

            correct = predictions.argmax(axis=1) == label
            acc = correct.sum().item() / correct.size(0)

            epoch_correct += correct.sum().item()
            epoch_count += correct.size(0)

            epoch_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)

            optimizer.step()

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

                correct = predictions.argmax(axis=1) == labels
                acc = correct.sum().item() / correct.size(0)

                val_epoch_correct += correct.sum().item()
                val_epoch_count += correct.size(0)
                val_epoch_loss += loss.item()

        print(f"{epoch_loss=}")
        print(f"epoch accuracy: {epoch_correct / epoch_count}")
        print(f"{val_epoch_loss=}")
        print(f"test epoch accuracy: {val_epoch_correct / val_epoch_count}")

    return model



def main(num_epochs, args, data_dir, model_type):
    torch.manual_seed(1234)
    final_model = train_classifier(num_epochs, args, data_dir, model_type)

if __name__ == '__main__':
    data_folder = r"C:\Users\rfgla\Documents\Ray\telerehab_exercise_feedback\data\gesture_sorted_data"
    model_type = "convolutional"

    # Config dict for convolutional classifier
    config = {
        "kernel_size": 64,
        "out_channels": 3,
        "n_classes": 8,
        "lr": 8e-4
    }

    # Config dict for transformer classifier
    """
    config = {
        "n_heads": 4,
        "dim_feedforward": 2048,
        "dropout": 0.2,
        "num_layers": 4,
        "lr": 8e-4
    }
    """
    main(num_epochs=50, args=config, data_path=data_folder, model_type=model_type)