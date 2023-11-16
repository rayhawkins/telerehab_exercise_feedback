import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from transformer_classifier import Classifier as TransformerClassifier
from convolutional_classifier import Classifier as ConvolutionalClassifier
from tunable_data import VideoData
from functools import partial
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
import ray.train
from ray.train import Checkpoint, report
import tempfile
import os
import warnings


def train_classifier(num_epochs, config, data_dir=None, model_type="convolutional"):
    # Create the argument parser with all of the arguments that the model will need
    pl.seed_everything(1234)
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    if model_type == "convolutional":
        parser = ConvolutionalClassifier.add_model_specific_args(parser)
    else:
        parser = TransformerClassifier.add_model_specific_args(parser)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)

    config_args = []
    for key in config.keys():
        config_args.append(key)
        config_args.append(str(config[key]))
    config_args.append("--data_path")
    config_args.append(data_dir)
    config_args.append("--gpus")
    config_args.append(str(1))
    config_args.append("--max_steps")
    config_args.append(str(200000))
    args = parser.parse_args(config_args)
    args.data_path = data_dir

    print("Creating data loaders ...")
    data = VideoData(args)
    # pre-make relevant cached files if necessary
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()

    args.n_classes = data.n_classes

    print("Instantiating model ...")
    if model_type == "convolutional":
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

    checkpoint = ray.train.get_checkpoint()
    if checkpoint:
        print("Loading from checkpoint ...")
        with checkpoint.as_directory() as ckpt_dir:
            model.load_state_dict(torch.load(os.path.join(ckpt_dir, "model.pt")))
            optimizer.load_state_dict(torch.load(os.path.join(ckpt_dir, "optimizer.pt")))
            start_epoch = (torch.load(os.path.join(ckpt_dir, "extra_state.pt"))["epoch"] + 1)
        print(f"Resuming from epoch {start_epoch} ...")
    else:
        start_epoch = 0

    print("Starting training ...")
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
            losses.append(epoch_loss)

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
        val_steps = 0
        for idx, batch in enumerate(val_loader):
            # Send batch to the gpu
            video, label = batch["video"], batch["label"]
            video, label = video.to(device), label.to(device)
            with torch.no_grad():
                predictions = model(video)
                val_loss = criterion(predictions, label)
                val_steps += 1

                correct = predictions.argmax(axis=1) == label

                val_epoch_correct += correct.sum().item()
                val_epoch_count += correct.size(0)
                val_epoch_loss += loss.item()

        print(f"{epoch_loss=}")
        print(f"epoch accuracy: {epoch_correct / epoch_count}")
        print(f"{val_epoch_loss=}")
        print(f"test epoch accuracy: {val_epoch_correct / val_epoch_count}")

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = None
            # Only the global rank 0 worker saves and reports the checkpoint
            if ray.train.get_context().get_world_rank() == 0:
                torch.save(
                    model.module.state_dict(),  # NOTE: Unwrap the model.
                    os.path.join(temp_checkpoint_dir, "model.pt"),
                )
                torch.save(optimizer.state_dict(),
                           os.path.join(temp_checkpoint_dir, "optimizer.pt"))
                torch.save({"epoch": epoch},
                           os.path.join(temp_checkpoint_dir, "extra_state.pt"))
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            ray.train.report({"loss": val_loss / val_steps},
                             checkpoint=checkpoint)

    print("Finished Training")


def main(num_samples=10, max_num_epochs=10, data_path=None, sequence_length=16, resolution=64, vqvae_path=None,
         model_type="convolutional"):
    # Initialize random seed
    pl.seed_everything(1234)

    # Create the tunable arguments for the convolutional classifier
    config = {
        "--vqvae": vqvae_path,
        "--batch_size": tune.choice([8, 16, 32]),
        "--sequence_length": sequence_length,
        "--lr": tune.uniform(1e-4, 8e-4),
        "--kernel_size": tune.choice([32, 64, 128]),
        "--max_epochs": str(max_num_epochs),
        "--n_classes": 8,
        "--out_channels": [1, 3]
    }

    # Create the tunable arguments for the transformer classifier
    """
    config = {
        "--vqvae": vqvae_path,
        "--batch_size": tune.choice([8, 16, 32]),
        "--sequence_length": sequence_length,
        "--lr": tune.uniform(1e-4, 8e-4),
        "--n_heads": tune.choice([2, 4, 8]),
        "--max_epochs": str(max_num_epochs),
        "--n_layers": tune.choice([2, 4, 8]),
        "--dim_feedforward": tune.choice([256, 512, 1024, 2048])
    }
    """

    print("Configuring scheduler ...")
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    print("Running tune ...")
    result = tune.run(
        partial(train_classifier, data_dir=data_path, model_type=model_type),
        resources_per_trial={"cpu": 16, "gpu": 1},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")


if __name__ == '__main__':
    data_folder = r"C:\Users\rfgla\Documents\Ray\telerehab_exercise_feedback\data\gesture_sorted_data"
    vqvae_path = r"C:\Users\rfgla\Documents\Ray\telerehab_exercise_feedback\VideoGPT-master\lightning_logs\version_23\checkpoints\epoch=60-step=188489.ckpt"
    model_type = "convolutional"
    sequence_length = 16  # must be same sequence length as used for vqvae
    main(num_samples=10, max_num_epochs=10, data_path=data_folder,
         sequence_length=sequence_length, vqvae_path=vqvae_path, model_type=model_type)

