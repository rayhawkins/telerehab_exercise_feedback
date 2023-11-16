import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from tunable_gpt import VideoGPT
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


def train_videogpt(config, data_dir=None):
    # Create the argument parser with all of the arguments that the model will need
    pl.seed_everything(1234)
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VideoGPT.add_model_specific_args(parser)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=128)
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

    args.class_cond = True
    args.class_cond_dim = data.n_classes if args.class_cond else None

    print("Instantiating model ...")
    model = VideoGPT(args)

    # Set training to GPU
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    model.to(device)

    print("Configuring optimizer ...")
    optimizer = model.configure_optimizers()[0]

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
    for epoch in range(start_epoch, int(config["--max_epochs"])):
        # Set model to train mode
        print("Setting model to train ...")
        model.train()
        torch.set_grad_enabled(True)
        running_loss = 0.0
        epoch_steps = 0
        print("Starting epoch ...")
        for batch_idx, batch in enumerate(train_loader):
            # Send batch data to the gpu
            video, label = batch["video"], batch["label"]
            video, label = video.to(device), label.to(device)

            optimizer.zero_grad()  # zero the parameter gradients

            # forward + backward + optimize
            loss = model.training_step(video, label, batch_idx)
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
            optimizer.step()

            losses.append(loss)

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if epoch_steps % 100 == 99:  # print every 100 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, epoch_steps + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0


        # Validation loss
        val_loss = 0.0
        val_steps = 0
        for batch_idx, batch in enumerate(val_loader):
            # Send batch data to the  gpu
            video, label = batch["video"], batch["label"]
            video, label = video.to(device), label.to(device)
            with torch.no_grad():
                loss = model.validation_step(video, label, batch_idx)
                val_loss += loss.cpu().numpy()
                val_steps += 1

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
            ray.train.report({"loss": val_loss / val_steps}, checkpoint=checkpoint)

    print("Finished Training")


def main(num_samples=10, max_num_epochs=10, data_path=None, sequence_length=16, resolution=64, vqvae_path=None):
    # Initialize random seed
    pl.seed_everything(1234)

    # Create the tunable arguments
    config = {
        "--vqvae": vqvae_path,
        "--n_cond_frames": tune.choice([1, 2, 4]),
        "--batch_size": tune.choice([8, 16, 32]),
        "--sequence_length": sequence_length,
        "--lr": tune.uniform(1e-4, 3e-4),
        "--resolution": resolution,
        "--heads": tune.choice([4, 8]),
        "--max_epochs": str(max_num_epochs)
    }

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
        partial(train_videogpt, data_dir=data_path),
        resources_per_trial={"cpu": 1, "gpu": 1},
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
    resolution = 64   # must be same resolution as used for vqvae
    sequence_length = 16  # must be same sequence length as used for vqvae
    main(num_samples=10, max_num_epochs=10, data_path=data_folder,
         sequence_length=sequence_length, resolution=resolution,
         vqvae_path=vqvae_path)

