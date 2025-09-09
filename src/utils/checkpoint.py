"""
Checkpoint utilities for saving and loading model states.

This module provides functions for saving and loading model checkpoints,
which include model weights, optimizer state, and training metrics.

Example:
    >>> save_checkpoint(model, optimizer, epoch, save_dir, metrics, "best_model.pt")
    >>> model, optimizer, start_epoch, metrics = load_checkpoint(model, optimizer, "best_model.pt")

"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2023-11-15"

import os

import torch


def save_checkpoint(
    model, optimizer, epoch, save_dir, metrics, filename="checkpoint.pt"
):
    """
    Save model checkpoint.

    Args:
        model (torch.nn.Module): Model to save
        optimizer (torch.optim.Optimizer): Optimizer to save
        epoch (int): Current epoch number
        save_dir (str): Directory to save the checkpoint
        metrics (dict): Metrics to save
        filename (str, optional): Filename for the checkpoint

    """
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    torch.save(checkpoint, os.path.join(save_dir, filename))
    print(f"Checkpoint saved to {os.path.join(save_dir, filename)}")


def load_checkpoint(model, optimizer=None, checkpoint_path=None):
    """
    Load model checkpoint

    Args:
        model (torch.nn.Module): Model to load weights into
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into
        checkpoint_path (str): Path to the checkpoint file

    Returns:
        tuple: (model, optimizer, start_epoch, metrics)

    """
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return model, optimizer, 0, None

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    metrics = checkpoint.get("metrics", None)

    return model, optimizer, start_epoch, metrics
