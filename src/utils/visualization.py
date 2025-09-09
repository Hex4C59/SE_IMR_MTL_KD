"""
Visualization utilities for training and evaluation.

This module provides functions for visualizing training progress,
model predictions, and evaluation results.

Example:
    >>> plot_learning_curves(train_losses, val_metrics, save_dir)

"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2023-11-15"

import os

import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves(train_losses, val_metrics, save_dir):
    """
    Plot and save learning curves.

    Args:
        train_losses (list): List of training losses per epoch
        val_metrics (list): List of validation metrics per epoch
        save_dir (str): Directory to save the plots

    """
    os.makedirs(save_dir, exist_ok=True)

    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot([m["loss"] for m in val_metrics], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    # Plot MSE metrics
    plt.figure(figsize=(10, 6))
    plt.plot([m["v_mse"] for m in val_metrics], label="Valence MSE")
    plt.plot([m["a_mse"] for m in val_metrics], label="Arousal MSE")
    plt.plot([m["d_mse"] for m in val_metrics], label="Dominance MSE")
    plt.plot([m["total_mse"] for m in val_metrics], label="Average MSE", linestyle="--")
    plt.title("Validation MSE by Dimension")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "mse_curve.png"))
    plt.close()

    # Plot R² metrics
    plt.figure(figsize=(10, 6))
    plt.plot([m["v_r2"] for m in val_metrics], label="Valence R²")
    plt.plot([m["a_r2"] for m in val_metrics], label="Arousal R²")
    plt.plot([m["d_r2"] for m in val_metrics], label="Dominance R²")
    plt.plot([m["total_r2"] for m in val_metrics], label="Average R²", linestyle="--")
    plt.title("Validation R² by Dimension")
    plt.xlabel("Epoch")
    plt.ylabel("R²")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "r2_curve.png"))
    plt.close()

    # Plot CCC metrics if available
    if "v_ccc" in val_metrics[0]:
        plt.figure(figsize=(10, 6))
        plt.plot([m["v_ccc"] for m in val_metrics], label="Valence CCC")
        plt.plot([m["a_ccc"] for m in val_metrics], label="Arousal CCC")
        plt.plot([m["d_ccc"] for m in val_metrics], label="Dominance CCC")
        plt.plot(
            [m["total_ccc"] for m in val_metrics], label="Average CCC", linestyle="--"
        )
        plt.title("Validation CCC by Dimension")
        plt.xlabel("Epoch")
        plt.ylabel("CCC")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "ccc_curve.png"))
        plt.close()


def plot_prediction_scatter(true_values, predictions, save_dir, dimension_names=None):
    """
    Plot scatter plots of true vs predicted values.

    Args:
        true_values (numpy.ndarray): Ground truth values
        predictions (numpy.ndarray): Predicted values
        save_dir (str): Directory to save the plots
        dimension_names (list, optional): Names of the dimensions

    """
    os.makedirs(save_dir, exist_ok=True)

    if dimension_names is None:
        dimension_names = ["Valence", "Arousal", "Dominance"]

    for i, name in enumerate(dimension_names):
        plt.figure(figsize=(8, 8))
        plt.scatter(true_values[:, i], predictions[:, i], alpha=0.5)

        # Add identity line
        min_val = min(np.min(true_values[:, i]), np.min(predictions[:, i]))
        max_val = max(np.max(true_values[:, i]), np.max(predictions[:, i]))
        plt.plot([min_val, max_val], [min_val, max_val], "r--")

        plt.title(f"{name}: True vs Predicted")
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.axis("equal")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"{name.lower()}_scatter.png"))
        plt.close()
