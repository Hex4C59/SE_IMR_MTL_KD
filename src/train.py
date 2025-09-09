"""
Training entry point for VAD regression model.

This module serves as the main entry point for training VAD regression models.
It handles argument parsing, configuration loading, and the training loop.

Example:
    >>> python src/train.py --config configs/config.yaml
"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2023-11-15"


import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import argparse
from tqdm import tqdm

from data.dataset import EmotionDataset, collate_fn
from models.vad_model import VADRegressionModel
from losses.regression_losses import get_loss_function
from metrics.regression_metrics import compute_metrics
from utils.visualization import plot_learning_curves
from utils.checkpoint import save_checkpoint, load_checkpoint


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # Move data to device
        features = batch["features"].to(device)
        vad = batch["vad"].to(device)

        # Forward pass
        outputs = model(features)

        # Calculate loss
        loss = criterion(outputs, vad)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update statistics
        total_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Calculate average loss
    avg_loss = total_loss / len(dataloader)

    return avg_loss


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0

    # For metrics calculation
    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for batch in progress_bar:
            # Move data to device
            features = batch["features"].to(device)
            vad = batch["vad"].to(device)

            # Forward pass
            outputs = model(features)

            # Calculate loss
            loss = criterion(outputs, vad)

            # Update statistics
            total_loss += loss.item()

            # Collect predictions and labels for metrics
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(vad.cpu().numpy())

            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Calculate average loss
    avg_loss = total_loss / len(dataloader)

    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate metrics
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss

    return metrics


def load_config(config_path):
    """
    Load configuration from YAML file
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """
    Main training function
    """
    parser = argparse.ArgumentParser(description="Train VAD regression model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set random seed
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Set device
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    save_dir = config["output"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Create datasets
    train_dataset = EmotionDataset(
        config["data"]["features_dir"], config["data"]["labels_file"], split="train"
    )
    val_dataset = EmotionDataset(
        config["data"]["features_dir"],
        config["data"]["labels_file"],
        split="validation",
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
    )

    # Create model
    model = VADRegressionModel(
        input_size=config["model"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        output_size=config["model"]["output_size"],
        dropout=config["model"]["dropout"],
    ).to(device)

    # Define loss function
    criterion = get_loss_function(config["training"].get("loss_function", "mse"))

    # Define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["training"]["scheduler"]["factor"],
        patience=config["training"]["scheduler"]["patience"],
        min_lr=config["training"]["scheduler"]["min_lr"],
        verbose=True,
    )

    # Early stopping parameters
    early_stopping_patience = config["training"]["early_stopping"]["patience"]
    early_stopping_min_delta = config["training"]["early_stopping"]["min_delta"]
    early_stopping_counter = 0
    best_val_loss = float("inf")

    # Training loop
    num_epochs = config["training"]["num_epochs"]
    train_losses = []
    val_metrics_list = []

    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        val_metrics_list.append(val_metrics)

        # Update learning rate
        scheduler.step(val_metrics["loss"])

        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}")
        print(f"Valence MSE: {val_metrics['v_mse']:.4f}, R²: {val_metrics['v_r2']:.4f}")
        print(f"Arousal MSE: {val_metrics['a_mse']:.4f}, R²: {val_metrics['a_r2']:.4f}")
        print(
            f"Dominance MSE: {val_metrics['d_mse']:.4f}, R²: {val_metrics['d_r2']:.4f}"
        )
        print(
            f"Average MSE: {val_metrics['total_mse']:.4f}, R²: {val_metrics['total_r2']:.4f}"
        )

        # Save checkpoint
        if val_metrics["loss"] < best_val_loss - early_stopping_min_delta:
            best_val_loss = val_metrics["loss"]
            early_stopping_counter = 0
            save_checkpoint(
                model, optimizer, epoch, save_dir, val_metrics, "best_model.pt"
            )
        else:
            early_stopping_counter += 1
            print(
                f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}"
            )

        # Save regular checkpoint
        if (epoch + 1) % config["output"]["save_interval"] == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                save_dir,
                val_metrics,
                f"checkpoint_epoch_{epoch + 1}.pt",
            )

        # Plot learning curves
        plot_learning_curves(train_losses, val_metrics_list, save_dir)

        # Early stopping
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Save final model
    save_checkpoint(
        model, optimizer, num_epochs, save_dir, val_metrics_list[-1], "final_model.pt"
    )

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
