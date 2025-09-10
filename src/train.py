#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py

Training entry point for VAD regression model.

Date: 2025-09-10
Author: Liu Yang
License: MIT
Project: https://github.com/Hex4C59/SE_IMR_MTL_KD
"""

import argparse
import os

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import EmotionDataset, collate_fn
from losses.multitask_losses import MultiTaskLoss
from metrics.regression_metrics import compute_metrics
from models.mtl_GRU_model import BaselineEmotionGRU
from utils.checkpoint import save_checkpoint
from utils.visualization import plot_training_summary


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch with multi-task learning."""
    model.train()
    total_loss = 0
    total_regression_loss = 0
    total_classification_loss = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        optimizer.zero_grad()

        features = batch["features"].to(device)
        vad = batch["vad"].to(device)
        emotion_class = batch["emotion_class"].to(device)
        use_for_classification = batch["use_for_classification"].to(device)

        outputs = model(features)

        targets = {
            "vad": vad,
            "emotion_class": emotion_class,
            "use_for_classification": use_for_classification,
        }

        loss_dict = criterion(outputs, targets)
        total_loss_batch = loss_dict["total_loss"]

        total_loss_batch.backward()
        optimizer.step()

        total_loss += total_loss_batch.item()
        total_regression_loss += loss_dict["regression_loss"].item()
        total_classification_loss += loss_dict["classification_loss"].item()

        progress_bar.set_postfix(
            {
                "total_loss": f"{total_loss_batch.item():.4f}",
                "reg_loss": f"{loss_dict['regression_loss'].item():.4f}",
                "cls_loss": f"{loss_dict['classification_loss'].item():.4f}",
            }
        )

    avg_total_loss = total_loss / len(dataloader)
    avg_regression_loss = total_regression_loss / len(dataloader)
    avg_classification_loss = total_classification_loss / len(dataloader)

    return {
        "total_loss": avg_total_loss,
        "regression_loss": avg_regression_loss,
        "classification_loss": avg_classification_loss,
    }


def validate(model, dataloader, criterion, device, use_classification=True):
    """Validate the model with single-task or multi-task learning."""
    model.eval()
    total_loss = 0
    total_regression_loss = 0
    total_classification_loss = 0

    all_regression_preds = []
    all_classification_preds = []
    all_vad_labels = []
    all_emotion_labels = []
    all_classification_masks = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for batch in progress_bar:
            features = batch["features"].to(device)
            vad = batch["vad"].to(device)
            
            outputs = model(features)

            targets = {"vad": vad}
            
            if use_classification:
                emotion_class = batch["emotion_class"].to(device)
                use_for_classification = batch["use_for_classification"].to(device)
                targets.update({
                    "emotion_class": emotion_class,
                    "use_for_classification": use_for_classification,
                })
                all_emotion_labels.append(emotion_class.cpu().numpy())
                all_classification_masks.append(use_for_classification.cpu().numpy())
                if "classification" in outputs:
                    all_classification_preds.append(outputs["classification"].cpu().numpy())

            loss_dict = criterion(outputs, targets)

            total_loss += loss_dict["total_loss"].item()
            total_regression_loss += loss_dict["regression_loss"].item()
            total_classification_loss += loss_dict["classification_loss"].item()

            all_regression_preds.append(outputs["regression"].cpu().numpy())
            all_vad_labels.append(vad.cpu().numpy())

            progress_bar.set_postfix({"loss": f"{loss_dict['total_loss'].item():.4f}"})

    avg_total_loss = total_loss / len(dataloader)
    avg_regression_loss = total_regression_loss / len(dataloader)
    avg_classification_loss = total_classification_loss / len(dataloader)

    all_regression_preds = np.concatenate(all_regression_preds, axis=0)
    all_vad_labels = np.concatenate(all_vad_labels, axis=0)

    regression_metrics = compute_metrics(all_vad_labels, all_regression_preds)

    classification_accuracy = 0.0
    if use_classification and all_classification_preds:
        all_classification_preds = np.concatenate(all_classification_preds, axis=0)
        all_emotion_labels = np.concatenate(all_emotion_labels, axis=0)
        all_classification_masks = np.concatenate(all_classification_masks, axis=0)

        valid_classification_mask = (all_classification_masks == True) & (
            all_emotion_labels >= 0
        )

        if valid_classification_mask.any():
            valid_classification_preds = all_classification_preds[valid_classification_mask]
            valid_emotion_labels = all_emotion_labels[valid_classification_mask]
            classification_preds = np.argmax(valid_classification_preds, axis=1)
            classification_accuracy = np.mean(
                classification_preds == valid_emotion_labels
            )
            print(
                f"Classification samples: {valid_classification_mask.sum()}/{len(all_emotion_labels)}"
            )
        else:
            print("No valid classification samples in validation set")

    metrics = {
        "loss": avg_total_loss,
        "total_loss": avg_total_loss,
        "regression_loss": avg_regression_loss,
        "classification_loss": avg_classification_loss,
        "classification_accuracy": classification_accuracy,
        **regression_metrics,
    }

    return metrics


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Train the VAD regression model."""
    parser = argparse.ArgumentParser(description="Train VAD regression model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Get use_classification from config
    use_classification = config["data"].get("use_classification", True)
    print(f"Training mode: {'Multi-task' if use_classification else 'Single-task (VAD regression only)'}")

    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    if config["device"] == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{config.get('gpu_id', 0)}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    save_dir = config["output"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    train_dataset = EmotionDataset(
        config["data"]["features_dir"], config["data"]["labels_file"], split="train"
    )
    val_dataset = EmotionDataset(
        config["data"]["features_dir"],
        config["data"]["labels_file"],
        split="validation",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
    )

    model = BaselineEmotionGRU(
        input_size=config["model"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        output_size=config["model"]["output_size"],
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
    ).to(device)

    criterion = MultiTaskLoss(
        classification_weight=config["training"]["classification_weight"],
        use_classification=use_classification,
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )

    early_stopping_patience = config["training"]["early_stopping"]["patience"]
    early_stopping_min_delta = config["training"]["early_stopping"]["min_delta"]
    early_stopping_counter = 0
    best_val_loss = float("inf")

    num_epochs = config["training"]["num_epochs"]
    train_losses_list = []
    val_metrics_list = []

    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss_dict = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses_list.append(train_loss_dict)

        val_metrics = validate(model, val_loader, criterion, device, use_classification)
        val_metrics_list.append(val_metrics)

        print(f"Train Total Loss: {train_loss_dict['total_loss']:.4f}")
        print(f"Train Regression Loss: {train_loss_dict['regression_loss']:.4f}")
        if use_classification:
            print(f"Train Classification Loss: {train_loss_dict['classification_loss']:.4f}")
            print(f"Val Classification Accuracy: {val_metrics['classification_accuracy']:.4f}")
        print(f"Val Total Loss: {val_metrics['total_loss']:.4f}")
        print(f"Valence CCC: {val_metrics['v_ccc']:.4f}")
        print(f"Arousal CCC: {val_metrics['a_ccc']:.4f}")
        print(f"Dominance CCC: {val_metrics['d_ccc']:.4f}")
        print(f"Average CCC: {val_metrics['total_ccc']:.4f}")

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

        if (epoch + 1) % config["output"]["save_interval"] == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                save_dir,
                val_metrics,
                f"checkpoint_epoch_{epoch + 1}.pt",
            )

        plot_training_summary(train_losses_list, val_metrics_list, save_dir, use_classification)

        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    save_checkpoint(
        model, optimizer, num_epochs, save_dir, val_metrics_list[-1], "final_model.pt"
    )

    print("\nTraining completed!")


if __name__ == "__main__":
    main()