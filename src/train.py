#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-line summary of the module.

Detailed description of the module.

Example :
    >>> example
"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2025-11-15"

import argparse
import os
from datetime import datetime

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
from utils.logger import ExperimentLogger
from utils.visualization import plot_training_summary


def _get_feature_id_from_config(config: dict) -> str:
    """
    Extract a short feature identifier from the config.

    Detailed description of the function.

    Args :
        config (dict): Configuration dictionary loaded from YAML.

    Returns :
        str: Short feature identifier to append to experiment dir.

    Raises :
        None
    """
    data = config.get("data", {}) if isinstance(config, dict) else {}
    candidates = [
        "feature_name",
        "feature_extractor",
        "feature_type",
        "features_name",
        "feature",
    ]
    for key in candidates:
        val = data.get(key)
        if val:
            return str(val).replace(" ", "-")
    features_dir = data.get("features_dir", "")
    if features_dir:
        base = os.path.basename(features_dir.rstrip(os.sep))
        if base:
            return base.replace(" ", "-")
    return "unknown_features"


def create_experiment_dir(
    base_dir: str, task_type: str = "train", feature_id: str = ""
) -> str:
    """
    Create experiment directory with timestamp and optional feature id.

    Detailed description of the function.

    Args :
        base_dir (str): Base directory for experiments.
        task_type (str): Type of task ('train', 'eval', 'test').
        feature_id (str): Optional short feature identifier to append.

    Returns :
        str: Path to created experiment directory.

    Raises :
        None
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{feature_id}" if feature_id else ""
    exp_name = f"{task_type}_{timestamp}{suffix}"
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    info_file = os.path.join(exp_dir, "experiment_info.txt")
    with open(info_file, "w") as f:
        f.write(f"Task Type: {task_type}\n")
        f.write(f"Feature: {feature_id}\n")
        f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Experiment Directory: {exp_dir}\n")

    print(f"Experiment directory created: {exp_dir}")
    return exp_dir


def train_epoch(model, dataloader, optimizer, criterion, device, logger):
    """Train for one epoch with multi-task learning."""
    model.train()
    total_loss = 0
    total_regression_loss = 0
    total_classification_loss = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(progress_bar):
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

        # Log batch details every 100 batches
        if batch_idx % 100 == 0:
            logger.log_batch_details(
                batch_idx,
                len(dataloader),
                total_loss_batch.item(),
                loss_dict["regression_loss"].item(),
                loss_dict["classification_loss"].item(),
            )

    avg_total_loss = total_loss / len(dataloader)
    avg_regression_loss = total_regression_loss / len(dataloader)
    avg_classification_loss = total_classification_loss / len(dataloader)

    train_loss_dict = {
        "total_loss": avg_total_loss,
        "regression_loss": avg_regression_loss,
        "classification_loss": avg_classification_loss,
    }

    logger.log_train_epoch_complete(
        avg_total_loss, avg_regression_loss, avg_classification_loss
    )

    return train_loss_dict


def validate(model, dataloader, criterion, device, logger, use_classification=True):
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
                targets.update(
                    {
                        "emotion_class": emotion_class,
                        "use_for_classification": use_for_classification,
                    }
                )
                all_emotion_labels.append(emotion_class.cpu().numpy())
                all_classification_masks.append(use_for_classification.cpu().numpy())
                if "classification" in outputs:
                    all_classification_preds.append(
                        outputs["classification"].cpu().numpy()
                    )

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

        valid_classification_mask = (all_classification_masks) & (
            all_emotion_labels >= 0
        )

        if valid_classification_mask.any():
            valid_classification_preds = all_classification_preds[
                valid_classification_mask
            ]
            valid_emotion_labels = all_emotion_labels[valid_classification_mask]
            classification_preds = np.argmax(valid_classification_preds, axis=1)
            classification_accuracy = np.mean(
                classification_preds == valid_emotion_labels
            )
            logger.log_classification_samples(
                valid_classification_mask.sum(), len(all_emotion_labels)
            )
        else:
            logger.warning("No valid classification samples in validation set")

    metrics = {
        "loss": avg_total_loss,
        "total_loss": avg_total_loss,
        "regression_loss": avg_regression_loss,
        "classification_loss": avg_classification_loss,
        "classification_accuracy": classification_accuracy,
        **regression_metrics,
    }

    logger.log_validation_complete(
        avg_total_loss,
        regression_metrics["total_ccc"],
        regression_metrics["v_ccc"],
        regression_metrics["a_ccc"],
        regression_metrics["d_ccc"],
    )

    return metrics


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_experiment_config(config, exp_dir):
    """Save experiment configuration to the experiment directory."""
    config_save_path = os.path.join(exp_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Configuration saved to: {config_save_path}")


def main():
    """Train the VAD regression model."""
    parser = argparse.ArgumentParser(description="Train VAD regression model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Get use_classification from config
    use_classification = config["data"].get("use_classification", True)

    # Determine feature id to append to experiment folder
    feature_id = _get_feature_id_from_config(config)

    # Create experiment directory with timestamp
    base_save_dir = config["output"]["save_dir"]
    exp_dir = create_experiment_dir(base_save_dir, "train", feature_id)

    # Setup logging
    logger = ExperimentLogger(exp_dir)

    # Log experiment start
    logger.log_experiment_start(args.config, use_classification)

    # Save experiment configuration
    save_experiment_config(config, exp_dir)
    logger.info("Experiment configuration saved")

    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    logger.info(f"Random seed set to: {seed}")

    if config["device"] == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{config.get('gpu_id', 0)}")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = EmotionDataset(
        config["data"]["features_dir"], config["data"]["labels_file"], split="train"
    )
    val_dataset = EmotionDataset(
        config["data"]["features_dir"],
        config["data"]["labels_file"],
        split="validation",
    )
    logger.log_dataset_info(len(train_dataset), len(val_dataset))

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

    # Initialize model
    logger.info("Initializing model...")
    model = BaselineEmotionGRU(
        input_size=config["model"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        output_size=config["model"]["output_size"],
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
        use_classification=use_classification,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log_model_info(total_params, trainable_params)

    criterion = MultiTaskLoss(
        classification_weight=config["training"]["classification_weight"],
        use_classification=use_classification,
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    logger.log_optimizer_info(
        config["training"]["learning_rate"], config["training"]["weight_decay"]
    )

    early_stopping_patience = config["training"]["early_stopping"]["patience"]
    early_stopping_min_delta = config["training"]["early_stopping"]["min_delta"]
    early_stopping_counter = 0
    best_val_loss = float("inf")
    best_epoch = 0

    num_epochs = config["training"]["num_epochs"]
    train_losses_list = []
    val_metrics_list = []

    # Log training configuration
    training_config = {
        "num_epochs": num_epochs,
        "early_stopping": config["training"]["early_stopping"],
        "batch_size": config["data"]["batch_size"],
    }
    logger.log_training_config(training_config)

    logger.info("Starting training loop...")
    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()
        logger.log_epoch_start(epoch + 1, num_epochs)

        train_loss_dict = train_epoch(
            model, train_loader, optimizer, criterion, device, logger
        )
        train_losses_list.append(train_loss_dict)

        val_metrics = validate(
            model, val_loader, criterion, device, logger, use_classification
        )
        val_metrics_list.append(val_metrics)

        # Log epoch summary
        epoch_time = (datetime.now() - epoch_start_time).total_seconds()
        logger.log_epoch_summary(
            epoch + 1, epoch_time, train_loss_dict, val_metrics, use_classification
        )

        # Log structured epoch results
        logger.log_epoch_results(
            epoch + 1, train_loss_dict, val_metrics, use_classification
        )

        # Early stopping check
        if val_metrics["loss"] < best_val_loss - early_stopping_min_delta:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            early_stopping_counter = 0
            save_checkpoint(
                model, optimizer, epoch, exp_dir, val_metrics, "best_model.pt"
            )
            logger.log_best_model_saved(best_val_loss)
        else:
            early_stopping_counter += 1
            logger.log_early_stopping_counter(
                early_stopping_counter, early_stopping_patience
            )

        if (epoch + 1) % config["output"]["save_interval"] == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                exp_dir,
                val_metrics,
                f"checkpoint_epoch_{epoch + 1}.pt",
            )
            logger.log_checkpoint_saved(epoch + 1)

        # Generate training plots
        plot_training_summary(
            train_losses_list, val_metrics_list, exp_dir, use_classification
        )

        if early_stopping_counter >= early_stopping_patience:
            logger.log_early_stopping_triggered(epoch + 1)
            break

    # Save final model
    save_checkpoint(
        model, optimizer, num_epochs, exp_dir, val_metrics_list[-1], "final_model.pt"
    )
    logger.info("Final model saved")

    # Save completion summary
    early_stopping_triggered = early_stopping_counter >= early_stopping_patience
    logger.save_completion_summary(
        len(train_losses_list),
        best_epoch,
        best_val_loss,
        val_metrics_list[-1]["total_ccc"],
        early_stopping_triggered,
    )

    # Log experiment end
    logger.log_experiment_end(
        best_epoch, best_val_loss, val_metrics_list[-1]["total_ccc"]
    )


if __name__ == "__main__":
    main()
