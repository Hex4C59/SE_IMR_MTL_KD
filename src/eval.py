#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate trained emotion regression/classification model.

This module loads a trained model and evaluates it on a specified split,
computing metrics and saving results with comprehensive logging.

Example:
    >>> python eval.py --config configs/eval.yaml --checkpoint runs/train_xxx/best_model.pt --split test

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
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import EmotionDataset, collate_fn
from losses.multitask_losses import MultiTaskLoss
from metrics.regression_metrics import compute_metrics
from models.mtl_GRU_model import BaselineEmotionGRU
from utils.logger import ExperimentLogger


def create_experiment_dir(base_dir: str, task_type: str = "eval") -> str:
    """Create experiment directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{task_type}_{timestamp}"
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    info_file = os.path.join(exp_dir, "experiment_info.txt")
    with open(info_file, "w") as f:
        f.write(f"Task Type: {task_type}\n")
        f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Experiment Directory: {exp_dir}\n")

    print(f"Evaluation directory created: {exp_dir}")
    return exp_dir


def evaluate_model(
    model, dataloader, criterion, device, use_classification=True, logger=None
):
    """Evaluate the model on given dataset."""
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
        progress_bar = tqdm(dataloader, desc="Evaluation")
        for batch_idx, batch in enumerate(progress_bar):
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

            if logger and batch_idx % 50 == 0:
                logger.info(f"Evaluation batch {batch_idx}/{len(dataloader)}")

    # Calculate metrics
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
            if logger:
                logger.info(
                    f"Classification samples: {valid_classification_mask.sum()}/{len(all_emotion_labels)}"
                )
        else:
            if logger:
                logger.warning("No valid classification samples in evaluation set")

    metrics = {
        "total_loss": total_loss / len(dataloader),
        "regression_loss": total_regression_loss / len(dataloader),
        "classification_loss": total_classification_loss / len(dataloader),
        "classification_accuracy": classification_accuracy,
        **regression_metrics,
    }

    if logger:
        logger.info(
            f"Evaluation completed - "
            f"Total Loss: {metrics['total_loss']:.6f}, "
            f"Avg CCC: {regression_metrics['total_ccc']:.6f}, "
            f"V/A/D CCC: {regression_metrics['v_ccc']:.4f}/"
            f"{regression_metrics['a_ccc']:.4f}/"
            f"{regression_metrics['d_ccc']:.4f}"
        )

    return metrics


def main():
    """Evaluate trained model."""
    parser = argparse.ArgumentParser(description="Evaluate VAD regression model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint file"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    use_classification = config["data"].get("use_classification", True)

    # Create experiment directory
    base_save_dir = config["output"]["save_dir"]
    exp_dir = create_experiment_dir(base_save_dir, "eval")

    # Setup logging
    logger = ExperimentLogger(exp_dir)
    logger.info("=" * 80)
    logger.info("EMOTION RECOGNITION EVALUATION STARTED")
    logger.info("=" * 80)
    logger.info(f"Evaluating on {args.split} split")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Experiment directory: {exp_dir}")

    # Save evaluation configuration
    eval_config = {
        "checkpoint_path": args.checkpoint,
        "split": args.split,
        "use_classification": use_classification,
        "original_config": config,
    }
    config_save_path = os.path.join(exp_dir, "eval_config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(eval_config, f, default_flow_style=False)
    logger.info("Evaluation configuration saved")

    if config["device"] == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{config.get('gpu_id', 0)}")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = EmotionDataset(
        config["data"]["features_dir"],
        config["data"]["labels_file"],
        split=args.split,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
    )
    logger.info(f"Dataset loaded: {len(dataset)} samples")

    # Load model
    logger.info("Loading model...")
    model = BaselineEmotionGRU(
        input_size=config["model"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        output_size=config["model"]["output_size"],
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
        use_classification=use_classification,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Model loaded from checkpoint")

    criterion = MultiTaskLoss(
        classification_weight=config["training"]["classification_weight"],
        use_classification=use_classification,
    )

    # Evaluate
    logger.info(f"Starting evaluation on {args.split} split...")
    metrics = evaluate_model(
        model, dataloader, criterion, device, use_classification, logger
    )

    # Save results
    results_file = os.path.join(exp_dir, f"eval_results_{args.split}.txt")
    with open(results_file, "w") as f:
        f.write(f"Evaluation Results on {args.split} split\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Regression Metrics:\n")
        f.write(f"Valence CCC: {metrics['v_ccc']:.4f}\n")
        f.write(f"Arousal CCC: {metrics['a_ccc']:.4f}\n")
        f.write(f"Dominance CCC: {metrics['d_ccc']:.4f}\n")
        f.write(f"Average CCC: {metrics['total_ccc']:.4f}\n\n")

        if use_classification:
            f.write(
                f"Classification Accuracy: {metrics['classification_accuracy']:.4f}\n\n"
            )

        f.write("Loss Values:\n")
        f.write(f"Total Loss: {metrics['total_loss']:.4f}\n")
        f.write(f"Regression Loss: {metrics['regression_loss']:.4f}\n")
        f.write(f"Classification Loss: {metrics['classification_loss']:.4f}\n")

    logger.info(f"Results saved to {results_file}")

    # Print final results
    logger.info("=" * 80)
    logger.info("EVALUATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Results saved in: {exp_dir}")
    logger.info(f"Average CCC: {metrics['total_ccc']:.4f}")
    if use_classification:
        logger.info(
            f"Classification Accuracy: {metrics['classification_accuracy']:.4f}"
        )
    logger.info(f"Evaluation logs: {os.path.join(exp_dir, 'training.log')}")

    print(f"Evaluation completed! Results saved in: {exp_dir}")
    print(f"Average CCC: {metrics['total_ccc']:.4f}")
    if use_classification:
        print(f"Classification Accuracy: {metrics['classification_accuracy']:.4f}")


if __name__ == "__main__":
    main()
