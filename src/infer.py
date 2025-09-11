#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run inference for emotion regression/classification model.

This module loads a trained model and runs inference on a specified split,
saving predictions and logging the process.

Example:
    >>> python src/infer.py \
        --config configs/infer.yaml \
        --checkpoint runs/train_xxx/best_model.pt \
        --split test

"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2025-11-15"

import argparse
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import EmotionDataset, collate_fn
from metrics.regression_metrics import compute_metrics
from models.mtl_GRU_model import BaselineEmotionGRU
from utils.logger import ExperimentLogger


def _get_feature_id_from_config(config: dict) -> str:
    """
    Extract a short feature identifier from the config.

    Args :
        config (dict): Configuration dictionary.

    Returns :
        str: Feature identifier string.
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
    base_dir: str,
    task_type: str = "test",
    write_info: bool = True,
    feature_id: str = "",
) -> str:
    """
    Create an experiment directory with timestamp and optional feature id.

    Args :
        base_dir (str): Base directory for experiments.
        task_type (str): Task type label.
        write_info (bool): Whether to write info file.
        feature_id (str): Optional feature id to append to folder name.

    Returns :
        str: Path to the created experiment directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{feature_id}" if feature_id else ""
    exp_name = f"{task_type}_{timestamp}{suffix}"
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    if write_info:
        info_file = os.path.join(exp_dir, "experiment_info.txt")
        with open(info_file, "w") as f:
            f.write(f"Task Type: {task_type}\n")
            f.write(f"Feature: {feature_id}\n")
            f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Experiment Directory: {exp_dir}\n")

    return exp_dir


def inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_classification: bool = True,
    logger: Optional[ExperimentLogger] = None,
) -> Dict[str, np.ndarray]:
    """
    Run inference on a dataset and return predictions and optional labels.

    Args:
        model: Trained PyTorch model.
        dataloader: DataLoader for the inference dataset.
        device: Torch device to run inference on.
        use_classification: Whether classification outputs are expected.
        logger: Optional ExperimentLogger for progress logging.

    Returns:
        A dict containing:
        - "filenames": list of file identifiers
        - "vad_predictions": np.ndarray shape (N, 3)
        - "vad_labels": np.ndarray shape (N, 3) if ground-truth available
        - "emotion_predictions": np.ndarray shape (N, num_classes) if available

    Raises:
        None

    Example:
        >>> # assumes model and dataloader exist
        >>> # results = inference(model, dataloader, torch.device("cpu"), False, None)
        >>> # isinstance(results["vad_predictions"], np.ndarray)
        True

    """
    model.eval()
    all_regression_preds: List[np.ndarray] = []
    all_classification_preds: List[np.ndarray] = []
    all_filenames: List[str] = []
    all_vad_labels: List[np.ndarray] = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Inference")
        for batch_idx, batch in enumerate(progress_bar):
            features = batch["features"].to(device)
            filenames = batch.get("filenames", [])

            outputs = model(features)

            all_regression_preds.append(outputs["regression"].cpu().numpy())
            all_filenames.extend(filenames)

            # collect ground-truth vad when provided by dataset/collate_fn
            if "vad" in batch and batch["vad"] is not None:
                all_vad_labels.append(batch["vad"].cpu().numpy())

            if use_classification and "classification" in outputs:
                all_classification_preds.append(outputs["classification"].cpu().numpy())

            if logger and batch_idx % 50 == 0:
                logger.info(f"Inference batch {batch_idx}/{len(dataloader)}")

    vad_preds = np.concatenate(all_regression_preds, axis=0)
    results: Dict[str, np.ndarray] = {
        "filenames": np.array(all_filenames, dtype=object),
        "vad_predictions": vad_preds,
    }

    if all_vad_labels:
        results["vad_labels"] = np.concatenate(all_vad_labels, axis=0)

    if use_classification and all_classification_preds:
        results["emotion_predictions"] = np.concatenate(
            all_classification_preds, axis=0
        )

    return results


def main() -> None:
    """
    Run inference with a trained model and save predictions.

    Args:
        None

    Returns:
        None

    Raises:
        None

    """
    parser = argparse.ArgumentParser(description="Inference with VAD regression model")
    parser.add_argument(
        "--config", type=str, default="configs/infer.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint file"
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "validation", "test"]
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    use_classification = config["data"].get("use_classification", True)

    feature_id = _get_feature_id_from_config(config)
    # Create experiment directory (do not write experiment_info.txt)
    base_save_dir = config["output"]["save_dir"]
    exp_dir = create_experiment_dir(
        base_save_dir, "test", write_info=False, feature_id=feature_id
    )

    # Setup logger
    logger = ExperimentLogger(exp_dir)

    # INFERRUN: disable epoch_results.log generation for this run only.
    # The logger implementation is shared; here we remove results_logger handlers
    # so that no epoch_results.log is produced when running infer.py.
    if hasattr(logger, "results_logger"):
        for h in list(logger.results_logger.handlers):
            try:
                logger.results_logger.removeHandler(h)
            except Exception:
                pass
            try:
                h.close()
            except Exception:
                pass
        # Remove file if already created
        epoch_file = os.path.join(exp_dir, "epoch_results.log")
        if os.path.exists(epoch_file):
            try:
                os.remove(epoch_file)
            except Exception:
                pass

    logger.info(f"Starting inference: split={args.split}, checkpoint={args.checkpoint}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Experiment directory: {exp_dir}")

    # copy config into result folder so infer config is saved with outputs
    try:
        shutil.copy(args.config, os.path.join(exp_dir, os.path.basename(args.config)))
    except Exception:
        # ignore copy failure (keep behavior simple)
        pass

    device = (
        torch.device(f"cuda:{config.get('gpu_id', 0)}")
        if config["device"] == "cuda" and torch.cuda.is_available()
        else torch.device("cpu")
    )
    logger.info(f"Using device: {device}")

    # Load dataset
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
    logger.info(f"Loaded {len(dataset)} samples for inference.")

    # Load model
    model = BaselineEmotionGRU(
        input_size=config["model"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        output_size=config["model"]["output_size"],
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
        use_classification=use_classification,
    ).to(device)

    # Load checkpoint (trusted source fallback)
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    except Exception:
        logger.warning(
            (
                "Safe torch.load failed; retrying with weights_only=False "
                "(trusted checkpoint)."
            )
        )
        checkpoint = torch.load(
            args.checkpoint, map_location=device, weights_only=False
        )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    logger.info("Model loaded from checkpoint.")

    # Run inference
    logger.info(f"Running inference on {args.split} split...")
    results = inference(model, dataloader, device, use_classification, logger)

    # If ground-truth VAD present, compute CCC metrics and log them
    if "vad_labels" in results:
        v_metrics = compute_metrics(results["vad_labels"], results["vad_predictions"])
        logger.info(
            "Inference metrics - "
            f"V_CCC: {v_metrics['v_ccc']:.4f}, A_CCC: {v_metrics['a_ccc']:.4f}, "
            f"D_CCC: {v_metrics['d_ccc']:.4f}, AVG_CCC: {v_metrics['total_ccc']:.4f}"
        )

    # Save predictions as npz
    predictions_file = os.path.join(exp_dir, f"predictions_{args.split}.npz")
    np.savez(predictions_file, **results, allow_pickle=True)
    logger.info(f"Predictions saved to {predictions_file}")

    # Save readable results: include forecast and true label (true label under forecast)
    readable_file = os.path.join(exp_dir, f"predictions_{args.split}.txt")
    with open(readable_file, "w") as f:
        f.write(f"Inference Results on {args.split} split\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Inference Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        vad_labels = results.get("vad_labels", None)
        for i, filename in enumerate(results["filenames"]):
            vad = results["vad_predictions"][i]
            f.write(f"File: {filename}\n")
            f.write(
                f"forecast results:VAD: [{vad[0]:.4f}, {vad[1]:.4f}, {vad[2]:.4f}]\n"
            )
            if vad_labels is not None:
                true_vad = vad_labels[i]
                f.write(
                    f"true label:VAD:[{true_vad[0]:.4f}, "
                    f"{true_vad[1]:.4f}, {true_vad[2]:.4f}]\n"
                )
            if "emotion_predictions" in results:
                emotion_probs = results["emotion_predictions"][i]
                predicted_class = int(np.argmax(emotion_probs))
                f.write(
                    f"Emotion Class: {predicted_class} "
                    f"(confidence: {emotion_probs[predicted_class]:.4f})\n"
                )
            f.write("\n")

    logger.info(f"Readable predictions saved to {readable_file}")
    logger.info("Inference completed successfully.")


if __name__ == "__main__":
    main()
