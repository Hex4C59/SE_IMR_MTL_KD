#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer.py

Inference script for trained VAD regression model.

Date: 2025-09-10
Author: Liu Yang
License: MIT
Project: https://github.com/Hex4C59/SE_IMR_MTL_KD
"""

import argparse
import os
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import EmotionDataset, collate_fn
from models.mtl_GRU_model import BaselineEmotionGRU


def create_experiment_dir(base_dir: str, task_type: str = "test") -> str:
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
    
    print(f"Inference directory created: {exp_dir}")
    return exp_dir


def inference(model, dataloader, device, use_classification=True):
    """Run inference on dataset."""
    model.eval()
    
    all_regression_preds = []
    all_classification_preds = []
    all_filenames = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Inference")
        for batch in progress_bar:
            features = batch["features"].to(device)
            filenames = batch["filenames"]
            
            outputs = model(features)
            
            all_regression_preds.append(outputs["regression"].cpu().numpy())
            all_filenames.extend(filenames)
            
            if use_classification and "classification" in outputs:
                all_classification_preds.append(outputs["classification"].cpu().numpy())

    all_regression_preds = np.concatenate(all_regression_preds, axis=0)
    
    results = {
        "filenames": all_filenames,
        "vad_predictions": all_regression_preds,
    }
    
    if use_classification and all_classification_preds:
        all_classification_preds = np.concatenate(all_classification_preds, axis=0)
        results["emotion_predictions"] = all_classification_preds

    return results


def main():
    """Run inference with trained model."""
    parser = argparse.ArgumentParser(description="Inference with VAD regression model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    use_classification = config["data"].get("use_classification", True)
    
    # Create experiment directory
    base_save_dir = config["output"]["save_dir"]
    exp_dir = create_experiment_dir(base_save_dir, "test")

    if config["device"] == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{config.get('gpu_id', 0)}")
    else:
        device = torch.device("cpu")

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

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Run inference
    print(f"Running inference on {args.split} split...")
    results = inference(model, dataloader, device, use_classification)

    # Save predictions
    predictions_file = os.path.join(exp_dir, f"predictions_{args.split}.npz")
    np.savez(predictions_file, **results)
    
    # Save readable results
    readable_file = os.path.join(exp_dir, f"predictions_{args.split}.txt")
    with open(readable_file, "w") as f:
        f.write(f"Inference Results on {args.split} split\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Inference Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for i, filename in enumerate(results["filenames"]):
            vad = results["vad_predictions"][i]
            f.write(f"File: {filename}\n")
            f.write(f"VAD: [{vad[0]:.4f}, {vad[1]:.4f}, {vad[2]:.4f}]\n")
            
            if "emotion_predictions" in results:
                emotion_probs = results["emotion_predictions"][i]
                predicted_class = np.argmax(emotion_probs)
                f.write(f"Emotion Class: {predicted_class} (confidence: {emotion_probs[predicted_class]:.4f})\n")
            
            f.write("\n")

    print(f"Inference completed! Results saved in: {exp_dir}")


if __name__ == "__main__":
    main()