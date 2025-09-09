"""
Evaluation script for VAD regression models.

This module handles the evaluation of trained models on test data,
generating comprehensive evaluation reports and visualizations.

Example:
    >>> python src/eval.py --config configs/config.yaml --model_path results/vad_regression/best_model.pt

"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2023-11-15"


import torch
from tqdm import tqdm


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model on a dataset.

    Args:
        model (torch.nn.Module): Model to evaluate
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation
        criterion (torch.nn.Module): Loss function
        device (torch.device): Device to run evaluation on

    Returns:
        tuple: (metrics, predictions, labels, filenames)

    """
    model.eval()
    total_loss = 0

    # For metrics calculation
    all_preds = []
    all_labels = []
    all_filenames = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        for batch in progress_bar:
            # Move data to device
            features = batch["features"].to(device)
            vad = batch["vad"].to(device)
