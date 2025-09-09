"""
Regression loss functions for emotion recognition tasks.

This module provides various loss functions for regression tasks,
particularly for VAD (Valence, Arousal, Dominance) prediction.

Example:
    >>> loss_fn = get_loss_function('mse')
    >>> loss = loss_fn(predictions, targets)

"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2023-11-15"

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcordanceLoss(nn.Module):
    """
    Concordance Correlation Coefficient (CCC) loss.

    CCC measures the agreement between two variables and is commonly
    used for emotion regression tasks.
    """

    def __init__(self):
        super(ConcordanceLoss, self).__init__()

    def forward(self, pred, target):
        """
        Calculate CCC loss.

        Args:
            pred (torch.Tensor): Predictions
            target (torch.Tensor): Ground truth values

        Returns:
            torch.Tensor: CCC loss (1 - CCC to convert to a loss)

        """
        # Calculate means
        pred_mean = torch.mean(pred, dim=0)
        target_mean = torch.mean(target, dim=0)

        # Calculate variances
        pred_var = torch.var(pred, dim=0, unbiased=False)
        target_var = torch.var(target, dim=0, unbiased=False)

        # Calculate covariance
        pred_centered = pred - pred_mean
        target_centered = target - target_mean
        covariance = torch.mean(pred_centered * target_centered, dim=0)

        # Calculate CCC
        numerator = 2 * covariance
        denominator = pred_var + target_var + (pred_mean - target_mean) ** 2
        ccc = numerator / (
            denominator + 1e-8
        )  # Add small epsilon to avoid division by zero

        # Return loss (1 - CCC)
        return torch.mean(1 - ccc)


class MSELoss(nn.Module):
    """Mean Squared Error loss with optional weighting."""

    def __init__(self, weights=None):
        """
        Initialize MSE loss with optional dimension weights.

        Args:
            weights (list, optional): Weights for each dimension

        """
        super(MSELoss, self).__init__()
        self.weights = weights

    def forward(self, pred, target):
        """
        Calculate MSE loss.

        Args:
            pred (torch.Tensor): Predictions
            target (torch.Tensor): Ground truth values

        Returns:
            torch.Tensor: MSE loss

        """
        if self.weights is not None:
            weights = torch.tensor(self.weights, device=pred.device)
            mse = F.mse_loss(pred, target, reduction="none")
            return torch.mean(mse * weights.unsqueeze(0))
        else:
            return F.mse_loss(pred, target)


def get_loss_function(loss_type, **kwargs):
    """
    Get the specified loss function.

    Args:
        loss_type (str): Type of loss function ('mse', 'ccc', etc.)
        **kwargs: Additional arguments for the loss function

    Returns:
        nn.Module: Loss function

    Raises:
        ValueError: If loss_type is not supported

    """
    if loss_type.lower() == "mse":
        return MSELoss(weights=kwargs.get("weights", None))
    elif loss_type.lower() == "ccc":
        return ConcordanceLoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_type}")
