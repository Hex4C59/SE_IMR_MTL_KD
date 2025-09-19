#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regression_losses.py.

Date: 2025-09-09
Author: Liu Yang
License: MIT
Project: https://github.com/your_project_homepage

CCC loss for emotion regression (VAD).

Implements the Concordance Correlation Coefficient (CCC) loss as a weighted combination
of CCC for valence, arousal, and dominance dimensions.

Example :
    >>> criterion = CCCLoss()
    >>> pred = torch.randn(32, 3)
    >>> target = torch.randn(32, 3)
    >>> loss = criterion(pred, target)
    >>> print(loss.shape)  # ()
"""

import torch
import torch.nn as nn


class CCCLoss(nn.Module):
    """
    Concordance Correlation Coefficient (CCC) loss for VAD regression.

    Args :
        alpha (float): Weight for valence CCC (default: 1/3).
        beta (float): Weight for arousal CCC (default: 1/3).

    Returns :
        torch.Tensor: Scalar CCC loss.

    Raises :
        None

    Examples :
        >>> criterion = CCCLoss()
        >>> pred = torch.randn(32, 3)
        >>> target = torch.randn(32, 3)
        >>> loss = criterion(pred, target)
        >>> print(loss.shape)  # ()
    """

    def __init__(self, alpha: float = 1 / 3, beta: float = 1 / 3):
        """
        Initialize CCCLoss with weights for valence and arousal.

        Args :
            alpha (float): Weight for valence CCC (default: 1/3).
            beta (float): Weight for arousal CCC (default: 1/3).
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute CCC loss for VAD dimensions.

        Args :
            pred (torch.Tensor): Predictions [batch_size, 3]
            target (torch.Tensor): Ground truth [batch_size, 3]

        Returns :
            torch.Tensor: Scalar CCC loss
        """
        ccc_v = self._ccc(pred[:, 0], target[:, 0])
        ccc_a = self._ccc(pred[:, 1], target[:, 1])
        ccc_d = self._ccc(pred[:, 2], target[:, 2])
        loss = 1 - (
            self.alpha * ccc_v
            + self.beta * ccc_a
            + (1 - self.alpha - self.beta) * ccc_d
        )
        return loss

    @staticmethod
    def _ccc(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute CCC for a single dimension.

        Args :
            x (torch.Tensor): Estimated values
            y (torch.Tensor): Ground truth values

        Returns :
            torch.Tensor: CCC value
        """
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)
        var_x = torch.var(x, unbiased=False)
        var_y = torch.var(y, unbiased=False)
        std_x = torch.sqrt(var_x + 1e-8)
        std_y = torch.sqrt(var_y + 1e-8)

        # Calculate correlation coefficient ρ
        cov_xy = torch.mean((x - mean_x) * (y - mean_y))
        rho = cov_xy / (std_x * std_y + 1e-8)

        # CCC formula: 2ρσxσy / (σx² + σy² + (μx - μy)²)
        ccc = (2 * rho * std_x * std_y) / (
            var_x + var_y + (mean_x - mean_y) ** 2 + 1e-8
        )
        return ccc
