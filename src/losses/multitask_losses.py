#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multitask_losses.py

Date: 2025-09-10
Author: Liu Yang
License: MIT
Project: https://github.com/Hex4C59/SE_IMR_MTL_KD

Multi-task loss for emotion regression/classification.
"""

import torch
import torch.nn as nn

from .regression_losses import CCCLoss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for regression (CCC) and classification (Cross-Entropy).

    Args:
        classification_weight (float): Weight for classification loss.
        alpha (float): Weight for valence CCC.
        beta (float): Weight for arousal CCC.
        use_classification (bool): Whether to use classification loss.

    Returns:
        dict: {'total_loss', 'regression_loss', 'classification_loss'}

    Raises:
        None

    Examples:
        >>> criterion = MultiTaskLoss(classification_weight=0.2, use_classification=True)
        >>> outputs = {
        ...     'regression': torch.randn(32, 3),
        ...     'classification': torch.randn(32, 7)
        ... }
        >>> targets = {
        ...     'vad': torch.randn(32, 3),
        ...     'emotion_class': torch.randint(0, 7, (32,)),
        ...     'use_for_classification': torch.ones(32, dtype=torch.bool)
        ... }
        >>> loss_dict = criterion(outputs, targets)
        >>> print(loss_dict['total_loss'].shape)
    """

    def __init__(
        self,
        classification_weight: float = 0.2,
        alpha: float = 1 / 3,
        beta: float = 1 / 3,
        use_classification: bool = True,
    ) -> None:
        super().__init__()
        self.classification_weight = classification_weight
        self.use_classification = use_classification
        self.regression_loss = CCCLoss(alpha=alpha, beta=beta)
        
        if self.use_classification:
            self.classification_loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, predictions: dict, targets: dict) -> dict:
        """
        Compute loss (single-task or multi-task based on configuration).

        Args:
            predictions (dict): Model outputs.
            targets (dict): Ground truth targets.

        Returns:
            dict: Losses.

        Raises:
            None

        Examples:
            >>> criterion = MultiTaskLoss(use_classification=False)
            >>> outputs = {'regression': torch.randn(8, 3)}
            >>> targets = {'vad': torch.randn(8, 3)}
            >>> out = criterion(outputs, targets)
            >>> print(out['total_loss'].shape)
        """
        reg_loss = self.regression_loss(predictions["regression"], targets["vad"])
        
        if not self.use_classification:
            # Single-task learning: only regression loss
            return {
                "total_loss": reg_loss,
                "regression_loss": reg_loss,
                "classification_loss": torch.tensor(0.0, device=reg_loss.device),
            }
        
        # Multi-task learning: regression + classification
        mask = targets.get("use_for_classification", None)
        if mask is not None and mask.any():
            cls_pred = predictions["classification"][mask]
            cls_target = targets["emotion_class"][mask]
            cls_loss = self.classification_loss(cls_pred, cls_target)
        else:
            cls_loss = torch.tensor(0.0, device=predictions["classification"].device)
        
        total_loss = reg_loss + self.classification_weight * cls_loss
        return {
            "total_loss": total_loss,
            "regression_loss": reg_loss,
            "classification_loss": cls_loss,
        }