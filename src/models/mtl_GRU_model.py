#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mtl_GRU_model.py

Multi-task GRU model for emotion recognition.

Date: 2025-09-10
Author: Liu Yang
License: MIT
Project: https://github.com/Hex4C59/SE_IMR_MTL_KD
"""

import torch
import torch.nn as nn


class BaselineEmotionGRU(nn.Module):
    """Baseline GRU model for emotion recognition.

    Args:
        input_size: Input feature dimension
        output_size: VAD regression output dimension (3)
        num_classes: Number of emotion classes (7)
        hidden_size: GRU hidden dimension
        num_layers: Number of GRU layers
        embedding_dim: Embedding dimension before output heads
        dropout: Dropout probability (deprecated, kept for compatibility)
        use_classification: Whether to include classification head

    Returns:
        Dictionary with regression and optionally classification outputs

    Raises:
        None

    Examples:
        >>> model = BaselineEmotionGRU(768, 3, 7, 128, 2, 128, 0.0, False)
        >>> features = torch.randn(32, 100, 768)
        >>> outputs = model(features)
        >>> print('classification' in outputs)  # False for single-task
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_classes: int = 7,
        hidden_size: int = 128,
        num_layers: int = 2,
        embedding_dim: int = 128,
        dropout: float = 0.2,
        use_classification: bool = True,
    ) -> None:
        super().__init__()
        self.use_classification = use_classification

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=False,
        )

        self.embedding = nn.Linear(hidden_size, embedding_dim)
        self.regression_head = nn.Linear(embedding_dim, output_size)

        if self.use_classification:
            self.classification_head = nn.Linear(embedding_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.constant_(param, 0)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass through the model.

        Args :
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)

        Returns :
            dict: Regression and optionally classification outputs

        Raises :
            None

        Examples :
            >>> model = BaselineEmotionGRU(768, 3, 7, use_classification=False)
            >>> x = torch.randn(32, 100, 768)
            >>> outputs = model(x)
            >>> print(outputs.keys())  # Only 'regression' for single-task
        """
        out, _ = self.gru(x)
        last_step = out[:, -1, :]
        embedding = self.embedding(last_step)
        regression_output = self.regression_head(embedding)

        result = {"regression": regression_output}

        if self.use_classification:
            classification_output = self.classification_head(embedding)
            result["classification"] = classification_output

        return result
