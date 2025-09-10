#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regression_metrics.py.

Metrics for regression tasks in emotion recognition.

Date: 2025-09-10
Author: Liu Yang
License: MIT
Project: https://github.com/Hex4C59/SE_IMR_MTL_KD
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def compute_ccc(y_true, y_pred):
    """
    Compute Concordance Correlation Coefficient.

    Args:
        y_true (numpy.ndarray): Ground truth values
        y_pred (numpy.ndarray): Predicted values

    Returns:
        float: CCC value

    Raises:
        None

    Examples:
        >>> y_true = np.array([1, 2, 3, 4, 5])
        >>> y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        >>> ccc = compute_ccc(y_true, y_pred)
        >>> print(f"CCC: {ccc:.4f}")
    """
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    numerator = 2 * covariance
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    return numerator / (denominator + 1e-8)


def compute_metrics(labels, predictions):
    """
    Compute regression metrics for VAD prediction.

    Args:
        labels (numpy.ndarray): Ground truth labels [batch_size, 3]
        predictions (numpy.ndarray): Predicted values [batch_size, 3]

    Returns:
        dict: Dictionary containing various metrics

    Raises:
        None

    Examples:
        >>> labels = np.random.randn(100, 3)
        >>> predictions = np.random.randn(100, 3)
        >>> metrics = compute_metrics(labels, predictions)
        >>> print(f"Average CCC: {metrics['total_ccc']:.4f}")
    """
    v_mse = mean_squared_error(labels[:, 0], predictions[:, 0])
    a_mse = mean_squared_error(labels[:, 1], predictions[:, 1])
    d_mse = mean_squared_error(labels[:, 2], predictions[:, 2])

    v_r2 = r2_score(labels[:, 0], predictions[:, 0])
    a_r2 = r2_score(labels[:, 1], predictions[:, 1])
    d_r2 = r2_score(labels[:, 2], predictions[:, 2])

    v_ccc = compute_ccc(labels[:, 0], predictions[:, 0])
    a_ccc = compute_ccc(labels[:, 1], predictions[:, 1])
    d_ccc = compute_ccc(labels[:, 2], predictions[:, 2])

    metrics = {
        "v_mse": v_mse,
        "a_mse": a_mse,
        "d_mse": d_mse,
        "v_r2": v_r2,
        "a_r2": a_r2,
        "d_r2": d_r2,
        "v_ccc": v_ccc,
        "a_ccc": a_ccc,
        "d_ccc": d_ccc,
        "total_mse": (v_mse + a_mse + d_mse) / 3,
        "total_r2": (v_r2 + a_r2 + d_r2) / 3,
        "total_ccc": (v_ccc + a_ccc + d_ccc) / 3,
    }

    return metrics