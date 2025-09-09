"""
Metrics for regression tasks in emotion recognition.

This module provides functions to calculate various metrics for
evaluating regression models, particularly for VAD prediction.

Example:
    >>> metrics = compute_metrics(labels, predictions)
    >>> print(f"MSE: {metrics['total_mse']}, R²: {metrics['total_r2']}")

"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2023-11-15"

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

    """
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    numerator = 2 * covariance
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    return numerator / (
        denominator + 1e-8
    )  # Add small epsilon to avoid division by zero


def compute_metrics(labels, predictions):
    """
    Compute regression metrics for VAD prediction.

    Args:
        labels (numpy.ndarray): Ground truth labels [batch_size, 3]
        predictions (numpy.ndarray): Predicted values [batch_size, 3]

    Returns:
        dict: Dictionary containing various metrics

    """
    # Calculate metrics for each dimension
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
