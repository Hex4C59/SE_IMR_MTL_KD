#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualization.py

Date: 2025-09-10
Author: Liu Yang
License: MIT
Project: https://github.com/Hex4C59/SE_IMR_MTL_KD

Visualization utilities for training progress and metrics.
"""

import os
from typing import Dict, List

import matplotlib.pyplot as plt


def plot_training_summary(
    train_losses: List[Dict[str, float]], 
    val_metrics: List[Dict[str, float]], 
    save_dir: str,
    use_classification: bool = True
) -> None:
    """Plot and save training progress summary.

    Args:
        train_losses: List of training loss dictionaries per epoch
        val_metrics: List of validation metrics dictionaries per epoch
        save_dir: Directory to save the plots
        use_classification: Whether classification task is used

    Returns:
        None

    Raises:
        None

    Examples:
        >>> train_losses = [{'total_loss': 1.0, 'regression_loss': 0.8}]
        >>> val_metrics = [{'total_loss': 0.9, 'total_ccc': 0.5}]
        >>> plot_training_summary(train_losses, val_metrics, "output/", True)
    """
    if not train_losses or not val_metrics:
        return

    epochs = range(1, len(train_losses) + 1)
    
    if use_classification:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training Progress (Multi-task Learning)', fontsize=16)
        
        # Total Loss
        train_total_losses = [loss['total_loss'] for loss in train_losses]
        val_total_losses = [metrics['total_loss'] for metrics in val_metrics]
        
        axes[0, 0].plot(epochs, train_total_losses, label="Train Total Loss", color='blue')
        axes[0, 0].plot(epochs, val_total_losses, label="Validation Total Loss", color='orange')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Regression Loss (CCC)
        train_reg_losses = [loss['regression_loss'] for loss in train_losses]
        val_reg_losses = [metrics['regression_loss'] for metrics in val_metrics]
        
        axes[0, 1].plot(epochs, train_reg_losses, label="Train Regression Loss", color='green')
        axes[0, 1].plot(epochs, val_reg_losses, label="Validation Regression Loss", color='red')
        axes[0, 1].set_title('Regression Loss (CCC)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Classification Loss
        train_cls_losses = [loss['classification_loss'] for loss in train_losses]
        val_cls_losses = [metrics['classification_loss'] for metrics in val_metrics]
        
        axes[1, 0].plot(epochs, train_cls_losses, label="Train Classification Loss", color='purple')
        axes[1, 0].plot(epochs, val_cls_losses, label="Validation Classification Loss", color='brown')
        axes[1, 0].set_title('Classification Loss (Cross-Entropy)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Validation Metrics
        avg_ccc = [metrics['total_ccc'] for metrics in val_metrics]
        cls_accuracy = [metrics['classification_accuracy'] for metrics in val_metrics]
        
        ax_ccc = axes[1, 1]
        ax_acc = ax_ccc.twinx()
        
        line1 = ax_ccc.plot(epochs, avg_ccc, label="Average CCC", color='green', linewidth=2)
        line2 = ax_acc.plot(epochs, cls_accuracy, label="Classification Accuracy", color='red', linewidth=2)
        
        ax_ccc.set_xlabel('Epoch')
        ax_ccc.set_ylabel('Average CCC', color='green')
        ax_acc.set_ylabel('Classification Accuracy', color='red')
        ax_ccc.set_title('Validation Metrics')
        ax_ccc.grid(True)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax_ccc.legend(lines, labels, loc='upper left')
        
    else:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Training Progress (Single-task VAD Regression)', fontsize=16)
        
        # Total Loss (Regression only)
        train_total_losses = [loss['total_loss'] for loss in train_losses]
        val_total_losses = [metrics['total_loss'] for metrics in val_metrics]
        
        axes[0].plot(epochs, train_total_losses, label="Train Regression Loss", color='blue')
        axes[0].plot(epochs, val_total_losses, label="Validation Regression Loss", color='orange')
        axes[0].set_title('Regression Loss (CCC)')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Validation CCC
        avg_ccc = [metrics['total_ccc'] for metrics in val_metrics]
        v_ccc = [metrics['v_ccc'] for metrics in val_metrics]
        a_ccc = [metrics['a_ccc'] for metrics in val_metrics]
        d_ccc = [metrics['d_ccc'] for metrics in val_metrics]
        
        axes[1].plot(epochs, avg_ccc, label="Average CCC", color='black', linewidth=3)
        axes[1].plot(epochs, v_ccc, label="Valence CCC", color='red', alpha=0.7)
        axes[1].plot(epochs, a_ccc, label="Arousal CCC", color='green', alpha=0.7)
        axes[1].plot(epochs, d_ccc, label="Dominance CCC", color='blue', alpha=0.7)
        axes[1].set_title('Validation CCC Metrics')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('CCC')
        axes[1].legend()
        axes[1].grid(True)

    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'training_progress.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training progress plot saved to: {save_path}")


def plot_detailed_metrics(val_metrics: List[Dict[str, float]], save_dir: str) -> None:
    """Plot detailed validation metrics over epochs.

    Args:
        val_metrics: List of validation metrics dictionaries per epoch
        save_dir: Directory to save the plots

    Returns:
        None

    Raises:
        None

    Examples:
        >>> val_metrics = [{'v_ccc': 0.5, 'a_ccc': 0.6, 'd_ccc': 0.4}]
        >>> plot_detailed_metrics(val_metrics, "output/")
    """
    if not val_metrics:
        return

    epochs = range(1, len(val_metrics) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Detailed Validation Metrics', fontsize=16)

    # CCC for each dimension
    v_ccc = [metrics['v_ccc'] for metrics in val_metrics]
    a_ccc = [metrics['a_ccc'] for metrics in val_metrics]
    d_ccc = [metrics['d_ccc'] for metrics in val_metrics]
    
    axes[0, 0].plot(epochs, v_ccc, label="Valence CCC", color='red')
    axes[0, 0].set_title('Valence CCC')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('CCC')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(epochs, a_ccc, label="Arousal CCC", color='green')
    axes[0, 1].set_title('Arousal CCC')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('CCC')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(epochs, d_ccc, label="Dominance CCC", color='blue')
    axes[1, 0].set_title('Dominance CCC')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('CCC')
    axes[1, 0].grid(True)

    # MSE for each dimension
    v_mse = [metrics['v_mse'] for metrics in val_metrics]
    a_mse = [metrics['a_mse'] for metrics in val_metrics]
    d_mse = [metrics['d_mse'] for metrics in val_metrics]
    
    axes[1, 1].plot(epochs, v_mse, label="Valence MSE", color='red', alpha=0.7)
    axes[1, 1].plot(epochs, a_mse, label="Arousal MSE", color='green', alpha=0.7)
    axes[1, 1].plot(epochs, d_mse, label="Dominance MSE", color='blue', alpha=0.7)
    axes[1, 1].set_title('Mean Squared Error')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MSE')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'detailed_metrics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Detailed metrics plot saved to: {save_path}")