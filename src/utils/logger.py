#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-line summary of the module.

Detailed description of the module.

Example :
    >>> example
"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2025-11-15"

import logging
import os
from datetime import datetime
from typing import Dict


class ExperimentLogger:
    """
    Centralized logging system for emotion recognition experiments.

    Args:
        exp_dir: Experiment directory to save log files
        name: Logger name (default: 'emotion_experiment')

    Returns:
        None

    Raises:
        None

    Examples:
        >>> logger = ExperimentLogger("runs/experiment_001")
        >>> logger.info("Training started")
        >>> logger.log_epoch_results(1, train_loss, val_metrics, True)

    """

    def __init__(self, exp_dir: str, name: str = "emotion_experiment"):
        self.exp_dir = exp_dir
        self.logger_name = name
        self.results_logger_name = f"{name}_results"

        # Setup loggers
        self.logger = self._setup_main_logger()
        self.results_logger = self._setup_results_logger()

        # Track if header has been logged
        self._header_logged = False

    def _setup_main_logger(self) -> logging.Logger:
        """Setup main logger for detailed training logs."""
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(logging.INFO)

        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Create formatter
        detailed_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        # File handler for detailed logs
        log_file = os.path.join(self.exp_dir, "training.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(detailed_formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(detailed_formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def _setup_results_logger(self) -> logging.Logger:
        """Setup results logger for structured epoch data."""
        results_logger = logging.getLogger(self.results_logger_name)
        results_logger.setLevel(logging.INFO)

        # Clear existing handlers
        for handler in results_logger.handlers[:]:
            results_logger.removeHandler(handler)

        # Simple formatter for CSV-like output
        simple_formatter = logging.Formatter("%(message)s")

        # File handler for epoch results
        results_file = os.path.join(self.exp_dir, "epoch_results.log")
        results_handler = logging.FileHandler(results_file)
        results_handler.setLevel(logging.INFO)
        results_handler.setFormatter(simple_formatter)

        results_logger.addHandler(results_handler)

        return results_logger

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def log_experiment_start(self, config_path: str, use_classification: bool) -> None:
        """Log experiment start information."""
        self.logger.info("=" * 80)
        self.logger.info("EMOTION RECOGNITION TRAINING STARTED")
        self.logger.info("=" * 80)
        self.logger.info(
            f"Training mode: {'Multi-task' if use_classification else 'Single-task (VAD regression only)'}"
        )
        self.logger.info(f"Experiment directory: {self.exp_dir}")
        self.logger.info(f"Configuration file: {config_path}")

    def log_experiment_end(
        self, best_epoch: int, best_val_loss: float, final_ccc: float
    ) -> None:
        """Log experiment completion information."""
        self.logger.info("=" * 80)
        self.logger.info("TRAINING COMPLETED SUCCESSFULLY")
        self.logger.info("=" * 80)
        self.logger.info(f"Results saved in: {self.exp_dir}")
        self.logger.info(
            f"Best epoch: {best_epoch} (validation loss: {best_val_loss:.6f})"
        )
        self.logger.info(f"Final average CCC: {final_ccc:.6f}")
        self.logger.info(f"Training logs: {os.path.join(self.exp_dir, 'training.log')}")
        self.logger.info(
            f"Epoch results: {os.path.join(self.exp_dir, 'epoch_results.log')}"
        )

    def log_model_info(self, total_params: int, trainable_params: int) -> None:
        """Log model parameter information."""
        self.logger.info(
            f"Model parameters: {total_params:,} total, {trainable_params:,} trainable"
        )

    def log_dataset_info(self, train_size: int, val_size: int) -> None:
        """Log dataset information."""
        self.logger.info(f"Train dataset size: {train_size}")
        self.logger.info(f"Validation dataset size: {val_size}")

    def log_training_config(self, config: Dict) -> None:
        """Log training configuration."""
        self.logger.info("Training configuration:")
        self.logger.info(f"  - Max epochs: {config.get('num_epochs')}")
        self.logger.info(
            f"  - Early stopping patience: {config.get('early_stopping', {}).get('patience')}"
        )
        self.logger.info(
            f"  - Early stopping min delta: {config.get('early_stopping', {}).get('min_delta')}"
        )
        self.logger.info(f"  - Batch size: {config.get('batch_size')}")

    def log_optimizer_info(self, lr: float, weight_decay: float) -> None:
        """Log optimizer information."""
        self.logger.info(f"Optimizer: Adam (lr={lr}, weight_decay={weight_decay})")

    def log_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """Log epoch start."""
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"EPOCH {epoch}/{total_epochs}")
        self.logger.info(f"{'=' * 60}")

    def log_epoch_summary(
        self,
        epoch: int,
        epoch_time: float,
        train_loss: Dict,
        val_metrics: Dict,
        use_classification: bool,
    ) -> None:
        """Log epoch summary."""
        self.logger.info(f"\nEpoch {epoch} Summary (Duration: {epoch_time:.1f}s):")
        self.logger.info(
            f"  Train - Total: {train_loss['total_loss']:.6f}, "
            f"Reg: {train_loss['regression_loss']:.6f}"
        )
        if use_classification:
            self.logger.info(f"  Train - Cls: {train_loss['classification_loss']:.6f}")
            self.logger.info(
                f"  Val - Cls Acc: {val_metrics['classification_accuracy']:.6f}"
            )
        self.logger.info(
            f"  Val - Total: {val_metrics['total_loss']:.6f}, "
            f"Reg: {val_metrics['regression_loss']:.6f}"
        )
        self.logger.info(
            f"  CCC - V: {val_metrics['v_ccc']:.6f}, "
            f"A: {val_metrics['a_ccc']:.6f}, "
            f"D: {val_metrics['d_ccc']:.6f}, "
            f"Avg: {val_metrics['total_ccc']:.6f}"
        )

    def log_train_epoch_complete(
        self, avg_total_loss: float, avg_reg_loss: float, avg_cls_loss: float
    ) -> None:
        """Log training epoch completion."""
        self.logger.info(
            f"Training epoch completed - "
            f"Total Loss: {avg_total_loss:.6f}, "
            f"Regression Loss: {avg_reg_loss:.6f}, "
            f"Classification Loss: {avg_cls_loss:.6f}"
        )

    def log_validation_complete(
        self,
        avg_total_loss: float,
        avg_ccc: float,
        v_ccc: float,
        a_ccc: float,
        d_ccc: float,
    ) -> None:
        """Log validation completion."""
        self.logger.info(
            f"Validation completed - "
            f"Total Loss: {avg_total_loss:.6f}, "
            f"Avg CCC: {avg_ccc:.6f}, "
            f"V/A/D CCC: {v_ccc:.4f}/{a_ccc:.4f}/{d_ccc:.4f}"
        )

    def log_best_model_saved(self, best_val_loss: float) -> None:
        """Log best model saved."""
        self.logger.info(
            f"★ New best model saved! Validation loss: {best_val_loss:.6f}"
        )

    def log_early_stopping_counter(self, counter: int, patience: int) -> None:
        """Log early stopping counter."""
        self.logger.info(f"Early stopping counter: {counter}/{patience}")

    def log_checkpoint_saved(self, epoch: int) -> None:
        """Log checkpoint saved."""
        self.logger.info(f"Checkpoint saved at epoch {epoch}")

    def log_early_stopping_triggered(self, epoch: int) -> None:
        """Log early stopping triggered."""
        self.logger.info(f"Early stopping triggered after {epoch} epochs")

    def log_batch_details(
        self,
        batch_idx: int,
        total_batches: int,
        total_loss: float,
        reg_loss: float,
        cls_loss: float,
    ) -> None:
        """Log batch details (debug level)."""
        self.logger.debug(
            f"Batch {batch_idx}/{total_batches}: "
            f"Total Loss: {total_loss:.6f}, "
            f"Reg Loss: {reg_loss:.6f}, "
            f"Cls Loss: {cls_loss:.6f}"
        )

    def log_classification_samples(
        self, valid_samples: int, total_samples: int
    ) -> None:
        """Log classification sample information."""
        self.logger.info(f"Classification samples: {valid_samples}/{total_samples}")

    def log_epoch_results(
        self, epoch: int, train_loss: Dict, val_metrics: Dict, use_classification: bool
    ) -> None:
        """
        Log structured epoch results to CSV-like file.

        Args:
            epoch: Current epoch number
            train_loss: Training loss dictionary
            val_metrics: Validation metrics dictionary
            use_classification: Whether classification is used

        Returns:
            None

        Raises:
            None

        Examples:
            >>> logger.log_epoch_results(1, train_loss, val_metrics, True)

        """
        # Log header if not already done
        if not self._header_logged:
            self._log_results_header(use_classification)
            self._header_logged = True

        # Log epoch data
        if use_classification:
            line = (
                f"{epoch},{train_loss['total_loss']:.6f},{train_loss['regression_loss']:.6f},"
                f"{train_loss['classification_loss']:.6f},{val_metrics['total_loss']:.6f},"
                f"{val_metrics['regression_loss']:.6f},{val_metrics['classification_loss']:.6f},"
                f"{val_metrics['classification_accuracy']:.6f},{val_metrics['v_ccc']:.6f},"
                f"{val_metrics['a_ccc']:.6f},{val_metrics['d_ccc']:.6f},"
                f"{val_metrics['total_ccc']:.6f},{val_metrics['v_mse']:.6f},"
                f"{val_metrics['a_mse']:.6f},{val_metrics['d_mse']:.6f},"
                f"{val_metrics['total_mse']:.6f}"
            )
        else:
            line = (
                f"{epoch},{train_loss['total_loss']:.6f},{train_loss['regression_loss']:.6f},"
                f"{val_metrics['total_loss']:.6f},{val_metrics['regression_loss']:.6f},"
                f"{val_metrics['v_ccc']:.6f},{val_metrics['a_ccc']:.6f},"
                f"{val_metrics['d_ccc']:.6f},{val_metrics['total_ccc']:.6f},"
                f"{val_metrics['v_mse']:.6f},{val_metrics['a_mse']:.6f},"
                f"{val_metrics['d_mse']:.6f},{val_metrics['total_mse']:.6f}"
            )

        self.results_logger.info(line)

    def _log_results_header(self, use_classification: bool) -> None:
        """Log header for results file."""
        if use_classification:
            header = (
                "Epoch,Train_Total_Loss,Train_Reg_Loss,Train_Cls_Loss,"
                "Val_Total_Loss,Val_Reg_Loss,Val_Cls_Loss,Val_Cls_Acc,"
                "V_CCC,A_CCC,D_CCC,Avg_CCC,V_MSE,A_MSE,D_MSE,Avg_MSE"
            )
        else:
            header = (
                "Epoch,Train_Total_Loss,Train_Reg_Loss,"
                "Val_Total_Loss,Val_Reg_Loss,"
                "V_CCC,A_CCC,D_CCC,Avg_CCC,V_MSE,A_MSE,D_MSE,Avg_MSE"
            )

        self.results_logger.info(header)

    def save_completion_summary(
        self,
        total_epochs: int,
        best_epoch: int,
        best_val_loss: float,
        final_ccc: float,
        early_stopping_triggered: bool,
    ) -> None:
        """Save training completion summary to file."""
        completion_time = datetime.now()
        completion_file = os.path.join(self.exp_dir, "training_completion.txt")

        with open(completion_file, "w") as f:
            f.write(
                f"Training completed at: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Total epochs: {total_epochs}\n")
            f.write(f"Best epoch: {best_epoch}\n")
            f.write(f"Best validation loss: {best_val_loss:.6f}\n")
            f.write(f"Final average CCC: {final_ccc:.6f}\n")
            f.write(
                f"Early stopping triggered: {'Yes' if early_stopping_triggered else 'No'}\n"
            )
