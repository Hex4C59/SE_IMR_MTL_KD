#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py

Created on 2025-09-09
Author: Liu Yang liuyang16@stu.sau.edu.cn
License: MIT License
Project: Speech Emotion Recognition with VAD Regression
python: >=3.8 
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import yaml
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from models.mtl_GRU_model import GRUModel


class VADRegressionModel(nn.Module):
    """
    Wrapper for GRUModel to handle VAD regression
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size=3, dropout=0.2):
        super(VADRegressionModel, self).__init__()
        self.model = GRUModel(input_size, hidden_size, num_layers, output_size, dropout)
        
    def forward(self, x):
        # The GRUModel has issues with its architecture, so we need to handle it properly
        # First GRU layer
        x, _ = self.model.gru_model[0](x)
        # Get the last time step output
        x = x[:, -1, :]
        # First linear layer
        x = self.model.gru_model[1](x)
        # We'll skip the rest of the model as it's not properly designed
        # Instead, we'll return the output directly as VAD predictions
        return x


class EmotionDataset(Dataset):
    """
    Dataset class for loading wav2vec2 features and emotion labels
    
    Args:
        features_dir: Directory containing wav2vec2 features
        labels_file: CSV file with emotion labels
        split: 'train', 'validation', or 'test'
    """
    def __init__(self, features_dir, labels_file, split='train'):
        self.features_dir = features_dir
        self.labels_df = pd.read_csv(labels_file)
        
        # Filter by split
        if split == 'train':
            self.labels_df = self.labels_df[self.labels_df['Split_Set'] == 'Train']
        elif split == 'validation':
            self.labels_df = self.labels_df[self.labels_df['Split_Set'] == 'Development']
        elif split == 'test':
            self.labels_df = self.labels_df[self.labels_df['Split_Set'] == 'Test']
        
        # List of valid samples (files that exist)
        self.valid_samples = []
        for idx, row in self.labels_df.iterrows():
            feature_path = os.path.join(self.features_dir, split, row['FileName'].replace('.wav', '.pt'))
            if os.path.exists(feature_path):
                self.valid_samples.append(idx)
        
        print(f"Loaded {len(self.valid_samples)} valid {split} samples")
        
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        sample_idx = self.valid_samples[idx]
        row = self.labels_df.iloc[sample_idx]
        
        # Load wav2vec2 features
        feature_path = os.path.join(self.features_dir, 
                                   'train' if row['Split_Set'] == 'Train' else 
                                   'validation' if row['Split_Set'] == 'Development' else 'test',
                                   row['FileName'].replace('.wav', '.pt'))
        features = torch.load(feature_path)
        
        # Get VAD labels
        valence = float(row['EmoVal'])
        arousal = float(row['EmoAct'])  # Activation is Arousal
        dominance = float(row['EmoDom'])
        
        # Combine VAD values into a single tensor
        vad = torch.tensor([valence, arousal, dominance], dtype=torch.float)
        
        return {
            'features': features,
            'vad': vad,
            'filename': row['FileName']
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences
    """
    # Sort batch by sequence length (descending)
    batch = sorted(batch, key=lambda x: x['features'].shape[0], reverse=True)
    
    # Get sequence lengths
    lengths = [x['features'].shape[0] for x in batch]
    max_length = max(lengths)
    
    # Pad sequences
    features = []
    for sample in batch:
        padded = torch.zeros(max_length, sample['features'].shape[1])
        padded[:sample['features'].shape[0], :] = sample['features']
        features.append(padded)
    
    # Stack tensors
    features = torch.stack(features)
    vad = torch.stack([x['vad'] for x in batch])
    
    return {
        'features': features,
        'lengths': lengths,
        'vad': vad,
        'filenames': [x['filename'] for x in batch]
    }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train for one epoch
    """
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # Move data to device
        features = batch['features'].to(device)
        vad = batch['vad'].to(device)
        
        # Forward pass
        outputs = model(features)
        
        # Calculate loss
        loss = criterion(outputs, vad)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss


def validate(model, dataloader, criterion, device):
    """
    Validate the model
    """
    model.eval()
    total_loss = 0
    
    # For metrics calculation
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for batch in progress_bar:
            # Move data to device
            features = batch['features'].to(device)
            vad = batch['vad'].to(device)
            
            # Forward pass
            outputs = model(features)
            
            # Calculate loss
            loss = criterion(outputs, vad)
            
            # Update statistics
            total_loss += loss.item()
            
            # Collect predictions and labels for metrics
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(vad.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader)
    
    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Calculate metrics for each dimension
    v_mse = mean_squared_error(all_labels[:, 0], all_preds[:, 0])
    a_mse = mean_squared_error(all_labels[:, 1], all_preds[:, 1])
    d_mse = mean_squared_error(all_labels[:, 2], all_preds[:, 2])
    
    v_r2 = r2_score(all_labels[:, 0], all_preds[:, 0])
    a_r2 = r2_score(all_labels[:, 1], all_preds[:, 1])
    d_r2 = r2_score(all_labels[:, 2], all_preds[:, 2])
    
    metrics = {
        'loss': avg_loss,
        'v_mse': v_mse,
        'a_mse': a_mse,
        'd_mse': d_mse,
        'v_r2': v_r2,
        'a_r2': a_r2,
        'd_r2': d_r2,
        'total_mse': (v_mse + a_mse + d_mse) / 3,
        'total_r2': (v_r2 + a_r2 + d_r2) / 3
    }
    
    return metrics


def plot_learning_curves(train_losses, val_metrics, save_dir):
    """
    Plot and save learning curves
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot([m['loss'] for m in val_metrics], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()
    
    # Plot MSE metrics
    plt.figure(figsize=(10, 6))
    plt.plot([m['v_mse'] for m in val_metrics], label='Valence MSE')
    plt.plot([m['a_mse'] for m in val_metrics], label='Arousal MSE')
    plt.plot([m['d_mse'] for m in val_metrics], label='Dominance MSE')
    plt.plot([m['total_mse'] for m in val_metrics], label='Average MSE', linestyle='--')
    plt.title('Validation MSE by Dimension')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'mse_curve.png'))
    plt.close()
    
    # Plot R² metrics
    plt.figure(figsize=(10, 6))
    plt.plot([m['v_r2'] for m in val_metrics], label='Valence R²')
    plt.plot([m['a_r2'] for m in val_metrics], label='Arousal R²')
    plt.plot([m['d_r2'] for m in val_metrics], label='Dominance R²')
    plt.plot([m['total_r2'] for m in val_metrics], label='Average R²', linestyle='--')
    plt.title('Validation R² by Dimension')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'r2_curve.png'))
    plt.close()


def save_checkpoint(model, optimizer, epoch, save_dir, metrics, filename='checkpoint.pt'):
    """
    Save model checkpoint
    """
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, os.path.join(save_dir, filename))
    print(f"Checkpoint saved to {os.path.join(save_dir, filename)}")


def load_config(config_path):
    """
    Load configuration from YAML file
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """
    Main training function
    """
    parser = argparse.ArgumentParser(description='Train VAD regression model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    save_dir = config['output']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # Create datasets
    train_dataset = EmotionDataset(
        config['data']['features_dir'],
        config['data']['labels_file'],
        split='train'
    )
    val_dataset = EmotionDataset(
        config['data']['features_dir'],
        config['data']['labels_file'],
        split='validation'
    )
    test_dataset = EmotionDataset(
        config['data']['features_dir'],
        config['data']['labels_file'],
        split='test'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    # Create model
    model = VADRegressionModel(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        output_size=config['model']['output_size'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['training']['scheduler']['factor'],
        patience=config['training']['scheduler']['patience'],
        min_lr=config['training']['scheduler']['min_lr'],
        verbose=True
    )
    
    # Early stopping parameters
    early_stopping_patience = config['training']['early_stopping']['patience']
    early_stopping_min_delta = config['training']['early_stopping']['min_delta']
    early_stopping_counter = 0
    best_val_loss = float('inf')
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    train_losses = []
    val_metrics_list = []
    
    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        val_metrics_list.append(val_metrics)
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}")
        print(f"Valence MSE: {val_metrics['v_mse']:.4f}, R²: {val_metrics['v_r2']:.4f}")
        print(f"Arousal MSE: {val_metrics['a_mse']:.4f}, R²: {val_metrics['a_r2']:.4f}")
        print(f"Dominance MSE: {val_metrics['d_mse']:.4f}, R²: {val_metrics['d_r2']:.4f}")
        print(f"Average MSE: {val_metrics['total_mse']:.4f}, R²: {val_metrics['total_r2']:.4f}")
        
        # Save checkpoint
        if val_metrics['loss'] < best_val_loss - early_stopping_min_delta:
            best_val_loss = val_metrics['loss']
            early_stopping_counter = 0
            save_checkpoint(model, optimizer, epoch, save_dir, val_metrics, 'best_model.pt')
        else:
            early_stopping_counter += 1
            print(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
        
        # Save regular checkpoint
        if (epoch + 1) % config['output']['save_interval'] == 0:
            save_checkpoint(model, optimizer, epoch, save_dir, val_metrics, f'checkpoint_epoch_{epoch+1}.pt')
        
        # Plot learning curves
        plot_learning_curves(train_losses, val_metrics_list, save_dir)
        
        # Early stopping
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_metrics = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Valence MSE: {test_metrics['v_mse']:.4f}, R²: {test_metrics['v_r2']:.4f}")
    print(f"Arousal MSE: {test_metrics['a_mse']:.4f}, R²: {test_metrics['a_r2']:.4f}")
    print(f"Dominance MSE: {test_metrics['d_mse']:.4f}, R²: {test_metrics['d_r2']:.4f}")
    print(f"Average MSE: {test_metrics['total_mse']:.4f}, R²: {test_metrics['total_r2']:.4f}")
    
    # Save final model
    save_checkpoint(model, optimizer, num_epochs, save_dir, test_metrics, 'final_model.pt')
    
    # Save test results
    with open(os.path.join(save_dir, 'test_results.txt'), 'w') as f:
        f.write(f"Test Loss: {test_metrics['loss']:.4f}\n")
        f.write(f"Valence MSE: {test_metrics['v_mse']:.4f}, R²: {test_metrics['v_r2']:.4f}\n")
        f.write(f"Arousal MSE: {test_metrics['a_mse']:.4f}, R²: {test_metrics['a_r2']:.4f}\n")
        f.write(f"Dominance MSE: {test_metrics['d_mse']:.4f}, R²: {test_metrics['d_r2']:.4f}\n")
        f.write(f"Average MSE: {test_metrics['total_mse']:.4f}, R²: {test_metrics['total_r2']:.4f}\n")
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()