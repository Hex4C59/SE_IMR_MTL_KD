#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extracting features using wav2vec2-base

Created on 2025-09-08
Author: Liu Yang liuyang16@stu.sau.edu.cn
License: MIT License
Project: https://github.com/Hex4C59/SE_IMR_MTL_KD
python: >=3.8 
"""
import os
import torch
import json
import soundfile as sf
import scipy.signal as signal
import torch.nn.functional as F


class Wav2vec2(object):
  def __init__(self, ckpt_path, max_chunk = 1600000):
    print(f"Loading Wav2Vec2 model from pytorch_model.bin: {ckpt_path}")
    
    # Load pytorch_model.bin file directly
    try:
        # Read configuration file
        config_path = os.path.join(ckpt_path, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"Model config: {config.get('model_type', 'wav2vec2')}")
        print(f"Hidden size: {config.get('hidden_size', 768)}")
        print(f"Num layers: {config.get('num_hidden_layers', 12)}")
        
        # Load model weights
        model_path = os.path.join(ckpt_path, 'pytorch_model.bin')
        print(f"Loading weights from: {model_path}")
        
        # Use weights_only=False to avoid version restrictions
        try:
            state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        except TypeError:
            # If weights_only parameter is not supported, use the old loading method
            state_dict = torch.load(model_path, map_location='cpu')
        
        print(f"Loaded {len(state_dict)} parameter tensors")
        
        # Create a simplified Wav2Vec2 model class
        from transformers.models.wav2vec2.modeling_wav2vec2 import (
            Wav2Vec2Model, 
            Wav2Vec2Config
        )
        
        # Create model from configuration
        model_config = Wav2Vec2Config(**config)
        self.model = Wav2Vec2Model(model_config)
        
        # Load weights
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval().cuda()
        
        # Create task configuration
        class SimpleTask:
            class Config:
                sample_rate = 16000  # Wav2Vec2 default sample rate
                normalize = True
            cfg = Config()
        
        self.task = SimpleTask()
        self.max_chunk = max_chunk
        self.is_hf_model = True
        self.config = config
        print("Successfully loaded Wav2Vec2 model from pytorch_model.bin")
        
    except Exception as e1:
        print(f"Failed to load from pytorch_model.bin: {e1}")
        print("Trying torch.hub as backup...")
        
        # Backup solution: use torch.hub
        try:
            self.model = torch.hub.load('pytorch/fairseq', 'wav2vec2_base_100h').eval().cuda()
            
            # Create task configuration
            class SimpleTask:
                class Config:
                    sample_rate = 16000
                    normalize = True
                cfg = Config()
            
            self.task = SimpleTask()
            self.max_chunk = max_chunk
            self.is_hf_model = False
            print("Successfully loaded model with torch.hub")
            
        except Exception as e2:
            print(f"torch.hub loading also failed: {e2}")
            raise RuntimeError(f"All loading methods failed. pytorch_model.bin error: {e1}, torch.hub error: {e2}")
    
  # Read audio data
  def read_audio(self, path):
    # Read audio data and sample rate
    wav, sr = sf.read(path)
    # If audio does not match the specified sample rate, resample to the specified sample rate
    if sr != self.task.cfg.sample_rate:
      num = int((wav.shape[0]) / sr * self.task.cfg.sample_rate)
      wav = signal.resample(wav, num)
      print(f'Resample {sr} to {self.task.cfg.sample_rate}')
    # Specify wav as one-dimensional data
    if wav.ndim == 2:
      wav = wav.mean(-1)
    assert wav.ndim == 1, wav.ndim
    # Return wav audio data
    return wav
  
  # Get features
  def get_feats(self, path, layer):
    '''
    Layer index starts from 1. (e.g. 1-12 for base model)
    '''
    x = self.read_audio(path)
    with torch.no_grad():
      x = torch.from_numpy(x).float().cuda()
      if self.task.cfg.normalize:
        x = F.layer_norm(x, x.shape)
      x = x.view(1, -1)

      feat = []
      for start in range(0, x.size(1), self.max_chunk):
        x_chunk = x[:, start: start + self.max_chunk]
        
        if hasattr(self, 'is_hf_model') and self.is_hf_model:
            # Use Hugging Face model to extract features
            try:
                # Get outputs from all hidden layers
                outputs = self.model(x_chunk, output_hidden_states=True)
                
                # Select the specified layer (layer starts counting from 1, but index starts from 0)
                if layer <= len(outputs.hidden_states):
                    feat_chunk = outputs.hidden_states[layer - 1]  # layer-1 because index starts from 0
                else:
                    # If the requested layer number exceeds the model depth, use the last layer
                    feat_chunk = outputs.last_hidden_state
                    print(f"Warning: Requested layer {layer} exceeds model depth, using last layer")
                
                feat.append(feat_chunk)
                
            except Exception as e:
                print(f"Error extracting features: {e}")
                # Backup solution: only get the last layer
                outputs = self.model(x_chunk)
                feat.append(outputs.last_hidden_state)
        else:
            # Use torch.hub loaded model
            try:
                # torch.hub model may have different interfaces
                if hasattr(self.model, 'extract_features'):
                    feat_chunk = self.model.extract_features(x_chunk)
                    if isinstance(feat_chunk, dict):
                        feat.append(feat_chunk['x'])
                    else:
                        feat.append(feat_chunk)
                else:
                    # Call the model directly
                    feat_chunk = self.model(x_chunk)
                    feat.append(feat_chunk)
            except Exception as e:
                print(f"Error with torch.hub model: {e}")
                feat_chunk = self.model(x_chunk)
                feat.append(feat_chunk)
        
    return torch.cat(feat, 1).squeeze(0)

def extract_w2v2(model: Wav2vec2, wavfile: str, feature_save_path: str) -> None:
  fea = model.get_feats(wavfile, layer=12)

  fea = fea.cpu().detach().numpy()   # (t, 768) for base model

  # Save to .pt file
  savefile = feature_save_path + '.pt'
  torch.save({'feature': fea}, savefile)

def process_split_folder(model: Wav2vec2, split_name: str, split_folder_path: str, feature_save_root: str) -> int:
  """
  Processing a single dataset and dividing it into folders (test/train/validation)
  """
  print(f"Start processing the {split_name} dataset...")
  
  # Create a corresponding feature save directory for each partition
  save_L12 = os.path.join(feature_save_root, 'wav2vec2-base-100h', split_name)
  
  if not os.path.exists(save_L12):
    os.makedirs(save_L12)
  
  # Count the number of files currently divided
  wav_files = []
  for file in os.listdir(split_folder_path):
    if file.endswith('.wav'):
      wav_files.append(file)
  
  total_files = len(wav_files)
  print(f"{split_name} folder has {total_files} wav files")
  
  processed_count = 0
  
  for filename in wav_files:
    # Complete path of wav file
    wavfile = os.path.join(split_folder_path, filename)
    
    # Get filename without extension
    feature_name = os.path.splitext(filename)[0]
    
    # Save feature files to feature directory
    feature_save_path = os.path.join(save_L12, feature_name)
    
    # Check if the feature files exists
    if os.path.exists(feature_save_path + '.pt'):
      processed_count += 1
      print(f'[{split_name}] {processed_count}/{total_files} - Skip existing feature files: {feature_name}')
      continue
    
    try:
      # Extract features
      extract_w2v2(model, wavfile, feature_save_path)
      processed_count += 1
      print(f'[{split_name}] {processed_count}/{total_files} - Finished features extraction: {feature_name}')
    except Exception as e:
      print(f'[{split_name}] Error - Processing files {filename} occurring error: {str(e)}')
      continue
  
  print(f"{split_name} dataset feature extraction completed! Processed {processed_count}/{total_files} files")
  return processed_count

def get_feature(model: Wav2vec2, feature_save_root: str, data_root: str) -> None:
  """
  The main feature extraction function processes the test, train, and validation folders
  """
  splits = ['test', 'train', 'validation']
  total_processed = 0
  for split_name in splits:
    split_folder_path = os.path.join(data_root, split_name)

    # Check folder if exist
    if not os.path.exists(split_folder_path):
      print(f"Warning: folder not exist: {split_folder_path}")
      continue
    
    # Check if folder is empty
    files_in_split = [f for f in os.listdir(split_folder_path) if f.endswith('.wav')]
    if not files_in_split:
      print(f"Warning: {split_name} folder is empty")
      continue
    
    # Process the currently partitioned folder
    processed_count = process_split_folder(model, split_name, split_folder_path, feature_save_root)
    total_processed += processed_count
    
  print(f"\nAll feature extraction completed!")
  print(f"Total processed files: {total_processed}")
  
  # Generate processing summary
  summary_file = os.path.join(feature_save_root, 'extraction_summary.txt')
  with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("Wav2Vec2 Feature Extraction Summary\n")
    f.write("="*50 + "\n\n")
    f.write(f"Data source: {data_root}\n")
    f.write(f"Feature save path: {feature_save_root}\n")
    f.write(f"Total processed files: {total_processed}\n\n")
    
    for split_name in splits:
      split_folder_path = os.path.join(data_root, split_name)
      if os.path.exists(split_folder_path):
        files_count = len([f for f in os.listdir(split_folder_path) if f.endswith('.wav')])
        f.write(f"{split_name}: {files_count} files\n")
  
  print(f"Processing summary saved to: {summary_file}")

if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  feature_save_root = '/mnt/shareEEx/liuyang/code/SE_IMR_MTL_KD/data/features_msp_1.6'
  ckpt_path = "/mnt/shareEEx/liuyang/code/SE_IMR_MTL_KD/pretrain_model/wav2vec2-base-100h"
  data_root = '/mnt/shareEEx/liuyang/code/SE_IMR_MTL_KD/data/organized_datasets/v1.6'

  print("Wav2Vec2 Feature Extraction Script")
  print("="*50)
  print(f"Data source: {data_root}")
  print(f"Feature save path: {feature_save_root}")
  print(f"Model path: {ckpt_path}")
  
  # Check PyTorch version
  torch_version = torch.__version__
  print(f"PyTorch version: {torch_version}")
  
  if not os.path.exists(data_root):
    print(f"Error: Data source directory does not exist: {data_root}")
    exit(1)
  
  if not os.path.exists(ckpt_path):
    print(f"Error: Model path does not exist: {ckpt_path}")
    exit(1)
  
  # Check necessary files
  required_files = ['pytorch_model.bin', 'config.json']
  for file in required_files:
    if not os.path.exists(os.path.join(ckpt_path, file)):
      print(f"Error: Missing necessary file: {file}")
      exit(1)
  
  # List files in the model directory
  print(f"\nFiles in the model directory:")
  for file in os.listdir(ckpt_path):
    print(f"  {file}")
  
  # Initialize the model
  print("\nLoading Wav2vec2 model...")
  try:
    model = Wav2vec2(ckpt_path)
    print("Model loaded successfully!")
  except Exception as e:
    print(f"Model loading failed: {e}")
    exit(1)
  
  # Start extracting features
  get_feature(model, feature_save_root, data_root)

  print("All Finished!!!")