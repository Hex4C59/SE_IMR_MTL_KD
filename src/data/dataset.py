#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset.py.

Dataset module for emotion recognition with wav2vec2 features.

Date: 2025-09-10
Author: Liu Yang
License: MIT
Project: https://github.com/Hex4C59/SE_IMR_MTL_KD
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    """
    Dataset class for loading wav2vec2 features and emotion labels.

    Args:
        features_dir: Directory containing wav2vec2 features
        labels_file: CSV file with emotion labels
        split: Dataset split ('train', 'validation', or 'test')
        use_classification: Whether to include classification task samples only

    Returns:
        Dictionary containing features, VAD values, emotion class, and metadata

    Raises:
        FileNotFoundError: If labels file or feature files don't exist

    Examples:
        >>> dataset = EmotionDataset("data/features", "labels.csv", "train")
        >>> sample = dataset[0]
        >>> print(sample['features'].shape)
    """

    def __init__(
        self,
        features_dir: str,
        labels_file: str,
        split: str = "train",
        use_classification: bool = True,
    ) -> None:
        self.features_dir = features_dir
        self.split = split
        self.use_classification = use_classification
        self.labels_df = pd.read_csv(labels_file)

        self.emotion_mapping = {
            "H": "Happy",
            "S": "Sad",
            "A": "Angry",
            "U": "Surprise",
            "F": "Fear",
            "D": "Disgust",
            "N": "Neutral",
            "C": None,
            "O": None,
            "X": None,
        }

        self.target_emotion_classes = [
            "Fear",
            "Angry",
            "Disgust",
            "Happy",
            "Neutral",
            "Sad",
            "Surprise",
        ]
        self.emotion_to_idx = {
            emotion: idx for idx, emotion in enumerate(self.target_emotion_classes)
        }

        print(f"Total samples in labels file: {len(self.labels_df)}")
        print(f"Columns: {self.labels_df.columns.tolist()}")
        if "Split_Set" in self.labels_df.columns:
            print(f"Split values: {self.labels_df['Split_Set'].unique()}")

        split_mapping = {"train": "Train", "validation": "Validation", "test": "Test"}
        target_split = split_mapping.get(split, split)
        mask = self.labels_df["Split_Set"] == target_split
        self.labels_df = self.labels_df[mask].reset_index(drop=True)

        print(f"Samples after split filtering: {len(self.labels_df)}")

        self._apply_emotion_mapping()
        self._validate_data()

    def _apply_emotion_mapping(self) -> None:
        """Apply emotion mapping and create classification task mask."""
        if "EmoClass" not in self.labels_df.columns:
            print("Warning: No 'EmoClass' column found")
            self.labels_df["mapped_emotion"] = None
            self.labels_df["use_for_classification"] = False
            return

        mapped_emotions = []
        classification_mask = []
        excluded_count = 0

        for emotion in self.labels_df["EmoClass"]:
            mapped = self.emotion_mapping.get(emotion, None)
            mapped_emotions.append(mapped)

            if mapped is None:
                classification_mask.append(False)
                excluded_count += 1
            else:
                classification_mask.append(True)

        self.labels_df["mapped_emotion"] = mapped_emotions
        self.labels_df["use_for_classification"] = classification_mask

        print("Emotion mapping statistics:")
        print(f"  Total samples: {len(self.labels_df)}")
        print(f"  Samples for classification: {sum(classification_mask)}")
        print(f"  Excluded from classification: {excluded_count}")

        valid_emotions = self.labels_df[self.labels_df["use_for_classification"]]
        if len(valid_emotions) > 0:
            emotion_counts = valid_emotions["mapped_emotion"].value_counts()
            print("  Distribution of 7 target classes:")
            for emotion, count in emotion_counts.items():
                print(f"    {emotion}: {count}")

    def _get_feature_filename(self, filename: str) -> str:
        """Convert label filename to feature filename format."""
        if filename.endswith(".wav"):
            return filename[:-4]
        return filename

    def _get_feature_path(self, filename: str) -> str:
        """Get the correct feature file path based on directory structure."""
        feature_filename = self._get_feature_filename(filename)
        split_subdir = {"train": "train", "validation": "validation", "test": "test"}
        subdir = split_subdir.get(self.split, self.split.lower())
        feature_path = os.path.join(self.features_dir, subdir, f"{feature_filename}.pt")
        return feature_path

    def _validate_data(self) -> None:
        """Validate that feature files exist and emotion classes are valid."""
        valid_indices = []
        missing_files = []

        split_subdir = {"train": "train", "validation": "validation", "test": "test"}
        subdir = split_subdir.get(self.split, self.split.lower())
        subdir_path = os.path.join(self.features_dir, subdir)

        print(f"Looking for features in: {subdir_path}")

        if os.path.exists(subdir_path):
            subdir_files = os.listdir(subdir_path)
            print(f"Files in {subdir} subdirectory: {len(subdir_files)}")
        else:
            print(f"Warning: Subdirectory {subdir_path} does not exist")

        for idx, row in self.labels_df.iterrows():
            filename = row["FileName"]
            feature_path = self._get_feature_path(filename)

            if not os.path.exists(feature_path):
                missing_files.append((filename, feature_path))
                continue

            valid_indices.append(idx)

        print(f"Missing feature files: {len(missing_files)}")

        if missing_files:
            print("Sample missing files:")
            for i, (label_name, feature_path) in enumerate(missing_files[:3]):
                print(f"  Label: {label_name} -> Feature: {feature_path}")

        self.labels_df = self.labels_df.iloc[valid_indices].reset_index(drop=True)

        classification_samples = sum(self.labels_df["use_for_classification"])
        regression_only_samples = len(self.labels_df) - classification_samples

        print(f"Loaded {len(self.labels_df)} total valid samples for {self.split} split")
        print(f"  - Used for BOTH regression AND classification: {classification_samples}")
        print(f"  - Used for regression ONLY: {regression_only_samples}")
        print(f"  - All {len(self.labels_df)} samples used for VAD regression training")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.labels_df)

    def __getitem__(self, idx: int) -> dict:
        """Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing features, VAD values, emotion class, and metadata

        Raises:
            FileNotFoundError: If feature file doesn't exist

        Examples:
            >>> dataset = EmotionDataset("data/features", "labels.csv")
            >>> sample = dataset[0]
            >>> print(sample.keys())
        """
        row = self.labels_df.iloc[idx]
        filename = row["FileName"]

        feature_path = self._get_feature_path(filename)

        try:
            features_data = torch.load(feature_path, map_location="cpu")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load feature file {feature_path}: {e}")

        if isinstance(features_data, dict):
            possible_keys = ["features", "feature", "last_hidden_state", "hidden_states"]
            features = None

            for key in possible_keys:
                if key in features_data:
                    features = features_data[key]
                    break

            if features is None:
                tensor_values = [
                    v for v in features_data.values() if isinstance(v, torch.Tensor)
                ]
                if tensor_values:
                    features = tensor_values[0]
                else:
                    raise ValueError(f"No tensor found in feature file {feature_path}")
        else:
            features = features_data

        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        elif not isinstance(features, torch.Tensor):
            features = torch.tensor(features)

        vad_raw = np.array([row["EmoVal"], row["EmoAct"], row["EmoDom"]], dtype=np.float32)
        vad = (vad_raw - 4.0) / 3.0

        use_for_classification = row["use_for_classification"]
        if use_for_classification and row["mapped_emotion"] is not None:
            emotion_idx = self.emotion_to_idx[row["mapped_emotion"]]
        else:
            emotion_idx = -1

        return {
            "features": features.float(),
            "vad": torch.FloatTensor(vad),
            "emotion_class": torch.LongTensor([emotion_idx]).squeeze(),
            "use_for_classification": use_for_classification,
            "filename": filename,
        }


def collate_fn(batch: list) -> dict:
    """Handle variable length sequences with custom collation.

    Args:
        batch: List of samples from the dataset

    Returns:
        Collated batch with padded sequences

    Raises:
        None

    Examples:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
        >>> batch = next(iter(loader))
        >>> print(batch['features'].shape)
    """
    features = [item["features"] for item in batch]

    vad = torch.stack([item["vad"] for item in batch])
    emotion_class = torch.stack([item["emotion_class"] for item in batch])
    use_for_classification = torch.tensor([item["use_for_classification"] for item in batch])
    filenames = [item["filename"] for item in batch]

    features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)

    return {
        "features": features,
        "vad": vad,
        "emotion_class": emotion_class,
        "use_for_classification": use_for_classification,
        "filenames": filenames,
    }