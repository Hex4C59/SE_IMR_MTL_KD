"""
One-line summary of the module.

Detailed description of the module.

Example:
    >>> example

"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2025-09-09"

import os

import pandas as pd
import torch
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    """
    Dataset class for loading wav2vec2 features and emotion labels.

    Args:
        features_dir: Directory containing wav2vec2 features
        labels_file: CSV file with emotion labels
        split: 'train', 'validation', or 'test'

    """

    def __init__(self, features_dir, labels_file, split="train"):
        self.features_dir = features_dir
        self.labels_df = pd.read_csv(labels_file)

        # Filter by split
        if split == "train":
            self.labels_df = self.labels_df[self.labels_df["Split_Set"] == "Train"]
        elif split == "validation":
            self.labels_df = self.labels_df[
                self.labels_df["Split_Set"] == "Development"
            ]
        elif split == "test":
            self.labels_df = self.labels_df[self.labels_df["Split_Set"] == "Test"]

        # List of valid samples (files that exist)
        self.valid_samples = []
        for idx, row in self.labels_df.iterrows():
            feature_path = os.path.join(
                self.features_dir, split, row["FileName"].replace(".wav", ".pt")
            )
            if os.path.exists(feature_path):
                self.valid_samples.append(idx)

        print(f"Loaded {len(self.valid_samples)} valid {split} samples")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sample_idx = self.valid_samples[idx]
        row = self.labels_df.iloc[sample_idx]

        # Load wav2vec2 features
        feature_path = os.path.join(
            self.features_dir,
            "train"
            if row["Split_Set"] == "Train"
            else "validation"
            if row["Split_Set"] == "Development"
            else "test",
            row["FileName"].replace(".wav", ".pt"),
        )
        features = torch.load(feature_path)

        # Get VAD labels
        valence = float(row["EmoVal"])
        arousal = float(row["EmoAct"])  # Activation is Arousal
        dominance = float(row["EmoDom"])

        # Combine VAD values into a single tensor
        vad = torch.tensor([valence, arousal, dominance], dtype=torch.float)

        return {"features": features, "vad": vad, "filename": row["FileName"]}


def collate_fn(batch):
    """Handle variable length sequences with custom collation."""
    # Sort batch by sequence length (descending)
    batch = sorted(batch, key=lambda x: x["features"].shape[0], reverse=True)

    # Get sequence lengths
    lengths = [x["features"].shape[0] for x in batch]
    max_length = max(lengths)

    # Pad sequences
    features = []
    for sample in batch:
        padded = torch.zeros(max_length, sample["features"].shape[1])
        padded[: sample["features"].shape[0], :] = sample["features"]
        features.append(padded)

    # Stack tensors
    features = torch.stack(features)
    vad = torch.stack([x["vad"] for x in batch])

    return {
        "features": features,
        "lengths": lengths,
        "vad": vad,
        "filenames": [x["filename"] for x in batch],
    }
