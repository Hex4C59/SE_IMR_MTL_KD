#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
t-SNE 可视化 wav2vec2-base-100h 第12层特征.

加载特征和标签，使用 t-SNE 降维并可视化，标签用于着色。

Example:
    >>> python visualize_tsne.py \
        --feature_dir /path/to/features \
        --label_file /path/to/labels.csv \
        --label_column EmoAct

"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2025-11-15"

import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE


def load_features(feature_dir: str, file_list: List[str]) -> np.ndarray:
    """
    Load and pool features from .pt files.

    Load features and apply mean pooling if needed.

    Args:
        feature_dir (str): Directory containing .pt feature files.
        file_list (List[str]): List of file base names.

    Returns:
        np.ndarray: Feature matrix (num_samples, feature_dim).

    Raises:
        None

    """
    features = []
    for base_name in file_list:
        pt_path = os.path.join(feature_dir, base_name + ".pt")
        feat = torch.load(pt_path, weights_only=False)
        tensor = feat["feature"]
        if hasattr(tensor, "ndim") and tensor.ndim == 2:
            tensor = tensor.mean(axis=0)
        features.append(np.asarray(tensor).astype(np.float32))
    return np.stack(features)


def main() -> None:
    """
    Parse arguments and visualize features with t-SNE.

    Load features and labels, apply t-SNE, and plot colored by label.

    Args:
        None

    Returns:
        None

    Raises:
        None

    """
    parser = argparse.ArgumentParser(
        description="t-SNE visualization for wav2vec2 features"
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        required=True,
        help="Directory containing .pt feature files",
    )
    parser.add_argument(
        "--label_file", type=str, required=True, help="CSV file containing labels"
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="EmoAct",
        help="Column name for label (default: EmoAct)",
    )
    parser.add_argument(
        "--split_set", type=str, default="Test", help="Split set name (default: Test)"
    )
    args = parser.parse_args()

    label_df = pd.read_csv(args.label_file, sep=",")
    label_df = label_df[label_df["Split_Set"] == args.split_set]
    label_df["base_name"] = label_df["FileName"].str.replace(".wav", "", regex=False)
    file_list = label_df["base_name"].tolist()
    labels = label_df[args.label_column].values

    features = load_features(args.feature_dir, file_list)
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    # 生成保存路径
    feature_type = os.path.basename(os.path.normpath(args.feature_dir))
    # 提取模型名称和数据集类型
    feature_dir_parts = os.path.normpath(args.feature_dir).split(os.sep)
    # 假设结构为 .../features_msp_1.6/<模型名>/<数据集类型>
    if len(feature_dir_parts) >= 2:
        model_name = feature_dir_parts[-2]
        dataset_type = feature_dir_parts[-1]
    else:
        model_name = "unknown_model"
        dataset_type = feature_dir_parts[-1] if feature_dir_parts else "unknown"

    label_map = {"EmoAct": "Act", "EmoVal": "Val", "EmoDom": "Dom"}
    label_short = label_map.get(args.label_column, args.label_column)
    save_dir = os.path.join("runs", "tsne", str(label_short))
    os.makedirs(save_dir, exist_ok=True)
    save_name = f"tsne_{model_name}_{dataset_type}_{label_short}.png"
    save_path = os.path.join(save_dir, save_name)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        features_2d[:, 0], features_2d[:, 1], c=labels, cmap="viridis", alpha=0.7
    )
    plt.colorbar(scatter, label=args.label_column)
    plt.title(f"t-SNE of {feature_type} Features ({label_short})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"t-SNE image saved to: {save_path}")


if __name__ == "__main__":
    main()
