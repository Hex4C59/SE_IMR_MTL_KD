#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_emotion_distribution.py.

Date: 2025-09-10
Author: Liu Yang
License: MIT
Project: https://github.com/Hex4C59/SE_IMR_MTL_KD

Script to analyze emotion class distribution in MSP-Podcast dataset.

This script examines the actual emotion classes present in the dataset
and their distribution across train/validation/test splits.

Example :
    >>> python scripts/check_emotion_distribution.py
"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2025-09-10"

import argparse
import os
import sys

import pandas as pd


def analyze_emotion_distribution(df: pd.DataFrame) -> None:
    """
    Analyze and print emotion class distribution.

    Args :
        df (pd.DataFrame): DataFrame containing emotion labels.

    Returns :
        None

    Raises :
        None

    Examples :
        >>> df = pd.read_csv("labels.csv")
        >>> analyze_emotion_distribution(df)
    """
    if 'EmoClass' not in df.columns:
        print("Warning: No 'EmoClass' column found")
        return

    emotion_counts = df['EmoClass'].value_counts()
    print("\nEmotion Class Distribution:")
    print("-" * 40)
    for emotion, count in emotion_counts.items():
        percentage = count / len(df) * 100
        print(f"  {emotion:8s}: {count:5d} ({percentage:5.2f}%)")


def analyze_split_distribution(df: pd.DataFrame) -> None:
    """
    Analyze emotion distribution by dataset splits.

    Args :
        df (pd.DataFrame): DataFrame containing emotion labels and splits.

    Returns :
        None

    Raises :
        None

    Examples :
        >>> df = pd.read_csv("labels.csv")
        >>> analyze_split_distribution(df)
    """
    if 'Split_Set' not in df.columns:
        print("Warning: No 'Split_Set' column found")
        return

    print("\nDistribution by Dataset Splits:")
    print("=" * 50)

    for split in ['Train', 'Validation', 'Test']:
        split_df = df[df['Split_Set'] == split]
        if len(split_df) == 0:
            continue

        print(f"\n{split} Dataset ({len(split_df)} samples):")
        print("-" * 30)

        if 'EmoClass' in df.columns:
            split_emotions = split_df['EmoClass'].value_counts()
            for emotion, count in split_emotions.items():
                percentage = count / len(split_df) * 100
                print(f"  {emotion:8s}: {count:5d} ({percentage:5.2f}%)")


def analyze_vad_statistics(df: pd.DataFrame) -> None:
    """
    Analyze VAD dimension statistics.

    Args :
        df (pd.DataFrame): DataFrame containing VAD values.

    Returns :
        None

    Raises :
        None

    Examples :
        >>> df = pd.read_csv("labels.csv")
        >>> analyze_vad_statistics(df)
    """
    vad_columns = ['EmoVal', 'EmoAct', 'EmoDom']
    if not all(col in df.columns for col in vad_columns):
        print("Warning: VAD columns not found")
        return

    print("\nVAD Statistics:")
    print("-" * 40)
    print(f"{'Dimension':10s} {'Mean':8s} {'Std':8s} {'Min':8s} {'Max':8s}")
    print("-" * 40)

    for dim in vad_columns:
        mean_val = df[dim].mean()
        std_val = df[dim].std()
        min_val = df[dim].min()
        max_val = df[dim].max()
        dim_name = {'EmoVal': 'Valence', 'EmoAct': 'Arousal', 'EmoDom': 'Dominance'}[dim]
        print(f"{dim_name:10s} {mean_val:8.3f} {std_val:8.3f} {min_val:8.3f} {max_val:8.3f}")


def check_data_quality(df: pd.DataFrame) -> None:
    """
    Check data quality and missing values.

    Args :
        df (pd.DataFrame): DataFrame to analyze.

    Returns :
        None

    Raises :
        None

    Examples :
        >>> df = pd.read_csv("labels.csv")
        >>> check_data_quality(df)
    """
    print("\nData Quality Analysis:")
    print("-" * 40)
    print(f"Dataset shape: {df.shape}")
    print(f"Total samples: {len(df)}")

    key_columns = ['EmoClass', 'EmoVal', 'EmoAct', 'EmoDom', 'Split_Set', 'FileName']
    existing_columns = [col for col in key_columns if col in df.columns]

    if existing_columns:
        print("\nMissing values:")
        for col in existing_columns:
            missing = df[col].isna().sum()
            percentage = missing / len(df) * 100
            print(f"  {col:12s}: {missing:5d} ({percentage:5.2f}%)")


def load_and_validate_data(labels_file: str) -> pd.DataFrame:
    """
    Load and validate the labels file.

    Args :
        labels_file (str): Path to the labels CSV file.

    Returns :
        pd.DataFrame: Loaded and validated dataframe.

    Raises :
        FileNotFoundError: If the labels file doesn't exist.
        pd.errors.EmptyDataError: If the file is empty.

    Examples :
        >>> df = load_and_validate_data("labels.csv")
        >>> print(df.shape)
    """
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Labels file not found: {labels_file}")

    df = pd.read_csv(labels_file)

    if df.empty:
        raise pd.errors.EmptyDataError("Labels file is empty")

    print(f"Successfully loaded {len(df)} samples from {labels_file}")
    print(f"Columns: {df.columns.tolist()}")

    return df


def main():
    """Run emotion distribution analysis."""
    parser = argparse.ArgumentParser(description="Analyze MSP-Podcast emotion distribution")
    parser.add_argument(
        "--labels_file",
        type=str,
        default="/mnt/shareEEx/liuyang/code/SE_IMR_MTL_KD/data/processed/labels/v1.6/label/labels_concensus.csv",
        help="Path to labels CSV file"
    )

    args = parser.parse_args()

    try:
        df = load_and_validate_data(args.labels_file)

        check_data_quality(df)
        analyze_emotion_distribution(df)
        analyze_split_distribution(df)
        analyze_vad_statistics(df)

        print(f"\n{'='*50}")
        print("Analysis completed successfully!")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
