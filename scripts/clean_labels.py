#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove label entries without corresponding audio files.

Scan the label CSV and remove rows where the audio file does not exist
in the specified audio directory. Save the cleaned CSV to a new file.

Example:
    >>> python clean_labels.py --label_file labels_concensus.csv --audio_dir organized_datasets/v1.6 --output_file labels_cleaned.csv

"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2025-11-15"

import argparse
import os

import pandas as pd


def filter_labels_by_audio(label_file: str, audio_dir: str, output_file: str) -> None:
    """
    Remove label entries without corresponding audio files.

    Args:
        label_file (str): Path to the label CSV file.
        audio_dir (str): Directory containing audio files (with subfolders).
        output_file (str): Path to save the cleaned label CSV.

    Returns:
        None

    Raises:
        None

    """
    df = pd.read_csv(label_file)
    # 构建所有音频文件的绝对路径集合
    audio_files = set()
    for root, _, files in os.walk(audio_dir):
        for f in files:
            audio_files.add(f)
    # 只保留标签中实际存在的音频文件
    exists_mask = df["FileName"].apply(lambda x: x in audio_files)
    df_clean = df[exists_mask]
    df_clean.to_csv(output_file, index=False)
    print(f"Cleaned label file saved to: {output_file}")


def main() -> None:
    """
    Parse arguments and run label cleaning.

    Args:
        None

    Returns:
        None

    Raises:
        None

    """
    parser = argparse.ArgumentParser(
        description="Remove label entries without corresponding audio files."
    )
    parser.add_argument(
        "--label_file", type=str, required=True, help="Path to label CSV file"
    )
    parser.add_argument(
        "--audio_dir", type=str, required=True, help="Directory containing audio files"
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to save cleaned CSV"
    )
    args = parser.parse_args()
    filter_labels_by_audio(args.label_file, args.audio_dir, args.output_file)


if __name__ == "__main__":
    main()
