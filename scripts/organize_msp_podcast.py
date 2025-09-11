#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Organize MSP-Podcast dataset into train, test, and validation splits.

This script parses the Partitions.txt file and copies audio files into
organized folders for each split. Supports dry-run and force-overwrite modes.

Example :
    >>> python scripts/organize_msp_podcast.py --version v1.6 --base-dir /path/to/project
"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2025-11-15"

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def parse_partitions_file(partitions_path: Path) -> Dict[str, List[str]]:
    """
    Parse Partitions.txt and return file lists for each split.

    Args :
        partitions_path (Path): Path to Partitions.txt.

    Returns :
        Dict[str, List[str]]: Dict with keys 'Train', 'Test', 'Validation'.

    Raises :
        FileNotFoundError: If partitions_path does not exist.
    """
    partitions = {"Train": [], "Test": [], "Validation": []}
    with partitions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ";" in line:
                partition, filename = line.split(";", 1)
                partition = partition.strip()
                filename = filename.strip()
                if partition in ["Train", "Training"]:
                    partitions["Train"].append(filename)
                elif partition in ["Test", "Testing"]:
                    partitions["Test"].append(filename)
                elif partition in ["Validation", "Valid", "Val"]:
                    partitions["Validation"].append(filename)
    return partitions


def copy_audio_files(
    source_dir: Path,
    target_dir: Path,
    file_list: List[str],
    partition_name: str,
    force: bool = False,
) -> Tuple[int, int, int]:
    """
    Copy audio files to target directory for a given partition.

    Args :
        source_dir (Path): Directory with source audio files.
        target_dir (Path): Output directory.
        file_list (List[str]): List of filenames to copy.
        partition_name (str): Partition name.
        force (bool): Overwrite existing files if True.

    Returns :
        Tuple[int, int, int]: (copied_count, skipped_count, missing_count).

    Raises :
        None
    """
    partition_dir = target_dir / partition_name.lower()
    partition_dir.mkdir(parents=True, exist_ok=True)
    copied_count = 0
    skipped_count = 0
    missing_files = []

    print(f"\nProcessing {partition_name} split...")
    print(f"Total files: {len(file_list)}")

    for filename in file_list:
        source_file = source_dir / filename
        target_file = partition_dir / filename

        if not source_file.exists():
            missing_files.append(filename)
            continue

        if target_file.exists() and not force:
            skipped_count += 1
            continue

        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target_file)
        copied_count += 1
        if copied_count % 100 == 0:
            print(f"Copied {copied_count} files...")

    print(f"{partition_name} split done:")
    print(f"  - Copied: {copied_count}")
    print(f"  - Skipped: {skipped_count}")
    print(f"  - Missing: {len(missing_files)}")
    print(f"  - Total processed: {copied_count + skipped_count}/{len(file_list)}")

    if missing_files:
        missing_file_path = target_dir / f"missing_files_{partition_name.lower()}.txt"
        with missing_file_path.open("w") as f:
            for missing_file in missing_files:
                f.write(f"{missing_file}\n")
        print(f"  - Missing file list saved to: {missing_file_path}")

    return copied_count, skipped_count, len(missing_files)


def organize_dataset(version: str, base_dir: str, force: bool = False) -> bool:
    """
    Organize MSP-Podcast dataset for a specific version.

    Args :
        version (str): Dataset version.
        base_dir (str): Project base directory.
        force (bool): Overwrite existing files if True.

    Returns :
        bool: True if organization succeeded.

    Raises :
        None
    """
    print(f"\nOrganizing MSP-Podcast {version} dataset...")

    base_path = Path(base_dir)
    audios_dir = base_path / "data" / "MSP-PODCAST-Publish-1.12" / "Audios"
    labels_dir = base_path / "data" / "labels" / version
    partitions_file = labels_dir / "Partitions.txt"
    output_dir = base_path / "data" / "organized_datasets" / version

    if not audios_dir.exists():
        print(f"Error: Audio directory not found: {audios_dir}")
        return False
    if not partitions_file.exists():
        print(f"Error: Partitions file not found: {partitions_file}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Audio source: {audios_dir}")
    print(f"Partitions file: {partitions_file}")
    print(f"Output dir: {output_dir}")

    partitions = parse_partitions_file(partitions_file)
    total_files = sum(len(files) for files in partitions.values())
    print("\nDataset stats:")
    print(f"Train: {len(partitions['Train'])}")
    print(f"Test: {len(partitions['Test'])}")
    print(f"Validation: {len(partitions['Validation'])}")
    print(f"Total: {total_files}")

    total_copied = 0
    total_skipped = 0
    total_missing = 0

    for partition_name, file_list in partitions.items():
        if file_list:
            copied, skipped, missing = copy_audio_files(
                audios_dir, output_dir, file_list, partition_name, force=force
            )
            total_copied += copied
            total_skipped += skipped
            total_missing += missing

    summary_file = output_dir / "dataset_summary.txt"
    with summary_file.open("w") as f:
        f.write(f"MSP-Podcast {version} dataset summary\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"Source audio: {audios_dir}\n")
        f.write(f"Partitions file: {partitions_file}\n")
        f.write(f"Output dir: {output_dir}\n\n")
        f.write("Stats:\n")
        f.write(f"Train: {len(partitions['Train'])}\n")
        f.write(f"Test: {len(partitions['Test'])}\n")
        f.write(f"Validation: {len(partitions['Validation'])}\n")
        f.write(f"Total files: {total_files}\n")
        f.write(f"Copied: {total_copied}\n")
        f.write(f"Skipped: {total_skipped}\n")
        f.write(f"Missing: {total_missing}\n")
        f.write(f"Processed: {total_copied + total_skipped}\n")

    print(f"\n{version} dataset organization complete!")
    print(f"Copied: {total_copied}")
    print(f"Skipped: {total_skipped}")
    if total_missing > 0:
        print(f"Missing: {total_missing}")
    print(f"Processed: {total_copied + total_skipped}/{total_files}")
    print(f"Summary saved to: {summary_file}")

    return True


def main() -> None:
    """
    Entry point for organizing MSP-Podcast dataset.

    Args :
        None

    Returns :
        None

    Raises :
        None
    """
    parser = argparse.ArgumentParser(
        description="Organize MSP-Podcast dataset into splits"
    )
    parser.add_argument(
        "--version",
        choices=["v1.3", "v1.6", "both"],
        default="both",
        help="Dataset version to organize (default: both)",
    )
    parser.add_argument(
        "--base-dir",
        default="/mnt/shareEEx/liuyang/code/SE_IMR_MTL_KD",
        help="Project base directory (default: /mnt/shareEEx/liuyang/code/SE_IMR_MTL_KD)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show stats only, do not copy files"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    base_dir = args.base_dir
    versions = ["v1.3", "v1.6"] if args.version == "both" else [args.version]
    success_count = 0

    for version in versions:
        labels_dir = Path(base_dir) / "data" / "labels" / version
        partitions_file = labels_dir / "Partitions.txt"
        output_dir = Path(base_dir) / "data" / "organized_datasets" / version

        if args.dry_run:
            if partitions_file.exists():
                print(f"\nAnalyzing {version}:")
                partitions = parse_partitions_file(partitions_file)
                total_files = sum(len(files) for files in partitions.values())
                total_existing = 0
                for partition_name, file_list in partitions.items():
                    partition_dir = output_dir / partition_name.lower()
                    existing_count = sum(
                        1
                        for filename in file_list
                        if (partition_dir / filename).exists()
                    )
                    total_existing += existing_count
                    print(
                        f"{partition_name}: {len(file_list)} files "
                        f"({existing_count} existing)"
                    )
                print(f"Total: {total_files} files ({total_existing} existing)")
                print(f"To copy: {total_files - total_existing} files")
            else:
                print(f"Error: {version} Partitions.txt not found")
        else:
            if organize_dataset(version, base_dir, force=args.force):
                success_count += 1

    if not args.dry_run:
        print(f"\nDone! Organized {success_count}/{len(versions)} dataset versions")


if __name__ == "__main__":
    main()
