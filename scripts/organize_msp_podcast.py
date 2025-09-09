#!/usr/bin/env python3
# filepath: /mnt/shareEEx/liuyang/code/SE_IMR_MTL_KD/utils/organize_msp_podcast.py

import os
import shutil
from pathlib import Path
import argparse

def parse_partitions_file(partitions_path):
    """
    解析Partitions.txt文件，返回按集合分类的文件列表
    """
    partitions = {'Train': [], 'Test': [], 'Validation': []}
    
    with open(partitions_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # 分割行内容 "集合; 文件名"
            if ';' in line:
                partition, filename = line.split(';', 1)
                partition = partition.strip()
                filename = filename.strip()
                
                # 统一分区名称
                if partition in ['Train', 'Training']:
                    partitions['Train'].append(filename)
                elif partition in ['Test', 'Testing']:
                    partitions['Test'].append(filename)
                elif partition in ['Validation', 'Valid', 'Val']:
                    partitions['Validation'].append(filename)
    
    return partitions

def copy_audio_files(source_dir, target_dir, file_list, partition_name):
    """
    复制音频文件到目标目录
    """
    partition_dir = target_dir / partition_name.lower()
    partition_dir.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    skipped_count = 0
    missing_files = []
    
    print(f"\n正在处理 {partition_name} 集合...")
    print(f"文件数量: {len(file_list)}")
    
    for filename in file_list:
        source_file = source_dir / filename
        target_file = partition_dir / filename
        
        # 检查目标文件是否已存在
        if target_file.exists():
            skipped_count += 1
            if skipped_count % 100 == 0:
                print(f"已跳过 {skipped_count} 个现有文件...")
            continue
        
        if source_file.exists():
            # 创建目标文件的父目录（如果需要）
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 复制文件
            shutil.copy2(source_file, target_file)
            copied_count += 1
            
            if copied_count % 100 == 0:
                print(f"已复制 {copied_count} 个文件...")
        else:
            missing_files.append(filename)
    
    print(f"{partition_name} 集合完成:")
    print(f"  - 复制了 {copied_count} 个新文件")
    print(f"  - 跳过了 {skipped_count} 个已存在文件")
    print(f"  - 缺失 {len(missing_files)} 个文件")
    print(f"  - 总计处理: {copied_count + skipped_count}/{len(file_list)} 个文件")
    
    if missing_files:
        # 保存缺失文件列表
        missing_file_path = target_dir / f"missing_files_{partition_name.lower()}.txt"
        with open(missing_file_path, 'w') as f:
            for missing_file in missing_files:
                f.write(f"{missing_file}\n")
        print(f"  - 缺失文件列表已保存到: {missing_file_path}")
    
    return copied_count, skipped_count, len(missing_files)

def organize_dataset(version, base_dir):
    """
    组织指定版本的数据集
    """
    print(f"\n开始组织 MSP-Podcast {version} 数据集...")
    
    # 定义路径
    base_path = Path(base_dir)
    audios_dir = base_path / "data" / "MSP-PODCAST-Publish-1.12" / "Audios"
    labels_dir = base_path / "data" / "labels" / version
    partitions_file = labels_dir / "Partitions.txt"
    output_dir = base_path / "data" / "organized_datasets" / version
    
    # 检查输入路径
    if not audios_dir.exists():
        print(f"错误: 音频目录不存在: {audios_dir}")
        return False
    
    if not partitions_file.exists():
        print(f"错误: Partitions文件不存在: {partitions_file}")
        return False
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"音频源目录: {audios_dir}")
    print(f"分区文件: {partitions_file}")
    print(f"输出目录: {output_dir}")
    
    # 解析分区文件
    partitions = parse_partitions_file(partitions_file)
    
    # 显示统计信息
    total_files = sum(len(files) for files in partitions.values())
    print(f"\n数据集统计:")
    print(f"训练集: {len(partitions['Train'])} 个文件")
    print(f"测试集: {len(partitions['Test'])} 个文件")
    print(f"验证集: {len(partitions['Validation'])} 个文件")
    print(f"总计: {total_files} 个文件")
    
    # 复制文件
    total_copied = 0
    total_skipped = 0
    total_missing = 0
    
    for partition_name, file_list in partitions.items():
        if file_list:  # 只处理非空的分区
            copied, skipped, missing = copy_audio_files(audios_dir, output_dir, file_list, partition_name)
            total_copied += copied
            total_skipped += skipped
            total_missing += missing
    
    # 保存分区信息
    summary_file = output_dir / "dataset_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"MSP-Podcast {version} 数据集组织摘要\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"源音频目录: {audios_dir}\n")
        f.write(f"分区文件: {partitions_file}\n")
        f.write(f"输出目录: {output_dir}\n\n")
        f.write(f"数据集统计:\n")
        f.write(f"训练集: {len(partitions['Train'])} 个文件\n")
        f.write(f"测试集: {len(partitions['Test'])} 个文件\n")
        f.write(f"验证集: {len(partitions['Validation'])} 个文件\n")
        f.write(f"总计文件: {total_files}\n")
        f.write(f"新复制文件: {total_copied}\n")
        f.write(f"跳过已存在文件: {total_skipped}\n")
        f.write(f"缺失文件: {total_missing}\n")
        f.write(f"成功处理文件: {total_copied + total_skipped}\n")
    
    print(f"\n{version} 数据集组织完成!")
    print(f"新复制了 {total_copied} 个文件")
    print(f"跳过了 {total_skipped} 个已存在的文件")
    if total_missing > 0:
        print(f"有 {total_missing} 个文件缺失")
    print(f"总共处理了 {total_copied + total_skipped}/{total_files} 个文件")
    print(f"数据集摘要已保存到: {summary_file}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='组织MSP-Podcast数据集')
    parser.add_argument('--version', choices=['v1.3', 'v1.6', 'both'], default='both',
                        help='要组织的数据集版本 (默认: both)')
    parser.add_argument('--base-dir', default='/mnt/shareEEx/liuyang/code/SE_IMR_MTL_KD',
                        help='项目基础目录 (默认: /mnt/shareEEx/liuyang/code/SE_IMR_MTL_KD)')
    parser.add_argument('--dry-run', action='store_true',
                        help='只显示统计信息，不实际复制文件')
    parser.add_argument('--force', action='store_true',
                        help='强制覆盖已存在的文件')
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("注意: 这是一次试运行，不会实际复制文件")
    
    if args.force:
        print("注意: 启用强制模式，将覆盖已存在的文件")
    
    base_dir = args.base_dir
    
    if args.version == 'both':
        versions = ['v1.3', 'v1.6']
    else:
        versions = [args.version]
    
    success_count = 0
    for version in versions:
        if args.dry_run:
            # 试运行模式：只解析和显示信息
            labels_dir = Path(base_dir) / "data" / "labels" / version
            partitions_file = labels_dir / "Partitions.txt"
            output_dir = Path(base_dir) / "data" / "organized_datasets" / version
            
            if partitions_file.exists():
                print(f"\n分析 {version} 版本:")
                partitions = parse_partitions_file(partitions_file)
                total_files = sum(len(files) for files in partitions.values())
                
                # 统计已存在的文件
                total_existing = 0
                for partition_name, file_list in partitions.items():
                    partition_dir = output_dir / partition_name.lower()
                    existing_count = 0
                    if partition_dir.exists():
                        for filename in file_list:
                            if (partition_dir / filename).exists():
                                existing_count += 1
                    total_existing += existing_count
                    print(f"{partition_name}: {len(file_list)} 个文件 ({existing_count} 个已存在)")
                
                print(f"总计: {total_files} 个文件 ({total_existing} 个已存在)")
                print(f"需要复制: {total_files - total_existing} 个文件")
            else:
                print(f"错误: {version} 的Partitions文件不存在")
        else:
            # 如果启用强制模式，修改copy_audio_files函数的行为
            if args.force:
                # 临时修改函数来支持强制覆盖
                original_copy_audio_files = copy_audio_files
                
                def force_copy_audio_files(source_dir, target_dir, file_list, partition_name):
                    partition_dir = target_dir / partition_name.lower()
                    partition_dir.mkdir(parents=True, exist_ok=True)
                    
                    copied_count = 0
                    missing_files = []
                    
                    print(f"\n正在处理 {partition_name} 集合 (强制模式)...")
                    print(f"文件数量: {len(file_list)}")
                    
                    for filename in file_list:
                        source_file = source_dir / filename
                        target_file = partition_dir / filename
                        
                        if source_file.exists():
                            # 创建目标文件的父目录（如果需要）
                            target_file.parent.mkdir(parents=True, exist_ok=True)
                            
                            # 复制文件（覆盖已存在的）
                            shutil.copy2(source_file, target_file)
                            copied_count += 1
                            
                            if copied_count % 100 == 0:
                                print(f"已复制 {copied_count} 个文件...")
                        else:
                            missing_files.append(filename)
                    
                    print(f"{partition_name} 集合完成: 复制了 {copied_count}/{len(file_list)} 个文件")
                    
                    if missing_files:
                        missing_file_path = target_dir / f"missing_files_{partition_name.lower()}.txt"
                        with open(missing_file_path, 'w') as f:
                            for missing_file in missing_files:
                                f.write(f"{missing_file}\n")
                        print(f"缺失文件列表已保存到: {missing_file_path}")
                    
                    return copied_count, 0, len(missing_files)  # 强制模式下跳过数为0
                
                # 临时替换函数
                globals()['copy_audio_files'] = force_copy_audio_files
            
            if organize_dataset(version, base_dir):
                success_count += 1
    
    if not args.dry_run:
        print(f"\n完成! 成功组织了 {success_count}/{len(versions)} 个数据集版本")

if __name__ == "__main__":
    main()