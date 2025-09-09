#!/usr/bin/env python3
# filepath: /mnt/shareEEx/liuyang/code/SE_IMR_MTL_KD/recover_missing_files.py

import os
import shutil
from pathlib import Path
import argparse

def read_missing_files(missing_file_path):
    """
    读取缺失文件列表
    """
    missing_files = []
    if Path(missing_file_path).exists():
        with open(missing_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                filename = line.strip()
                if filename:
                    missing_files.append(filename)
    return missing_files

def search_and_copy_files(missing_files, removed_files_dir, target_dir, partition_name):
    """
    在removed_files目录中搜索缺失文件并复制到目标目录
    """
    found_count = 0
    not_found = []
    
    print(f"\n正在处理 {partition_name} 集合的缺失文件...")
    print(f"需要查找的文件数量: {len(missing_files)}")
    
    target_partition_dir = target_dir / partition_name.lower()
    
    for filename in missing_files:
        found = False
        
        # 在所有removed版本文件夹中搜索
        for version_dir in removed_files_dir.iterdir():
            if version_dir.is_dir():
                source_file = version_dir / filename
                if source_file.exists():
                    # 找到文件，复制到目标目录
                    target_file = target_partition_dir / filename
                    
                    # 确保目标目录存在
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # 复制文件
                    shutil.copy2(source_file, target_file)
                    print(f"✓ 恢复文件: {filename} (从 {version_dir.name})")
                    found_count += 1
                    found = True
                    break
        
        if not found:
            not_found.append(filename)
    
    print(f"{partition_name} 集合恢复完成: 找到并恢复了 {found_count}/{len(missing_files)} 个文件")
    
    if not_found:
        print(f"仍然缺失的文件数量: {len(not_found)}")
        # 保存仍然缺失的文件列表
        still_missing_file = target_dir / f"still_missing_{partition_name.lower()}.txt"
        with open(still_missing_file, 'w') as f:
            for missing_file in not_found:
                f.write(f"{missing_file}\n")
        print(f"仍然缺失的文件列表已保存到: {still_missing_file}")
    
    return found_count, len(not_found)

def recover_missing_files(version, base_dir):
    """
    恢复指定版本数据集的缺失文件
    """
    print(f"\n开始恢复 MSP-Podcast {version} 数据集的缺失文件...")
    
    # 定义路径
    base_path = Path(base_dir)
    removed_files_dir = base_path / "data" / "MSP-PODCAST-Publish-1.12" / "Previous_releases_information" / "files_removed_from_prev_versions"
    organized_dir = base_path / "data" / "organized_datasets" / version
    
    # 检查目录是否存在
    if not removed_files_dir.exists():
        print(f"错误: removed_files目录不存在: {removed_files_dir}")
        return False
    
    if not organized_dir.exists():
        print(f"错误: 组织的数据集目录不存在: {organized_dir}")
        return False
    
    print(f"搜索目录: {removed_files_dir}")
    print(f"目标目录: {organized_dir}")
    
    # 显示removed_files目录中的版本文件夹
    version_dirs = [d for d in removed_files_dir.iterdir() if d.is_dir()]
    print(f"\n可用的removed版本文件夹: {[d.name for d in version_dirs]}")
    
    total_recovered = 0
    total_still_missing = 0
    
    # 处理每个分区的缺失文件
    partitions = ['train', 'test', 'validation']
    
    for partition in partitions:
        missing_file_path = organized_dir / f"missing_files_{partition}.txt"
        
        if missing_file_path.exists():
            missing_files = read_missing_files(missing_file_path)
            
            if missing_files:
                recovered, still_missing = search_and_copy_files(
                    missing_files, removed_files_dir, organized_dir, partition
                )
                total_recovered += recovered
                total_still_missing += still_missing
            else:
                print(f"\n{partition} 集合没有缺失文件")
        else:
            print(f"\n{partition} 集合的缺失文件列表不存在: {missing_file_path}")
    
    # 更新摘要文件
    summary_file = organized_dir / "recovery_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"MSP-Podcast {version} 数据集文件恢复摘要\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"搜索目录: {removed_files_dir}\n")
        f.write(f"目标目录: {organized_dir}\n\n")
        f.write(f"恢复统计:\n")
        f.write(f"成功恢复: {total_recovered} 个文件\n")
        f.write(f"仍然缺失: {total_still_missing} 个文件\n")
        f.write(f"\n可用的removed版本: {[d.name for d in version_dirs]}\n")
    
    print(f"\n{version} 数据集文件恢复完成!")
    print(f"总共恢复了 {total_recovered} 个文件")
    if total_still_missing > 0:
        print(f"仍有 {total_still_missing} 个文件无法找到")
    print(f"恢复摘要已保存到: {summary_file}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='恢复MSP-Podcast数据集的缺失文件')
    parser.add_argument('--version', choices=['v1.3', 'v1.6', 'both'], default='v1.3',
                        help='要恢复的数据集版本 (默认: v1.3)')
    parser.add_argument('--base-dir', default='/mnt/shareEEx/liuyang/code/SE_IMR_MTL_KD',
                        help='项目基础目录 (默认: /mnt/shareEEx/liuyang/code/SE_IMR_MTL_KD)')
    parser.add_argument('--dry-run', action='store_true',
                        help='只显示会恢复的文件，不实际复制')
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("注意: 这是一次试运行，不会实际复制文件")
    
    base_dir = args.base_dir
    
    if args.version == 'both':
        versions = ['v1.3', 'v1.6']
    else:
        versions = [args.version]
    
    success_count = 0
    for version in versions:
        if args.dry_run:
            # 试运行模式：只显示信息
            base_path = Path(base_dir)
            organized_dir = base_path / "data" / "organized_datasets" / version
            
            print(f"\n分析 {version} 版本的缺失文件:")
            partitions = ['train', 'test', 'validation']
            total_missing = 0
            
            for partition in partitions:
                missing_file_path = organized_dir / f"missing_files_{partition}.txt"
                if missing_file_path.exists():
                    missing_files = read_missing_files(missing_file_path)
                    print(f"{partition}: {len(missing_files)} 个缺失文件")
                    total_missing += len(missing_files)
                else:
                    print(f"{partition}: 无缺失文件列表")
            
            print(f"总计缺失: {total_missing} 个文件")
        else:
            if recover_missing_files(version, base_dir):
                success_count += 1
    
    if not args.dry_run:
        print(f"\n完成! 成功处理了 {success_count}/{len(versions)} 个数据集版本")

if __name__ == "__main__":
    main()