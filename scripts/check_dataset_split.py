import pandas as pd
import os
import glob

def check_dataset_split(csv_file_path, v16_base_path):
    """
    检查v1.6数据集的划分是否符合labels_consensus.csv的要求
    
    Args:
        csv_file_path: labels_consensus.csv文件路径
        v16_base_path: v1.6数据集的基础路径，包含test、train、validation文件夹
    """
    
    # 读取标签文件
    print("读取labels_consensus.csv文件...")
    try:
        labels_df = pd.read_csv(csv_file_path)
        print(f"成功读取{len(labels_df)}条记录")
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return
    
    # 检查必要的列是否存在
    required_columns = ['FileName', 'EmoClass', 'EmoAct', 'EmoVal', 'EmoDom', 'SpkrID', 'Gender', 'Split_Set']
    missing_columns = [col for col in required_columns if col not in labels_df.columns]
    if missing_columns:
        print(f"CSV文件缺少必要的列: {missing_columns}")
        return
    
    # 按Split_Set分组
    split_groups = labels_df.groupby('Split_Set')
    print(f"\n标签文件中的数据集划分:")
    for split_name, group in split_groups:
        print(f"  {split_name}: {len(group)}个文件")
    
    # 检查v1.6目录结构
    splits_to_check = ['test', 'train', 'validation']
    results = {}
    
    for split_name in splits_to_check:
        split_path = os.path.join(v16_base_path, split_name)
        
        if not os.path.exists(split_path):
            print(f"\n警告: 目录不存在 - {split_path}")
            continue
            
        print(f"\n检查 {split_name} 数据集...")
        
        # 获取该split在CSV中应该包含的文件
        expected_files = set()
        if split_name in labels_df['Split_Set'].values:
            expected_files = set(labels_df[labels_df['Split_Set'] == split_name]['FileName'].values)
        
        # 获取实际存在的文件（支持常见的音频格式）
        audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.aac']
        actual_files = set()
        
        for ext in audio_extensions:
            files = glob.glob(os.path.join(split_path, ext))
            actual_files.update([os.path.basename(f) for f in files])
        
        # 也检查子目录
        for root, dirs, files in os.walk(split_path):
            for file in files:
                if any(file.lower().endswith(ext[1:]) for ext in audio_extensions):
                    actual_files.add(file)
        
        # 比较结果
        missing_files = expected_files - actual_files
        extra_files = actual_files - expected_files
        correct_files = expected_files & actual_files
        
        results[split_name] = {
            'expected_count': len(expected_files),
            'actual_count': len(actual_files),
            'correct_count': len(correct_files),
            'missing_count': len(missing_files),
            'extra_count': len(extra_files),
            'missing_files': missing_files,
            'extra_files': extra_files
        }
        
        print(f"  预期文件数: {len(expected_files)}")
        print(f"  实际文件数: {len(actual_files)}")
        print(f"  正确匹配数: {len(correct_files)}")
        print(f"  缺失文件数: {len(missing_files)}")
        print(f"  多余文件数: {len(extra_files)}")
        
        if missing_files:
            print(f"  缺失的文件 (前10个): {list(missing_files)[:10]}")
            if len(missing_files) > 10:
                print(f"    ... 还有 {len(missing_files) - 10} 个文件")
        
        if extra_files:
            print(f"  多余的文件 (前10个): {list(extra_files)[:10]}")
            if len(extra_files) > 10:
                print(f"    ... 还有 {len(extra_files) - 10} 个文件")
    
    # 生成总结报告
    print(f"\n{'='*50}")
    print("总结报告:")
    print(f"{'='*50}")
    
    total_errors = 0
    for split_name, result in results.items():
        accuracy = (result['correct_count'] / result['expected_count'] * 100) if result['expected_count'] > 0 else 0
        print(f"\n{split_name.upper()} 数据集:")
        print(f"  准确率: {accuracy:.2f}%")
        print(f"  匹配度: {result['correct_count']}/{result['expected_count']}")
        
        if result['missing_count'] > 0 or result['extra_count'] > 0:
            print(f"  ❌ 发现问题: 缺失{result['missing_count']}个文件，多余{result['extra_count']}个文件")
            total_errors += result['missing_count'] + result['extra_count']
        else:
            print(f"  ✅ 完全匹配")
    
    if total_errors == 0:
        print(f"\n🎉 所有数据集划分都符合labels_consensus.csv的要求！")
    else:
        print(f"\n⚠️  发现 {total_errors} 个不匹配的问题需要处理")
    
    # 保存详细报告到文件
    report_path = "dataset_split_check_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("数据集划分检查报告\n")
        f.write("="*50 + "\n\n")
        
        for split_name, result in results.items():
            f.write(f"{split_name.upper()} 数据集:\n")
            f.write(f"预期文件数: {result['expected_count']}\n")
            f.write(f"实际文件数: {result['actual_count']}\n")
            f.write(f"正确匹配数: {result['correct_count']}\n")
            f.write(f"缺失文件数: {result['missing_count']}\n")
            f.write(f"多余文件数: {result['extra_count']}\n\n")
            
            if result['missing_files']:
                f.write("缺失的文件:\n")
                for file in sorted(result['missing_files']):
                    f.write(f"  - {file}\n")
                f.write("\n")
            
            if result['extra_files']:
                f.write("多余的文件:\n")
                for file in sorted(result['extra_files']):
                    f.write(f"  - {file}\n")
                f.write("\n")
            
            f.write("-" * 30 + "\n\n")
    
    print(f"\n详细报告已保存到: {report_path}")

def main():
    # 配置文件路径
    csv_file_path = "/mnt/shareEEx/liuyang/code/SE_IMR_MTL_KD/data/labels/v1.6/label/labels_concensus.csv"  # 修改为你的CSV文件路径
    v16_base_path = "/mnt/shareEEx/liuyang/code/SE_IMR_MTL_KD/data/organized_datasets/v1.6"  # 修改为你的v1.6数据集路径
    
    print("数据集划分验证脚本")
    print("="*50)
    
    # 检查文件是否存在
    if not os.path.exists(csv_file_path):
        print(f"错误: 找不到文件 {csv_file_path}")
        return
    
    if not os.path.exists(v16_base_path):
        print(f"错误: 找不到目录 {v16_base_path}")
        return
    
    # 执行检查
    check_dataset_split(csv_file_path, v16_base_path)

if __name__ == "__main__":
    main()