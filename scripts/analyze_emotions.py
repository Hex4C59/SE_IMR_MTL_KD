#!/usr/bin/env python3
import pandas as pd
import numpy as np
from collections import Counter

def analyze_emotion_distribution():
    labels_file = "/mnt/shareEEx/liuyang/code/SE_IMR_MTL_KD/data/processed/labels/v1.6/label/labels_concensus.csv"
    df = pd.read_csv(labels_file)
    
    print("原始情感分布:")
    emotion_counts = df['EmoClass'].value_counts()
    print(emotion_counts)
    
    # 检查你的映射
    emotion_mapping = {
        'H': 'Happy', 'S': 'Sad', 'A': 'Angry', 'U': 'Surprise',
        'F': 'Fear', 'D': 'Disgust', 'N': 'Neutral',
        'C': None, 'O': None, 'X': None,
    }
    
    valid_emotions = [k for k, v in emotion_mapping.items() if v is not None]
    valid_df = df[df['EmoClass'].isin(valid_emotions)]
    
    print(f"\n有效分类样本: {len(valid_df)}/{len(df)} ({len(valid_df)/len(df)*100:.1f}%)")
    
    # 按数据集划分检查
    for split in ['Train', 'Validation', 'Test']:
        split_df = valid_df[valid_df['Split_Set'] == split]
        print(f"\n{split} 数据集 ({len(split_df)} 样本):")
        split_emotions = split_df['EmoClass'].value_counts()
        for emotion, count in split_emotions.items():
            mapped_name = emotion_mapping[str(emotion)]
            print(f"  {emotion} ({mapped_name}): {count} ({count/len(split_df)*100:.1f}%)")

if __name__ == "__main__":
    analyze_emotion_distribution()