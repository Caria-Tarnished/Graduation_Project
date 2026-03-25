# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

df = pd.read_csv(r'e:\Projects\Graduation_Project\data\processed\train_3cls.csv', encoding='utf-8', on_bad_lines='skip')

print("=== 基础统计 ===")
print(f"总样本数: {len(df)}")
print(f"收益率列: {[c for c in df.columns if 'ret' in c.lower() or 'delta' in c.lower()]}")

# 检查关键列
ret_col = 'ret_post' if 'ret_post' in df.columns else None
if ret_col:
    print(f"\n=== {ret_col} 收益率分布 ===")
    print(df[ret_col].describe())
    print(f"零值比例: {(df[ret_col] == 0).mean():.3f}")
    print(f"绝对值小于0.0001的比例(近零): {(df[ret_col].abs() < 0.0001).mean():.3f}")
    print(f"绝对值小于0.0005的比例(极小): {(df[ret_col].abs() < 0.0005).mean():.3f}")

print("\n=== 按来源分类的指标分布 ===")
if ret_col:
    for src in df['source'].unique():
        sub = df[df['source'] == src]
        print(f"\n  来源={src} ({len(sub)} 条)")
        print(f"    标签分布: {sub['label'].value_counts().to_dict()}")
        print(f"    收益率中位数: {sub[ret_col].median():.6f}")
        print(f"    收益率标准差: {sub[ret_col].std():.6f}")
        print(f"    近零收益(|ret|<1e-4)比例: {(sub[ret_col].abs() < 1e-4).mean():.3f}")

print("\n=== 标签与收益率的关联性验证 ===")
if ret_col:
    for lbl in [-1, 0, 1]:
        sub = df[df['label'] == lbl]
        print(f"  标签={lbl}: 平均收益率={sub[ret_col].mean():.6f}, 标准差={sub[ret_col].std():.6f}, 样本数={len(sub)}")

print("\n=== delta_event_sec 检查(事件时间对齐质量) ===")
if 'delta_event_sec' in df.columns:
    print(df['delta_event_sec'].describe())
    print(f"  对齐偏差超过60秒的比例: {(df['delta_event_sec'].abs() > 60).mean():.3f}")
    print(f"  对齐偏差超过300秒的比例: {(df['delta_event_sec'].abs() > 300).mean():.3f}")
