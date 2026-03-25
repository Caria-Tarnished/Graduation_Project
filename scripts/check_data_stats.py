# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from collections import defaultdict

df_train = pd.read_csv(r'e:\Projects\Graduation_Project\data\processed\train_3cls.csv', encoding='utf-8', on_bad_lines='skip')
print('=== 训练集总行数 ===', len(df_train))
label_counts = df_train['label'].value_counts().sort_index()
print('label 分布:')
for lbl, cnt in label_counts.items():
    print(f'  {lbl}: {cnt} ({cnt/len(df_train)*100:.1f}%)')

source_counts = df_train['source'].value_counts()
print('数据来源分布:')
print(source_counts)

df_pred = pd.read_csv(r'e:\Projects\Graduation_Project\reports\bert_3cls_enhanced_v1\pred_test.csv', encoding='utf-8')
print()
print('=== 测试集总行数 ===', len(df_pred))
print('真实标签分布:', df_pred['label'].value_counts().sort_index().to_dict())
print('预测标签分布:', df_pred['pred'].value_counts().sort_index().to_dict())

labels = [-1, 0, 1]
tp = defaultdict(int)
fp = defaultdict(int)
fn = defaultdict(int)
for _, row in df_pred.iterrows():
    t, p = row['label'], row['pred']
    if t == p:
        tp[t] += 1
    else:
        fp[p] += 1
        fn[t] += 1

print()
print('Per-Class Results:')
f1s = []
for l in labels:
    prec = tp[l] / (tp[l] + fp[l] + 1e-9)
    rec = tp[l] / (tp[l] + fn[l] + 1e-9)
    f1 = 2*prec*rec/(prec+rec+1e-9)
    f1s.append(f1)
    print(f'  cls {l}: P={prec:.4f}, R={rec:.4f}, F1={f1:.4f}, TP={tp[l]}, FP={fp[l]}, FN={fn[l]}')

print(f'  Macro F1: {np.mean(f1s):.4f}')
acc = (df_pred['label'] == df_pred['pred']).mean()
print(f'  Accuracy: {acc:.4f}')
