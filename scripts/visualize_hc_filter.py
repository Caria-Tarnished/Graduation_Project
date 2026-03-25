# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = r"e:\Projects\Graduation_Project"
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
OUT_DIR  = os.path.join(BASE_DIR, "thesis_assets", "charts")
os.makedirs(OUT_DIR, exist_ok=True)

df_orig = pd.read_csv(os.path.join(DATA_DIR, "train_3cls.csv"), encoding="utf-8", on_bad_lines="skip")
df_hc   = pd.read_csv(os.path.join(DATA_DIR, "train_hc_3cls.csv"), encoding="utf-8", on_bad_lines="skip")

# ─── 图1：收益率分布对比（原始 vs 高置信度）─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
colors = {-1: 'tomato', 0: 'steelblue', 1: 'mediumseagreen'}
label_names = {-1: '利空 (-1)', 0: '中立 (0)', 1: '利多 (+1)'}

for ax, df, title in zip(axes, [df_orig, df_hc], ['原始标注数据集', '高置信度过滤后数据集']):
    for lbl in [-1, 0, 1]:
        sub = df[df['label'] == lbl]['ret_post']
        sub_clipped = sub.clip(-0.006, 0.006)  # 截断极端值以便可视化
        ax.hist(sub_clipped, bins=60, alpha=0.55, color=colors[lbl], label=f"{label_names[lbl]} n={len(sub)}")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('ret_post（事件后15分钟收益率）', fontsize=11)
    ax.set_ylabel('样本数量', fontsize=11)
    ax.legend(fontsize=10)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')

plt.suptitle('高置信度过滤前后：各类别收益率分布对比', fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'hc_filter_ret_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print("已生成: hc_filter_ret_distribution.png")

# ─── 图2：过滤前后各类均值/标准差对比柱状图 ─────────────────────────
labels_order = [-1, 0, 1]
orig_means = [df_orig[df_orig['label'] == l]['ret_post'].mean() for l in labels_order]
hc_means   = [df_hc[df_hc['label'] == l]['ret_post'].mean()   for l in labels_order]
orig_stds  = [df_orig[df_orig['label'] == l]['ret_post'].std() for l in labels_order]
hc_stds    = [df_hc[df_hc['label'] == l]['ret_post'].std()   for l in labels_order]

x = np.arange(3)
w = 0.3

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 均值对比
axes[0].bar(x - w/2, orig_means, w, label='原始', color='#90CAF9', alpha=0.9)
axes[0].bar(x + w/2, hc_means,   w, label='过滤后', color='#1565C0', alpha=0.9)
axes[0].set_xticks(x)
axes[0].set_xticklabels([label_names[l] for l in labels_order])
axes[0].set_ylabel('平均收益率 ret_post')
axes[0].set_title('各类别收益率均值对比')
axes[0].legend()
axes[0].axhline(0, color='gray', linewidth=0.8)

# 标准差对比（噪声水平）
axes[1].bar(x - w/2, orig_stds, w, label='原始', color='#EF9A9A', alpha=0.9)
axes[1].bar(x + w/2, hc_stds,   w, label='过滤后', color='#B71C1C', alpha=0.9)
axes[1].set_xticks(x)
axes[1].set_xticklabels([label_names[l] for l in labels_order])
axes[1].set_ylabel('收益率标准差（噪声水平）')
axes[1].set_title('各类别收益率标准差对比（越低越好）')
axes[1].legend()

plt.suptitle('高置信度过滤对标注纯度的改善效果', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'hc_filter_purity_improvement.png'), dpi=300, bbox_inches='tight')
plt.close()
print("已生成: hc_filter_purity_improvement.png")

# ─── 图3：数据集规模变化对比 ─────────────────────────────────────────
splits = ['训练集', '验证集', '测试集']
orig_sizes = [
    len(df_orig),
    len(pd.read_csv(os.path.join(DATA_DIR, "val_3cls.csv"), encoding="utf-8", on_bad_lines="skip")),
    len(pd.read_csv(os.path.join(DATA_DIR, "test_3cls.csv"), encoding="utf-8", on_bad_lines="skip")),
]
hc_sizes = [
    len(df_hc),
    len(pd.read_csv(os.path.join(DATA_DIR, "val_hc_3cls.csv"), encoding="utf-8", on_bad_lines="skip")),
    len(pd.read_csv(os.path.join(DATA_DIR, "test_hc_3cls.csv"), encoding="utf-8", on_bad_lines="skip")),
]

x = np.arange(3)
fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - w/2, orig_sizes, w, label='原始数据集', color='#A5D6A7')
b2 = ax.bar(x + w/2, hc_sizes,   w, label='高置信度过滤后', color='#2E7D32')

for bar, val in zip(b1, orig_sizes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, str(val), ha='center', fontsize=9)
for bar, val in zip(b2, hc_sizes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, str(val), ha='center', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(splits)
ax.set_ylabel('样本数量')
ax.set_title('高置信度过滤前后各分割数据集规模对比')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'hc_filter_size_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("已生成: hc_filter_size_comparison.png")
print("\n所有对比图已保存至 thesis_assets/charts/")
