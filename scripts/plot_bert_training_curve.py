# -*- coding: utf-8 -*-
"""绘制 BERT 3分类训练曲线（基于 Colab 训练日志）"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 从 Colab 训练日志提取的数据
eval_data = [
    (0.2491, 0.2690, 1.102),
    (0.4981, 0.2947, 1.101),
    (0.7472, 0.3661, 1.091),
    (0.9963, 0.3778, 1.086),
    (1.244,  0.3184, 1.110),
    (1.493,  0.3631, 1.085),
    (1.742,  0.3622, 1.088),
    (1.991,  0.3411, 1.088),
    (2.239,  0.3672, 1.108),
    (2.488,  0.3655, 1.095),
    (2.737,  0.3580, 1.105),
    (2.986,  0.3576, 1.111),
    (3.234,  0.3614, 1.111),
]

train_data = [
    (0.1245, 2.239),
    (0.2491, 2.237),
    (0.3736, 2.214),
    (0.4981, 2.209),
    (0.6227, 2.214),
    (0.7472, 2.209),
    (0.8717, 2.206),
    (0.9963, 2.199),
    (1.120,  2.134),
    (1.244,  2.155),
    (1.369,  2.173),
    (1.493,  2.182),
    (1.618,  2.187),
    (1.742,  2.159),
    (1.867,  2.143),
    (1.991,  2.159),
    (2.115,  2.100),
    (2.239,  2.113),
    (2.364,  2.143),
    (2.488,  2.121),
    (2.613,  2.118),
    (2.737,  2.111),
    (2.862,  2.132),
    (2.986,  2.123),
    (3.110,  2.045),
    (3.234,  2.086),
    (3.359,  2.088),
]

epochs_eval = [d[0] for d in eval_data]
f1_scores = [d[1] for d in eval_data]
eval_loss = [d[2] for d in eval_data]
epochs_train = [d[0] for d in train_data]
train_loss = [d[1] for d in train_data]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# 左图：Macro F1
ax1.plot(epochs_eval, f1_scores, 'o-', color='#2196F3', linewidth=2, markersize=5, label='Macro F1')
best_idx = f1_scores.index(max(f1_scores))
ax1.annotate(f'Best: {f1_scores[best_idx]:.4f}\n(epoch {epochs_eval[best_idx]:.1f})',
             xy=(epochs_eval[best_idx], f1_scores[best_idx]),
             xytext=(epochs_eval[best_idx] + 0.5, f1_scores[best_idx] + 0.015),
             arrowprops=dict(arrowstyle='->', color='#E53935'),
             fontsize=9, color='#E53935', fontweight='bold')
ax1.axhline(y=0.3778, color='gray', linestyle='--', alpha=0.5, label='Best F1=0.3778')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Macro F1', fontsize=11)
ax1.set_title('Validation Macro F1', fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.2, 0.45)

# 右图：Loss
ax2.plot(epochs_train, train_loss, '-', color='#FF9800', linewidth=1.5, alpha=0.8, label='Train Loss')
ax2.plot(epochs_eval, eval_loss, 's-', color='#4CAF50', linewidth=2, markersize=5, label='Eval Loss')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Loss', fontsize=11)
ax2.set_title('Training & Validation Loss', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('thesis_assets/charts/bert_3cls_training_curve.png', dpi=200, bbox_inches='tight')
print('Saved: thesis_assets/charts/bert_3cls_training_curve.png')
plt.close()
