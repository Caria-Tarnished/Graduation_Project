# -*- coding: utf-8 -*-
"""绘制 QLoRA v2 训练损失曲线（基于 Colab 训练日志，500条样本）"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 从 Colab v2 训练日志提取的数据（500条样本，3 epoch，96 steps）
# 每10步记录一次 loss
train_log = [
    (10,  0.32, 2.612),
    (20,  0.64, 2.498),
    (30,  0.96, 2.353),
    (40,  1.256, 2.070),
    (50,  1.576, 1.656),
    (60,  1.896, 1.353),
    (70,  2.192, 1.153),
    (80,  2.512, 1.062),
    (90,  2.832, 0.986),
]

steps = [d[0] for d in train_log]
epochs = [d[1] for d in train_log]
losses = [d[2] for d in train_log]

fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(steps, losses, 'o-', color='#E53935', linewidth=2, markersize=6)

# 标注起止值
ax.annotate(f'{losses[0]:.2f}', xy=(steps[0], losses[0]),
            xytext=(steps[0]+5, losses[0]+0.05),
            fontsize=9, color='#333')
ax.annotate(f'{losses[-1]:.2f}', xy=(steps[-1], losses[-1]),
            xytext=(steps[-1]-15, losses[-1]+0.15),
            fontsize=9, color='#333')

# 添加 epoch 分隔线
for ep in [1, 2]:
    # 找到最接近该 epoch 的 step
    for s, e, _ in train_log:
        if abs(e - ep) < 0.2:
            ax.axvline(x=s, color='gray', linestyle=':', alpha=0.5)
            ax.text(s+1, max(losses)*0.95, f'Epoch {ep}', fontsize=8, color='gray')
            break

ax.set_xlabel('Training Steps', fontsize=11)
ax.set_ylabel('Training Loss', fontsize=11)
ax.set_title('QLoRA Training Loss (v2, 500 samples, 3 epochs)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.5, 3.0)

plt.tight_layout()
plt.savefig('thesis_assets/charts/qlora_training_loss.png', dpi=200, bbox_inches='tight')
print('Saved: thesis_assets/charts/qlora_training_loss.png')
plt.close()
