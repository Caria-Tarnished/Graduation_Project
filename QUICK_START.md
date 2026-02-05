# 快速开始：Colab 训练 Phase 1

## ? 3 步开始训练

### 步骤 1: 本地提交代码（1 分钟）

```bash
# 在本地项目目录运行
git add .
git commit -m "Fix Colab training compatibility"
git push
```

### 步骤 2: Colab 拉取代码（1 分钟）

在 Colab 新建单元格运行：
```python
from google.colab import drive
drive.mount('/content/drive')

!pip install -U transformers datasets evaluate accelerate -q

import os
if not os.path.exists('/content/Graduation_Project'):
    !git clone https://github.com/Caria-Tarnished/Graduation_Project.git
else:
    !cd /content/Graduation_Project && git pull  # 确保拉取最新修复

%cd /content/Graduation_Project
```

### 步骤 3: 开始训练（1-1.5 小时）

复制粘贴 `colab_phase1_cells.txt` 中的单元格 2-4，依次运行。

或者直接运行：
```python
# 生成数据（如果不存在）
!python scripts/modeling/build_enhanced_dataset.py \
  --input_dir /content/Graduation_Project/data/processed \
  --output_dir /content/Graduation_Project/data/processed

# 开始训练
!python scripts/modeling/bert_finetune_cls.py \
  --train_csv /content/Graduation_Project/data/processed/train_enhanced.csv \
  --val_csv /content/Graduation_Project/data/processed/val_enhanced.csv \
  --test_csv /content/Graduation_Project/data/processed/test_enhanced.csv \
  --output_dir /content/drive/MyDrive/Graduation_Project/experiments/bert_enhanced_v1 \
  --label_col label_multi_cls \
  --model_name hfl/chinese-roberta-wwm-ext \
  --class_weight auto \
  --epochs 5 \
  --lr 1e-5 \
  --max_length 384 \
  --train_bs 16 \
  --eval_bs 32 \
  --gradient_accumulation_steps 2 \
  --warmup_ratio 0.06 \
  --weight_decay 0.01 \
  --eval_steps 100 \
  --save_steps 100 \
  --early_stopping_patience 0
```

## ? 训练成功标志

看到这个输出就成功了：
```
完成：模型与评估结果已保存至 /content/drive/MyDrive/Graduation_Project/experiments/bert_enhanced_v1
```

## ? 查看结果

```python
import json

OUTPUT_DIR = '/content/drive/MyDrive/Graduation_Project/experiments/bert_enhanced_v1'

with open(f'{OUTPUT_DIR}/metrics_test.json', 'r') as f:
    metrics = json.load(f)

print(f"Test Macro F1: {metrics['eval_macro_f1']:.4f}")
print(f"Test Accuracy: {metrics['eval_accuracy']:.4f}")

# 目标：macro_f1 > 0.35（基线是 0.163）
```

## ? 期望结果

- **Test Macro F1**: 0.35 - 0.50（提升 100%+）
- **稀有类别 F1**: > 0（Classes 3, 4, 5 开始被预测）

## ?? 如果报错

### 错误 1: `AssertionError: EarlyStoppingCallback`
**原因**: 代码未更新
**解决**: 确保在 Colab 中运行了 `git pull`

### 错误 2: 数据文件不存在
**原因**: 增强数据未生成
**解决**: 运行单元格 2（数据生成脚本）

### 错误 3: CUDA out of memory
**原因**: GPU 内存不足
**解决**: 减小 batch size
```python
--train_bs 8 \
--gradient_accumulation_steps 4 \
```

## ? 完整文件参考

所有训练相关文件：
- `colab_phase1_cells.txt` - 完整的 Colab 单元格
- `COLAB_FIX_SUMMARY.md` - 修复说明
- `COLAB_TRAINING_CHECKLIST.md` - 详细检查清单
- `COLAB_TRAINING_COMMAND.md` - 命令参考

## ? 提示

1. **确保 GPU 已启用**: Runtime → Change runtime type → GPU
2. **监控训练**: 每 100 steps 会输出一次评估结果
3. **保存结果**: 训练结果自动保存到 Google Drive
4. **可以忽略的警告**: HuggingFace 403、UNEXPECTED weights

---

**准备好了吗？开始训练吧！** ?
