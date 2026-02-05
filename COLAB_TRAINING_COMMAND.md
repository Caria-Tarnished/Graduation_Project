# Colab 训练命令（Phase 1 增强版）

## 问题修复说明

1. **路径修正**：数据在 `/content/Graduation_Project/data/processed/`（从 GitHub 拉取的代码仓库中）
2. **禁用 EarlyStoppingCallback**：设置 `--early_stopping_patience 0` 以避免兼容性问题
3. **HuggingFace 403 警告**：可以忽略，不影响训练

## 正确的训练命令

在 Colab 中运行以下命令：

```python
# Phase 1 训练：增强数据 + 自动类权重（无早停）
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

**注意**：
- `--early_stopping_patience 0` 禁用早停（避免 Colab 兼容性问题）
- 输出目录在 Drive 中（`/content/drive/MyDrive/...`），训练结果会自动保存
- 数据路径在代码仓库中（`/content/Graduation_Project/...`）

## 预计训练时间

- **T4 GPU**: 约 1-1.5 小时
- **总 steps**: 约 2,355 steps（5 epochs × 471 steps/epoch）
- **评估频率**: 每 100 steps

## 训练完成后查看结果

```python
import json

OUTPUT_DIR = '/content/drive/MyDrive/Graduation_Project/experiments/bert_enhanced_v1'

# 查看测试集指标
with open(f'{OUTPUT_DIR}/metrics_test.json', 'r') as f:
    metrics = json.load(f)

print(f"Test Accuracy: {metrics['eval_accuracy']:.4f}")
print(f"Test Macro F1: {metrics['eval_macro_f1']:.4f}")

# 查看分类报告
with open(f'{OUTPUT_DIR}/report_test.txt', 'r') as f:
    print(f.read())
```

## 如果仍然报错

### 错误 1: `AssertionError: EarlyStoppingCallback requires IntervalStrategy`

**解决方案**：已通过设置 `--early_stopping_patience 0` 解决

### 错误 2: 数据文件找不到

**检查数据是否存在**：
```python
import os
files = [
    '/content/Graduation_Project/data/processed/train_enhanced.csv',
    '/content/Graduation_Project/data/processed/val_enhanced.csv',
    '/content/Graduation_Project/data/processed/test_enhanced.csv'
]
for f in files:
    print(f"{'?' if os.path.exists(f) else '?'} {f}")
```

**如果文件不存在**：
1. 确认你已经 `git push` 了增强数据生成脚本
2. 在 Colab 中运行数据增强脚本：
```python
!python scripts/modeling/build_enhanced_dataset.py \
  --input_dir /content/Graduation_Project/data/processed \
  --output_dir /content/Graduation_Project/data/processed
```

### 错误 3: HuggingFace 403 Forbidden

**这是警告，不是错误**，可以安全忽略。如果想消除警告：
```python
# 设置 HuggingFace token（可选）
from huggingface_hub import login
login(token="your_hf_token_here")  # 从 https://huggingface.co/settings/tokens 获取
```

## 完整的 Colab 单元格序列

### 单元格 1: 环境准备
```python
# 挂载 Drive
from google.colab import drive
drive.mount('/content/drive')

# 安装依赖
!pip install -U transformers datasets evaluate accelerate -q

# 克隆/更新代码
import os
if not os.path.exists('/content/Graduation_Project'):
    !git clone https://github.com/Caria-Tarnished/Graduation_Project.git
else:
    !cd /content/Graduation_Project && git pull

%cd /content/Graduation_Project
```

### 单元格 2: 验证数据
```python
# 检查增强数据是否存在
import os
import pandas as pd

files = {
    'train': '/content/Graduation_Project/data/processed/train_enhanced.csv',
    'val': '/content/Graduation_Project/data/processed/val_enhanced.csv',
    'test': '/content/Graduation_Project/data/processed/test_enhanced.csv'
}

print("数据文件检查:")
all_exist = True
for name, path in files.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"? {name}: {len(df)} 样本")
    else:
        print(f"? {name}: 文件不存在")
        all_exist = False

if not all_exist:
    print("\n?? 数据文件缺失，正在生成...")
    !python scripts/modeling/build_enhanced_dataset.py \
      --input_dir /content/Graduation_Project/data/processed \
      --output_dir /content/Graduation_Project/data/processed
```

### 单元格 3: 运行训练
```python
# Phase 1 训练
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

### 单元格 4: 查看结果
```python
# 查看训练结果
import json
import pandas as pd

OUTPUT_DIR = '/content/drive/MyDrive/Graduation_Project/experiments/bert_enhanced_v1'

# 测试集指标
with open(f'{OUTPUT_DIR}/metrics_test.json', 'r') as f:
    test_metrics = json.load(f)

print("="*80)
print("Phase 1 训练结果")
print("="*80)
print(f"Test Accuracy: {test_metrics['eval_accuracy']:.4f}")
print(f"Test Macro F1: {test_metrics['eval_macro_f1']:.4f}")

# 与基线对比
baseline_f1 = 0.163
improvement = (test_metrics['eval_macro_f1'] - baseline_f1) / baseline_f1 * 100
print(f"\n与基线对比:")
print(f"  基线: {baseline_f1:.4f}")
print(f"  增强: {test_metrics['eval_macro_f1']:.4f}")
print(f"  提升: {improvement:+.1f}%")

# 分类报告
print("\n" + "="*80)
print("分类报告")
print("="*80)
with open(f'{OUTPUT_DIR}/report_test.txt', 'r') as f:
    print(f.read())

# 预测分析
pred_df = pd.read_csv(f'{OUTPUT_DIR}/pred_test.csv')
print("\n预测分布:")
print(pred_df['pred'].value_counts().sort_index())

# 稀有类别
rare_classes = [3, 4, 5]
print("\n稀有类别预测:")
for cls in rare_classes:
    pred_count = (pred_df['pred'] == cls).sum()
    true_count = (pred_df['label'] == cls).sum()
    print(f"  Class {cls}: 真实={true_count}, 预测={pred_count}")
```
