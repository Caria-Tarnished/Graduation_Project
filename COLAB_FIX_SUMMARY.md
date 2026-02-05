# Colab 训练错误修复总结

## 问题诊断

你遇到的错误是：
```
AssertionError: EarlyStoppingCallback requires IntervalStrategy of steps or epoch
```

这是因为 `EarlyStoppingCallback` 在 Colab 的 transformers 版本中对 `eval_strategy` 的检查更严格。

## 已修复的问题

### 1. EarlyStoppingCallback 条件判断
**问题**：即使设置 `--early_stopping_patience 0`，回调仍可能被添加
**修复**：明确检查 `patience > 0`，确保 0 值不会触发早停

### 2. compute_loss 参数兼容性
**问题**：新版 transformers 需要 `num_items_in_batch` 参数
**修复**：已添加 `num_items_in_batch=None` 参数

### 3. 数据路径
**问题**：文档中路径不一致
**修复**：统一使用 `/content/Graduation_Project/data/processed/`

## 需要执行的步骤

### 步骤 1: 提交修复后的代码到 GitHub

在本地运行：
```bash
git add scripts/modeling/bert_finetune_cls.py
git add colab_phase1_cells.txt
git commit -m "Fix EarlyStoppingCallback and compute_loss compatibility issues"
git push
```

### 步骤 2: 在 Colab 中拉取最新代码

在 Colab 的第一个单元格运行后，确保代码已更新：
```python
!cd /content/Graduation_Project && git pull
```

你应该看到类似输出：
```
Updating xxxxx..yyyyy
Fast-forward
 scripts/modeling/bert_finetune_cls.py | 5 +++--
 1 file changed, 3 insertions(+), 2 deletions(-)
```

### 步骤 3: 运行训练

直接使用 `colab_phase1_cells.txt` 中的单元格 4：
```python
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

## 可以忽略的警告

### HuggingFace 403 Forbidden
```
huggingface_hub.errors.HfHubHTTPError: 403 Forbidden: Discussions are disabled for this repo.
```
**这是警告，不是错误**。模型仍会正常加载和训练。

### UNEXPECTED/MISSING 权重
```
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |
classifier.bias                            | MISSING    |
```
**这是正常的**。我们在做序列分类任务，不是预训练任务，所以预训练的 MLM 头会被丢弃，新的分类头会被初始化。

## 预期训练时间

- **GPU**: T4 (Colab 免费版)
- **总时长**: 约 1-1.5 小时
- **总 steps**: ~2,355 (5 epochs × 471 steps/epoch)
- **评估频率**: 每 100 steps

## 训练成功的标志

你应该看到类似输出：
```
Training: 100%|██████████| 2355/2355 [1:23:45<00:00, 2.13s/it]
Evaluating: 100%|██████████| 196/196 [00:15<00:00, 12.45it/s]
{'eval_loss': 1.234, 'eval_accuracy': 0.567, 'eval_macro_f1': 0.345}
完成：模型与评估结果已保存至 /content/drive/MyDrive/Graduation_Project/experiments/bert_enhanced_v1
```

## 如果仍然报错

### 错误：数据文件不存在
**解决方案**：在单元格 2 中会自动检测并生成数据

### 错误：CUDA out of memory
**解决方案**：减小 batch size
```python
--train_bs 8 \
--gradient_accumulation_steps 4 \
```

### 错误：transformers 版本问题
**解决方案**：重新安装
```python
!pip uninstall transformers -y
!pip install transformers==4.36.0 -q
```

## 下一步

训练完成后：
1. 查看 `metrics_test.json` 中的 `eval_macro_f1`
2. 与基线 (0.163) 对比
3. 检查稀有类别 (3, 4, 5) 的 F1 分数
4. 如果 F1 > 0.35，Phase 1 成功！
5. 如果稀有类别仍然 F1=0，考虑 Phase 2（合成数据生成）
