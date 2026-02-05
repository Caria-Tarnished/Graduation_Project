# Colab 训练检查清单

## ? 在本地完成（提交代码前）

- [x] 修复 `bert_finetune_cls.py` 中的 EarlyStoppingCallback 问题
- [x] 修复 `compute_loss` 方法的参数兼容性
- [x] 更新 `colab_phase1_cells.txt` 训练命令
- [ ] **提交代码到 GitHub**
  ```bash
  git add scripts/modeling/bert_finetune_cls.py
  git add colab_phase1_cells.txt
  git add COLAB_FIX_SUMMARY.md
  git commit -m "Fix Colab training compatibility issues"
  git push
  ```

## ? 在 Colab 完成

### 1. 环境准备
- [ ] 挂载 Google Drive
- [ ] 安装依赖包
- [ ] 克隆/拉取最新代码（**重要：确保拉取了最新修复**）
  ```python
  !cd /content/Graduation_Project && git pull
  ```
- [ ] 验证代码已更新（检查 git pull 输出）

### 2. 数据准备
- [ ] 检查增强数据文件是否存在
- [ ] 如果不存在，运行数据增强脚本
- [ ] 预览增强数据（可选）

### 3. 训练
- [ ] 运行训练命令（单元格 4）
- [ ] 监控训练进度（约 1-1.5 小时）
- [ ] 确认没有报错

### 4. 结果分析
- [ ] 查看测试集指标（`metrics_test.json`）
- [ ] 对比基线性能（基线 macro_f1 = 0.163）
- [ ] 检查分类报告（`report_test.txt`）
- [ ] 分析稀有类别预测情况（Classes 3, 4, 5）

## ? 成功标准

### Phase 1 目标
- **Test Macro F1 > 0.35**（相比基线 0.163 提升 >100%）
- **稀有类别 F1 > 0**（Classes 3, 4, 5 开始被预测）

### 如果达到目标
? Phase 1 成功！可以考虑：
- 进一步调优超参数
- 尝试更大的模型
- 增加训练轮数

### 如果未达到目标
?? 需要 Phase 2：
- 合成数据生成（SMOTE/Back-translation）
- 手动调整类权重
- 数据增强策略优化

## ? 预期输出文件

训练完成后，以下文件应该存在于 `/content/drive/MyDrive/Graduation_Project/experiments/bert_enhanced_v1/`：

- `metrics_val.json` - 验证集指标
- `metrics_test.json` - 测试集指标
- `report_test.txt` - 详细分类报告
- `pred_test.csv` - 测试集预测结果
- `eval_results.json` - 汇总评估结果
- `best/` - 最优模型目录
  - `config.json`
  - `pytorch_model.bin`
  - `tokenizer_config.json`
  - 等

## ? 常见问题

### Q: HuggingFace 403 Forbidden 警告
**A**: 可以忽略，不影响训练

### Q: UNEXPECTED/MISSING 权重警告
**A**: 正常现象，分类头会被重新初始化

### Q: CUDA out of memory
**A**: 减小 batch size 到 8，增加 gradient_accumulation_steps 到 4

### Q: 训练速度很慢
**A**: 确认使用了 GPU（Runtime → Change runtime type → GPU）

### Q: 数据文件找不到
**A**: 单元格 2 会自动检测并生成，确保运行了该单元格

## ? 记录训练结果

训练完成后，记录以下信息：

```
训练日期: ___________
训练时长: ___________
GPU 类型: ___________

结果:
- Test Accuracy: ___________
- Test Macro F1: ___________
- 相比基线提升: ___________

稀有类别 F1:
- Class 3: ___________
- Class 4: ___________
- Class 5: ___________

备注:
___________________________________________
___________________________________________
```
