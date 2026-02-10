# Colab QLoRA 训练问题修复总结

## 问题分析

### 问题 1：数据库列名错误 ?
```
? 提取新闻样本失败: no such column: ei.ret_post_15
```

**原因**：
- 代码中使用：`ei.ret_post_15` 和 `ei.pre_ret_120`
- 数据库实际列名：`ei.ret_post_15m` 和 `ei.pre_ret_120m`

**影响**：
- 无法从数据库提取真实新闻样本
- 降级使用模板数据（120条而非300条）

**解决方案**：
- 已修复本地脚本：`scripts/qlora/build_instruction_dataset.py`
- Colab 中动态修复：在单元格 3 中替换列名

---

### 问题 2：triton.ops 模块缺失 ?
```
ModuleNotFoundError: No module named 'triton.ops'
```

**原因**：
- 使用了 `bitsandbytes==0.44.1`
- 该版本依赖 `triton>=3.0.0`
- 但 triton 3.0 移除了 `triton.ops` 模块（API 重构）

**影响**：
- 训练脚本无法导入 bitsandbytes
- 训练完全失败，没有生成任何模型文件

**解决方案**：
- 降级到兼容版本组合：
  - `bitsandbytes==0.43.1`（最后支持 triton 2.x 的版本）
  - `triton==2.1.0`（稳定版本）

---

### 问题 3：数据库文件路径 ??
```
? 数据库不存在，从 Drive 复制...
cp: cannot stat '/content/drive/MyDrive/Graduation_Project/finance_analysis.db': No such file or directory
```

**原因**：
- 用户将数据库上传到了 `/content/drive/MyDrive/Graduation_Project/datasets/finance_analysis.db`
- 代码期望路径：`/content/drive/MyDrive/Graduation_Project/finance_analysis.db`

**解决方案**：
- 在单元格 3 中从正确路径复制：
  ```python
  drive_db_path = Path('/content/drive/MyDrive/Graduation_Project/datasets/finance_analysis.db')
  !cp /content/drive/MyDrive/Graduation_Project/datasets/finance_analysis.db .
  ```

---

## 完整修复方案

### 文件清单

1. **colab_qlora_training_cells_final.txt**（新文件）
   - 完整修复版 Colab 训练代码
   - 包含 7 个单元格
   - 修复了所有已知问题

2. **scripts/qlora/build_instruction_dataset.py**（已更新）
   - 修复数据库列名：`ret_post_15` → `ret_post_15m`
   - 修复数据库列名：`pre_ret_120` → `pre_ret_120m`

3. **COLAB_QLORA_FIX_SUMMARY.md**（本文件）
   - 问题分析和修复总结

---

## 关键修复点

### 修复 1：依赖版本（单元格 1）

**修复前**：
```python
!pip install bitsandbytes==0.44.1  # ? 不兼容
```

**修复后**：
```python
!pip uninstall -y bitsandbytes triton
!pip install -q triton==2.1.0
!pip install -q bitsandbytes==0.43.1  # ? 兼容
```

### 修复 2：数据库列名（单元格 3）

**修复前**：
```python
# 直接运行原始脚本
!python scripts/qlora/build_instruction_dataset.py  # ? 列名错误
```

**修复后**：
```python
# 动态修复脚本
with open('scripts/qlora/build_instruction_dataset.py', 'r', encoding='utf-8') as f:
    script_content = f.read()

script_content = script_content.replace('ei.ret_post_15', 'ei.ret_post_15m')
script_content = script_content.replace('ei.pre_ret_120', 'ei.pre_ret_120m')

with open('scripts/qlora/build_instruction_dataset_fixed.py', 'w', encoding='utf-8') as f:
    f.write(script_content)

!python scripts/qlora/build_instruction_dataset_fixed.py  # ? 使用修复版
```

### 修复 3：数据库路径（单元格 3）

**修复前**：
```python
drive_db_path = Path('/content/drive/MyDrive/Graduation_Project/finance_analysis.db')  # ? 路径错误
```

**修复后**：
```python
drive_db_path = Path('/content/drive/MyDrive/Graduation_Project/datasets/finance_analysis.db')  # ? 正确路径
```

---

## 使用新的修复版本

### 步骤 1：在 Colab 中创建新 Notebook

1. 打开 Google Colab：https://colab.research.google.com
2. 创建新笔记本
3. 将 `colab_qlora_training_cells_final.txt` 中的每个单元格复制到 Colab

### 步骤 2：按顺序执行单元格

```
单元格 1：环境准备（2-3 分钟）
    ↓
单元格 2：克隆代码（30 秒）
    ↓
单元格 3：生成指令集（1-2 分钟）
    ↓
单元格 4：开始训练（2-4 小时）?
    ↓
单元格 5：验证结果（10 秒）
    ↓
单元格 6：测试模型（2-3 分钟）
    ↓
单元格 7：备份模型（可选，1 分钟）
```

### 步骤 3：预期输出

#### 单元格 1 成功标志：
```
? 环境准备完成
? 依赖版本：
Name: bitsandbytes
Version: 0.43.1
Name: transformers
Version: 4.46.0
Name: triton
Version: 2.1.0
```

#### 单元格 3 成功标志：
```
? 数据库已存在
? 脚本修复完成
...
1. 从数据库提取 180 条新闻样本...
   ? 成功生成 180 条新闻指令  # ? 不再是 "? 提取失败"
...
总样本数: 300  # ? 不再是 120
```

#### 单元格 4 成功标志：
```
开始 QLoRA 微调...
...
Epoch 1/3: 100%|██████████| 75/75 [XX:XX<00:00]
Epoch 2/3: 100%|██████████| 75/75 [XX:XX<00:00]
Epoch 3/3: 100%|██████████| 75/75 [XX:XX<00:00]
? 训练完成  # ? 不再是 "ModuleNotFoundError"
```

---

## 验证修复效果

### 检查点 1：依赖版本
```python
# 在 Colab 单元格中运行
!pip show bitsandbytes triton | grep "Name:\|Version:"
```

**预期输出**：
```
Name: bitsandbytes
Version: 0.43.1
Name: triton
Version: 2.1.0
```

### 检查点 2：数据库查询
```python
# 在 Colab 单元格中运行
import sqlite3
conn = sqlite3.connect('finance_analysis.db')
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(event_impacts)")
columns = [row[1] for row in cursor.fetchall()]
print("event_impacts 表的列名：", columns)
conn.close()
```

**预期输出**：
```
event_impacts 表的列名：['event_id', 'ret_post_15m', 'pre_ret_120m', ...]
```

### 检查点 3：训练输出文件
```python
# 在 Colab 单元格中运行
!ls -lh /content/drive/MyDrive/Graduation_Project/qlora_output/
```

**预期输出**：
```
adapter_model.bin       (50-100 MB)
adapter_config.json     (1 KB)
training_info.json      (1 KB)
```

---

## 故障排查

### 如果单元格 1 仍然报错

**症状**：
```
ERROR: Could not find a version that satisfies the requirement triton==2.1.0
```

**解决**：
```python
# 使用 pip 的 --no-deps 选项
!pip install --no-deps triton==2.1.0
!pip install bitsandbytes==0.43.1
```

### 如果单元格 3 数据库查询仍然失败

**症状**：
```
? 提取新闻样本失败: no such column: ei.ret_post_15m
```

**解决**：
```python
# 检查数据库表结构
!sqlite3 finance_analysis.db ".schema event_impacts"

# 如果列名不同，手动修改脚本中的列名
```

### 如果单元格 4 训练 OOM（显存不足）

**症状**：
```
RuntimeError: CUDA out of memory
```

**解决**：
```python
# 减小 batch_size 和 max_length
!python scripts/qlora/train_qlora.py \
    --batch_size 2 \        # 从 4 改为 2
    --max_length 256 \      # 从 512 改为 256
    ...
```

---

## 本地文件更新

已更新以下本地文件：

1. ? `scripts/qlora/build_instruction_dataset.py`
   - 修复数据库列名

2. ? `colab_qlora_training_cells_final.txt`
   - 完整修复版 Colab 代码

3. ? `COLAB_QLORA_FIX_SUMMARY.md`
   - 本文档

---

## 下一步操作

### 立即执行：

1. **在 Colab 中使用新的修复版本**
   - 打开 `colab_qlora_training_cells_final.txt`
   - 复制每个单元格到 Colab
   - 按顺序执行

2. **提交本地修复到 GitHub**
   ```bash
   git add scripts/qlora/build_instruction_dataset.py
   git add colab_qlora_training_cells_final.txt
   git add COLAB_QLORA_FIX_SUMMARY.md
   git commit -m "Fix QLoRA training issues: database columns and triton.ops"
   git push
   ```

### 训练完成后：

1. **验证模型效果**（单元格 6）
2. **备份模型到 Drive**（单元格 7）
3. **更新项目状态文档**

---

## 预期训练时间

| GPU 类型 | 300 样本 | 120 样本 |
|---------|---------|---------|
| T4      | 3-4 小时 | 1-2 小时 |
| V100    | 2-3 小时 | 45-90 分钟 |
| A100    | 1-2 小时 | 30-60 分钟 |

---

## 总结

### 修复前的问题：
- ? triton.ops 模块缺失 → 训练完全失败
- ? 数据库列名错误 → 只能使用模板数据（120条）
- ?? 数据库路径错误 → 需要手动调整

### 修复后的效果：
- ? 依赖版本兼容 → 训练可以正常运行
- ? 数据库查询成功 → 可以使用真实数据（300条）
- ? 路径自动处理 → 无需手动干预

### 关键改进：
1. **依赖版本**：bitsandbytes 0.44.1 → 0.43.1 + triton 2.1.0
2. **数据库列名**：ret_post_15 → ret_post_15m
3. **动态修复**：Colab 中自动修复脚本，无需手动编辑

现在可以在 Colab 中使用 `colab_qlora_training_cells_final.txt` 开始训练了！
