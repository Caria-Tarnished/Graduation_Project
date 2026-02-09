# QLoRA 微调工作流程

**更新时间**: 2026-02-09  
**目标**: 微调 Deepseek-7B 模型，提升财经领域专业性  
**硬件**: Google Colab T4 GPU（免费版）  
**时间**: 约 4-6 小时（数据准备 + 训练 + 测试）

---

## 1. 工作流程概览

```
┌─────────────────────────────────────────────────────────────┐
│  阶段 1: 准备指令集数据（1-2 小时）                          │
│  - 从 finance_analysis.db 提取真实案例                      │
│  - 生成 Instruction-Input-Output 格式数据                   │
│  - 目标：300 条指令（60% 新闻 + 20% 市场分析 + 20% 财报）   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  阶段 2: Colab 训练（2-4 小时）                              │
│  - 加载 Deepseek-7B 模型（4-bit 量化）                      │
│  - 配置 LoRA（rank=8, alpha=16）                            │
│  - 训练 3 epochs                                            │
│  - 保存 LoRA 权重（约 30-50 MB）                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  阶段 3: 测试与集成（1 小时）                                │
│  - 在 Colab 上测试微调效果                                  │
│  - 下载 LoRA 权重到本地                                     │
│  - （可选）集成到本地 Agent 系统                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 阶段 1：准备指令集数据

### 2.1 本地生成指令集

在本地运行以下命令：

```powershell
# 生成 300 条指令
python scripts/qlora/build_instruction_dataset.py `
    --db finance_analysis.db `
    --output data/qlora/instructions.jsonl `
    --num_samples 300
```

**输出**：
- 文件：`data/qlora/instructions.jsonl`
- 格式：每行一个 JSON 对象
- 大小：约 100-200 KB

**数据分布**：
- 60%（180 条）：从真实新闻生成（基于 finance_analysis.db）
- 20%（60 条）：市场分析指令（预期兑现、观望信号等）
- 20%（60 条）：财报问答指令（模拟）

### 2.2 查看生成的数据

```powershell
# 查看前 3 条指令
python -c "import json; [print(f'\n{i+1}. {json.loads(line)}') for i, line in enumerate(open('data/qlora/instructions.jsonl', encoding='utf-8')) if i < 3]"
```

### 2.3 上传到 GitHub

```powershell
git add data/qlora/instructions.jsonl
git commit -m "Add QLoRA instruction dataset"
git push
```

**注意**：确保 `.gitignore` 允许提交 `data/qlora/` 目录。

---

## 3. 阶段 2：Colab 训练

### 3.1 打开 Colab 笔记本

1. 访问 [Google Colab](https://colab.research.google.com/)
2. 创建新笔记本
3. 选择运行时类型：**GPU（T4）**

### 3.2 复制训练单元格

打开 `colab_qlora_training_cells.txt`，按顺序复制并运行以下单元格：

#### 单元格 1：环境准备（约 5 分钟）
- 检查 GPU
- 挂载 Google Drive
- 安装依赖（transformers, peft, bitsandbytes 等）

#### 单元格 2：克隆代码仓库（约 1 分钟）
- 从 GitHub 克隆或更新代码

#### 单元格 3：生成指令集（约 1 分钟）
- 如果本地已生成，会直接使用
- 如果未生成，会在 Colab 上生成

#### 单元格 4：开始训练（约 2-4 小时）⏰
- 加载 Deepseek-7B 模型（4-bit 量化）
- 配置 LoRA
- 训练 3 epochs
- 保存到 Google Drive

**重要**：
- ⚠️ 保持 Colab 页面打开，避免断开连接
- ⚠️ 建议使用 Colab Pro（更稳定）
- ⚠️ 训练过程中可以查看 Loss 曲线

#### 单元格 5：验证结果（约 1 分钟）
- 检查输出文件
- 查看训练信息
- 确认权重文件大小

#### 单元格 6：测试模型（约 5 分钟，可选）
- 加载微调后的模型
- 测试推理效果
- 对比微调前后的输出

#### 单元格 7：下载权重（可选）
- 打包 LoRA 权重
- 下载到本地电脑

### 3.3 训练参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `model_name` | deepseek-ai/deepseek-llm-7b-chat | 基础模型 |
| `num_epochs` | 3 | 训练轮数 |
| `batch_size` | 4 | 批大小 |
| `learning_rate` | 2e-4 | 学习率 |
| `max_length` | 512 | 最大序列长度 |
| `lora_r` | 8 | LoRA rank |
| `lora_alpha` | 16 | LoRA alpha |
| `lora_dropout` | 0.05 | Dropout 比例 |

### 3.4 预期输出

训练完成后，在 Google Drive 的 `Graduation_Project/qlora_output/` 目录下会生成：

```
qlora_output/
├── adapter_model.bin          # LoRA 权重文件（约 30-50 MB）
├── adapter_config.json        # LoRA 配置文件
├── training_info.json         # 训练信息
└── checkpoint-*/              # 训练 checkpoint（可选）
```

---

## 4. 阶段 3：测试与集成

### 4.1 在 Colab 上测试

运行单元格 6，测试微调后的模型：

```python
# 测试案例
test_cases = [
    {
        "instruction": "分析以下财经快讯对市场的影响",
        "input": "美联储宣布加息25个基点"
    }
]

# 查看输出
# 预期：模型会生成专业的财经分析
```

### 4.2 下载权重到本地

**方案 A：通过 Colab 下载**
- 运行单元格 7
- 下载 `qlora_weights.zip`
- 解压到 `models/qlora/adapter/`

**方案 B：通过 Google Drive 下载**
- 打开 Google Drive
- 找到 `Graduation_Project/qlora_output/`
- 下载 `adapter_model.bin` 和 `adapter_config.json`
- 保存到 `models/qlora/adapter/`

### 4.3 本地测试（可选）

```powershell
# 测试微调后的模型
python scripts/qlora/test_qlora_model.py `
    --adapter_path models/qlora/adapter `
    --test_cases "美联储宣布加息25个基点"
```

**注意**：本地测试需要较大内存（约 8GB），如果电脑配置不够，可以跳过此步骤。

### 4.4 集成到 Agent（可选）

如果要在本地 Agent 中使用微调后的模型，需要修改 `app/adapters/llm/deepseek_client.py`：

```python
# 方案 A：使用 Colab 部署的 API（推荐）
# 在 Colab 上使用 ngrok 开启 API 接口
# 本地调用该接口

# 方案 B：本地加载 LoRA 权重（需要大内存）
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-llm-7b-chat",
    load_in_4bit=True,
    device_map='auto'
)
model = PeftModel.from_pretrained(model, "models/qlora/adapter")
```

**推荐**：答辩时使用 Deepseek API，展示 LoRA 权重文件和训练日志即可。

---

## 5. 答辩准备

### 5.1 需要准备的材料

1. **训练日志截图**
   - Loss 曲线
   - 训练参数
   - 训练时间

2. **LoRA 权重文件**
   - `adapter_model.bin`（约 30-50 MB）
   - `adapter_config.json`
   - `training_info.json`

3. **测试结果对比**
   - 微调前的输出
   - 微调后的输出
   - 对比说明

### 5.2 答辩话术

**问题 1：为什么要做 QLoRA 微调？**

> "通用大模型虽然能力强大，但在财经领域的专业性不足。通过 QLoRA 微调，我们可以：
> 1. 让模型学习财经领域的专业术语和分析方法
> 2. 提升模型对市场走势的理解能力
> 3. 使模型输出更符合财经分析师的表达习惯"

**问题 2：为什么使用 QLoRA 而不是完整微调？**

> "QLoRA 是一种参数高效微调方法，相比完整微调有以下优势：
> 1. 显存占用低：4-bit 量化 + LoRA，T4 GPU 即可训练 7B 模型
> 2. 训练速度快：只训练约 0.1% 的参数，训练时间大幅缩短
> 3. 权重文件小：LoRA 权重仅 30-50 MB，便于存储和部署"

**问题 3：为什么答辩时使用 API 而不是本地模型？**

> "考虑到演示流畅度和硬件限制：
> 1. 本地显存不足（MX570 2GB），无法加载 7B 模型
> 2. 使用 API 可以保证推理速度和稳定性
> 3. 我已经完成了微调工作（展示训练日志和权重文件）
> 4. 实际部署时可以在云端服务器上加载微调后的模型"

**问题 4：微调效果如何评估？**

> "我们通过以下方式评估微调效果：
> 1. 定性评估：对比微调前后的输出质量
> 2. 人工评估：邀请财经专业人士评估输出的专业性
> 3. 实际应用：在 Agent 系统中测试，观察用户反馈"

---

## 6. 常见问题

### Q1: 训练过程中断开连接怎么办？

**A**: 训练会自动保存 checkpoint，可以从最近的 checkpoint 继续训练：

```python
# 在训练命令中添加 --resume_from_checkpoint
!python scripts/qlora/train_qlora.py \
    --resume_from_checkpoint /content/drive/MyDrive/Graduation_Project/qlora_output/checkpoint-100 \
    ...
```

### Q2: 显存不足怎么办？

**A**: 减小批大小或序列长度：

```python
# 减小批大小
--batch_size 2

# 减小序列长度
--max_length 256
```

### Q3: 如何评估微调效果？

**A**: 
1. 运行单元格 6，测试模型输出
2. 对比微调前后的输出质量
3. 使用人工评估（邀请财经专业人士）

### Q4: 如何在本地使用微调后的模型？

**A**: 
1. 下载 LoRA 权重文件
2. 使用 PEFT 库加载（参考 `test_qlora_model.py`）
3. 注意：需要较大内存（约 8GB）

### Q5: 训练数据不够怎么办？

**A**: 
1. 增加数据生成脚本中的样本数量
2. 使用 GPT-4 或 Kimi 生成更多指令
3. 从其他数据源（如财经新闻网站）爬取数据

---

## 7. 时间规划

| 阶段 | 任务 | 预计时间 |
|------|------|----------|
| 阶段 1 | 准备指令集数据 | 1-2 小时 |
| 阶段 2 | Colab 训练 | 2-4 小时 |
| 阶段 3 | 测试与集成 | 1 小时 |
| **总计** | | **4-7 小时** |

**建议**：
- 第 1 天：完成阶段 1（数据准备）
- 第 2 天：完成阶段 2（Colab 训练）
- 第 3 天：完成阶段 3（测试与集成）

---

## 8. 下一步

完成 QLoRA 微调后：

1. **更新 Project_Status.md**
   - 记录 QLoRA 微调完成
   - 添加训练参数和结果

2. **准备答辩材料**
   - 截图训练日志
   - 准备对比测试结果
   - 整理答辩话术

3. **可选：集成到 Agent**
   - 在 Colab 上部署 API
   - 本地调用微调后的模型

---

**祝训练顺利！** 🎓
