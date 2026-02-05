# 毕设项目优化实施方案：Engine A 重构与数据增强 (v2.0)

**版本日期：** 2026-02-05
**基于文档：** 金融大模型综述 (Lecture 1-10)
**核心目标：** 解决 Engine A (BERT) 在“利好/利空兑现”逻辑上的缺陷，通过 Input Augmentation (输入增强) 提升多分类 F1 Score。

---

## 1. 核心改进策略概览 (Strategic Overview)

[cite_start]根据 *Lecture 7 & 8* 关于 "Time Series Textualization" (时序文本化) 的理论 [cite: 822, 824, 888]，我们将原本独立的“新闻文本”与“市场行情”在**输入端**进行融合。

* **当前问题：** BERT 仅能看到新闻文本 $T$，无法感知发布前的市场状态 $S_{pre}$。导致 $P(Label | T)$ 在“兑现”类样本上失效。
* **改进方案：** 构建新的输入 $X = f(S_{pre}) + T$。即将发布前的 K 线趋势转化为自然语言前缀（Prefix），强行注入上下文。
* **预期效果：** 模型将学习到模式：`[前期大涨]` + `利好消息` $\rightarrow$ `利好兑现 (Sell)`。

---

## 2. 数据重构方案 (Data Refactoring Plan)

**现状：** 数据分散在 `finance.db` (新闻/日历) 和 `finance_analysis.db` (分钟级 K 线)。时间跨度：2024-12-02 至 2026-01-31。
**结论：** **必须重构**。现有的 `train_multi_labeled.csv` 仅包含结果标签 (Label)，缺少输入的上下文特征 (Features)。

### 2.1 重构目标

生成一个新的训练数据集 `train_enhanced.csv`，其中 `text` 列不再是纯新闻，而是 **"Context-Aware Prompt"**。

### 2.2 详细重构步骤 (Step-by-Step Implementation)

请使用 Python (Pandas + SQLite) 实现以下管线。建议在 `scripts/modeling/data_refactor.py` 中新建脚本。

#### 步骤 1：时间对齐与窗口定义

* **输入：**
  * 新闻事件集合 $E$ (来自 `finance.db`)
  * 分钟级 K 线 $P$ (来自 `finance_analysis.db`，标的如 XAUUSD)
* **逻辑：**
  对于每一个新闻事件 $e_i$ (发生时间 $t_i$)，我们需要截取两个窗口：
  1. **前视窗口 (Lookback Window):** $[t_i - 120min, t_i)$。用于计算 `pre_ret` (前期趋势)。
  2. **后视窗口 (Outcome Window):** $[t_i, t_i + 30min]$。用于计算 `post_ret` (作为 Ground Truth Label)。

#### 步骤 2：趋势特征计算 (Feature Engineering)

对于每个事件，计算以下指标：

* **Pre-Return (`pre_ret`):** `(Price_open_at_t - Price_open_at_t-120m) / Price_open_at_t-120m`
* **Volatility (`volatility`):** 前 120 分钟内的高低价差 `(High - Low) / Open`。

#### 步骤 3：文本化映射 (Textualization Logic)

这是 Input Augmentation 的核心。定义映射函数 $f(pre\_ret, vol)$ 生成前缀：

| 市场状态           | 判定条件 (阈值需根据分布调整)              | 生成前缀 (Prefix) | 含义             |
| :----------------- | :----------------------------------------- | :---------------- | :--------------- |
| **大涨**     | `pre_ret > 0.5%`                         | `[前期大涨]`    | 市场情绪极度乐观 |
| **大跌**     | `pre_ret < -0.5%`                        | `[前期大跌]`    | 市场情绪极度悲观 |
| **剧烈震荡** | `abs(pre_ret) < 0.2%` AND `vol > 0.8%` | `[剧烈震荡]`    | 多空分歧巨大     |
| **阴跌**     | `-0.5% < pre_ret < -0.1%`                | `[弱势下跌]`    | 情绪偏弱         |
| **盘整**     | 其他情况                                   | `[横盘震荡]`    | 情绪平稳         |

#### 步骤 4：构建增强文本 (Prompt Construction)

将前缀与原始新闻标题/内容拼接。建议加入**宏观数据**的特殊处理。

* **普通新闻：**
  * 原：`美联储宣布维持利率不变。`
  * 新：`[前期大涨] 美联储宣布维持利率不变。`
* **宏观数据 (Data Release)：**
  * 原：`美国1月CPI年率录得2.8%，预期2.9%。`
  * 新：`[前期大跌] [通胀数据] 美国1月CPI年率录得2.8%，预期2.9%。`
  * *(注：如果能从 JSON 字段提取 actual/forecast，直接拼在文本里效果更好)*

### 2.3 代码逻辑伪代码 (For Agent)

```python
def generate_enhanced_text(row):
    # 1. 获取前期趋势前缀
    prefix = get_trend_prefix(row['pre_ret'], row['volatility'])
  
    # 2. (可选) 获取宏观数据前缀
    if row['type'] == 'economic_data':
        data_info = f"[公布值{row['actual']} 预测值{row['forecast']}] "
    else:
        data_info = ""
    
    # 3. 拼接
    return f"{prefix} {data_info}{row['content']}"

# 应用到 DataFrame
df['text_input'] = df.apply(generate_enhanced_text, axis=1)
```


## 3. 模型训练升级指南 (Model Training Upgrade)

基于 *Lecture 2 & 6* 的建议 ，我们不仅要改数据，还要升级模型策略。

### 3.1 模型底座替换 (Model Selection)

放弃通用的 `hfl/chinese-roberta-wwm-ext`，改用金融预训练模型：

* **首选：** `Langboat/mengzi-bert-base-fin` (孟子金融 BERT)
* **备选：** `IDEA-CCNL/Erlangshen-RoBERTa-110M-Financial`

### 3.2 损失函数调整 (Loss Function)

解决类别不平衡（Class 3/4 样本极少）的关键。在 `Trainer` 中重写 `compute_loss`：

* 引入 **Focal Loss** 或  **Weighted CrossEntropy** 。
* **权重设置：** 给予 Class 3 (利好兑现) 和 Class 4 (利空兑现) 更高的权重（例如 5.0），给予 Class 0 (中性) 较低权重（例如 0.5）。

### 3.3 数据增强 (Synthetic Data Augmentation)

利用 LLM API 生成“合成样本”以补充稀缺类别。

* **Action:** 编写脚本，调用 Gemini/GPT，生成 200 条“利好兑现”和“利空兑现”的文本样本，混入训练集。
* **Prompt 示例:** *"请生成 5 条财经快讯，内容本身是利好的（如降息、业绩大增），但语境设定为市场前期已经透支了预期（Priced-in），因此属于'利好兑现'。请直接返回 JSON 格式。"*

---

## 4. 系统集成接口 (Integration with Agent)

当模型训练完成后，Engine A 将作为一个 **Tool** 被 Engine B (Agent) 调用。

* **工具名称：** `MarketSentimentAnalyzer`
* **输入参数：** `news_text` (当前新闻), `current_trend` (当前K线计算出的趋势前缀)
* **工作流：**
  1. Agent 收到用户请求：“现在的非农数据怎么看？”
  2. Agent 调用 `fetch_prices` 获取最近 2 小时 XAUUSD 数据。
  3. Agent 计算得出趋势为“大涨”。
  4. Agent 构造文本 `[前期大涨] 非农数据公布...`。
  5. Agent 调用 Engine A (BERT) 进行预测。
  6. Engine A 返回 `Label: 4 (利空兑现)`。
  7. Agent 输出：“虽然数据看似不错，但考虑到前期已经大涨，模型判断可能出现利好兑现，建议谨慎追多。”

---

## 5. 立即行动清单 (Action Items)

1. [ ] **新建脚本** `scripts/data_processor/build_enhanced_dataset.py`。
2. [ ] **实现逻辑** ：连接 SQLite，按上述“步骤 1-4”清洗 2024.12-2026.01 的数据。
3. [ ] **生成文件** ：`data/processed/train_enhanced_v1.csv`。
4. [ ] **下载权重** ：从 HuggingFace 获取孟子金融 BERT。
5. [ ] **修改训练代码** ：在 `bert_finetune_cls.py` 中加入 Focal Loss 类。
