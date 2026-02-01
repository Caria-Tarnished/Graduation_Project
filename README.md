# 毕设项目：基于混合NLP模型的财经分析系统设计与实现

# 1. 项目总览 (Project Overview)

**核心理念：** 本项目旨在构建一个智能财经分析Agent，解决传统金融NLP任务中“高频快讯”与“深度财报”难以兼顾的问题。系统采用“双引擎”架构，并由一个经过金融领域微调的大语言模型（LLM）担任“大脑”进行统一调度。

**双引擎架构：**

1. **高频快讯引擎 (Engine A):** 基于微调的 **BERT** 模型，负责对实时财经快讯进行毫秒级的情感分类（利好/利空），并结合K线数据进行相关性验证。
2. **深度财报引擎 (Engine B):** 基于 **RAG (****检索增强生成)** 技术，负责对低频的长篇财报进行深度内容检索、摘要与问答。

**最终产出：** 一个基于 **Streamlit** 的Web交互系统，用户可以通过自然语言与Agent对话，Agent自主调用工具进行分析，并输出文字结论与交互式K线图表。

## 2. 阶段一：多源数据获取与存储 (Data Pipeline)

**目标：** 获取并清洗多模态金融数据，为双引擎准备“弹药”。

| **任务模块**              | **具体内容**                                             | **推荐技术栈**                                      | **产出物**                |
| ------------------------------- | -------------------------------------------------------------- | --------------------------------------------------------- | ------------------------------- |
| **1. K**线数据            | 获取目标标的（如A股个股、黄金、美股）的分钟级或日级OHLCV数据。 | `Tushare`(A股),`yfinance`(美股/大宗),`baostock`     | `market_data.csv`             |
| **2.** **高频快讯** | 爬取金十数据、财联社等网站的7x24小时直播快讯（时间戳+内容）。  | `Requests`,`BeautifulSoup`,`Selenium`(处理动态加载) | `fast_news.csv`               |
| **3.** **深度财报** | 下载上市公司的季度/年度财报（PDF格式）。                       | `巨潮资讯网`爬虫,`PyMuPDF`(PDF解析)                   | PDF文件夹&`reports_text.json` |
| **4.** **数据存储** | 建立本地SQLite数据库，统一管理结构化与非结构化数据。           | `SQLite`,`SQLAlchemy`                                 | `finance.db`                  |

## 3. 阶段二：构建高频快讯引擎 (Engine A - BERT)

目标： 训练一个能“读懂”市场情绪的分类器。

硬件策略： 标注在本地完成，模型训练在云端 (Google Colab) 完成。

| **任务模块**              | **具体内容**                                                                               | **技术细节**                                                                         |
| ------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| **1.** **代理标注** | \*\*(核心创新)\*\*利用K线走势反向标注新闻情感。例如：新闻发布后30分钟内涨幅>0.5%标记为“利好”。 | 使用 `pandas.merge_asof`进行时间戳对齐；设定阈值规则生成Labels。                         |
| **2.** **云端微调** | 在\*\*Google Colab (T4 GPU)\*\*上，使用标注好的数据微调 `bert-base-chinese`模型。              | 利用HuggingFace `Transformers`库进行训练；保存模型权重 `bert_sentiment.pth`(约400MB)。 |
| **3.** **本地推理** | 在本地电脑上加载微调后的模型，进行CPU推理（BERT较轻量，CPU推理延迟<0.5秒）。                     | 编写 `predict_sentiment(text)`函数，供Agent调用。                                        |

## 4. 阶段三：构建深度财报引擎 (Engine B - RAG)

目标： 让系统具备“翻阅”长文档的能力。

硬件策略： 全流程可在本地 CPU 环境下完成（ChromaDB 轻量高效）。

| **任务模块**                | **具体内容**                                  | **技术细节**                                                                 |
| --------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **1.** **文档切片**   | 将PDF文本按语义或字符长度切割成小块(Chunks)。       | 使用 `LangChain`的 `RecursiveCharacterTextSplitter`，设置 `chunk_size=500`。 |
| **2.** **向量化**     | 将文本块转换为向量(Embedding)。                     | 使用 `BAAI/bge-m3`或 `m3e-base`模型(支持CPU运行)。                             |
| **3.** **向量库构建** | 将向量存入本地向量数据库，实现语义检索。            | 使用 `ChromaDB`(推荐，无需安装服务)或 `FAISS`。                                |
| **4.** **检索器封装** | 封装检索函数，输入问题，返回最相关的Top-3文档片段。 | 编写 `retrieve_docs(query)`函数，供Agent调用。                                   |

## 5. 阶段四：Agent“大脑”定制 (LLM Fine-Tuning)

目标： 解决通用大模型不懂“行话”的问题，打造专业财经分析师。

硬件策略： 必须在云端 (Google Colab) 完成，本地显存不够。

| **任务模块**                | **具体内容**                                              | **技术细节**                                                                 |
| --------------------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **1.** **指令集构建** | 构建"Instruction-Input-Output"格式的微调数据集（约200-500条）。 | 数据来源：利用GPT-4或Kimi辅助生成财经问答对；格式：JSONL。                         |
| **2. QLoRA微调**            | (核心创新)使用QLoRA技术在单张T4 GPU (Colab)上微调7B模型。       | 基础模型：`Qwen-7B`或 `Deepseek-7B`；工具：`PEFT`,`TRL`,`bitsandbytes`。 |
| **3.** **权重合并**   | 训练结束后，导出LoRA Adapter权重文件(仅几十MB)。                | 将权重文件下载至本地保存。                                                         |

## 6. 阶段五：系统总成与部署 (System Assembly)

目标： 克服本地算力瓶颈，实现流畅的 Web Demo。

最终方案： 本地UI (Streamlit) + 云端大脑 (API)

| **组件**           | **运行位置** | **实现方案**                                                                                                                                                                                   |
| ------------------------ | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **用户界面(UI)**   | **本地PC**   | 使用 `Streamlit`构建聊天窗口和图表展示页。                                                                                                                                                         |
| **K**线可视化      | **本地PC**   | 使用 `Pyecharts`或 `Plotly`绘制交互式K线图。                                                                                                                                                     |
| **快讯引擎(BERT)** | **本地PC**   | 使用CPU运行BERT模型进行实时分类（轻负载）。                                                                                                                                                          |
| **财报引擎(RAG)**  | **本地PC**   | 使用 `ChromaDB`进行本地检索（轻负载）。                                                                                                                                                            |
| **Agent**大脑(LLM) | **云端API**  | \*\*(关键策略)\*\*本地不加载7B模型。Agent逻辑通过API调用云端大模型（如阿里通义千问API或Deepseek API）。*注：若必须展示微调成果，可在Colab上部署微调后的模型并开启API接口(ngrok)，本地调用该接口。* |

## 7. 附录：针对 MX570 (2G显存) 的具体操作指南

由于本地显存无法支撑大模型训练和推理，请严格执行以下“云+端”分离策略：

1. **训练阶段 (Training Phase):**
   * 所有涉及 **GPU** 的训练任务（BERT微调、LLM QLoRA微调），**必须** 将数据上传至 Google Drive，然后在 **Google Colab** 笔记本中挂载云端硬盘进行训练。
   * 训练完成后，仅将模型权重文件（`<span lang="EN-US">.pth</span>` 或 `<span lang="EN-US">adapter</span>` 文件夹）下载回本地。
2. **演示阶段 (Demo Phase):**
   * **不要尝试在本地运行 Qwen-7B 或 Deepseek-7B**，这会导致电脑死机。
   * **推荐方案：** 在代码中使用 `<span lang="EN-US">LangChain</span>` 的       `<span lang="EN-US">ChatOpenAI</span>` 或 `<span lang="EN-US">ChatTongyi</span>` 类，填入 API Key，直接调用云端商用模型作为 Agent 的推理核心。这是答辩时最稳定、最流畅的方案。
   * **话术准备：** 当老师问及微调工作时，展示你在 Colab 上的训练日志、Loss 曲线以及训练好的 LoRA 权重文件截图，证明你完成了微调工作，只是为了演示流畅度而使用了 API 推理（或者解释为使用了私有云部署）。

## 8. 毕设创新点总结 (用于开题/论文)

1. **架构创新：** 提出了 BERT+RAG 的双引擎架构，有效融合了高频时序数据（快讯）与低频非结构化数据（财报）的分析能力。
2. **数据处理创新：** 引入“基于K线走势的代理标注 (Proxy Labeling)”方法，解决了金融领域缺乏大规模情感标注数据的痛点。
3. **工程实现创新：** 基于 Agent 思想，实现了从数据检索、模型推理到可视化呈现的端到端自动化流程。
4. **模型定制：** 利用 PEFT (QLoRA) 技术对通用大模型进行了金融领域的低资源微调，提升了模型在特定领域的表现。

---

## 9. 分钟级冲击建模脚本（Baseline / BERT）

基于 `finance_analysis.db` 导出的 `train_30m_labeled.csv / val_30m_labeled.csv / test_30m_labeled.csv`，提供两套开箱即用的训练脚本。

### 9.1 基线：TF-IDF + LinearSVC（CPU 即可）

```powershell
python scripts/modeling/baseline_tfidf_svm.py ^
  --train_csv data\processed\train_30m_labeled.csv ^
  --val_csv   data\processed\val_30m_labeled.csv ^
  --test_csv  data\processed\test_30m_labeled.csv ^
  --output_dir models\baseline_tfidf_svm
```

输出：

- tfidf.pkl / model.pkl（可复用推理）
- metrics_val.json / metrics_test.json、report_*.txt
- pred_test.csv（含 event_id 若存在）

可选参数：`--no_prefix` 关闭前缀特征；`--analyzer char|word`、`--ngram_min/max` 等。

### 9.2 BERT 中文微调（RoBERTa-wwm-ext）

安装依赖（首次）：

```powershell
pip install -U transformers datasets accelerate evaluate
# 无 GPU 可安装 CPU 版 torch：
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

训练与评估：

```powershell
python scripts/modeling/bert_finetune_cls.py ^
  --train_csv data\processed\train_30m_labeled.csv ^
  --val_csv   data\processed\val_30m_labeled.csv ^
  --test_csv  data\processed\test_30m_labeled.csv ^
  --output_dir models\bert_xauusd_cls ^
  --epochs 2 --lr 2e-5 --max_length 256
```

输出：

- models/bert_xauusd_cls/best（最优权重）
- metrics_val.json / metrics_test.json、report_test.txt
- pred_test.csv（-1/0/1 标签）

### 9.3 多窗口数据集（可选）

当前 30 分钟窗口数据集仅包含 `window_min=30` 的样本。若希望每个事件形成 4 条样本（5/10/15/30 分钟），可生成“多窗口版”数据集：

```powershell
$code = @'
import sqlite3, pandas as pd, os, json
c = sqlite3.connect("finance_analysis.db")
q = """
select
  ei.event_id, e.source, e.ts_local as event_ts_local, e.ts_utc as event_ts_utc,
  e.country, e.name, e.content, e.star, e.previous, e.consensus, e.actual,
  e.affect, e.detail_url, e.important, e.hot, e.indicator_name, e.unit,
  ei.ticker, ei.window_min, ei.price_event, ei.price_future, ei.delta, ei.ret
from event_impacts ei
join events e on e.event_id = ei.event_id
where ei.ticker='XAUUSD' and ei.window_min in (5,10,15,30)
order by e.ts_local asc, ei.window_min asc
"""
df = pd.read_sql_query(q, c); c.close()

# 文本优先 content，其次 name
df["text"] = df["content"].fillna("").astype(str)
mask = df["text"].str.len()==0
df.loc[mask, "text"] = df.loc[mask, "name"].fillna("").astype(str)

# 时间切分
df["event_ts_local"] = pd.to_datetime(df["event_ts_local"], errors="coerce")
t1, t2, t3 = pd.Timestamp("2025-08-01 00:00:00"), pd.Timestamp("2025-11-01 00:00:00"), pd.Timestamp("2026-02-01 00:00:00")
train = df[df["event_ts_local"] < t1].copy()
val   = df[(df["event_ts_local"] >= t1) & (df["event_ts_local"] < t2)].copy()
test  = df[(df["event_ts_local"] >= t2) & (df["event_ts_local"] < t3)].copy()

# 按窗口在训练集上分别计算阈值（更贴合各窗口分布）
def thresholds_per_window(train_df):
    qmap = {}
    for w, g in train_df.dropna(subset=["ret"]).groupby("window_min"):
        qmap[w] = (
            float(g["ret"].quantile(0.30)),
            float(g["ret"].quantile(0.70)),
        )
    return qmap

q = thresholds_per_window(train)
def lab(row):
    r = row["ret"]
    w = int(row["window_min"])
    lo, hi = q.get(w, (-0.001, 0.001))
    if pd.isna(r): return 0
    return -1 if r <= lo else (1 if r >= hi else 0)

for part in (train, val, test):
    part["label"] = part.apply(lab, axis=1)

keep = [
  "event_id","event_ts_local","event_ts_utc","source","country","name","content","text",
  "star","previous","consensus","actual","affect","detail_url","important","hot","indicator_name","unit",
  "ticker","window_min","price_event","price_future","delta","ret","label"
]
os.makedirs(r"data\processed", exist_ok=True)
train[keep].to_csv(r"data\processed\train_multi_labeled.csv", index=False, encoding="utf-8")
val[keep].to_csv(  r"data\processed\val_multi_labeled.csv",   index=False, encoding="utf-8")
test[keep].to_csv( r"data\processed\test_multi_labeled.csv",  index=False, encoding="utf-8")
print("done")
'@
$code | python -
```

训练脚本依旧可用，若希望模型“感知窗口”，建议保留 `window_min`（可编码为前缀 token 或数值特征）。
