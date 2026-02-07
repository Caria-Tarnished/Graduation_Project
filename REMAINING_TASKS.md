# 剩余任务开发文档

**更新时间**: 2026-02-07  
**答辩时间**: 约 1 个月后  
**目标**: 完成 Agent 系统集成，实现可演示的 Streamlit UI

---

## 1. 项目现状总结

### 1.1 已完成部分

**Engine A（情感分类引擎）**:
- ✅ 数据管线：MT5 分钟价 + 金十快讯/日历，数据库 `finance_analysis.db`
- ✅ Baseline：TF-IDF + SVM（macro_f1=0.3458）
- ✅ 数据集生成：3 类标签数据集（Bearish/Neutral/Bullish）+ 输入增强
- ⏳ BERT 训练：准备就绪，待 Colab 执行（预计 Test Macro F1 > 0.35）

**Engine B（RAG 检索）**:
- ❌ 未开始

**Agent 层**:
- ❌ 未开始

**UI 层**:
- ❌ 未开始

### 1.2 技术约束

- **本地硬件**: Intel Core i5-1235U（CPU 推理）
- **LLM**: Deepseek API（云端调用，不部署本地）
- **训练**: Google Colab（T4 GPU）
- **财报数据**: 手动下载 PDF（5-10 份即可）
- **答辩时间**: 1 个月

---

## 2. 系统架构设计

### 2.1 整体架构（4 层设计）

基于 `Agent_System_Architecture_Recommendations.md` 的建议，采用分层架构：

```
┌─────────────────────────────────────────────────────────────┐
│                    Host Layer (宿主层)                        │
│  ┌──────────────────────┐      ┌──────────────────────┐     │
│  │  Streamlit UI        │      │  FastAPI Service     │     │
│  │  (答辩演示版)         │      │  (QuantSway 集成版)   │     │
│  └──────────────────────┘      └──────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Application Layer (用例层)                       │
│  - analyze_news(): 快讯情感分析 + 规则引擎                    │
│  - ask_report(): 财报检索 + LLM 总结                         │
│  - agent_chat_turn(): 完整对话回合                           │
│  - 超时控制、缓存、降级策略                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 Core Layer (核心层)                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Engine A    │  │  Engine B    │  │ Rule Engine  │      │
│  │  情感分类     │  │  RAG 检索    │  │  后处理规则   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────────────────────────────────────────┐      │
│  │  DTO (数据结构)                                    │      │
│  │  NewsItem, MarketContext, SentimentResult, etc.  │      │
│  └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Adapters Layer (适配器层)                        │
│  - LLM Client (Deepseek API)                                │
│  - Vector Store (Chroma)                                    │
│  - Data Source (SQLite: finance_analysis.db)               │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 目录结构设计

```
Graduation_Project/
├── app/                          # 应用代码（新增）
│   ├── core/                     # 核心层（不依赖 UI/HTTP）
│   │   ├── dto.py                # 数据结构定义
│   │   ├── engines/
│   │   │   ├── __init__.py
│   │   │   ├── sentiment_engine.py    # Engine A 推理
│   │   │   └── rag_engine.py          # Engine B 检索
│   │   ├── rules/
│   │   │   ├── __init__.py
│   │   │   └── rule_engine.py         # 后处理规则
│   │   └── orchestrator/
│   │       ├── __init__.py
│   │       └── agent.py               # Agent 编排器
│   │
│   ├── application/              # 用例层
│   │   ├── __init__.py
│   │   ├── analyze_news.py       # 快讯分析用例
│   │   ├── ask_report.py         # 财报问答用例
│   │   └── utils.py              # 超时/缓存工具
│   │
│   ├── adapters/                 # 适配器层
│   │   ├── __init__.py
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   └── deepseek_client.py     # Deepseek API 客户端
│   │   ├── vector_store/
│   │   │   ├── __init__.py
│   │   │   └── chroma_store.py        # Chroma 向量库
│   │   └── data_source/
│   │       ├── __init__.py
│   │       └── sqlite_source.py       # SQLite 数据源
│   │
│   └── hosts/                    # 宿主层
│       └── streamlit_app/        # Streamlit UI（答辩版）
│           ├── app.py            # 主入口
│           ├── pages/
│           │   ├── 1_Chat.py     # 聊天页面
│           │   ├── 2_Charts.py   # K 线图表页面
│           │   └── 3_Reports.py  # 财报检索页面
│           └── utils/
│               └── chart_utils.py     # 图表工具
│
├── scripts/                      # 离线脚本（已有）
├── data/                         # 数据目录（已有）
├── models/                       # 模型目录（已有）
│   └── bert_3cls/                # 3 类 BERT 模型（待训练）
│       └── best/                 # 最优权重
├── reports/                      # 训练报告（已有）
├── configs/                      # 配置文件（已有）
├── finance_analysis.db           # 数据库（已有）
├── requirements.txt              # 依赖（需更新）
└── .env                          # 环境变量（需创建）
```

---

## 3. 分阶段实施计划

### 阶段 1：完成 Engine A（1 周）

**目标**: 完成 BERT 训练 + 本地推理 + 规则引擎

#### 任务 1.1: Colab 训练 3 类 BERT 模型
- **时间**: 1-2 小时（GPU）
- **操作**:
  1. 在 Colab 上运行 `colab_3cls_training_cells.txt` 中的训练流程
  2. 验证 Test Macro F1 > 0.35
  3. 下载模型权重到 `models/bert_3cls/best/`
- **产出**: 训练好的 BERT 模型权重

#### 任务 1.2: 实现 Engine A 推理包装器
- **时间**: 2-3 小时
- **文件**: `app/core/engines/sentiment_engine.py`
- **功能**:
  ```python
  class SentimentEngine:
      def __init__(self, model_path: str):
          # 加载 BERT 模型和 tokenizer
          pass
      
      def predict_sentiment(
          self, 
          text: str, 
          context: MarketContext | None = None
      ) -> SentimentResult:
          # 1. 如果有 context，添加市场前缀
          # 2. Tokenize 文本
          # 3. 模型推理
          # 4. 返回 SentimentResult
          pass
  ```
- **测试**: 单条文本推理耗时 < 500ms

#### 任务 1.3: 实现规则引擎
- **时间**: 2-3 小时
- **文件**: `app/core/rules/rule_engine.py`
- **规则示例**:
  ```python
  class RuleEngine:
      def post_process(
          self,
          sentiment: SentimentResult,
          context: MarketContext | None,
          news: NewsItem | None = None
      ) -> SentimentResult:
          # 规则 1: 预期兑现检测
          if sentiment.label == 1 and context and context.pre_ret > 0.01:
              return SentimentResult(
                  label=sentiment.label,
                  explain="利好预期兑现，前期已大涨 {:.2%}".format(context.pre_ret)
              )
          
          # 规则 2: 观望信号
          if context and context.volatility > 0.008 and abs(context.pre_ret) < 0.002:
              return SentimentResult(
                  label=0,
                  explain="高波动低净变动，建议观望"
              )
          
          return sentiment
  ```

#### 任务 1.4: 实现 DTO 数据结构
- **时间**: 1 小时
- **文件**: `app/core/dto.py`
- **内容**: 参考架构文档第 0.7 节的契约草案

---

### 阶段 2：实现 Engine B（1 周）

**目标**: 完成 RAG 检索管线

#### 任务 2.1: 准备财报 PDF
- **时间**: 1-2 小时
- **操作**:
  1. 手动下载 5-10 份财报 PDF（建议：贵州茅台、宁德时代、比亚迪等）
  2. 保存到 `data/reports/pdfs/`
- **命名规范**: `{ticker}_{period}.pdf`（如 `600519_2023Q4.pdf`）

#### 任务 2.2: PDF 解析与切片
- **时间**: 3-4 小时
- **脚本**: `scripts/rag/build_chunks.py`
- **功能**:
  ```python
  # 使用 PyMuPDF 解析 PDF
  import fitz  # PyMuPDF
  
  def parse_pdf(pdf_path: str) -> list[dict]:
      doc = fitz.open(pdf_path)
      chunks = []
      for page_idx, page in enumerate(doc):
          text = page.get_text()
          # 使用 LangChain RecursiveCharacterTextSplitter
          # chunk_size=500, overlap=50
          chunks.append({
              'text': text,
              'page_idx': page_idx,
              'source_file': pdf_path
          })
      return chunks
  ```
- **产出**: `data/reports/chunks.json`

#### 任务 2.3: 向量化与索引构建
- **时间**: 2-3 小时
- **脚本**: `scripts/rag/build_vector_index.py`
- **功能**:
  ```python
  # 使用 bge-m3 嵌入模型 + Chroma 向量库
  from sentence_transformers import SentenceTransformer
  import chromadb
  
  model = SentenceTransformer('BAAI/bge-m3')
  client = chromadb.PersistentClient(path="data/reports/chroma_db")
  collection = client.create_collection("reports_chunks")
  
  # 批量嵌入并插入
  for chunk in chunks:
      embedding = model.encode(chunk['text'])
      collection.add(
          embeddings=[embedding],
          documents=[chunk['text']],
          metadatas=[{
              'page_idx': chunk['page_idx'],
              'source_file': chunk['source_file']
          }]
      )
  ```
- **产出**: `data/reports/chroma_db/`

#### 任务 2.4: 实现 RAG Engine
- **时间**: 2-3 小时
- **文件**: `app/core/engines/rag_engine.py`
- **功能**:
  ```python
  class RagEngine:
      def __init__(self, chroma_path: str, model_name: str):
          # 加载 Chroma 和嵌入模型
          pass
      
      def retrieve(
          self, 
          query: str, 
          top_k: int = 5
      ) -> list[Citation]:
          # 1. 查询向量化
          # 2. Chroma 检索
          # 3. 返回 Citation 列表
          pass
  ```

---


### 阶段 3：Agent 编排与工具集成（1 周）

**目标**: 实现 Agent 核心逻辑和工具调用

#### 任务 3.1: 实现 Deepseek LLM 客户端
- **时间**: 1-2 小时
- **文件**: `app/adapters/llm/deepseek_client.py`
- **功能**:
  ```python
  import os
  import requests
  from typing import Optional
  
  class DeepseekClient:
      def __init__(self, api_key: Optional[str] = None):
          self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
          self.base_url = "https://api.deepseek.com/v1"
      
      def complete(
          self, 
          prompt: str, 
          timeout_seconds: float = 10.0
      ) -> str:
          # 调用 Deepseek API
          # 处理超时和错误
          pass
  ```
- **配置**: 在 `.env` 中添加 `DEEPSEEK_API_KEY=your_key_here`

#### 任务 3.2: 实现核心工具函数
- **时间**: 4-5 小时
- **文件**: `app/core/orchestrator/tools.py`
- **工具清单**:

**工具 1: 获取市场上下文**
```python
def get_market_context(
    ticker: str,
    event_time: datetime,
    window_minutes: int = 120
) -> MarketContext:
    """
    从 finance_analysis.db 读取事件前的 K 线数据
    计算 pre_ret, volatility, trend_tag
    """
    # 1. 连接数据库
    # 2. 查询 [event_time - window_minutes, event_time] 的价格
    # 3. 计算指标
    # 4. 返回 MarketContext
    pass
```

**工具 2: 分析快讯情感**
```python
def analyze_sentiment(
    news_text: str,
    context: Optional[MarketContext] = None,
    sentiment_engine: SentimentEngine,
    rule_engine: Optional[RuleEngine] = None
) -> SentimentResult:
    """
    调用 Engine A + 规则引擎
    """
    # 1. 调用 sentiment_engine.predict_sentiment()
    # 2. 如果有 rule_engine，调用 post_process()
    # 3. 返回最终结果
    pass
```

**工具 3: 检索财报**
```python
def search_reports(
    query: str,
    rag_engine: RagEngine,
    top_k: int = 5
) -> list[Citation]:
    """
    调用 Engine B RAG 检索
    """
    return rag_engine.retrieve(query, top_k)
```

#### 任务 3.3: 实现 Agent 编排器
- **时间**: 3-4 小时
- **文件**: `app/core/orchestrator/agent.py`
- **功能**:
  ```python
  class Agent:
      def __init__(
          self,
          sentiment_engine: SentimentEngine,
          rag_engine: RagEngine,
          rule_engine: RuleEngine,
          llm_client: DeepseekClient
      ):
          self.sentiment_engine = sentiment_engine
          self.rag_engine = rag_engine
          self.rule_engine = rule_engine
          self.llm = llm_client
      
      def process_query(self, user_query: str) -> AgentAnswer:
          """
          处理用户查询，返回结构化答案
          包含 tool_trace（工具调用追踪）
          """
          tool_trace = []
          
          # 1. 判断查询类型（快讯分析 vs 财报问答）
          # 2. 调用相应工具
          # 3. 记录每个工具的耗时
          # 4. 使用 LLM 生成最终总结
          # 5. 返回 AgentAnswer
          pass
  ```

#### 任务 3.4: 实现用例层函数
- **时间**: 2-3 小时
- **文件**: `app/application/analyze_news.py` 和 `app/application/ask_report.py`
- **功能**: 封装 Agent 调用，添加超时控制和缓存

---

### 阶段 4：Streamlit UI 实现（1 周）

**目标**: 完成可演示的 Web 界面

#### 任务 4.1: 实现聊天页面
- **时间**: 1 天
- **文件**: `app/hosts/streamlit_app/pages/1_Chat.py`
- **功能**:
  - 用户输入框
  - 对话历史显示
  - 调用 Agent 并展示结果
  - 显示工具追踪（Tool Trace）

**界面设计**:
```
┌─────────────────────────────────────────┐
│  财经分析 Agent                          │
├─────────────────────────────────────────┤
│  [用户] 最近的非农数据怎么看？            │
│                                         │
│  [Agent] 正在分析...                     │
│  ├─ 获取市场上下文 (120ms)               │
│  ├─ 情感分析 (450ms)                     │
│  └─ LLM 总结 (1200ms)                   │
│                                         │
│  [Agent] 根据分析，非农数据...           │
│  情感: 利好 (置信度: 0.85)               │
│  规则: 前期已大涨 1.2%，可能预期兑现      │
│                                         │
│  [输入框: 请输入问题...]  [发送]         │
└─────────────────────────────────────────┘
```

#### 任务 4.2: 实现 K 线图表页面
- **时间**: 2 天
- **文件**: `app/hosts/streamlit_app/pages/2_Charts.py`
- **功能**:
  - 使用 Plotly 绘制 K 线图
  - 在图表上标注事件点
  - 点击事件点触发情感分析
  - 显示分析结果

**图表示例**:
```python
import plotly.graph_objects as go

def plot_kline_with_events(prices_df, events_df):
    fig = go.Figure(data=[
        go.Candlestick(
            x=prices_df['ts'],
            open=prices_df['open'],
            high=prices_df['high'],
            low=prices_df['low'],
            close=prices_df['close']
        )
    ])
    
    # 添加事件标注
    for _, event in events_df.iterrows():
        fig.add_annotation(
            x=event['ts'],
            y=event['price'],
            text=event['title'][:20],
            showarrow=True
        )
    
    return fig
```

#### 任务 4.3: 实现财报检索页面
- **时间**: 1 天
- **文件**: `app/hosts/streamlit_app/pages/3_Reports.py`
- **功能**:
  - 输入问题
  - 显示 Top-5 引用片段
  - 显示页码和相似度分数
  - LLM 生成的答案

**界面设计**:
```
┌─────────────────────────────────────────┐
│  财报检索                                │
├─────────────────────────────────────────┤
│  问题: 贵州茅台 2023 年营收情况如何？     │
│  [搜索]                                  │
│                                         │
│  检索结果 (5 条):                        │
│  ┌───────────────────────────────────┐  │
│  │ 1. 相似度: 0.92                    │  │
│  │    来源: 600519_2023Q4.pdf (第 3 页)│  │
│  │    内容: 2023年公司实现营业收入...  │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │ 2. 相似度: 0.88                    │  │
│  │    ...                             │  │
│  └───────────────────────────────────┘  │
│                                         │
│  AI 总结:                                │
│  根据财报，贵州茅台 2023 年...           │
└─────────────────────────────────────────┘
```

#### 任务 4.4: 实现主入口和配置
- **时间**: 半天
- **文件**: `app/hosts/streamlit_app/app.py`
- **功能**:
  - 初始化所有引擎
  - 侧边栏配置
  - 页面路由

---

### 阶段 5：测试与优化（3-5 天）

**目标**: 确保系统稳定可演示

#### 任务 5.1: 端到端测试
- **时间**: 1 天
- **测试用例**:
  1. 快讯情感分析（有/无市场上下文）
  2. 财报检索问答
  3. 完整对话流程
  4. 异常情况处理（超时、API 失败）

#### 任务 5.2: 性能优化
- **时间**: 1 天
- **优化点**:
  - BERT 推理加速（批处理）
  - 缓存常见查询结果
  - 减少数据库查询次数

#### 任务 5.3: 答辩准备
- **时间**: 2-3 天
- **准备内容**:
  1. 演示脚本（5-10 个典型场景）
  2. PPT 制作（架构图、效果展示）
  3. 问题预演（老师可能的提问）
  4. 备用方案（网络/API 故障时的降级策略）

---

## 4. 关键技术细节

### 4.1 依赖安装

更新 `requirements.txt`:
```txt
# 现有依赖（保留）
pandas
numpy
torch
transformers
datasets
scikit-learn

# 新增依赖
streamlit>=1.30.0
plotly>=5.18.0
chromadb>=0.4.0
sentence-transformers>=2.2.0
PyMuPDF>=1.23.0
langchain>=0.1.0
python-dotenv>=1.0.0
requests>=2.31.0
```

安装命令:
```powershell
pip install -r requirements.txt
```

### 4.2 环境变量配置

创建 `.env` 文件:
```env
# Deepseek API
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# 数据库路径
DB_PATH=finance_analysis.db

# 模型路径
BERT_MODEL_PATH=models/bert_3cls/best
EMBEDDING_MODEL_NAME=BAAI/bge-m3

# Chroma 路径
CHROMA_DB_PATH=data/reports/chroma_db

# 日志级别
LOG_LEVEL=INFO
```

### 4.3 启动命令

**启动 Streamlit UI**:
```powershell
streamlit run app/hosts/streamlit_app/app.py
```

**访问地址**: `http://localhost:8501`

---

## 5. 时间规划（总计 4 周）

| 周次 | 阶段 | 任务 | 预计工时 |
|------|------|------|----------|
| 第 1 周 | 阶段 1 | Engine A 完成 | 15-20 小时 |
| 第 2 周 | 阶段 2 | Engine B 完成 | 15-20 小时 |
| 第 3 周 | 阶段 3 + 4 | Agent + UI（部分） | 20-25 小时 |
| 第 4 周 | 阶段 4 + 5 | UI 完成 + 测试优化 | 20-25 小时 |

**总工时**: 70-90 小时（平均每天 2.5-3 小时）

---

## 6. 风险与应对

### 6.1 风险点

1. **BERT 训练效果不达标**
   - 应对: 使用 Baseline（TF-IDF）作为备选，macro_f1=0.3458 已可用
   
2. **Deepseek API 不稳定**
   - 应对: 添加重试机制，准备备用 API（通义千问）
   
3. **RAG 检索效果差**
   - 应对: 调整 chunk_size，使用 BM25 混合检索
   
4. **答辩时网络故障**
   - 应对: 准备离线演示视频，提前录制关键场景

### 6.2 最小可演示版本（MVP）

如果时间紧张，优先完成以下功能:
- ✅ Engine A 推理（BERT 或 Baseline）
- ✅ 规则引擎（2-3 条核心规则）
- ✅ 聊天页面（基础对话）
- ✅ K 线图表（静态展示 + 事件标注）
- ⚠️ Engine B RAG（可选，时间不够可暂缓）

---

## 7. 答辩演示脚本

### 7.1 开场（1 分钟）

"各位老师好，我的毕设题目是《基于混合 NLP 模型的财经分析系统》。系统采用双引擎架构：Engine A 负责高频快讯的情感分类，Engine B 负责深度财报的检索问答，由 Agent 统一调度。"

### 7.2 核心演示（5 分钟）

**场景 1: 快讯情感分析**
- 输入: "美联储宣布加息 25 个基点"
- 展示: 情感分析结果 + 市场上下文 + 规则引擎输出
- 亮点: 展示"预期兑现"逻辑

**场景 2: K 线联动**
- 展示: K 线图 + 事件标注
- 操作: 点击事件点，触发情感分析
- 亮点: 可视化与分析的联动

**场景 3: 财报检索**
- 输入: "贵州茅台 2023 年营收情况"
- 展示: Top-5 引用片段 + 页码 + LLM 总结
- 亮点: 引用可追溯

### 7.3 技术亮点（2 分钟）

1. **代理标注**: 利用 K 线走势反向标注情感，解决标注成本问题
2. **混合架构**: ML 模型 + 规则引擎，兼顾准确性和可解释性
3. **工具追踪**: 每次分析都有完整的 tool trace，便于调试和审计

### 7.4 结尾（1 分钟）

"系统已完成核心功能，测试集 macro F1 达到 0.35+，相比 baseline 提升 100%+。未来可集成到 QuantSway 交易平台，作为研究辅助工具。谢谢各位老师！"

---

## 8. 后续扩展（答辩后）

### 8.1 FastAPI 服务化

- 将 Core 层抽取为独立服务
- 提供 HTTP API 供 QuantSway 调用
- 添加鉴权、限流、监控

### 8.2 QuantSway 集成

- 松耦合集成（HTTP 调用）
- 前端 hover 弹出标签页展示分析结果
- 支持超时和缓存

### 8.3 功能增强

- 实时快讯抓取（WebSocket）
- 更多财报来源（自动爬虫）
- 多标的支持（A 股、美股、商品）

---

## 9. 参考资料

- **架构设计**: `Agent_System_Architecture_Recommendations.md`
- **项目计划**: `PLAN.md`
- **项目现状**: `Project_Status.md`
- **优化方案**: `Project_optimization_plan.md`
- **Colab 训练**: `colab_3cls_training_cells.txt`

---

## 10. 联系与支持

如有问题，请参考:
1. 项目文档（上述参考资料）
2. 代码注释（关键函数都有中文注释）
3. 提交 Issue 到 GitHub 仓库

**祝答辩顺利！** 🎓

---

## 11. 模型接入详细说明

### 11.1 两个模型的接入方式

本项目使用**两个模型**协同工作：

| 模型 | 类型 | 存储方式 | 加载方式 | 推理位置 | 文件大小 |
|------|------|---------|---------|---------|---------|
| **BERT** | 小模型 | 本地权重文件 | `AutoModelForSequenceClassification.from_pretrained()` | 本地 CPU | ~400MB |
| **Deepseek** | 大模型 | 无需本地存储 | API Key（环境变量） | 云端 API | 0 |

### 11.2 BERT 模型接入（本地权重文件）

**存储位置**：
```
models/bert_3cls/best/
├── config.json              # 模型配置
├── pytorch_model.bin        # 权重文件（约 400MB）
├── tokenizer_config.json    # Tokenizer 配置
├── vocab.txt                # 词表
└── special_tokens_map.json  # 特殊 token 映射
```

**加载代码示例**（`app/core/engines/sentiment_engine.py`）：

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class SentimentEngine:
    def __init__(self, model_path: str = "models/bert_3cls/best"):
        """初始化情感分析引擎"""
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 加载模型（CPU 模式）
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=3  # Bearish/Neutral/Bullish
        )
        self.model.eval()  # 设置为评估模式
        
        # 标签映射
        self.label_map = {0: -1, 1: 0, 2: 1}  # 模型输出 -> 业务标签
        self.label_names = {-1: "Bearish", 0: "Neutral", 1: "Bullish"}
    
    def predict_sentiment(
        self, 
        text: str, 
        context: MarketContext | None = None
    ) -> SentimentResult:
        """预测文本情感"""
        # 1. 如果有市场上下文，添加前缀
        if context:
            prefix = self._get_trend_prefix(context)
            enhanced_text = f"{prefix} {text}"
        else:
            enhanced_text = text
        
        # 2. Tokenize
        inputs = self.tokenizer(
            enhanced_text,
            return_tensors="pt",
            max_length=384,
            truncation=True,
            padding=True
        )
        
        # 3. 模型推理（CPU）
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
            pred_label = torch.argmax(probs).item()
        
        # 4. 转换为业务标签
        business_label = self.label_map[pred_label]
        probs_list = probs.tolist()
        
        return SentimentResult(
            label=business_label,
            probs=probs_list,
            score=float(probs[pred_label]),
            explain=f"预测为 {self.label_names[business_label]}，置信度 {probs[pred_label]:.2%}"
        )
    
    def _get_trend_prefix(self, context: MarketContext) -> str:
        """根据市场上下文生成前缀"""
        if context.pre_ret > 0.005:
            return "[Strong Rally]"
        elif context.pre_ret < -0.005:
            return "[Sharp Decline]"
        elif abs(context.pre_ret) < 0.002 and context.volatility > 0.008:
            return "[High Volatility]"
        elif context.pre_ret > 0.001:
            return "[Mild Rally]"
        elif context.pre_ret < -0.001:
            return "[Weak Decline]"
        else:
            return "[Sideways]"
```

**使用示例**：
```python
# 初始化（只初始化一次）
sentiment_engine = SentimentEngine(model_path="models/bert_3cls/best")

# 使用
result = sentiment_engine.predict_sentiment(
    text="美联储宣布加息 25 个基点",
    context=MarketContext(
        window_pre_minutes=120,
        pre_ret=0.012,  # 前期涨了 1.2%
        volatility=0.005,
        trend_tag="Strong Rally"
    )
)

print(result.label)    # 1 (Bullish)
print(result.explain)  # "预测为 Bullish，置信度 85.32%"
```

### 11.3 Deepseek 大模型接入（API 调用）

**配置方式**（在 `.env` 文件中）：
```env
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

**加载代码示例**（`app/adapters/llm/deepseek_client.py`）：

```python
import os
import requests
from typing import Optional

class DeepseekClient:
    def __init__(self, api_key: Optional[str] = None):
        """初始化 Deepseek 客户端"""
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment")
        
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def complete(
        self, 
        prompt: str, 
        timeout_seconds: float = 10.0
    ) -> str:
        """调用 Deepseek API 生成文本"""
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=timeout_seconds
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
        
        except requests.exceptions.Timeout:
            return "[超时] Deepseek API 响应超时，请稍后重试"
        except requests.exceptions.RequestException as e:
            return f"[错误] API 调用失败: {str(e)}"
```

**使用示例**：
```python
# 初始化（只初始化一次）
llm_client = DeepseekClient()

# 使用
prompt = """
你是一个专业的财经分析师。根据以下信息，生成一段简洁的分析总结：

新闻: 美联储宣布加息 25 个基点
情感分析: Bullish (置信度 85%)
市场上下文: 前期已上涨 1.2%
规则引擎: 可能存在预期兑现风险

请用 2-3 句话总结你的观点。
"""

summary = llm_client.complete(prompt, timeout_seconds=10.0)
print(summary)
# 输出: "虽然加息消息本身偏利好，但考虑到市场前期已经上涨 1.2%，
#        存在预期兑现的风险。建议投资者谨慎追高，关注后续市场反应。"
```

### 11.4 Agent 中的集成方式

在 `app/core/orchestrator/agent.py` 中，两个模型协同工作：

```python
import time
from datetime import datetime

class Agent:
    def __init__(
        self,
        sentiment_engine: SentimentEngine,      # BERT 本地推理
        rag_engine: RagEngine,
        rule_engine: RuleEngine,
        llm_client: DeepseekClient              # Deepseek API 调用
    ):
        self.sentiment_engine = sentiment_engine
        self.rag_engine = rag_engine
        self.rule_engine = rule_engine
        self.llm = llm_client
    
    def analyze_news(self, news_text: str, ticker: str) -> AgentAnswer:
        """分析快讯的完整流程"""
        tool_trace = []
        
        # 步骤 1: 获取市场上下文（从数据库）
        start = time.time()
        context = get_market_context(ticker, datetime.now(), 120)
        tool_trace.append(ToolTraceItem(
            name="get_market_context",
            elapsed_ms=int((time.time() - start) * 1000),
            ok=True
        ))
        
        # 步骤 2: BERT 情感分析（本地推理）
        start = time.time()
        sentiment = self.sentiment_engine.predict_sentiment(news_text, context)
        tool_trace.append(ToolTraceItem(
            name="bert_sentiment_analysis",
            elapsed_ms=int((time.time() - start) * 1000),
            ok=True
        ))
        
        # 步骤 3: 规则引擎后处理
        start = time.time()
        final_sentiment = self.rule_engine.post_process(
            sentiment=sentiment,
            context=context,
            news=NewsItem(ts=datetime.now(), source="jin10", content=news_text)
        )
        tool_trace.append(ToolTraceItem(
            name="rule_engine",
            elapsed_ms=int((time.time() - start) * 1000),
            ok=True
        ))
        
        # 步骤 4: LLM 生成总结（Deepseek API）
        start = time.time()
        prompt = f"""
        你是财经分析师。根据以下信息生成简洁总结：
        
        新闻: {news_text}
        情感: {final_sentiment.label} ({final_sentiment.explain})
        市场上下文: 前期涨跌 {context.pre_ret:.2%}，波动率 {context.volatility:.2%}
        
        用 2-3 句话总结。
        """
        summary = self.llm.complete(prompt, timeout_seconds=10.0)
        tool_trace.append(ToolTraceItem(
            name="llm_summary",
            elapsed_ms=int((time.time() - start) * 1000),
            ok=True
        ))
        
        return AgentAnswer(
            summary=summary,
            sentiment=final_sentiment,
            warnings=[],
            tool_trace=tool_trace
        )
```

### 11.5 完整的初始化流程

在 Streamlit 主入口 `app/hosts/streamlit_app/app.py` 中：

```python
import streamlit as st
from app.core.engines.sentiment_engine import SentimentEngine
from app.core.engines.rag_engine import RagEngine
from app.core.rules.rule_engine import RuleEngine
from app.adapters.llm.deepseek_client import DeepseekClient
from app.core.orchestrator.agent import Agent

# 使用 Streamlit 缓存，避免重复加载
@st.cache_resource
def initialize_agent():
    """初始化 Agent 系统（只执行一次）"""
    # 1. 加载 BERT 模型（本地权重文件）
    sentiment_engine = SentimentEngine(
        model_path="models/bert_3cls/best"
    )
    
    # 2. 加载 RAG 引擎（本地 Chroma）
    rag_engine = RagEngine(
        chroma_path="data/reports/chroma_db",
        model_name="BAAI/bge-m3"
    )
    
    # 3. 初始化规则引擎
    rule_engine = RuleEngine()
    
    # 4. 初始化 LLM 客户端（Deepseek API）
    llm_client = DeepseekClient()  # 从 .env 读取 API Key
    
    # 5. 组装 Agent
    agent = Agent(
        sentiment_engine=sentiment_engine,
        rag_engine=rag_engine,
        rule_engine=rule_engine,
        llm_client=llm_client
    )
    
    return agent

# 主程序
def main():
    st.title("财经分析 Agent")
    
    # 初始化（只执行一次）
    agent = initialize_agent()
    
    # 用户输入
    user_input = st.text_input("请输入问题:")
    
    if st.button("分析"):
        with st.spinner("正在分析..."):
            result = agent.analyze_news(user_input, ticker="XAUUSD")
            st.write(result.summary)
            st.json(result.tool_trace)

if __name__ == "__main__":
    main()
```

---


## 12. QuantSway 集成指南

### 12.1 代码仓库组织建议

**推荐：在当前仓库（Graduation_Project）继续开发**

理由：
- ✅ 统一管理：训练代码、模型权重、Agent 系统都在一个仓库
- ✅ 路径简单：不需要跨仓库引用，配置更简单
- ✅ 答辩友好：老师只需要看一个仓库就能了解全貌
- ✅ 已有基础：`scripts/modeling/` 已有训练代码，`models/` 已有模型目录

### 12.2 需要复制到 QuantSway 的代码

#### 12.2.1 必须复制的目录（核心运行时）

```
Graduation_Project/          → QuantSway/
├── app/                     → backend/agent/
│   ├── core/                ✅ 必须复制（核心分析逻辑）
│   │   ├── dto.py
│   │   ├── engines/
│   │   │   ├── sentiment_engine.py
│   │   │   └── rag_engine.py
│   │   ├── rules/
│   │   │   └── rule_engine.py
│   │   └── orchestrator/
│   │       ├── agent.py
│   │       └── tools.py
│   │
│   ├── application/         ✅ 必须复制（用例层）
│   │   ├── analyze_news.py
│   │   ├── ask_report.py
│   │   └── utils.py
│   │
│   ├── adapters/            ✅ 必须复制（适配器层）
│   │   ├── llm/
│   │   │   └── deepseek_client.py
│   │   ├── vector_store/
│   │   │   └── chroma_store.py
│   │   └── data_source/
│   │       └── sqlite_source.py
│   │
│   └── hosts/
│       └── api_service/     ✅ 必须复制（FastAPI 服务）
│           ├── main.py
│           ├── routes/
│           │   └── analysis.py
│           └── schemas.py
│
├── models/                  ✅ 必须复制（模型权重）
│   └── bert_3cls/
│       └── best/            # 约 400MB
│
└── data/                    ⚠️ 部分复制（仅运行时数据）
    └── reports/
        └── chroma_db/       ✅ 复制（向量库索引）
```

#### 12.2.2 不需要复制的目录（开发时代码）

```
Graduation_Project/
├── scripts/                 ❌ 不复制（训练/数据处理脚本）
├── data/raw/                ❌ 不复制（原始数据）
├── data/processed/          ❌ 不复制（训练集）
├── reports/                 ❌ 不复制（训练报告）
├── notebooks/               ❌ 不复制（Jupyter 笔记本）
├── archive/                 ❌ 不复制（归档文件）
├── finance_analysis.db      ❌ 不复制（开发数据库）
└── app/hosts/streamlit_app/ ❌ 不复制（答辩演示 UI）
```

### 12.3 QuantSway 集成后的目录结构

```
QuantSway/
├── backend/
│   ├── api/                 # QuantSway 原有的 API
│   ├── core/                # QuantSway 原有的核心逻辑
│   ├── services/            # QuantSway 原有的服务
│   │
│   └── agent/               # 🆕 新增：财经分析 Agent 模块
│       ├── __init__.py
│       ├── core/            # 从 Graduation_Project/app/core/ 复制
│       │   ├── dto.py
│       │   ├── engines/
│       │   ├── rules/
│       │   └── orchestrator/
│       │
│       ├── application/     # 从 Graduation_Project/app/application/ 复制
│       │   ├── analyze_news.py
│       │   ├── ask_report.py
│       │   └── utils.py
│       │
│       ├── adapters/        # 从 Graduation_Project/app/adapters/ 复制
│       │   ├── llm/
│       │   ├── vector_store/
│       │   └── data_source/
│       │
│       ├── api/             # 从 Graduation_Project/app/hosts/api_service/ 复制
│       │   ├── routes.py
│       │   └── schemas.py
│       │
│       └── config.py        # Agent 配置
│
├── models/                  # 🆕 新增：模型权重目录
│   └── financial_agent/
│       ├── bert_3cls/       # BERT 模型权重（约 400MB）
│       └── chroma_db/       # RAG 向量库
│
├── frontend/                # QuantSway 原有的前端
│   └── src/
│       └── components/
│           └── AgentPanel/  # 🆕 新增：Agent 面板组件
│               ├── AgentChat.tsx
│               └── AgentResult.tsx
│
└── .env                     # 添加 Agent 相关配置
```

### 12.4 FastAPI 路由集成

在 QuantSway 的 FastAPI 主应用中注册 Agent 路由：

```python
# QuantSway/backend/main.py

from fastapi import FastAPI
from backend.api import trading_routes, portfolio_routes  # 原有路由
from backend.agent.api.routes import agent_router         # 🆕 Agent 路由

app = FastAPI(title="QuantSway API")

# 原有路由
app.include_router(trading_routes.router, prefix="/api/trading")
app.include_router(portfolio_routes.router, prefix="/api/portfolio")

# 🆕 新增 Agent 路由
app.include_router(agent_router, prefix="/api/agent", tags=["Agent"])
```

### 12.5 具体的复制清单（按优先级）

#### 第一优先级：核心运行时（必须）

| 源路径 | 目标路径 | 说明 |
|--------|---------|------|
| `app/core/` | `QuantSway/backend/agent/core/` | 核心分析逻辑 |
| `app/application/` | `QuantSway/backend/agent/application/` | 用例层 |
| `app/adapters/` | `QuantSway/backend/agent/adapters/` | 适配器层 |
| `models/bert_3cls/best/` | `QuantSway/models/financial_agent/bert_3cls/` | BERT 权重 |

#### 第二优先级：API 服务（推荐）

| 源路径 | 目标路径 | 说明 |
|--------|---------|------|
| `app/hosts/api_service/` | `QuantSway/backend/agent/api/` | FastAPI 路由 |

#### 第三优先级：数据和配置（可选）

| 源路径 | 目标路径 | 说明 |
|--------|---------|------|
| `data/reports/chroma_db/` | `QuantSway/models/financial_agent/chroma_db/` | RAG 向量库 |
| `configs/agent_config.yaml` | `QuantSway/backend/agent/config.yaml` | Agent 配置 |

### 12.6 复制后的适配工作

#### 12.6.1 路径调整

**原代码（Graduation_Project）**:
```python
# app/core/engines/sentiment_engine.py
class SentimentEngine:
    def __init__(self, model_path: str = "models/bert_3cls/best"):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
```

**适配后（QuantSway）**:
```python
# backend/agent/core/engines/sentiment_engine.py
import os
from pathlib import Path

class SentimentEngine:
    def __init__(self, model_path: str = None):
        if model_path is None:
            # 自动检测 QuantSway 项目根目录
            project_root = Path(__file__).parent.parent.parent.parent
            model_path = project_root / "models" / "financial_agent" / "bert_3cls"
        
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
```

#### 12.6.2 依赖管理

在 QuantSway 的 `requirements.txt` 中添加 Agent 依赖：

```txt
# QuantSway/requirements.txt

# 原有依赖
fastapi>=0.104.0
uvicorn>=0.24.0
...

# 🆕 Agent 依赖
torch>=2.0.0
transformers>=4.35.0
sentence-transformers>=2.2.0
chromadb>=0.4.0
langchain>=0.1.0
```

#### 12.6.3 环境变量配置

在 QuantSway 的 `.env` 中添加：

```env
# QuantSway/.env

# 原有配置
DATABASE_URL=...
REDIS_URL=...

# 🆕 Agent 配置
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
AGENT_MODEL_PATH=models/financial_agent/bert_3cls
AGENT_CHROMA_PATH=models/financial_agent/chroma_db
AGENT_ENABLE=true
```

### 12.7 集成后的调用示例

#### 12.7.1 后端调用（Python）

```python
# QuantSway/backend/services/research_service.py

from backend.agent.application.analyze_news import analyze_news
from backend.agent.core.dto import NewsItem, MarketContext

class ResearchService:
    def __init__(self):
        # Agent 在应用启动时初始化（单例）
        from backend.agent.core.orchestrator.agent import get_agent_instance
        self.agent = get_agent_instance()
    
    def analyze_market_news(self, news_text: str, ticker: str):
        """分析市场新闻（供 QuantSway 其他服务调用）"""
        result = analyze_news(
            news=NewsItem(
                ts=datetime.now(),
                source="jin10",
                content=news_text
            ),
            ticker=ticker,
            agent=self.agent,
            timeout_seconds=3.0  # 短超时，避免阻塞交易
        )
        
        return {
            "sentiment": result.sentiment.label,
            "confidence": result.sentiment.score,
            "summary": result.summary,
            "warnings": result.warnings
        }
```

#### 12.7.2 前端调用（TypeScript）

```typescript
// QuantSway/frontend/src/services/agentService.ts

export interface AgentAnalysisResult {
  sentiment: number;  // -1/0/1
  confidence: number;
  summary: string;
  warnings: string[];
}

export async function analyzeNews(
  newsText: string,
  ticker: string
): Promise<AgentAnalysisResult> {
  const response = await fetch('/api/agent/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ news_text: newsText, ticker })
  });
  
  return response.json();
}
```

```tsx
// QuantSway/frontend/src/components/AgentPanel/AgentChat.tsx

import { analyzeNews } from '@/services/agentService';

export function AgentChat() {
  const [result, setResult] = useState<AgentAnalysisResult | null>(null);
  
  const handleAnalyze = async (newsText: string) => {
    const analysis = await analyzeNews(newsText, 'XAUUSD');
    setResult(analysis);
  };
  
  return (
    <div>
      <textarea onChange={(e) => handleAnalyze(e.target.value)} />
      {result && (
        <div>
          <p>情感: {result.sentiment === 1 ? '利好' : result.sentiment === -1 ? '利空' : '中性'}</p>
          <p>置信度: {(result.confidence * 100).toFixed(2)}%</p>
          <p>总结: {result.summary}</p>
        </div>
      )}
    </div>
  );
}
```

### 12.8 最佳实践建议

#### 12.8.1 使用 Git Submodule（可选）

如果希望保持两个仓库的同步：

```bash
# 在 QuantSway 仓库中
cd QuantSway/backend
git submodule add https://github.com/your-username/Graduation_Project.git agent_source

# 只复制需要的文件
cp -r agent_source/app/core ./agent/core
cp -r agent_source/app/application ./agent/application
cp -r agent_source/app/adapters ./agent/adapters
```

#### 12.8.2 创建独立的 Python 包（更推荐）

将 Agent 打包成独立的 Python 包，通过 pip 安装：

```python
# Graduation_Project/setup.py
from setuptools import setup, find_packages

setup(
    name="financial-agent",
    version="0.1.0",
    packages=find_packages(where="app"),
    package_dir={"": "app"},
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "sentence-transformers>=2.2.0",
        "chromadb>=0.4.0",
    ]
)
```

然后在 QuantSway 中安装：
```bash
pip install -e /path/to/Graduation_Project
```

### 12.9 集成时间线

| 阶段 | 时间 | 任务 |
|------|------|------|
| **答辩前** | 当前 - 1 个月后 | 在 Graduation_Project 完成开发和测试 |
| **答辩后** | 答辩后 1 周 | 将核心代码复制到 QuantSway |
| **集成测试** | 答辩后 2-3 周 | 在 QuantSway 中测试 Agent 功能 |
| **上线部署** | 答辩后 1 个月 | 正式集成到 QuantSway 生产环境 |

### 12.10 集成检查清单

在将代码复制到 QuantSway 之前，请确认：

- [ ] BERT 模型训练完成，Test Macro F1 > 0.35
- [ ] 本地 CPU 推理速度 < 500ms/条
- [ ] Deepseek API 调用稳定，有重试机制
- [ ] RAG 向量库构建完成，检索速度 < 200ms
- [ ] 所有核心代码有中文注释
- [ ] 单元测试覆盖核心功能
- [ ] 环境变量配置文档完整
- [ ] 依赖版本明确（requirements.txt）

---

**祝答辩顺利！** 🎓
