# Core Layer（核心层）

核心层包含 Agent 系统的核心业务逻辑，不依赖任何外部框架（如 FastAPI、Streamlit）。

## 目录结构

```
app/core/
├── dto.py              # 数据传输对象（DTO）定义
├── engines/            # 分析引擎
│   ├── sentiment_engine.py  # Engine A：情感分析引擎
│   └── rag_engine.py        # Engine B：RAG 检索引擎
├── rules/              # 规则引擎
│   └── rule_engine.py       # 后处理规则引擎
└── orchestrator/       # Agent 编排器
    ├── agent.py             # Agent 核心逻辑
    └── tools.py             # 工具函数集合
```

## DTO 数据结构

### 1. MarketContext（市场上下文）

表示新闻发布前的市场状态，用于情感分析时提供背景信息。

```python
from app.core.dto import MarketContext
from datetime import datetime

context = MarketContext(
    ticker="XAUUSD",
    window_pre_minutes=120,
    pre_ret=0.015,           # 前期涨幅 1.5%
    volatility=0.008,        # 波动率 0.8%
    trend_tag="Strong Rally",
    price_start=2650.0,
    price_end=2690.0,
    ts_start=datetime(2026, 2, 7, 10, 0),
    ts_end=datetime(2026, 2, 7, 12, 0)
)
```

### 2. NewsItem（新闻条目）

表示一条财经快讯或日历事件。

```python
from app.core.dto import NewsItem
from datetime import datetime

news = NewsItem(
    ts=datetime(2026, 2, 7, 12, 0),
    source="jin10_flash",
    content="美联储宣布维持利率不变",
    star=5,
    country="美国"
)
```

### 3. SentimentResult（情感分析结果）

Engine A（BERT + 规则引擎）的输出。

```python
from app.core.dto import SentimentResult

sentiment = SentimentResult(
    label=1,                 # 1=利好, 0=中性, -1=利空
    score=0.85,              # 置信度 85%
    explain="预测为利好，置信度 85%",
    base_label=1,
    base_score=0.85,
    rule_triggered=None,     # 未触发规则
    probs=[0.05, 0.10, 0.85] # [bearish, neutral, bullish]
)
```

### 4. Citation（引用片段）

Engine B（RAG）的检索结果。

```python
from app.core.dto import Citation

citation = Citation(
    text="2023年公司实现营业收入1234.56亿元，同比增长15.2%。",
    source_file="600519_2023Q4.pdf",
    page_idx=3,              # 第 4 页（从 0 开始）
    score=0.92               # 相似度 92%
)
```

### 5. ToolTraceItem（工具调用追踪）

记录 Agent 执行过程中每个工具的调用情况。

```python
from app.core.dto import ToolTraceItem

trace = ToolTraceItem(
    name="get_market_context",
    elapsed_ms=120,          # 耗时 120ms
    ok=True,
    input_summary="ticker=XAUUSD, window=120min",
    output_summary="pre_ret=1.5%, volatility=0.8%"
)
```

### 6. AgentAnswer（Agent 最终答案）

包含分析结果、工具追踪、警告信息等。

```python
from app.core.dto import AgentAnswer, SentimentResult, ToolTraceItem
from datetime import datetime

answer = AgentAnswer(
    summary="虽然消息利好，但前期已大涨1.5%，可能是利好预期兑现，建议谨慎追多。",
    sentiment=SentimentResult(
        label=1,
        score=0.85,
        explain="预测为利好，置信度 85%"
    ),
    citations=[],
    warnings=["前期涨幅较大，注意回调风险"],
    tool_trace=[
        ToolTraceItem(
            name="get_market_context",
            elapsed_ms=120,
            ok=True
        )
    ],
    query="美联储宣布维持利率不变",
    query_type="news_analysis",
    ts=datetime(2026, 2, 7, 12, 0, 30)
)
```

### 7. EngineConfig（引擎配置）

用于初始化各个引擎的配置参数。

```python
from app.core.dto import EngineConfig

config = EngineConfig(
    bert_model_path="models/bert_3cls/best",
    bert_max_length=384,
    bert_device="cpu",
    chroma_path="data/reports/chroma_db",
    embedding_model="BAAI/bge-m3",
    rag_top_k=5,
    priced_in_threshold=0.01,
    high_volatility_threshold=0.015,
    low_net_change_threshold=0.002,
    llm_api_key="sk-test123",
    llm_timeout=10.0,
    llm_max_tokens=500,
    db_path="finance_analysis.db",
    log_level="INFO"
)
```

## 辅助函数

### sentiment_label_to_text()

将情感标签转换为中文文本。

```python
from app.core.dto import sentiment_label_to_text

print(sentiment_label_to_text(1))   # "利好"
print(sentiment_label_to_text(0))   # "中性"
print(sentiment_label_to_text(-1))  # "利空"
```

### sentiment_label_to_english()

将情感标签转换为英文文本。

```python
from app.core.dto import sentiment_label_to_english

print(sentiment_label_to_english(1))   # "bullish"
print(sentiment_label_to_english(0))   # "neutral"
print(sentiment_label_to_english(-1))  # "bearish"
```

## 设计原则

1. **类型安全**：所有字段都有明确的类型注解
2. **不可变性**：使用 dataclass，鼓励不可变数据
3. **合理默认值**：可选字段提供合理的默认值
4. **清晰文档**：每个类和字段都有详细的中文注释
5. **易于测试**：简单的数据结构，易于创建测试数据

## 测试

运行 DTO 测试：

```bash
python app/core/dto.py
```

预期输出：

```
================================================================================
DTO 数据结构测试
================================================================================

1. MarketContext
标的: XAUUSD
前期收益率: 1.50%
趋势标签: Strong Rally

2. NewsItem
来源: jin10_flash
内容: 美联储宣布维持利率不变
重要性: 5星

...

================================================================================
所有 DTO 测试通过！
================================================================================
```

## 下一步

- [ ] 实现 Engine A 推理包装器（`engines/sentiment_engine.py`）
- [ ] 实现规则引擎（`rules/rule_engine.py`）
- [ ] 实现 Engine B RAG 引擎（`engines/rag_engine.py`）
- [ ] 实现 Agent 编排器（`orchestrator/agent.py`）
- [ ] 实现工具函数集合（`orchestrator/tools.py`）
