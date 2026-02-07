# -*- coding: utf-8 -*-
"""
数据传输对象（DTO）定义

本模块定义了 Agent 系统中各层之间传递的数据结构。
遵循"契约优先"原则，确保各层之间的接口清晰、类型安全。

设计原则：
- 使用 dataclass 简化定义
- 所有字段都有明确的类型注解
- 提供合理的默认值
- 包含必要的验证逻辑
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any


# ============================================================================
# 市场数据相关 DTO
# ============================================================================

@dataclass
class MarketContext:
    """
    市场上下文（发布前的 K 线状态）
    
    用于情感分析时提供市场背景信息，帮助模型理解新闻发布时的市场状态。
    """
    ticker: str                          # 标的代码（如 "XAUUSD"）
    window_pre_minutes: int              # 回看窗口（分钟数，如 120）
    pre_ret: float                       # 前期收益率（如 0.015 表示涨 1.5%）
    volatility: float                    # 波动率（高低价差/开盘价，如 0.008）
    trend_tag: str                       # 趋势标签（如 "Strong Rally"）
    
    # 可选字段：价格详情
    price_start: Optional[float] = None  # 窗口起始价
    price_end: Optional[float] = None    # 窗口结束价（事件发布时）
    price_high: Optional[float] = None   # 窗口内最高价
    price_low: Optional[float] = None    # 窗口内最低价
    
    # 时间戳
    ts_start: Optional[datetime] = None  # 窗口起始时间
    ts_end: Optional[datetime] = None    # 窗口结束时间（事件发布时）


@dataclass
class NewsItem:
    """
    新闻条目
    
    表示一条财经快讯或日历事件。
    """
    ts: datetime                         # 发布时间（本地时区）
    source: str                          # 来源（如 "jin10_flash", "jin10_calendar"）
    content: str                         # 新闻内容/标题
    
    # 可选字段：元数据
    event_id: Optional[str] = None       # 事件 ID（数据库主键）
    country: Optional[str] = None        # 国家/地区
    name: Optional[str] = None           # 指标名称（日历事件）
    star: Optional[int] = None           # 重要性星级（1-5）
    
    # 宏观数据字段
    previous: Optional[float] = None     # 前值
    consensus: Optional[float] = None    # 预期值
    actual: Optional[float] = None       # 实际值
    
    # 其他元数据
    url: Optional[str] = None            # 详情链接
    extra: Optional[Dict[str, Any]] = None  # 额外信息（JSON）


# ============================================================================
# 分析结果相关 DTO
# ============================================================================

@dataclass
class SentimentResult:
    """
    情感分析结果
    
    Engine A（BERT + 规则引擎）的输出。
    """
    label: int                           # 情感标签（-1=利空, 0=中性, 1=利好）
    score: float                         # 置信度（0-1）
    explain: str                         # 解释文本
    
    # 可选字段：详细信息
    base_label: Optional[int] = None     # BERT 基础预测（规则引擎前）
    base_score: Optional[float] = None   # BERT 置信度
    rule_triggered: Optional[str] = None # 触发的规则名称（如 "priced_in", "watch"）
    probs: Optional[List[float]] = None  # 三类概率分布 [bearish, neutral, bullish]


@dataclass
class Citation:
    """
    引用片段（RAG 检索结果）
    
    Engine B 的输出单元。
    """
    text: str                            # 引用文本
    source_file: str                     # 来源文件（如 "600519_2023Q4.pdf"）
    page_idx: int                        # 页码（从 0 开始）
    score: float                         # 相似度分数（0-1）
    
    # 可选字段：元数据
    chunk_id: Optional[str] = None       # 切片 ID
    metadata: Optional[Dict[str, Any]] = None  # 额外元数据


@dataclass
class ToolTraceItem:
    """
    工具调用追踪条目
    
    记录 Agent 执行过程中每个工具的调用情况。
    """
    name: str                            # 工具名称（如 "get_market_context"）
    elapsed_ms: int                      # 耗时（毫秒）
    ok: bool                             # 是否成功
    
    # 可选字段：详细信息
    error: Optional[str] = None          # 错误信息（失败时）
    input_summary: Optional[str] = None  # 输入摘要
    output_summary: Optional[str] = None # 输出摘要


@dataclass
class AgentAnswer:
    """
    Agent 最终答案
    
    包含分析结果、工具追踪、警告信息等。
    """
    summary: str                         # LLM 生成的总结
    sentiment: Optional[SentimentResult] = None  # 情感分析结果（快讯分析）
    citations: List[Citation] = field(default_factory=list)  # 引用片段（财报问答）
    warnings: List[str] = field(default_factory=list)        # 警告信息
    tool_trace: List[ToolTraceItem] = field(default_factory=list)  # 工具追踪
    
    # 可选字段：元数据
    query: Optional[str] = None          # 用户查询
    query_type: Optional[str] = None     # 查询类型（"news_analysis", "report_qa"）
    ts: Optional[datetime] = None        # 生成时间


# ============================================================================
# 配置相关 DTO
# ============================================================================

@dataclass
class EngineConfig:
    """
    引擎配置
    
    用于初始化各个引擎的配置参数。
    """
    # Engine A（情感分析）
    bert_model_path: str = "models/bert_3cls/best"
    bert_max_length: int = 384
    bert_device: str = "cpu"
    
    # Engine B（RAG）
    chroma_path: str = "data/reports/chroma_db"
    embedding_model: str = "BAAI/bge-m3"
    rag_top_k: int = 5
    
    # 规则引擎
    priced_in_threshold: float = 0.01    # 预期兑现阈值（1%）
    high_volatility_threshold: float = 0.015  # 高波动阈值（1.5%）
    low_net_change_threshold: float = 0.002   # 低净变动阈值（0.2%）
    
    # LLM
    llm_api_key: Optional[str] = None    # Deepseek API Key
    llm_timeout: float = 10.0            # 超时时间（秒）
    llm_max_tokens: int = 500            # 最大生成 token 数
    
    # 数据源
    db_path: str = "finance_analysis.db"
    
    # 其他
    log_level: str = "INFO"


# ============================================================================
# 辅助函数
# ============================================================================

def sentiment_label_to_text(label: int) -> str:
    """
    将情感标签转换为文本
    
    Args:
        label: 情感标签（-1/0/1）
    
    Returns:
        文本描述（"利空"/"中性"/"利好"）
    """
    mapping = {
        -1: "利空",
        0: "中性",
        1: "利好"
    }
    return mapping.get(label, "未知")


def sentiment_label_to_english(label: int) -> str:
    """
    将情感标签转换为英文
    
    Args:
        label: 情感标签（-1/0/1）
    
    Returns:
        英文描述（"bearish"/"neutral"/"bullish"）
    """
    mapping = {
        -1: "bearish",
        0: "neutral",
        1: "bullish"
    }
    return mapping.get(label, "unknown")


if __name__ == "__main__":
    # 测试代码
    print("=" * 80)
    print("DTO 数据结构测试")
    print("=" * 80)
    
    # 测试 MarketContext
    print("\n1. MarketContext")
    context = MarketContext(
        ticker="XAUUSD",
        window_pre_minutes=120,
        pre_ret=0.015,
        volatility=0.008,
        trend_tag="Strong Rally",
        price_start=2650.0,
        price_end=2690.0,
        ts_start=datetime(2026, 2, 7, 10, 0),
        ts_end=datetime(2026, 2, 7, 12, 0)
    )
    print(f"标的: {context.ticker}")
    print(f"前期收益率: {context.pre_ret:.2%}")
    print(f"趋势标签: {context.trend_tag}")
    
    # 测试 NewsItem
    print("\n2. NewsItem")
    news = NewsItem(
        ts=datetime(2026, 2, 7, 12, 0),
        source="jin10_flash",
        content="美联储宣布维持利率不变",
        star=5,
        country="美国"
    )
    print(f"来源: {news.source}")
    print(f"内容: {news.content}")
    print(f"重要性: {news.star}星")
    
    # 测试 SentimentResult
    print("\n3. SentimentResult")
    sentiment = SentimentResult(
        label=1,
        score=0.85,
        explain="预测为利好，置信度 85%",
        base_label=1,
        base_score=0.85,
        rule_triggered=None,
        probs=[0.05, 0.10, 0.85]
    )
    print(f"情感: {sentiment_label_to_text(sentiment.label)}")
    print(f"置信度: {sentiment.score:.2%}")
    print(f"解释: {sentiment.explain}")
    
    # 测试 Citation
    print("\n4. Citation")
    citation = Citation(
        text="2023年公司实现营业收入1234.56亿元，同比增长15.2%。",
        source_file="600519_2023Q4.pdf",
        page_idx=3,
        score=0.92
    )
    print(f"来源: {citation.source_file} (第 {citation.page_idx + 1} 页)")
    print(f"相似度: {citation.score:.2%}")
    print(f"内容: {citation.text[:50]}...")
    
    # 测试 ToolTraceItem
    print("\n5. ToolTraceItem")
    trace = ToolTraceItem(
        name="get_market_context",
        elapsed_ms=120,
        ok=True,
        input_summary="ticker=XAUUSD, window=120min",
        output_summary="pre_ret=1.5%, volatility=0.8%"
    )
    print(f"工具: {trace.name}")
    print(f"耗时: {trace.elapsed_ms}ms")
    print(f"状态: {'成功' if trace.ok else '失败'}")
    
    # 测试 AgentAnswer
    print("\n6. AgentAnswer")
    answer = AgentAnswer(
        summary="虽然消息利好，但前期已大涨1.5%，可能是利好预期兑现，建议谨慎追多。",
        sentiment=sentiment,
        citations=[],
        warnings=["前期涨幅较大，注意回调风险"],
        tool_trace=[trace],
        query="美联储宣布维持利率不变",
        query_type="news_analysis",
        ts=datetime(2026, 2, 7, 12, 0, 30)
    )
    print(f"查询类型: {answer.query_type}")
    print(f"总结: {answer.summary}")
    print(f"情感: {sentiment_label_to_text(answer.sentiment.label)}")
    print(f"警告: {answer.warnings}")
    print(f"工具调用: {len(answer.tool_trace)} 个")
    
    # 测试 EngineConfig
    print("\n7. EngineConfig")
    config = EngineConfig(
        bert_model_path="models/bert_3cls/best",
        chroma_path="data/reports/chroma_db",
        llm_api_key="sk-test123",
        db_path="finance_analysis.db"
    )
    print(f"BERT 模型路径: {config.bert_model_path}")
    print(f"Chroma 路径: {config.chroma_path}")
    print(f"数据库路径: {config.db_path}")
    print(f"LLM 超时: {config.llm_timeout}秒")
    
    print("\n" + "=" * 80)
    print("所有 DTO 测试通过！")
    print("=" * 80)
