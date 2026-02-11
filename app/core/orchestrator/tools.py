# -*- coding: utf-8 -*-
"""
Agent 工具函数集合

提供 Agent 使用的各种工具函数：
1. get_market_context: 获取市场上下文
2. analyze_sentiment: 分析快讯情感
3. search_reports: 检索财报

使用示例：
    from app.core.orchestrator.tools import get_market_context
    
    context = get_market_context(
        ticker="XAUUSD",
        event_time=datetime.now(),
        window_minutes=120
    )
"""
import sqlite3
from datetime import datetime, timedelta
from typing import Optional
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from app.core.dto import MarketContext, SentimentResult, Citation


def get_market_context(
    ticker: str,
    event_time: datetime,
    window_minutes: int = 120,
    db_path: str = "finance_analysis.db"
) -> Optional[MarketContext]:
    """
    从数据库获取市场上下文
    
    Args:
        ticker: 标的代码（如 "XAUUSD"）
        event_time: 事件时间
        window_minutes: 回看窗口（分钟数）
        db_path: 数据库路径
    
    Returns:
        MarketContext 对象，如果数据不足则返回 None
    """
    try:
        # 计算时间窗口
        window_start = event_time - timedelta(minutes=window_minutes)
        
        # 连接数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 查询价格数据
        query = """
        SELECT ts_local, open, high, low, close
        FROM prices_m1
        WHERE ticker = ?
          AND ts_local >= ?
          AND ts_local <= ?
        ORDER BY ts_local ASC
        """
        
        cursor.execute(query, (
            ticker,
            window_start.strftime('%Y-%m-%d %H:%M:%S'),
            event_time.strftime('%Y-%m-%d %H:%M:%S')
        ))
        
        rows = cursor.fetchall()
        conn.close()
        
        if len(rows) < 2:
            # 数据不足，无法计算指标
            # 打印调试信息
            print(f"[DEBUG] get_market_context 失败:")
            print(f"  ticker={ticker}, event_time={event_time}")
            print(f"  window_start={window_start}, window_end={event_time}")
            print(f"  查询到的行数: {len(rows)}")
            print(f"  原因: 数据不足（需要至少2行数据）")
            return None
        
        # 提取价格数据
        timestamps = [datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S') for row in rows]
        opens = [row[1] for row in rows]
        highs = [row[2] for row in rows]
        lows = [row[3] for row in rows]
        closes = [row[4] for row in rows]
        
        # 计算指标
        price_start = opens[0]
        price_end = closes[-1]
        price_high = max(highs)
        price_low = min(lows)
        
        # 前期收益率
        pre_ret = (price_end - price_start) / price_start if price_start > 0 else 0.0
        
        # 波动率（高低价差 / 起始价）
        volatility = (price_high - price_low) / price_start if price_start > 0 else 0.0
        
        # 趋势标签
        if pre_ret > 0.005:
            trend_tag = "Strong Rally"
        elif pre_ret < -0.005:
            trend_tag = "Sharp Decline"
        elif abs(pre_ret) < 0.002 and volatility > 0.008:
            trend_tag = "High Volatility"
        elif pre_ret > 0.001:
            trend_tag = "Mild Rally"
        elif pre_ret < -0.001:
            trend_tag = "Weak Decline"
        else:
            trend_tag = "Sideways"
        
        return MarketContext(
            ticker=ticker,
            window_pre_minutes=window_minutes,
            pre_ret=pre_ret,
            volatility=volatility,
            trend_tag=trend_tag,
            price_start=price_start,
            price_end=price_end,
            price_high=price_high,
            price_low=price_low,
            ts_start=timestamps[0],
            ts_end=timestamps[-1]
        )
    
    except Exception as e:
        print(f"[DEBUG] get_market_context 异常:")
        print(f"  ticker={ticker}, event_time={event_time}")
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_sentiment(
    news_text: str,
    context: Optional[MarketContext] = None,
    sentiment_engine = None,
    rule_engine = None
) -> SentimentResult:
    """
    分析快讯情感
    
    Args:
        news_text: 新闻文本
        context: 市场上下文（可选）
        sentiment_engine: 情感分析引擎（Engine A）
        rule_engine: 规则引擎（可选）
    
    Returns:
        SentimentResult 对象
    """
    if sentiment_engine is None:
        # 如果没有提供引擎，返回默认结果
        return SentimentResult(
            label=0,
            score=0.5,
            explain="未加载情感分析引擎，返回默认中性结果"
        )
    
    try:
        # 1. 从 context 提取参数（如果 context 为 None，使用默认值）
        if context is not None:
            pre_ret = context.pre_ret
            range_ratio = context.volatility
        else:
            pre_ret = 0.0
            range_ratio = 0.0
        
        # 2. 调用情感分析引擎（使用 analyze 方法）
        result = sentiment_engine.analyze(
            text=news_text,
            pre_ret=pre_ret,
            range_ratio=range_ratio
        )
        
        # 3. 将 final_sentiment 映射为 label（-1/0/1）
        sentiment_map = {
            "bearish": -1,
            "bearish_priced_in": -1,
            "neutral": 0,
            "watch": 0,
            "bullish": 1,
            "bullish_priced_in": 1
        }
        
        final_sentiment = result.get('final_sentiment', 'neutral')
        label = sentiment_map.get(final_sentiment, 0)
        score = result.get('base_confidence', 0.5)
        
        # 4. 构建解释文本
        explanation = result.get('explanation', '')
        recommendation = result.get('recommendation', '')
        if recommendation:
            explanation = f"{explanation}\n{recommendation}"
        
        # 5. 转换为 SentimentResult
        sentiment = SentimentResult(
            label=label,
            score=score,
            explain=explanation
        )
        
        # 6. 如果有规则引擎，进行后处理（注意：规则引擎已经在 sentiment_engine 中应用了）
        # 这里保留接口以备将来使用
        if rule_engine is not None:
            from app.core.dto import NewsItem
            news_item = NewsItem(
                ts=datetime.now(),
                source="unknown",
                content=news_text
            )
            sentiment = rule_engine.post_process(sentiment, context, news_item)
        
        return sentiment
    
    except Exception as e:
        print(f"情感分析失败: {e}")
        import traceback
        traceback.print_exc()
        return SentimentResult(
            label=0,
            score=0.0,
            explain=f"情感分析失败: {str(e)}"
        )


def search_reports(
    query: str,
    rag_engine,
    top_k: int = 5
) -> list:
    """
    检索财报
    
    Args:
        query: 查询文本
        rag_engine: RAG 检索引擎（Engine B）
        top_k: 返回前 k 个结果
    
    Returns:
        Citation 列表
    """
    if rag_engine is None:
        # 如果没有提供引擎，返回空列表
        return []
    
    try:
        citations = rag_engine.retrieve(query, top_k=top_k)
        return citations
    
    except Exception as e:
        print(f"财报检索失败: {e}")
        return []


# 测试代码
if __name__ == "__main__":
    print("=" * 80)
    print("工具函数测试")
    print("=" * 80)
    
    # 测试 1: 获取市场上下文
    print("\n测试 1: 获取市场上下文")
    print("-" * 80)
    
    test_time = datetime(2026, 1, 15, 12, 0, 0)
    context = get_market_context(
        ticker="XAUUSD",
        event_time=test_time,
        window_minutes=120
    )
    
    if context:
        print(f"✓ 成功获取市场上下文")
        print(f"  标的: {context.ticker}")
        print(f"  时间窗口: {context.window_pre_minutes} 分钟")
        print(f"  前期收益率: {context.pre_ret:.4f} ({context.pre_ret*100:.2f}%)")
        print(f"  波动率: {context.volatility:.4f} ({context.volatility*100:.2f}%)")
        print(f"  趋势标签: {context.trend_tag}")
        print(f"  价格范围: {context.price_start:.2f} -> {context.price_end:.2f}")
    else:
        print("✗ 未能获取市场上下文（数据不足或数据库不存在）")
    
    # 测试 2: 情感分析（无引擎）
    print("\n测试 2: 情感分析（无引擎）")
    print("-" * 80)
    
    news_text = "美联储宣布加息 25 个基点"
    sentiment = analyze_sentiment(news_text, context=context)
    
    print(f"新闻: {news_text}")
    print(f"结果: {sentiment.explain}")
    
    # 测试 3: 财报检索（无引擎）
    print("\n测试 3: 财报检索（无引擎）")
    print("-" * 80)
    
    query = "黄金价格走势"
    citations = search_reports(query, rag_engine=None, top_k=3)
    
    print(f"查询: {query}")
    print(f"结果数: {len(citations)}")
    
    print("\n" + "=" * 80)
    print("工具函数测试完成")
    print("=" * 80)
    print("\n说明:")
    print("- 测试 1 需要 finance_analysis.db 数据库")
    print("- 测试 2 和 3 在没有引擎时返回默认结果")
    print("- 完整测试需要初始化 SentimentEngine 和 RagEngine")
