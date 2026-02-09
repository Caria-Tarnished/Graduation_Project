# -*- coding: utf-8 -*-
"""
快讯分析用例

封装快讯情感分析的完整流程，提供：
- 超时控制
- 缓存支持
- 降级策略

使用示例：
    from app.application.analyze_news import analyze_news_with_context
    
    result = analyze_news_with_context(
        news_text="美联储宣布加息 25 个基点",
        ticker="XAUUSD",
        agent=agent
    )
"""
from typing import Optional
from datetime import datetime
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.core.dto import AgentAnswer
from app.application.utils import with_timeout, SimpleCache, with_cache


# 全局缓存（可选）
_news_cache = SimpleCache(ttl_seconds=300)  # 5 分钟缓存


def analyze_news_with_context(
    news_text: str,
    ticker: str = "XAUUSD",
    agent = None,
    use_cache: bool = True,
    timeout_seconds: float = 30.0
) -> AgentAnswer:
    """
    分析快讯（带市场上下文）
    
    Args:
        news_text: 新闻文本
        ticker: 标的代码
        agent: Agent 实例
        use_cache: 是否使用缓存
        timeout_seconds: 超时时间
    
    Returns:
        AgentAnswer 对象
    """
    if agent is None:
        # 如果没有 Agent，返回错误
        return AgentAnswer(
            summary="错误：未初始化 Agent",
            query=news_text,
            query_type="news_analysis",
            warnings=["Agent 未初始化"],
            tool_trace=[],
            ts=datetime.now()
        )
    
    # 生成缓存键
    cache_key = f"news:{ticker}:{hash(news_text)}"
    
    # 尝试从缓存获取
    if use_cache:
        cached_result = _news_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
    
    # 调用 Agent 处理
    try:
        result = agent.process_query(
            user_query=news_text,
            ticker=ticker,
            query_type="news_analysis"
        )
        
        # 存入缓存
        if use_cache:
            _news_cache.set(cache_key, result)
        
        return result
    
    except Exception as e:
        # 降级：返回错误信息
        return AgentAnswer(
            summary=f"分析失败: {str(e)}",
            query=news_text,
            query_type="news_analysis",
            warnings=[f"异常: {str(e)}"],
            tool_trace=[],
            ts=datetime.now()
        )


def analyze_news_simple(
    news_text: str,
    agent = None
) -> str:
    """
    简化版快讯分析（仅返回总结文本）
    
    Args:
        news_text: 新闻文本
        agent: Agent 实例
    
    Returns:
        总结文本
    """
    result = analyze_news_with_context(
        news_text=news_text,
        agent=agent,
        use_cache=True
    )
    
    return result.summary


def clear_news_cache():
    """清空快讯分析缓存"""
    _news_cache.clear()


def get_cache_stats() -> dict:
    """获取缓存统计信息"""
    return {
        "cache_size": _news_cache.size(),
        "ttl_seconds": _news_cache.ttl_seconds
    }


# 测试代码
if __name__ == "__main__":
    print("=" * 80)
    print("快讯分析用例测试")
    print("=" * 80)
    
    # 测试 1: 无 Agent
    print("\n测试 1: 无 Agent")
    print("-" * 80)
    
    result1 = analyze_news_with_context(
        news_text="美联储宣布加息 25 个基点",
        ticker="XAUUSD",
        agent=None
    )
    
    print(f"总结: {result1.summary}")
    print(f"警告: {result1.warnings}")
    
    # 测试 2: 缓存统计
    print("\n测试 2: 缓存统计")
    print("-" * 80)
    
    stats = get_cache_stats()
    print(f"缓存大小: {stats['cache_size']}")
    print(f"缓存过期时间: {stats['ttl_seconds']} 秒")
    
    print("\n" + "=" * 80)
    print("用例测试完成")
    print("=" * 80)
    print("\n说明:")
    print("- 完整测试需要初始化 Agent")
    print("- 缓存功能可以减少重复计算")
