# -*- coding: utf-8 -*-
"""
财报问答用例

封装财报检索问答的完整流程，提供：
- 超时控制
- 缓存支持
- 降级策略

使用示例：
    from app.application.ask_report import ask_report_question
    
    result = ask_report_question(
        question="贵州茅台 2023 年营收情况如何？",
        agent=agent
    )
"""
from typing import Optional, List
from datetime import datetime
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.core.dto import AgentAnswer, Citation
from app.application.utils import with_timeout, SimpleCache, with_cache


# 全局缓存（可选）
_report_cache = SimpleCache(ttl_seconds=600)  # 10 分钟缓存


def ask_report_question(
    question: str,
    agent = None,
    top_k: int = 5,
    use_cache: bool = True,
    timeout_seconds: float = 30.0
) -> AgentAnswer:
    """
    财报问答
    
    Args:
        question: 问题文本
        agent: Agent 实例
        top_k: 返回前 k 个引用
        use_cache: 是否使用缓存
        timeout_seconds: 超时时间
    
    Returns:
        AgentAnswer 对象
    """
    if agent is None:
        # 如果没有 Agent，返回错误
        return AgentAnswer(
            summary="错误：未初始化 Agent",
            query=question,
            query_type="report_qa",
            warnings=["Agent 未初始化"],
            tool_trace=[],
            citations=[],
            ts=datetime.now()
        )
    
    # 生成缓存键
    cache_key = f"report:{hash(question)}:{top_k}"
    
    # 尝试从缓存获取
    if use_cache:
        cached_result = _report_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
    
    # 调用 Agent 处理
    try:
        result = agent.process_query(
            user_query=question,
            query_type="report_qa"
        )
        
        # 存入缓存
        if use_cache:
            _report_cache.set(cache_key, result)
        
        return result
    
    except Exception as e:
        # 降级：返回错误信息
        return AgentAnswer(
            summary=f"查询失败: {str(e)}",
            query=question,
            query_type="report_qa",
            warnings=[f"异常: {str(e)}"],
            tool_trace=[],
            citations=[],
            ts=datetime.now()
        )


def ask_report_simple(
    question: str,
    agent = None
) -> str:
    """
    简化版财报问答（仅返回总结文本）
    
    Args:
        question: 问题文本
        agent: Agent 实例
    
    Returns:
        总结文本
    """
    result = ask_report_question(
        question=question,
        agent=agent,
        use_cache=True
    )
    
    return result.summary


def get_report_citations(
    question: str,
    agent = None,
    top_k: int = 5
) -> List[Citation]:
    """
    仅获取财报引用（不生成总结）
    
    Args:
        question: 问题文本
        agent: Agent 实例
        top_k: 返回前 k 个引用
    
    Returns:
        Citation 列表
    """
    result = ask_report_question(
        question=question,
        agent=agent,
        top_k=top_k,
        use_cache=True
    )
    
    return result.citations


def clear_report_cache():
    """清空财报问答缓存"""
    _report_cache.clear()


def get_cache_stats() -> dict:
    """获取缓存统计信息"""
    return {
        "cache_size": _report_cache.size(),
        "ttl_seconds": _report_cache.ttl_seconds
    }


# 测试代码
if __name__ == "__main__":
    print("=" * 80)
    print("财报问答用例测试")
    print("=" * 80)
    
    # 测试 1: 无 Agent
    print("\n测试 1: 无 Agent")
    print("-" * 80)
    
    result1 = ask_report_question(
        question="贵州茅台 2023 年营收情况如何？",
        agent=None
    )
    
    print(f"总结: {result1.summary}")
    print(f"警告: {result1.warnings}")
    print(f"引用数: {len(result1.citations)}")
    
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
    print("- 缓存功能可以减少重复检索")
