# -*- coding: utf-8 -*-
"""
Agent 编排器

负责协调各个引擎和工具，处理用户查询：
1. 判断查询类型（快讯分析 vs 财报问答）
2. 调用相应的工具和引擎
3. 记录工具调用追踪
4. 使用 LLM 生成最终总结
5. 支持查询结果缓存（性能优化）

使用示例：
    from app.core.orchestrator.agent import Agent
    
    agent = Agent(
        sentiment_engine=sentiment_engine,
        rag_engine=rag_engine,
        rule_engine=rule_engine,
        llm_client=llm_client,
        enable_cache=True  # 启用缓存
    )
    
    answer = agent.process_query("美联储加息对黄金有什么影响？")
"""
import time
from datetime import datetime
from typing import Optional
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from app.core.dto import AgentAnswer, ToolTraceItem, NewsItem
from app.core.orchestrator.tools import (
    get_market_context,
    analyze_sentiment,
    search_reports
)
from app.core.utils.cache import get_cache


class Agent:
    """Agent 编排器"""
    
    def __init__(
        self,
        sentiment_engine=None,
        rag_engine=None,
        rule_engine=None,
        llm_client=None,
        db_path: str = "finance_analysis.db",
        enable_cache: bool = True,
        cache_ttl: int = 300  # 5 分钟
    ):
        """
        初始化 Agent
        
        Args:
            sentiment_engine: 情感分析引擎（Engine A）
            rag_engine: RAG 检索引擎（Engine B）
            rule_engine: 规则引擎
            llm_client: LLM 客户端（Deepseek）
            db_path: 数据库路径
            enable_cache: 是否启用缓存
            cache_ttl: 缓存过期时间（秒）
        """
        self.sentiment_engine = sentiment_engine
        self.rag_engine = rag_engine
        self.rule_engine = rule_engine
        self.llm = llm_client
        self.db_path = db_path
        self.enable_cache = enable_cache
        
        # 初始化缓存
        if enable_cache:
            self.query_cache = get_cache("agent_query", maxsize=100, ttl=cache_ttl)
            self.market_context_cache = get_cache("market_context", maxsize=50, ttl=60)
            self.rag_cache = get_cache("rag_retrieval", maxsize=100, ttl=600)
        else:
            self.query_cache = None
            self.market_context_cache = None
            self.rag_cache = None
    
    def process_query(
        self,
        user_query: str,
        ticker: str = "XAUUSD",
        query_type: Optional[str] = None
    ) -> AgentAnswer:
        """
        处理用户查询
        
        Args:
            user_query: 用户查询文本
            ticker: 标的代码（默认 XAUUSD）
            query_type: 查询类型（"news_analysis" 或 "report_qa"，None 则自动判断）
        
        Returns:
            AgentAnswer 对象
        """
        # 尝试从缓存获取结果
        if self.enable_cache and self.query_cache is not None:
            cache_key = (user_query, ticker, query_type)
            cached_result = self.query_cache.get(cache_key)
            if cached_result is not None:
                # 添加缓存命中标记
                cached_result.tool_trace.insert(0, ToolTraceItem(
                    name="cache_hit",
                    elapsed_ms=0,
                    ok=True,
                    input_summary=f"query={user_query[:50]}...",
                    output_summary="从缓存返回结果"
                ))
                return cached_result
        
        tool_trace = []
        warnings = []
        
        # 1. 判断查询类型
        if query_type is None:
            query_type = self._detect_query_type(user_query)
        
        # 2. 根据查询类型调用不同的处理流程
        if query_type == "news_analysis":
            result = self._process_news_analysis(
                user_query, ticker, tool_trace, warnings
            )
        elif query_type == "report_qa":
            result = self._process_report_qa(
                user_query, tool_trace, warnings
            )
        else:
            # 未知查询类型，返回默认回复
            result = AgentAnswer(
                summary="抱歉，我无法理解您的问题。请尝试询问财经快讯分析或财报相关问题。",
                query=user_query,
                query_type="unknown",
                warnings=["无法识别查询类型"],
                tool_trace=[],
                ts=datetime.now()
            )
        
        # 存入缓存
        if self.enable_cache and self.query_cache is not None:
            cache_key = (user_query, ticker, query_type)
            self.query_cache.set(cache_key, result)
        
        return result
    
    def _detect_query_type(self, query: str) -> str:
        """
        检测查询类型
        
        Args:
            query: 用户查询
        
        Returns:
            "news_analysis" 或 "report_qa"
        """
        # 简单的关键词匹配
        report_keywords = ["财报", "营收", "利润", "年报", "季报", "业绩"]
        news_keywords = ["加息", "降息", "非农", "CPI", "GDP", "美联储", "央行"]
        
        query_lower = query.lower()
        
        # 检查财报关键词
        if any(keyword in query for keyword in report_keywords):
            return "report_qa"
        
        # 检查快讯关键词
        if any(keyword in query for keyword in news_keywords):
            return "news_analysis"
        
        # 默认为快讯分析
        return "news_analysis"
    
    def _process_news_analysis(
        self,
        news_text: str,
        ticker: str,
        tool_trace: list,
        warnings: list
    ) -> AgentAnswer:
        """
        处理快讯分析查询
        
        Args:
            news_text: 新闻文本
            ticker: 标的代码
            tool_trace: 工具追踪列表
            warnings: 警告列表
        
        Returns:
            AgentAnswer 对象
        """
        # 步骤 1: 获取市场上下文（带缓存）
        start = time.time()
        
        # 尝试从缓存获取
        context = None
        cache_hit = False
        if self.enable_cache and self.market_context_cache is not None:
            cache_key = (ticker, datetime.now().strftime('%Y-%m-%d %H:%M'), 120)
            context = self.market_context_cache.get(cache_key)
            if context is not None:
                cache_hit = True
        
        # 缓存未命中，查询数据库
        if context is None:
            context = get_market_context(
                ticker=ticker,
                event_time=datetime.now(),
                window_minutes=120,
                db_path=self.db_path
            )
            # 存入缓存
            if self.enable_cache and self.market_context_cache is not None and context is not None:
                cache_key = (ticker, datetime.now().strftime('%Y-%m-%d %H:%M'), 120)
                self.market_context_cache.set(cache_key, context)
        
        elapsed_ms = int((time.time() - start) * 1000)
        
        tool_trace.append(ToolTraceItem(
            name="get_market_context" + (" (cached)" if cache_hit else ""),
            elapsed_ms=elapsed_ms,
            ok=context is not None,
            error=None if context else "数据不足或数据库不存在",
            input_summary=f"ticker={ticker}, window=120min",
            output_summary=f"pre_ret={context.pre_ret:.2%}, trend={context.trend_tag}" if context else None
        ))
        
        if context is None:
            warnings.append("无法获取市场上下文，分析结果可能不准确")
        
        # 步骤 2: 情感分析
        start = time.time()
        sentiment = analyze_sentiment(
            news_text=news_text,
            context=context,
            sentiment_engine=self.sentiment_engine,
            rule_engine=self.rule_engine
        )
        elapsed_ms = int((time.time() - start) * 1000)
        
        tool_trace.append(ToolTraceItem(
            name="sentiment_analysis",
            elapsed_ms=elapsed_ms,
            ok=True,
            input_summary=f"text_length={len(news_text)}",
            output_summary=f"label={sentiment.label}, score={sentiment.score:.2f}"
        ))
        
        # 步骤 3: LLM 生成总结
        summary = self._generate_summary_for_news(
            news_text, sentiment, context, tool_trace
        )
        
        return AgentAnswer(
            summary=summary,
            sentiment=sentiment,
            citations=[],
            warnings=warnings,
            tool_trace=tool_trace,
            query=news_text,
            query_type="news_analysis",
            ts=datetime.now()
        )
    
    def _process_report_qa(
        self,
        query: str,
        tool_trace: list,
        warnings: list
    ) -> AgentAnswer:
        """
        处理财报问答查询
        
        Args:
            query: 查询文本
            tool_trace: 工具追踪列表
            warnings: 警告列表
        
        Returns:
            AgentAnswer 对象
        """
        # 步骤 1: RAG 检索（带缓存）
        start = time.time()
        
        # 尝试从缓存获取
        citations = None
        cache_hit = False
        if self.enable_cache and self.rag_cache is not None:
            cache_key = (query, 5)  # query + top_k
            citations = self.rag_cache.get(cache_key)
            if citations is not None:
                cache_hit = True
        
        # 缓存未命中，执行检索
        if citations is None:
            citations = search_reports(
                query=query,
                rag_engine=self.rag_engine,
                top_k=5
            )
            # 存入缓存
            if self.enable_cache and self.rag_cache is not None:
                cache_key = (query, 5)
                self.rag_cache.set(cache_key, citations)
        
        elapsed_ms = int((time.time() - start) * 1000)
        
        tool_trace.append(ToolTraceItem(
            name="rag_retrieval" + (" (cached)" if cache_hit else ""),
            elapsed_ms=elapsed_ms,
            ok=len(citations) > 0,
            error=None if len(citations) > 0 else "未找到相关内容",
            input_summary=f"query={query[:50]}...",
            output_summary=f"found {len(citations)} citations"
        ))
        
        if len(citations) == 0:
            warnings.append("未找到相关财报内容")
        
        # 步骤 2: LLM 生成总结
        summary = self._generate_summary_for_report(
            query, citations, tool_trace
        )
        
        return AgentAnswer(
            summary=summary,
            sentiment=None,
            citations=citations,
            warnings=warnings,
            tool_trace=tool_trace,
            query=query,
            query_type="report_qa",
            ts=datetime.now()
        )
    
    def _generate_summary_for_news(
        self,
        news_text: str,
        sentiment,
        context,
        tool_trace: list
    ) -> str:
        """
        为快讯分析生成 LLM 总结
        
        Args:
            news_text: 新闻文本
            sentiment: 情感分析结果
            context: 市场上下文
            tool_trace: 工具追踪列表
        
        Returns:
            总结文本
        """
        if self.llm is None:
            # 如果没有 LLM，返回简单总结
            from app.core.dto import sentiment_label_to_text
            label_text = sentiment_label_to_text(sentiment.label)
            
            summary = f"情感分析结果：{label_text}（置信度 {sentiment.score:.2%}）\n"
            summary += f"解释：{sentiment.explain}\n"
            
            if context:
                summary += f"\n市场背景：前期{context.trend_tag}，"
                summary += f"涨跌幅 {context.pre_ret:.2%}，波动率 {context.volatility:.2%}"
            
            return summary
        
        # 使用 LLM 生成总结
        start = time.time()
        
        # 构建提示词
        from app.core.dto import sentiment_label_to_text
        label_text = sentiment_label_to_text(sentiment.label)
        
        prompt = f"""你是一个专业的财经分析师。请根据以下信息，生成一段简洁的分析总结（2-3句话）：

新闻内容：{news_text}

情感分析：{label_text}（置信度 {sentiment.score:.2%}）
分析说明：{sentiment.explain}
"""
        
        if context:
            prompt += f"""
市场背景：
- 趋势：{context.trend_tag}
- 前期涨跌：{context.pre_ret:.2%}
- 波动率：{context.volatility:.2%}
"""
        
        prompt += "\n请用专业但易懂的语言总结，重点说明对市场的影响。"
        
        try:
            summary = self.llm.complete(prompt, timeout_seconds=10.0)
            elapsed_ms = int((time.time() - start) * 1000)
            
            tool_trace.append(ToolTraceItem(
                name="llm_summary",
                elapsed_ms=elapsed_ms,
                ok=not summary.startswith("[错误]"),
                error=summary if summary.startswith("[错误]") else None,
                input_summary=f"prompt_length={len(prompt)}",
                output_summary=f"summary_length={len(summary)}"
            ))
            
            return summary
        
        except Exception as e:
            # LLM 调用失败，返回简单总结
            return self._generate_summary_for_news(news_text, sentiment, context, [])
    
    def _generate_summary_for_report(
        self,
        query: str,
        citations: list,
        tool_trace: list
    ) -> str:
        """
        为财报问答生成 LLM 总结
        
        Args:
            query: 查询文本
            citations: 引用片段列表
            tool_trace: 工具追踪列表
        
        Returns:
            总结文本
        """
        if len(citations) == 0:
            return "抱歉，未找到相关财报内容。请尝试其他关键词。"
        
        if self.llm is None:
            # 如果没有 LLM，返回简单总结
            summary = f"找到 {len(citations)} 条相关内容：\n\n"
            for i, citation in enumerate(citations[:3], 1):
                summary += f"{i}. {citation.text[:100]}...\n"
                summary += f"   来源：{citation.source_file}（相似度 {citation.score:.2%}）\n\n"
            return summary
        
        # 使用 LLM 生成总结
        start = time.time()
        
        # 构建提示词
        context_text = "\n\n".join([
            f"[引用 {i+1}] {citation.text}"
            for i, citation in enumerate(citations[:3])
        ])
        
        prompt = f"""你是一个专业的财经分析师。请根据以下财报内容，回答用户的问题。

用户问题：{query}

相关财报内容：
{context_text}

请用2-3句话简洁回答，并在回答中标注引用来源（如"根据引用1..."）。
"""
        
        try:
            summary = self.llm.complete(prompt, timeout_seconds=10.0)
            elapsed_ms = int((time.time() - start) * 1000)
            
            tool_trace.append(ToolTraceItem(
                name="llm_summary",
                elapsed_ms=elapsed_ms,
                ok=not summary.startswith("[错误]"),
                error=summary if summary.startswith("[错误]") else None,
                input_summary=f"prompt_length={len(prompt)}",
                output_summary=f"summary_length={len(summary)}"
            ))
            
            return summary
        
        except Exception as e:
            # LLM 调用失败，返回简单总结
            return self._generate_summary_for_report(query, citations, [])
    
    def get_cache_stats(self) -> dict:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计信息字典
        """
        if not self.enable_cache:
            return {"enabled": False}
        
        stats = {"enabled": True}
        
        if self.query_cache is not None:
            stats["query_cache"] = self.query_cache.get_stats()
        
        if self.market_context_cache is not None:
            stats["market_context_cache"] = self.market_context_cache.get_stats()
        
        if self.rag_cache is not None:
            stats["rag_cache"] = self.rag_cache.get_stats()
        
        return stats
    
    def clear_cache(self) -> None:
        """清空所有缓存"""
        if self.enable_cache:
            if self.query_cache is not None:
                self.query_cache.clear()
            if self.market_context_cache is not None:
                self.market_context_cache.clear()
            if self.rag_cache is not None:
                self.rag_cache.clear()


# 测试代码
if __name__ == "__main__":
    print("=" * 80)
    print("Agent 编排器测试")
    print("=" * 80)
    
    # 初始化 Agent（无引擎，启用缓存）
    agent = Agent(enable_cache=True)
    
    # 测试 1: 快讯分析
    print("\n测试 1: 快讯分析")
    print("-" * 80)
    
    query1 = "美联储宣布加息 25 个基点"
    answer1 = agent.process_query(query1, ticker="XAUUSD")
    
    print(f"查询: {query1}")
    print(f"查询类型: {answer1.query_type}")
    print(f"总结: {answer1.summary}")
    print(f"工具调用: {len(answer1.tool_trace)} 个")
    for trace in answer1.tool_trace:
        status = "✓" if trace.ok else "✗"
        print(f"  {status} {trace.name} ({trace.elapsed_ms}ms)")
    
    # 测试 2: 重复查询（测试缓存）
    print("\n测试 2: 重复查询（测试缓存）")
    print("-" * 80)
    
    answer2 = agent.process_query(query1, ticker="XAUUSD")
    print(f"查询: {query1}")
    print(f"工具调用: {len(answer2.tool_trace)} 个")
    for trace in answer2.tool_trace:
        status = "✓" if trace.ok else "✗"
        print(f"  {status} {trace.name} ({trace.elapsed_ms}ms)")
    
    # 测试 3: 缓存统计
    print("\n测试 3: 缓存统计")
    print("-" * 80)
    
    stats = agent.get_cache_stats()
    print(f"缓存统计: {stats}")
    
    print("\n" + "=" * 80)
    print("Agent 测试完成")
    print("=" * 80)
    print("\n说明:")
    print("- 当前测试未加载引擎，使用默认行为")
    print("- 完整功能需要初始化所有引擎（SentimentEngine, RagEngine, LLM）")
    print("- 缓存功能已启用，重复查询会从缓存返回结果")
