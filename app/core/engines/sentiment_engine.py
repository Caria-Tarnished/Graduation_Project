# -*- coding: utf-8 -*-
"""
情感分析引擎适配器

将 SentimentAnalyzer 适配为 Agent 工具所需的接口。
提供 predict_sentiment() 方法，接受 news_text 和 MarketContext。

使用示例:
    from app.core.engines.sentiment_engine import SentimentEngine
    
    engine = SentimentEngine(
        model_path="models/bert_3cls/best",
        device="cpu"
    )
    
    sentiment = engine.predict_sentiment(news_text, context)
"""
import sys
from pathlib import Path
from typing import Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from app.services.sentiment_analyzer import SentimentAnalyzer
from app.core.dto import MarketContext, SentimentResult


class SentimentEngine:
    """
    情感分析引擎适配器
    
    将 SentimentAnalyzer 适配为 Agent 工具所需的接口
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        max_length: int = 384
    ):
        """
        初始化情感分析引擎
        
        Args:
            model_path: BERT 模型路径
            device: 设备（"cpu" 或 "cuda"）
            max_length: 最大序列长度
        """
        self.analyzer = SentimentAnalyzer(
            model_path=model_path,
            device=device,
            max_length=max_length
        )
    
    def predict_sentiment(
        self,
        news_text: str,
        context: Optional[MarketContext] = None
    ) -> SentimentResult:
        """
        预测情感
        
        Args:
            news_text: 新闻文本
            context: 市场上下文（可选）
        
        Returns:
            SentimentResult 对象
        """
        # 提取市场上下文参数
        pre_ret = 0.0
        range_ratio = 0.0
        
        if context:
            pre_ret = context.pre_ret
            range_ratio = context.volatility
        
        # 调用 SentimentAnalyzer
        result = self.analyzer.analyze(
            text=news_text,
            pre_ret=pre_ret,
            range_ratio=range_ratio
        )
        
        # 转换为 SentimentResult
        # 将 final_sentiment 映射回标签
        sentiment_map = {
            "bearish": -1,
            "neutral": 0,
            "bullish": 1,
            "bearish_priced_in": -1,
            "bullish_priced_in": 1,
            "watch": 0
        }
        
        label = sentiment_map.get(result["final_sentiment"], 0)
        score = result["base_confidence"]
        
        # 构建解释文本
        explain = result["explanation"]
        if result["recommendation"]:
            explain += f"\n{result['recommendation']}"
        
        return SentimentResult(
            label=label,
            score=score,
            explain=explain,
            base_label=sentiment_map.get(result["base_sentiment"], 0),
            base_score=result["base_confidence"],
            rule_triggered=result["final_sentiment"] if result["final_sentiment"] not in ["bearish", "neutral", "bullish"] else None
        )
    
    def predict_sentiment_batch(
        self,
        news_texts: list,
        contexts: Optional[list] = None
    ) -> list:
        """
        批量预测情感
        
        Args:
            news_texts: 新闻文本列表
            contexts: 市场上下文列表（可选）
        
        Returns:
            SentimentResult 对象列表
        """
        batch_size = len(news_texts)
        
        # 提取市场上下文参数
        pre_rets = []
        range_ratios = []
        
        if contexts is None:
            contexts = [None] * batch_size
        
        for context in contexts:
            if context:
                pre_rets.append(context.pre_ret)
                range_ratios.append(context.volatility)
            else:
                pre_rets.append(0.0)
                range_ratios.append(0.0)
        
        # 调用 SentimentAnalyzer 批处理
        results = self.analyzer.analyze_batch(
            texts=news_texts,
            pre_rets=pre_rets,
            range_ratios=range_ratios
        )
        
        # 转换为 SentimentResult 列表
        sentiment_map = {
            "bearish": -1,
            "neutral": 0,
            "bullish": 1,
            "bearish_priced_in": -1,
            "bullish_priced_in": 1,
            "watch": 0
        }
        
        sentiment_results = []
        for result in results:
            label = sentiment_map.get(result["final_sentiment"], 0)
            score = result["base_confidence"]
            
            # 构建解释文本
            explain = result["explanation"]
            if result["recommendation"]:
                explain += f"\n{result['recommendation']}"
            
            sentiment_results.append(SentimentResult(
                label=label,
                score=score,
                explain=explain,
                base_label=sentiment_map.get(result["base_sentiment"], 0),
                base_score=result["base_confidence"],
                rule_triggered=result["final_sentiment"] if result["final_sentiment"] not in ["bearish", "neutral", "bullish"] else None
            ))
        
        return sentiment_results


# 测试代码
if __name__ == "__main__":
    print("=" * 80)
    print("情感分析引擎适配器测试")
    print("=" * 80)
    
    try:
        # 创建引擎
        engine = SentimentEngine(
            model_path="models/bert_3cls/best",
            device="cpu"
        )
        
        # 创建测试上下文
        context = MarketContext(
            ticker="XAUUSD",
            window_pre_minutes=120,
            pre_ret=0.015,  # 前期涨1.5%
            volatility=0.008,  # 波动率0.8%
            trend_tag="Strong Rally"
        )
        
        # 测试预测
        news_text = "美联储宣布降息25个基点"
        sentiment = engine.predict_sentiment(news_text, context)
        
        print(f"\n新闻: {news_text}")
        print(f"标签: {sentiment.label}")
        print(f"置信度: {sentiment.score:.2%}")
        print(f"解释: {sentiment.explain}")
        print(f"基础标签: {sentiment.base_label}")
        print(f"触发规则: {sentiment.rule_triggered}")
        
        print("\n" + "=" * 80)
        print("测试完成")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n错误: {e}")
        print("\n提示：请确保 BERT 模型已训练并复制到 models/bert_3cls/best/")
