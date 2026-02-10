# -*- coding: utf-8 -*-
"""
情感分析服务（Engine A）：BERT 3类分类 + 后处理规则引擎

架构设计：
- ML 模型：专注于可学习的 3 类基础方向（Bearish/Neutral/Bullish）
- 规则引擎：处理"预期兑现"和"建议观望"等复杂逻辑
- 优势：更可解释、可维护、可调试；规则阈值可灵活调整

使用示例：
    from app.services.sentiment_analyzer import SentimentAnalyzer
    
    analyzer = SentimentAnalyzer(model_path="models/bert_3cls/best")
    result = analyzer.analyze(
        text="美联储宣布维持利率不变",
        pre_ret=0.015,  # 前120分钟涨幅1.5%
        range_ratio=0.008  # 波动率0.8%
    )
    print(result)
    # {
    #   "base_sentiment": "bullish",
    #   "base_confidence": 0.65,
    #   "final_sentiment": "bullish_priced_in",
    #   "explanation": "虽然消息利好，但前期已大涨1.50%，可能是利好预期兑现",
    #   "recommendation": "建议谨慎追多"
    # }
"""
import os
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SentimentAnalyzer:
    """
    情感分析器：BERT 3类分类 + 后处理规则引擎
    
    标签映射：
    - -1: Bearish（利空/看跌）
    - 0: Neutral（中性）
    - 1: Bullish（利好/看涨）
    """
    
    # 标签映射
    LABEL_MAP = {
        -1: "bearish",
        0: "neutral",
        1: "bullish"
    }
    
    # 规则引擎阈值（可根据实际情况调整）
    PRICED_IN_THRESHOLD = 0.01  # 前期涨跌幅阈值（1%）
    HIGH_VOLATILITY_THRESHOLD = 0.015  # 高波动阈值（1.5%）
    LOW_NET_CHANGE_THRESHOLD = 0.002  # 低净变动阈值（0.2%）
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        max_length: int = 384
    ):
        """
        初始化情感分析器
        
        Args:
            model_path: BERT 模型路径（包含 config.json 和权重文件）
            device: 设备（"cpu" 或 "cuda"），默认自动检测
            max_length: 最大序列长度
        """
        self.model_path = model_path
        self.max_length = max_length
        
        # 自动检测设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # 加载模型和分词器
        self._load_model()
    
    def _load_model(self):
        """加载 BERT 模型和分词器"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"模型路径不存在: {self.model_path}\n"
                f"请确保已完成训练并将模型权重复制到该路径。"
            )
        
        print(f"正在加载模型: {self.model_path}")
        print(f"设备: {self.device}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # 加载模型
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path
        )
        self.model.to(self.device)
        self.model.eval()
        
        print("模型加载完成")
    
    def _predict_base_sentiment(
        self,
        text: str
    ) -> Tuple[int, float]:
        """
        使用 BERT 模型预测基础情感
        
        Args:
            text: 输入文本（已包含市场上下文前缀）
        
        Returns:
            (label, confidence): 标签（-1/0/1）和置信度
        """
        # 分词
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_label = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, pred_label].item()
        
        # 将模型输出（0/1/2）映射回标签（-1/0/1）
        # 假设模型训练时标签顺序为：0->-1, 1->0, 2->1
        label_mapping = {0: -1, 1: 0, 2: 1}
        label = label_mapping.get(pred_label, 0)
        
        return label, confidence
    
    def _apply_priced_in_rule(
        self,
        base_label: int,
        pre_ret: float
    ) -> Optional[str]:
        """
        应用"预期兑现"规则
        
        规则逻辑：
        - 如果 BERT 输出="利好" AND 前期涨幅>阈值 → "利好预期兑现"
        - 如果 BERT 输出="利空" AND 前期跌幅>阈值 → "利空预期兑现"
        
        Args:
            base_label: BERT 基础预测（-1/0/1）
            pre_ret: 前120分钟收益率
        
        Returns:
            如果触发规则，返回 "bullish_priced_in" 或 "bearish_priced_in"
            否则返回 None
        """
        # 利好预期兑现：BERT预测利好 + 前期大涨
        if base_label == 1 and pre_ret > self.PRICED_IN_THRESHOLD:
            return "bullish_priced_in"
        
        # 利空预期兑现：BERT预测利空 + 前期大跌
        if base_label == -1 and pre_ret < -self.PRICED_IN_THRESHOLD:
            return "bearish_priced_in"
        
        return None
    
    def _apply_watch_rule(
        self,
        range_ratio: float,
        abs_ret: float
    ) -> bool:
        """
        应用"建议观望"规则
        
        规则逻辑：
        - 如果 高波动 AND 低净变动 → "建议观望"
        
        Args:
            range_ratio: 波动率（高低价差/开盘价）
            abs_ret: 绝对收益率
        
        Returns:
            是否触发观望规则
        """
        # 高波动 + 低净变动 = 多空分歧，建议观望
        if (range_ratio > self.HIGH_VOLATILITY_THRESHOLD and
            abs_ret < self.LOW_NET_CHANGE_THRESHOLD):
            return True
        
        return False
    
    def _build_enhanced_text(
        self,
        text: str,
        pre_ret: float,
        range_ratio: float
    ) -> str:
        """
        构建增强文本（添加市场上下文前缀）
        
        Args:
            text: 原始新闻文本
            pre_ret: 前120分钟收益率
            range_ratio: 波动率
        
        Returns:
            增强后的文本
        """
        # 根据市场状态选择前缀
        if pre_ret > 0.01:
            prefix = "[Strong Rally]"
        elif pre_ret < -0.01:
            prefix = "[Sharp Decline]"
        elif 0.003 < pre_ret <= 0.01:
            prefix = "[Mild Rally]"
        elif -0.01 <= pre_ret < -0.003:
            prefix = "[Weak Decline]"
        elif abs(pre_ret) < 0.003 and range_ratio > 0.015:
            prefix = "[High Volatility]"
        else:
            prefix = "[Sideways]"
        
        return f"{prefix} {text}"
    
    def analyze(
        self,
        text: str,
        pre_ret: float = 0.0,
        range_ratio: float = 0.0,
        actual: Optional[float] = None,
        consensus: Optional[float] = None
    ) -> Dict:
        """
        分析新闻情感（BERT + 规则引擎）
        
        Args:
            text: 新闻文本
            pre_ret: 前120分钟收益率（默认0）
            range_ratio: 波动率（默认0）
            actual: 实际值（宏观数据，可选）
            consensus: 预期值（宏观数据，可选）
        
        Returns:
            分析结果字典：
            {
                "base_sentiment": "bearish/neutral/bullish",
                "base_confidence": 0.65,
                "final_sentiment": "bearish/neutral/bullish/bullish_priced_in/bearish_priced_in/watch",
                "explanation": "解释文本",
                "recommendation": "建议文本"
            }
        """
        # 1. 构建增强文本
        enhanced_text = self._build_enhanced_text(text, pre_ret, range_ratio)
        
        # 2. BERT 基础预测
        base_label, confidence = self._predict_base_sentiment(enhanced_text)
        base_sentiment = self.LABEL_MAP[base_label]
        
        # 3. 应用规则引擎
        final_sentiment = base_sentiment
        explanation = f"基础情感分析：{base_sentiment}（置信度：{confidence:.2%}）"
        recommendation = ""
        
        # 规则1：预期兑现
        priced_in = self._apply_priced_in_rule(base_label, pre_ret)
        if priced_in:
            final_sentiment = priced_in
            direction = "利好" if base_label == 1 else "利空"
            trend = "大涨" if pre_ret > 0 else "大跌"
            explanation = (
                f"虽然消息{direction}，但前期已{trend}{abs(pre_ret):.2%}，"
                f"可能是{direction}预期兑现"
            )
            recommendation = "建议谨慎追多" if base_label == 1 else "建议谨慎追空"
        
        # 规则2：建议观望（优先级高于预期兑现）
        abs_ret = abs(pre_ret)
        if self._apply_watch_rule(range_ratio, abs_ret):
            final_sentiment = "watch"
            explanation = (
                f"市场波动率较高（{range_ratio:.2%}），但净变动较小（{abs_ret:.2%}），"
                f"多空分歧较大"
            )
            recommendation = "建议观望，等待方向明确"
        
        return {
            "base_sentiment": base_sentiment,
            "base_confidence": confidence,
            "final_sentiment": final_sentiment,
            "explanation": explanation,
            "recommendation": recommendation,
            "market_context": {
                "pre_ret": pre_ret,
                "range_ratio": range_ratio,
                "enhanced_text": enhanced_text
            }
        }
    
    def analyze_batch(
        self,
        texts: list,
        pre_rets: Optional[list] = None,
        range_ratios: Optional[list] = None
    ) -> list:
        """
        批量分析新闻情感（BERT 批处理 + 规则引擎）
        
        批处理优势：
        - 减少模型调用次数
        - 提高 GPU 利用率
        - 降低推理延迟
        
        Args:
            texts: 新闻文本列表
            pre_rets: 前120分钟收益率列表（可选，默认全0）
            range_ratios: 波动率列表（可选，默认全0）
        
        Returns:
            分析结果列表，每个元素为 analyze() 返回的字典
        """
        batch_size = len(texts)
        
        # 填充默认值
        if pre_rets is None:
            pre_rets = [0.0] * batch_size
        if range_ratios is None:
            range_ratios = [0.0] * batch_size
        
        # 验证输入长度
        if len(pre_rets) != batch_size or len(range_ratios) != batch_size:
            raise ValueError(
                f"输入长度不匹配：texts={batch_size}, "
                f"pre_rets={len(pre_rets)}, range_ratios={len(range_ratios)}"
            )
        
        # 1. 构建增强文本（批量）
        enhanced_texts = [
            self._build_enhanced_text(text, pre_ret, range_ratio)
            for text, pre_ret, range_ratio in zip(texts, pre_rets, range_ratios)
        ]
        
        # 2. BERT 批量推理
        base_labels, confidences = self._predict_base_sentiment_batch(enhanced_texts)
        
        # 3. 应用规则引擎（逐个处理）
        results = []
        for i in range(batch_size):
            base_label = base_labels[i]
            confidence = confidences[i]
            pre_ret = pre_rets[i]
            range_ratio = range_ratios[i]
            enhanced_text = enhanced_texts[i]
            
            base_sentiment = self.LABEL_MAP[base_label]
            final_sentiment = base_sentiment
            explanation = f"基础情感分析：{base_sentiment}（置信度：{confidence:.2%}）"
            recommendation = ""
            
            # 规则1：预期兑现
            priced_in = self._apply_priced_in_rule(base_label, pre_ret)
            if priced_in:
                final_sentiment = priced_in
                direction = "利好" if base_label == 1 else "利空"
                trend = "大涨" if pre_ret > 0 else "大跌"
                explanation = (
                    f"虽然消息{direction}，但前期已{trend}{abs(pre_ret):.2%}，"
                    f"可能是{direction}预期兑现"
                )
                recommendation = "建议谨慎追多" if base_label == 1 else "建议谨慎追空"
            
            # 规则2：建议观望
            abs_ret = abs(pre_ret)
            if self._apply_watch_rule(range_ratio, abs_ret):
                final_sentiment = "watch"
                explanation = (
                    f"市场波动率较高（{range_ratio:.2%}），但净变动较小（{abs_ret:.2%}），"
                    f"多空分歧较大"
                )
                recommendation = "建议观望，等待方向明确"
            
            results.append({
                "base_sentiment": base_sentiment,
                "base_confidence": confidence,
                "final_sentiment": final_sentiment,
                "explanation": explanation,
                "recommendation": recommendation,
                "market_context": {
                    "pre_ret": pre_ret,
                    "range_ratio": range_ratio,
                    "enhanced_text": enhanced_text
                }
            })
        
        return results
    
    def _predict_base_sentiment_batch(
        self,
        texts: list
    ) -> Tuple[list, list]:
        """
        使用 BERT 模型批量预测基础情感
        
        Args:
            texts: 输入文本列表（已包含市场上下文前缀）
        
        Returns:
            (labels, confidences): 标签列表和置信度列表
        """
        # 批量分词
        inputs = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 批量推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_labels = torch.argmax(probs, dim=-1).cpu().numpy()
            confidences = torch.max(probs, dim=-1).values.cpu().numpy()
        
        # 将模型输出（0/1/2）映射回标签（-1/0/1）
        label_mapping = {0: -1, 1: 0, 2: 1}
        labels = [label_mapping.get(int(pred), 0) for pred in pred_labels]
        confidences = [float(conf) for conf in confidences]
        
        return labels, confidences


# 便捷函数：创建全局分析器实例
_global_analyzer: Optional[SentimentAnalyzer] = None


def get_analyzer(
    model_path: str = "models/bert_3cls/best",
    force_reload: bool = False
) -> SentimentAnalyzer:
    """
    获取全局情感分析器实例（单例模式）
    
    Args:
        model_path: 模型路径
        force_reload: 是否强制重新加载模型
    
    Returns:
        SentimentAnalyzer 实例
    """
    global _global_analyzer
    
    if _global_analyzer is None or force_reload:
        _global_analyzer = SentimentAnalyzer(model_path=model_path)
    
    return _global_analyzer


if __name__ == "__main__":
    # 测试代码
    print("=" * 80)
    print("情感分析器测试")
    print("=" * 80)
    
    # 创建分析器（需要先训练模型并复制权重）
    try:
        analyzer = SentimentAnalyzer(
            model_path="models/bert_3cls/best"
        )
        
        # 测试案例1：普通利好消息
        print("\n测试案例1：普通利好消息")
        result = analyzer.analyze(
            text="美联储宣布降息25个基点",
            pre_ret=0.002,  # 前期小涨0.2%
            range_ratio=0.005  # 波动率0.5%
        )
        print(f"基础情感: {result['base_sentiment']}")
        print(f"最终情感: {result['final_sentiment']}")
        print(f"解释: {result['explanation']}")
        print(f"建议: {result['recommendation']}")
        
        # 测试案例2：利好预期兑现
        print("\n测试案例2：利好预期兑现")
        result = analyzer.analyze(
            text="美联储宣布降息25个基点",
            pre_ret=0.015,  # 前期大涨1.5%
            range_ratio=0.008  # 波动率0.8%
        )
        print(f"基础情感: {result['base_sentiment']}")
        print(f"最终情感: {result['final_sentiment']}")
        print(f"解释: {result['explanation']}")
        print(f"建议: {result['recommendation']}")
        
        # 测试案例3：建议观望
        print("\n测试案例3：建议观望")
        result = analyzer.analyze(
            text="美国CPI数据公布，符合预期",
            pre_ret=0.001,  # 前期几乎无变动
            range_ratio=0.020  # 高波动2.0%
        )
        print(f"基础情感: {result['base_sentiment']}")
        print(f"最终情感: {result['final_sentiment']}")
        print(f"解释: {result['explanation']}")
        print(f"建议: {result['recommendation']}")
        
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("\n提示：请先完成以下步骤：")
        print("1. 在 Colab 上完成 BERT 3类模型训练")
        print("2. 使用 sync_results.py 同步训练结果到本地")
        print("3. 将最优模型权重复制到 models/bert_3cls/best/")
