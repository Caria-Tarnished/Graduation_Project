# -*- coding: utf-8 -*-
"""
QLoRA 对比实验重新评分（v2 评分标准）

基于系统实际需求重新设计评分维度：
  - 不再奖励"长度"本身，而是评估"是否在合理范围内"
  - JSON 任务重点评估格式合法性和字段完整性
  - 情感分析任务要求有明确结论
  - 所有任务评估"是否有不必要的废话或重复"

用法：
    python scripts/qlora_rescore.py --input thesis_assets/qlora_comparison_results_v2.json
"""
import json
import re
import argparse
from pathlib import Path


# ============================================================
# 各任务类型的理想长度范围（字符数）
# ============================================================
IDEAL_LENGTH = {
    "新闻情感分析": (80, 300),      # 简短判断 + 理由
    "JSON结构化提取": (50, 400),    # JSON 本身不会太长
    "经济指标解读": (100, 400),     # 数据对比 + 影响
    "市场分析": (150, 600),         # 允许稍长
    "财报问答": (100, 500),         # 分析+建议
}


def score_has_conclusion(response, question):
    """是否有明确的结论/判断（0-5）"""
    conclusion_markers = [
        "看涨", "看跌", "中性", "利好", "利空", "Bullish", "Bearish", "Neutral",
        "建议", "综合判断", "综合来看", "总体而言", "因此",
        "超预期", "不及预期", "符合预期",
        "风险较高", "风险较低", "风险可控",
    ]
    count = sum(1 for m in conclusion_markers if m in response)

    # 情感分析类必须有明确方向判断
    if "情感" in question.get("category", "") or "判断" in question.get("instruction", ""):
        direction_words = ["看涨", "看跌", "中性", "利好", "利空", "Bullish", "Bearish", "Neutral"]
        has_direction = any(w in response for w in direction_words)
        if has_direction and count >= 2:
            return 5
        elif has_direction:
            return 4
        elif count >= 1:
            return 2
        else:
            return 1

    if count >= 3:
        return 5
    elif count >= 2:
        return 4
    elif count >= 1:
        return 3
    else:
        return 1


def score_json_quality(response, question):
    """JSON 格式质量（仅对 JSON 类任务评分，0-5）"""
    inst = question.get("instruction", "")
    if "JSON" not in inst and "json" not in inst:
        return None  # 非 JSON 任务不评分

    # 提取 JSON 块
    json_match = re.search(r'[\{\[].*[\}\]]', response, re.DOTALL)
    if not json_match:
        return 0  # 完全没有 JSON

    try:
        parsed = json.loads(json_match.group())
    except json.JSONDecodeError:
        return 1  # 有 JSON 结构但不合法

    # 检查字段完整性
    expected_fields = []
    if "sentiment" in inst.lower():
        expected_fields = ["sentiment", "market_reaction", "impact_duration", "confidence"]
    elif "country" in inst.lower():
        expected_fields = ["country", "actual", "previous", "consensus", "surprise", "direction"]

    if expected_fields:
        if isinstance(parsed, dict):
            present = sum(1 for f in expected_fields if f in parsed)
            ratio = present / len(expected_fields)
            if ratio >= 0.8:
                return 5
            elif ratio >= 0.5:
                return 4
            else:
                return 3
        elif isinstance(parsed, list) and len(parsed) > 0:
            return 4  # 返回了数组，格式不完全匹配但合法
    else:
        return 5  # 合法 JSON 且没有特定字段要求

    return 3


def score_length_appropriateness(response, question):
    """长度是否在合理范围内（0-5）"""
    category = question.get("category", "其他")
    min_len, max_len = IDEAL_LENGTH.get(category, (80, 500))
    length = len(response)

    if length < 20:
        return 1  # 过短，基本没回答
    elif length < min_len * 0.5:
        return 2  # 偏短
    elif min_len <= length <= max_len:
        return 5  # 理想范围
    elif length <= max_len * 1.5:
        return 4  # 稍长但可接受
    elif length <= max_len * 2:
        return 3  # 偏长
    else:
        return 2  # 过于冗长


def score_finance_relevance(response, question):
    """内容与财经分析的相关性（0-5）"""
    # 检查是否包含与输入相关的分析（而非泛泛而谈）
    input_text = question.get("input", "")

    # 提取输入中的关键实体
    entities = []
    for keyword in ["美联储", "加息", "CPI", "非农", "GDP", "黄金", "XAUUSD",
                     "苹果", "特斯拉", "英伟达", "微软", "茅台", "中国", "美国",
                     "降准", "PCE", "PMI", "国债", "半导体",
                     "营收", "净利润", "毛利率", "现金流", "资产负债", "PE", "PB", "ROE"]:
        if keyword in input_text:
            entities.append(keyword)

    if not entities:
        # 输入中没有可识别的实体，只检查基本财经术语
        basic_terms = ["市场", "投资", "经济", "金融", "价格", "交易", "风险"]
        term_count = sum(1 for t in basic_terms if t in response)
        return min(5, 2 + term_count)

    # 检查回答是否引用了输入中的实体
    mentioned = sum(1 for e in entities if e in response)
    ratio = mentioned / len(entities) if entities else 0

    if ratio >= 0.5 and len(response) > 50:
        return 5
    elif ratio >= 0.3:
        return 4
    elif ratio >= 0.1:
        return 3
    else:
        return 2  # 回答与输入关联度低


def score_no_repetition(response):
    """无重复/乱码（0-5）"""
    if len(response) < 20:
        return 3

    # 检查句子级重复
    sentences = re.split(r'[。！？\n]', response)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if len(sentences) >= 2:
        unique_ratio = len(set(sentences)) / len(sentences)
        if unique_ratio < 0.5:
            return 1
        elif unique_ratio < 0.7:
            return 2

    # 检查短语级重复
    if len(response) > 100:
        # 检查 20 字的滑动窗口是否有大量重复
        window = 20
        chunks = [response[i:i+window] for i in range(0, len(response)-window, window)]
        if len(chunks) > 2:
            unique_ratio = len(set(chunks)) / len(chunks)
            if unique_ratio < 0.5:
                return 1
            elif unique_ratio < 0.7:
                return 3

    return 5


def evaluate_response_v2(question, response):
    """v2 评分：面向系统实际需求"""
    scores = {}

    scores["结论明确性"] = score_has_conclusion(response, question)
    scores["长度合理性"] = score_length_appropriateness(response, question)
    scores["内容相关性"] = score_finance_relevance(response, question)
    scores["无重复/乱码"] = score_no_repetition(response)

    json_score = score_json_quality(response, question)
    if json_score is not None:
        scores["JSON格式"] = json_score

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default="thesis_assets/qlora_comparison_results_v2.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"文件不存在: {input_path}")
        print("请先将 Colab 上的结果文件同步到本地")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    details = data.get("details", [])
    if not details:
        print("结果文件中没有 details 数据")
        return

    print("=" * 70)
    print("QLoRA 对比实验 - v2 评分标准重新评估")
    print("=" * 70)

    base_all_scores = []
    ft_all_scores = []

    # 逐题重新评分
    for item in details:
        q = {
            "id": item["id"],
            "category": item["category"],
            "instruction": item.get("instruction", ""),
            "input": item.get("input", ""),
        }

        base_scores = evaluate_response_v2(q, item["base_response"])
        ft_scores = evaluate_response_v2(q, item["ft_response"])

        base_avg = round(sum(base_scores.values()) / len(base_scores), 2)
        ft_avg = round(sum(ft_scores.values()) / len(ft_scores), 2)

        base_all_scores.append({"id": q["id"], "category": q["category"],
                                 "scores": base_scores, "avg": base_avg})
        ft_all_scores.append({"id": q["id"], "category": q["category"],
                               "scores": ft_scores, "avg": ft_avg})

        diff = ft_avg - base_avg
        marker = "+" if diff > 0 else ""
        print(f"  Q{q['id']:02d} ({q['category']}) "
              f"原始={base_avg:.1f}  微调={ft_avg:.1f}  {marker}{diff:.1f}")

    # 按类别统计
    categories = sorted(set(item["category"] for item in details))
    print(f"\n{'='*70}")
    print("按类别汇总")
    print(f"{'='*70}")

    for cat in categories:
        bc = [s for s in base_all_scores if s["category"] == cat]
        fc = [s for s in ft_all_scores if s["category"] == cat]
        ba = sum(s["avg"] for s in bc) / len(bc)
        fa = sum(s["avg"] for s in fc) / len(fc)
        diff = fa - ba
        print(f"\n{cat} ({len(bc)}题):")
        print(f"  原始模型: {ba:.2f}")
        print(f"  微调模型: {fa:.2f}")
        print(f"  变化: {'+' if diff >= 0 else ''}{diff:.2f}")

    # 总体
    bo = sum(s["avg"] for s in base_all_scores) / len(base_all_scores)
    fo = sum(s["avg"] for s in ft_all_scores) / len(ft_all_scores)
    pct = (fo - bo) / bo * 100 if bo > 0 else 0

    print(f"\n{'='*70}")
    print(f"总体: 原始={bo:.2f}  微调={fo:.2f}  变化={fo-bo:+.2f} ({pct:+.1f}%)")
    print(f"{'='*70}")

    # 各维度对比
    all_dims = set()
    for s in base_all_scores + ft_all_scores:
        all_dims.update(s["scores"].keys())

    print("\n各维度详细对比:")
    for dim in sorted(all_dims):
        bd = [s["scores"][dim] for s in base_all_scores if dim in s["scores"]]
        fd = [s["scores"][dim] for s in ft_all_scores if dim in s["scores"]]
        if bd and fd:
            bavg = sum(bd) / len(bd)
            favg = sum(fd) / len(fd)
            print(f"  {dim}: 原始={bavg:.2f}  微调={favg:.2f}  变化={favg-bavg:+.2f}")

    # 输出论文表格
    print(f"\n{'='*70}")
    print("论文表格（v2 评分标准）")
    print(f"{'='*70}")
    print("\n| 评估维度 | 原始模型 | QLoRA微调后 | 变化 |")
    print("| --- | --- | --- | --- |")
    for dim in sorted(all_dims):
        bd = [s["scores"][dim] for s in base_all_scores if dim in s["scores"]]
        fd = [s["scores"][dim] for s in ft_all_scores if dim in s["scores"]]
        if bd and fd:
            bavg = sum(bd) / len(bd)
            favg = sum(fd) / len(fd)
            print(f"| {dim} | {bavg:.2f} | {favg:.2f} | {favg-bavg:+.2f} |")
    print(f"| **综合平均** | **{bo:.2f}** | **{fo:.2f}** | **{fo-bo:+.2f}** |")


if __name__ == "__main__":
    main()
