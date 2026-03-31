# -*- coding: utf-8 -*-
"""
QLoRA 微调前后对比实验 - Colab 运行脚本
============================================================
用途：在相同的 20 个测试问题上，分别使用原始 Deepseek-7B 和 QLoRA 微调后的模型生成答案，
      从格式规范性、财经术语准确性、JSON 结构完整性、回答长度等维度进行量化对比。

运行方式：在 Colab 中按顺序执行以下 Cell。
前置条件：
  - QLoRA 训练已完成，LoRA 权重在 /content/drive/MyDrive/Graduation_Project/qlora_output/
  - Colab Runtime 类型为 GPU (T4 即可)

注意：本脚本会先加载原始模型跑完所有问题，释放显存后再加载微调模型，避免 OOM。
"""

# ============================================================
# Cell 1: 环境准备
# ============================================================
# !pip install -q transformers peft accelerate bitsandbytes>=0.46.1
# from google.colab import drive
# drive.mount('/content/drive')

import os
import gc
import json
import re
import time
import torch
from pathlib import Path
from datetime import datetime

# 配置
BASE_MODEL = "deepseek-ai/deepseek-llm-7b-chat"
LORA_PATH = "/content/drive/MyDrive/Graduation_Project/qlora_output"
OUTPUT_PATH = "/content/drive/MyDrive/Graduation_Project/thesis_assets/qlora_comparison_results.json"

# ============================================================
# Cell 2: 定义 20 个测试问题（覆盖系统的典型使用场景）
# ============================================================

TEST_QUESTIONS = [
    # --- 新闻情感分析类（6题）---
    {
        "id": 1,
        "category": "新闻情感分析",
        "instruction": "分析以下财经新闻对市场的影响，并给出看涨/看跌/中性的判断",
        "input": "美联储宣布加息25个基点，符合市场预期。美联储主席鲍威尔表示未来可能暂停加息。"
    },
    {
        "id": 2,
        "category": "新闻情感分析",
        "instruction": "分析以下财经新闻对市场的影响，并给出看涨/看跌/中性的判断",
        "input": "苹果公司发布2025年第一季度财报，营收同比增长5%至1200亿美元，超出分析师预期。"
    },
    {
        "id": 3,
        "category": "新闻情感分析",
        "instruction": "分析以下财经新闻对市场的影响，并给出看涨/看跌/中性的判断",
        "input": "中国央行宣布降准0.5个百分点，释放约1.2万亿元长期流动性。"
    },
    {
        "id": 4,
        "category": "新闻情感分析",
        "instruction": "分析以下财经新闻对相关股票的影响",
        "input": "特斯拉宣布召回120万辆汽车，涉及自动驾驶软件问题。"
    },
    {
        "id": 5,
        "category": "新闻情感分析",
        "instruction": "分析以下财经新闻中的关键财务指标",
        "input": "英伟达2024财年第四季度营收达221亿美元，同比增长265%，数据中心收入创历史新高。"
    },
    {
        "id": 6,
        "category": "新闻情感分析",
        "instruction": "判断以下新闻是否存在'预期兑现'现象",
        "input": "市场此前已连续上涨三周，今日公布的GDP数据好于预期，但大盘冲高回落收跌1.2%。"
    },
    # --- JSON 结构化提取类（4题）---
    {
        "id": 7,
        "category": "JSON结构化提取",
        "instruction": "请提取以下新闻中的关键财务指标，以JSON格式输出，包含字段：company, metric, value, period",
        "input": "微软2024年第三季度云服务Azure收入增长29%，总营收达618亿美元。"
    },
    {
        "id": 8,
        "category": "JSON结构化提取",
        "instruction": "请提取以下新闻中的关键信息，以JSON格式输出",
        "input": "高盛将贵州茅台目标价上调至2100元，评级维持买入，预计2025年净利润增长12%。"
    },
    {
        "id": 9,
        "category": "JSON结构化提取",
        "instruction": "请从以下文本中提取事件信息，以JSON格式输出，包含字段：event_type, entity, impact, magnitude",
        "input": "沙特阿美宣布将4月份亚洲客户原油官方售价每桶下调2美元，为连续第二个月下调。"
    },
    {
        "id": 10,
        "category": "JSON结构化提取",
        "instruction": "请提取以下新闻中的所有数字指标，以JSON数组格式输出",
        "input": "比亚迪3月销量达30.2万辆，同比增长46%，新能源车渗透率首次突破50%，出口量达2.8万辆。"
    },
    # --- 市场分析与解释类（5题）---
    {
        "id": 11,
        "category": "市场分析",
        "instruction": "解释以下市场现象的可能原因",
        "input": "美国10年期国债收益率突破5%，但黄金价格不跌反涨，达到历史新高。"
    },
    {
        "id": 12,
        "category": "市场分析",
        "instruction": "分析以下宏观经济数据对A股市场的影响",
        "input": "中国3月CPI同比上涨0.1%，PPI同比下降2.8%，社会融资规模增量5.38万亿元。"
    },
    {
        "id": 13,
        "category": "市场分析",
        "instruction": "解释什么是量化宽松政策以及其对股票市场的影响",
        "input": ""
    },
    {
        "id": 14,
        "category": "市场分析",
        "instruction": "分析美元指数走强对新兴市场的影响",
        "input": "美元指数DXY本周突破107关口，创近两年新高，多个新兴市场货币大幅贬值。"
    },
    {
        "id": 15,
        "category": "市场分析",
        "instruction": "分析以下行业趋势",
        "input": "全球半导体行业资本支出2024年预计达1700亿美元，其中AI芯片占比从15%提升至35%。"
    },
    # --- 财报问答类（5题）---
    {
        "id": 16,
        "category": "财报问答",
        "instruction": "根据以下财报摘要回答：该公司的盈利能力如何？",
        "input": "某公司2024年报：营收500亿元(+8%)，毛利率45.2%(+1.5pp)，净利润80亿元(+15%)，经营性现金流120亿元。"
    },
    {
        "id": 17,
        "category": "财报问答",
        "instruction": "根据以下信息判断该公司的财务健康状况",
        "input": "资产负债率72%，流动比率0.8，速动比率0.5，利息保障倍数2.1，应收账款周转天数从45天增加到68天。"
    },
    {
        "id": 18,
        "category": "财报问答",
        "instruction": "对比以下两家公司的估值水平",
        "input": "公司A：PE 25倍，PB 3.2倍，ROE 15%，营收增速20%。公司B：PE 15倍，PB 1.8倍，ROE 12%，营收增速8%。"
    },
    {
        "id": 19,
        "category": "财报问答",
        "instruction": "解读以下现金流数据，判断公司是否存在财务风险",
        "input": "经营活动现金流-5亿元，投资活动现金流-30亿元，筹资活动现金流+40亿元，期末现金余额15亿元。"
    },
    {
        "id": 20,
        "category": "财报问答",
        "instruction": "分析以下研发投入数据，评估公司的创新能力",
        "input": "研发费用50亿元，占营收比例10%，同比增长25%。研发人员占比35%，专利申请数量同比增长40%。"
    },
]

# ============================================================
# Cell 3: 评估函数
# ============================================================

def evaluate_response(question, response):
    """
    对单个回答进行多维度自动评分（0-5分制）
    """
    scores = {}

    # 1. 回答长度（太短说明敷衍，太长说明废话多）
    length = len(response)
    if length < 20:
        scores["回答完整性"] = 1
    elif length < 50:
        scores["回答完整性"] = 2
    elif length < 100:
        scores["回答完整性"] = 3
    elif length < 500:
        scores["回答完整性"] = 4
    else:
        scores["回答完整性"] = 5

    # 2. JSON 格式规范性（仅对 JSON 提取类题目评分）
    if "JSON" in question["instruction"] or "json" in question["instruction"]:
        # 尝试提取 JSON
        json_match = re.search(r'[\{\[].*[\}\]]', response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                scores["JSON格式"] = 5
            except json.JSONDecodeError:
                scores["JSON格式"] = 2  # 有 JSON 结构但不合法
        else:
            scores["JSON格式"] = 0

    # 3. 财经术语使用（检查是否包含专业术语）
    finance_terms = [
        "利率", "通胀", "GDP", "CPI", "PPI", "营收", "净利润", "毛利率",
        "估值", "PE", "PB", "ROE", "现金流", "资产负债", "流动性",
        "看涨", "看跌", "中性", "利好", "利空", "预期", "超预期",
        "加息", "降息", "量化宽松", "紧缩", "降准", "汇率",
        "bullish", "bearish", "neutral", "revenue", "earnings",
        "同比", "环比", "增速", "波动", "风险", "收益率"
    ]
    term_count = sum(1 for term in finance_terms if term in response)
    if term_count >= 5:
        scores["财经术语"] = 5
    elif term_count >= 3:
        scores["财经术语"] = 4
    elif term_count >= 1:
        scores["财经术语"] = 3
    else:
        scores["财经术语"] = 1

    # 4. 逻辑连贯性（简单检查：是否有分析过程，而非直接给结论）
    logic_markers = ["因为", "因此", "所以", "由于", "导致", "表明", "意味着",
                     "一方面", "另一方面", "首先", "其次", "综合来看",
                     "because", "therefore", "indicates", "suggests"]
    logic_count = sum(1 for m in logic_markers if m in response)
    if logic_count >= 3:
        scores["逻辑连贯性"] = 5
    elif logic_count >= 2:
        scores["逻辑连贯性"] = 4
    elif logic_count >= 1:
        scores["逻辑连贯性"] = 3
    else:
        scores["逻辑连贯性"] = 2

    # 5. 是否出现明显的错误/幻觉标志（重复、乱码、无关内容）
    if response.count(response[:20]) > 3 and len(response) > 100:
        scores["无重复/乱码"] = 1  # 严重重复
    elif len(set(response)) < 20 and len(response) > 50:
        scores["无重复/乱码"] = 2  # 字符种类太少
    else:
        scores["无重复/乱码"] = 5

    return scores


def generate_batch(model, tokenizer, questions, model_name="model"):
    """
    对一组问题逐个生成回答，记录耗时
    """
    results = []
    for q in questions:
        prompt = f"User: {q['instruction']}\n{q['input']}\n\nAssistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        elapsed = time.time() - start_time

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

        scores = evaluate_response(q, response)

        results.append({
            "id": q["id"],
            "category": q["category"],
            "instruction": q["instruction"],
            "input": q["input"],
            "response": response,
            "scores": scores,
            "avg_score": round(sum(scores.values()) / len(scores), 2),
            "time_seconds": round(elapsed, 2),
            "response_length": len(response)
        })

        print(f"  [{model_name}] Q{q['id']:02d} ({q['category']}) "
              f"avg={results[-1]['avg_score']:.1f} len={len(response)} "
              f"time={elapsed:.1f}s")

    return results


# ============================================================
# Cell 4: 加载原始模型并生成回答
# ============================================================

print("=" * 60)
print("Phase 1: 原始 Deepseek-7B (无微调)")
print("=" * 60)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

print("加载原始模型 (4-bit)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
base_model.eval()
print("原始模型加载完成\n")

base_results = generate_batch(base_model, tokenizer, TEST_QUESTIONS, "原始模型")

# 释放显存
del base_model
gc.collect()
torch.cuda.empty_cache()
print("\n原始模型已释放显存")

# ============================================================
# Cell 5: 加载微调模型并生成回答
# ============================================================

print("\n" + "=" * 60)
print("Phase 2: QLoRA 微调后的 Deepseek-7B")
print("=" * 60)

from peft import PeftModel

print("加载基础模型 (4-bit)...")
base_model_ft = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

print("加载 LoRA 权重...")
ft_model = PeftModel.from_pretrained(base_model_ft, LORA_PATH)
ft_model.eval()
print("微调模型加载完成\n")

ft_results = generate_batch(ft_model, tokenizer, TEST_QUESTIONS, "微调模型")

del ft_model, base_model_ft
gc.collect()
torch.cuda.empty_cache()
print("\n微调模型已释放显存")

# ============================================================
# Cell 6: 汇总对比结果
# ============================================================

print("\n" + "=" * 60)
print("对比结果汇总")
print("=" * 60)

# 按类别统计
categories = sorted(set(q["category"] for q in TEST_QUESTIONS))
summary = {"by_category": {}, "overall": {}, "details": []}

for cat in categories:
    base_cat = [r for r in base_results if r["category"] == cat]
    ft_cat = [r for r in ft_results if r["category"] == cat]

    base_avg = sum(r["avg_score"] for r in base_cat) / len(base_cat)
    ft_avg = sum(r["avg_score"] for r in ft_cat) / len(ft_cat)
    base_time = sum(r["time_seconds"] for r in base_cat) / len(base_cat)
    ft_time = sum(r["time_seconds"] for r in ft_cat) / len(ft_cat)
    base_len = sum(r["response_length"] for r in base_cat) / len(base_cat)
    ft_len = sum(r["response_length"] for r in ft_cat) / len(ft_cat)

    summary["by_category"][cat] = {
        "num_questions": len(base_cat),
        "base_avg_score": round(base_avg, 2),
        "ft_avg_score": round(ft_avg, 2),
        "improvement": round(ft_avg - base_avg, 2),
        "base_avg_time": round(base_time, 2),
        "ft_avg_time": round(ft_time, 2),
        "base_avg_length": round(base_len, 1),
        "ft_avg_length": round(ft_len, 1),
    }

    print(f"\n{cat} ({len(base_cat)}题):")
    print(f"  原始模型  平均分={base_avg:.2f}  平均耗时={base_time:.1f}s  平均长度={base_len:.0f}字")
    print(f"  微调模型  平均分={ft_avg:.2f}  平均耗时={ft_time:.1f}s  平均长度={ft_len:.0f}字")
    print(f"  提升: {'+' if ft_avg >= base_avg else ''}{ft_avg - base_avg:.2f}")

# 总体统计
base_overall = sum(r["avg_score"] for r in base_results) / len(base_results)
ft_overall = sum(r["avg_score"] for r in ft_results) / len(ft_results)
base_time_overall = sum(r["time_seconds"] for r in base_results) / len(base_results)
ft_time_overall = sum(r["time_seconds"] for r in ft_results) / len(ft_results)

summary["overall"] = {
    "base_avg_score": round(base_overall, 2),
    "ft_avg_score": round(ft_overall, 2),
    "improvement": round(ft_overall - base_overall, 2),
    "improvement_pct": round((ft_overall - base_overall) / base_overall * 100, 1) if base_overall > 0 else 0,
    "base_avg_time": round(base_time_overall, 2),
    "ft_avg_time": round(ft_time_overall, 2),
    "num_questions": len(TEST_QUESTIONS),
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

# 各评估维度的详细对比
all_dims = set()
for r in base_results + ft_results:
    all_dims.update(r["scores"].keys())

print(f"\n{'='*60}")
print(f"总体: 原始={base_overall:.2f}  微调={ft_overall:.2f}  "
      f"提升={ft_overall - base_overall:+.2f} ({summary['overall']['improvement_pct']:+.1f}%)")
print(f"{'='*60}")

print("\n各维度详细对比:")
for dim in sorted(all_dims):
    base_dim_scores = [r["scores"].get(dim, 0) for r in base_results if dim in r["scores"]]
    ft_dim_scores = [r["scores"].get(dim, 0) for r in ft_results if dim in r["scores"]]
    if base_dim_scores and ft_dim_scores:
        b_avg = sum(base_dim_scores) / len(base_dim_scores)
        f_avg = sum(ft_dim_scores) / len(ft_dim_scores)
        print(f"  {dim}: 原始={b_avg:.2f}  微调={f_avg:.2f}  提升={f_avg - b_avg:+.2f}")

# 保存详细结果
for i in range(len(base_results)):
    summary["details"].append({
        "id": base_results[i]["id"],
        "category": base_results[i]["category"],
        "instruction": base_results[i]["instruction"],
        "input": base_results[i]["input"],
        "base_response": base_results[i]["response"],
        "ft_response": ft_results[i]["response"],
        "base_scores": base_results[i]["scores"],
        "ft_scores": ft_results[i]["scores"],
        "base_avg": base_results[i]["avg_score"],
        "ft_avg": ft_results[i]["avg_score"],
    })

# 保存到 Drive
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n详细结果已保存到: {OUTPUT_PATH}")
print("完成！请将此文件同步到本地后用于填充论文对比表格。")

# ============================================================
# Cell 7: 生成论文可用的表格数据（方便直接粘贴）
# ============================================================

print("\n" + "=" * 60)
print("论文表格数据 (直接可用)")
print("=" * 60)

print("\n| 评估维度 | 原始模型 | QLoRA微调后 | 提升 |")
print("| --- | --- | --- | --- |")

for dim in sorted(all_dims):
    base_dim_scores = [r["scores"].get(dim, 0) for r in base_results if dim in r["scores"]]
    ft_dim_scores = [r["scores"].get(dim, 0) for r in ft_results if dim in r["scores"]]
    if base_dim_scores and ft_dim_scores:
        b_avg = sum(base_dim_scores) / len(base_dim_scores)
        f_avg = sum(ft_dim_scores) / len(ft_dim_scores)
        diff = f_avg - b_avg
        print(f"| {dim} | {b_avg:.2f} | {f_avg:.2f} | {diff:+.2f} |")

print(f"| **综合平均** | **{base_overall:.2f}** | **{ft_overall:.2f}** | **{ft_overall - base_overall:+.2f}** |")
print(f"\n平均生成耗时: 原始={base_time_overall:.1f}s  微调={ft_time_overall:.1f}s")
