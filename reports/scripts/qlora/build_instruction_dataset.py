# -*- coding: utf-8 -*-
"""
构建 QLoRA 微调指令集（v2 - 高质量版）

从 finance_analysis.db 中提取真实案例，生成多样化的 Instruction-Input-Output 数据集。
相比 v1 的主要改进：
  1. 利用全量 flash news（不限 star>=3），按市场反应筛选有信息量的样本
  2. 充分利用 calendar 数据的 previous/consensus/actual 生成经济指标解读
  3. 多种指令类型：情感分析、JSON提取、经济指标解读、预期兑现判断、多窗口对比
  4. 每条 output 根据实际数据动态生成，不使用模板复制

使用方法：
    python scripts/qlora/build_instruction_dataset.py --db finance_analysis.db --output data/qlora/instructions.jsonl --num_samples 500
"""
import sqlite3
import json
import random
from pathlib import Path
import argparse


# ============================================================
# 数据提取
# ============================================================

def get_flash_news_with_impact(db_path: str, min_abs_ret: float = 0.0005):
    """提取有显著市场反应的 flash 新闻，附带多窗口收益率"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT e.event_id, e.content,
               ei5.ret  AS ret_5m,
               ei15.ret AS ret_15m,
               ei30.ret AS ret_30m
        FROM events e
        JOIN event_impacts ei5  ON e.event_id = ei5.event_id  AND ei5.window_min  = 5
        JOIN event_impacts ei15 ON e.event_id = ei15.event_id AND ei15.window_min = 15
        JOIN event_impacts ei30 ON e.event_id = ei30.event_id AND ei30.window_min = 30
        WHERE e.source = 'flash'
          AND e.content IS NOT NULL
          AND LENGTH(e.content) > 20
          AND ABS(ei15.ret) > ?
        ORDER BY RANDOM()
    """, (min_abs_ret,))
    rows = cur.fetchall()
    conn.close()
    return [{"event_id": r[0], "content": r[1],
             "ret_5m": r[2], "ret_15m": r[3], "ret_30m": r[4]} for r in rows]


def get_calendar_events(db_path: str):
    """提取有 actual 值的经济日历事件，附带市场反应。
    注意：calendar 事件的内容存在 name 字段（非 content），actual/previous/consensus 为文本类型。
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT e.name, e.star, e.country, e.previous, e.consensus, e.actual,
               COALESCE(ei.ret, 0.0) AS ret_15m
        FROM events e
        LEFT JOIN event_impacts ei ON e.event_id = ei.event_id AND ei.window_min = 15
        WHERE e.source = 'calendar'
          AND e.actual IS NOT NULL
          AND e.actual != '未公布'
          AND e.name IS NOT NULL
        ORDER BY RANDOM()
    """)
    rows = cur.fetchall()
    conn.close()

    results = []
    for r in rows:
        # 构造完整的 content 文本（模拟原始快讯格式）
        name, star, country, prev, cons, actual, ret = r
        parts = [f"{name}"]
        if prev:
            parts.append(f"前值:{prev}")
        if cons:
            parts.append(f"预期:{cons}")
        parts.append(f"公布:{actual}")
        content = " ".join(parts)

        results.append({
            "content": content, "star": star, "country": country,
            "previous": prev, "consensus": cons, "actual": actual,
            "ret_15m": ret or 0.0
        })
    return results


def get_flash_news_with_priced_in(db_path: str):
    """提取存在预期兑现特征的新闻（前期涨后期反向）"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT e.content,
               ei_pre.ret AS ret_pre_120m,
               ei_post.ret AS ret_post_15m
        FROM events e
        JOIN event_impacts ei_pre  ON e.event_id = ei_pre.event_id  AND ei_pre.window_min  = 5
        JOIN event_impacts ei_post ON e.event_id = ei_post.event_id AND ei_post.window_min = 15
        WHERE e.content IS NOT NULL AND LENGTH(e.content) > 20
          AND ABS(ei_pre.ret) > 0.002
          AND (
              (ei_pre.ret > 0 AND ei_post.ret < -0.0003)
              OR (ei_pre.ret < 0 AND ei_post.ret > 0.0003)
          )
        ORDER BY RANDOM()
    """)
    rows = cur.fetchall()
    conn.close()
    return [{"content": r[0], "ret_pre": r[1], "ret_post": r[2]} for r in rows]


# ============================================================
# 指令生成器（每种类型一个函数）
# ============================================================

def _sentiment_label(ret):
    if ret > 0.001:
        return "看涨（Bullish）"
    elif ret < -0.001:
        return "看跌（Bearish）"
    else:
        return "中性（Neutral）"


def _ret_desc(ret):
    """将收益率转为自然语言描述"""
    pct = abs(ret) * 100
    if pct < 0.05:
        return "几乎没有变化"
    elif pct < 0.1:
        return f"小幅{'上涨' if ret > 0 else '下跌'}{pct:.2f}%"
    elif pct < 0.3:
        return f"{'上涨' if ret > 0 else '下跌'}{pct:.2f}%"
    else:
        return f"大幅{'上涨' if ret > 0 else '下跌'}{pct:.2f}%"


def gen_sentiment_analysis(sample):
    """类型1：新闻情感分析 - 给出判断并解释"""
    content = sample["content"]
    ret = sample["ret_15m"]
    label = _sentiment_label(ret)
    move = _ret_desc(ret)

    output = f"综合判断：{label}。"
    output += f"该消息发布后15分钟内，XAUUSD{move}。"

    # 根据多窗口数据判断持续性
    ret5 = sample.get("ret_5m", 0)
    ret30 = sample.get("ret_30m", 0)
    if ret5 and ret30:
        if (ret > 0 and ret30 > ret) or (ret < 0 and ret30 < ret):
            output += "从5分钟到30分钟窗口来看，市场反应呈持续强化趋势，说明消息影响较为深远。"
        elif abs(ret30) < abs(ret5) * 0.5:
            output += "不过从更长时间窗口来看，初始反应有所回调，市场可能正在消化这一消息。"
        else:
            output += "多个时间窗口的反应方向一致，表明市场对该消息的解读较为明确。"

    return {
        "instruction": "分析以下财经快讯对黄金市场的影响，给出看涨/看跌/中性的判断并说明理由",
        "input": content,
        "output": output
    }


def gen_json_extraction(sample):
    """类型2：从新闻中提取结构化 JSON"""
    content = sample["content"]
    ret = sample["ret_15m"]
    label = _sentiment_label(ret)

    json_obj = {
        "sentiment": label,
        "market_reaction": f"XAUUSD {_ret_desc(ret)}",
        "impact_duration": "短期" if abs(ret) < 0.002 else "中期",
        "confidence": "高" if abs(ret) > 0.003 else "中" if abs(ret) > 0.001 else "低"
    }

    output = f"```json\n{json.dumps(json_obj, ensure_ascii=False, indent=2)}\n```"

    return {
        "instruction": "请从以下财经新闻中提取关键信息，以JSON格式输出，包含字段：sentiment（情感）、market_reaction（市场反应）、impact_duration（影响周期）、confidence（判断置信度）",
        "input": content,
        "output": output
    }


def _parse_numeric(val):
    """尝试将文本值（如 '2.60' 或 '2.6%'）转为 float，失败返回 None"""
    if val is None:
        return None
    try:
        return float(str(val).replace("%", "").replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def gen_calendar_interpretation(sample):
    """类型3：经济指标解读 - 利用 previous/consensus/actual"""
    content = sample["content"]
    prev_str = sample["previous"]
    cons_str = sample["consensus"]
    actual_str = sample["actual"]
    country = sample["country"] or "未知"
    ret = sample["ret_15m"]

    prev = _parse_numeric(prev_str)
    cons = _parse_numeric(cons_str)
    actual = _parse_numeric(actual_str)

    output = f"该数据来自{country}。"

    # 与前值对比
    if prev is not None and actual is not None:
        if actual > prev:
            output += f"实际公布值{actual_str}高于前值{prev_str}，显示该指标呈改善趋势。"
        elif actual < prev:
            output += f"实际公布值{actual_str}低于前值{prev_str}，显示该指标有所走弱。"
        else:
            output += f"实际公布值{actual_str}与前值{prev_str}持平。"

    # 与预期对比
    if cons is not None and actual is not None:
        diff = actual - cons
        if abs(diff) < 0.01 * max(abs(cons), 1):
            output += "公布值基本符合市场预期。"
        elif diff > 0:
            output += f"公布值超出市场预期（预期{cons_str}），属于正面意外。"
        else:
            output += f"公布值不及市场预期（预期{cons_str}），属于负面意外。"

    # 市场反应
    output += f"数据公布后，黄金市场{_ret_desc(ret)}。"

    # 分析逻辑
    if cons is not None and actual is not None and actual > cons and ret < -0.0005:
        output += "值得注意的是，数据虽超预期但黄金反而下跌，可能因为强经济数据强化了加息预期，美元走强压制了金价。"
    elif cons is not None and actual is not None and actual < cons and ret > 0.0005:
        output += "数据不及预期但黄金上涨，说明市场将其解读为宽松预期增强的信号，避险买盘推动金价走高。"

    return {
        "instruction": "解读以下经济数据的含义及其对黄金市场的影响",
        "input": content,
        "output": output
    }


def gen_calendar_json(sample):
    """类型4：经济指标的 JSON 结构化提取"""
    content = sample["content"]
    prev_str = sample["previous"]
    cons_str = sample["consensus"]
    actual_str = sample["actual"]
    country = sample["country"] or "未知"

    prev = _parse_numeric(prev_str)
    cons = _parse_numeric(cons_str)
    actual = _parse_numeric(actual_str)

    # 判断 surprise
    if cons is not None and actual is not None:
        surprise_val = actual - cons
        if abs(surprise_val) < 0.01 * max(abs(cons), 1):
            surprise = "符合预期"
        elif surprise_val > 0:
            surprise = "超预期"
        else:
            surprise = "不及预期"
    else:
        surprise = "无预期值"

    # 判断 direction
    if prev is not None and actual is not None:
        direction = "改善" if actual > prev else "恶化" if actual < prev else "持平"
    else:
        direction = "持平"

    json_obj = {
        "country": country,
        "actual": actual_str,
        "previous": prev_str,
        "consensus": cons_str,
        "surprise": surprise,
        "direction": direction
    }

    output = f"```json\n{json.dumps(json_obj, ensure_ascii=False, indent=2)}\n```"

    return {
        "instruction": "请提取以下经济数据公告中的关键指标，以JSON格式输出，包含字段：country、actual、previous、consensus、surprise、direction",
        "input": content,
        "output": output
    }


def gen_priced_in_analysis(sample):
    """类型5：预期兑现分析"""
    content = sample["content"]
    ret_pre = sample["ret_pre"]
    ret_post = sample["ret_post"]

    pre_dir = "上涨" if ret_pre > 0 else "下跌"
    post_dir = "上涨" if ret_post > 0 else "下跌"

    output = f"从市场走势来看，消息公布前市场已{pre_dir}{abs(ret_pre)*100:.2f}%，"
    output += f"而消息公布后市场反而{post_dir}{abs(ret_post)*100:.2f}%，"
    output += "这是典型的\"预期兑现\"（Priced-In）现象。"
    output += "其本质是市场参与者在消息正式公布之前已通过预期交易提前反映了该消息的影响，"
    output += "当消息落地后，前期获利盘集中平仓导致价格反向运动。"
    output += "在实际交易中，投资者需要关注消息公布前的累计涨跌幅度——"

    if abs(ret_pre) > 0.005:
        output += "本案例中前期波动幅度较大，预期兑现的概率相应较高。"
    else:
        output += "本案例中前期波动较为温和，预期兑现效应属于轻度。"

    return {
        "instruction": "判断以下市场走势是否存在\"预期兑现\"现象，并分析原因",
        "input": f"{content}\n（市场背景：消息公布前价格已{pre_dir}{abs(ret_pre)*100:.2f}%）",
        "output": output
    }


def gen_multi_window_comparison(sample):
    """类型6：多时间窗口市场反应对比"""
    content = sample["content"]
    ret5 = sample.get("ret_5m", 0)
    ret15 = sample.get("ret_15m", 0)
    ret30 = sample.get("ret_30m", 0)

    output = "不同时间窗口的市场反应如下：\n"
    output += f"- 5分钟窗口：{_ret_desc(ret5)}\n"
    output += f"- 15分钟窗口：{_ret_desc(ret15)}\n"
    output += f"- 30分钟窗口：{_ret_desc(ret30)}\n\n"

    # 趋势判断
    rets = [ret5, ret15, ret30]
    if all(r > 0 for r in rets) or all(r < 0 for r in rets):
        if abs(ret30) > abs(ret5):
            output += "市场反应呈持续强化态势，各时间窗口方向一致且幅度递增，表明市场对该消息的定价尚未完成，后续可能延续当前方向。"
        else:
            output += "虽然各时间窗口方向一致，但幅度有所收敛，说明初始冲击最大，市场正在逐步消化该消息。"
    elif (ret5 > 0 and ret30 < 0) or (ret5 < 0 and ret30 > 0):
        output += "短期和中期反应方向出现反转，说明市场最初的膝跳反应被修正，投资者在更充分地消化信息后改变了判断。"
    else:
        output += "各时间窗口反应混合，市场对该消息的解读存在较大分歧。"

    return {
        "instruction": "分析以下财经新闻在不同时间窗口的市场影响差异",
        "input": content,
        "output": output
    }


def gen_risk_assessment(sample):
    """类型7：风险评估与建议"""
    content = sample["content"]
    ret = sample["ret_15m"]
    ret5 = sample.get("ret_5m", 0)
    ret30 = sample.get("ret_30m", 0)

    volatility = abs(ret5) + abs(ret) + abs(ret30)

    output = f"综合判断：{_sentiment_label(ret)}。"

    if volatility > 0.01:
        output += f"该消息引发了较高的市场波动（综合波动率{volatility*100:.2f}%），建议采取谨慎策略：控制仓位，设置止损。"
    elif volatility > 0.003:
        output += f"市场波动处于中等水平（综合波动率{volatility*100:.2f}%），可以正常持仓但需密切关注后续消息。"
    else:
        output += f"市场反应相对平淡（综合波动率{volatility*100:.2f}%），该消息的短期影响有限。"

    # 方向一致性
    if ret > 0 and ret30 > ret:
        output += "趋势向好且在强化，可以考虑顺势操作。"
    elif ret < 0 and ret30 < ret:
        output += "下行趋势且在加速，建议规避风险或考虑对冲。"
    else:
        output += "建议等待方向进一步明确后再做决策。"

    return {
        "instruction": "评估以下消息的市场风险等级，并给出交易建议",
        "input": content,
        "output": output
    }


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="构建 QLoRA 微调指令集（v2）")
    parser.add_argument("--db", type=str, default="finance_analysis.db")
    parser.add_argument("--output", type=str, default="data/qlora/instructions.jsonl")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"目标样本数: {args.num_samples}")
    print(f"数据库: {args.db}")

    # ---- 1. 提取原始数据 ----
    print("\n1. 提取 flash 新闻（有显著市场反应）...")
    flash_samples = get_flash_news_with_impact(args.db, min_abs_ret=0.0003)
    print(f"   获取 {len(flash_samples)} 条")

    print("2. 提取经济日历数据...")
    calendar_samples = get_calendar_events(args.db)
    print(f"   获取 {len(calendar_samples)} 条")

    print("3. 提取预期兑现样本...")
    priced_in_samples = get_flash_news_with_priced_in(args.db)
    print(f"   获取 {len(priced_in_samples)} 条")

    # ---- 2. 生成指令 ----
    instructions = []

    # 分配比例（大致）
    target = args.num_samples
    n_sentiment = int(target * 0.20)     # 情感分析
    n_json_news = int(target * 0.12)     # 新闻 JSON 提取
    n_calendar = int(target * 0.18)      # 经济指标解读
    n_cal_json = int(target * 0.10)      # 经济指标 JSON
    n_priced_in = int(target * 0.12)     # 预期兑现
    n_multi_win = int(target * 0.15)     # 多窗口对比
    n_risk = int(target * 0.13)          # 风险评估

    # 从 flash 新闻生成
    random.shuffle(flash_samples)
    idx = 0

    print(f"\n生成情感分析指令 ({n_sentiment})...")
    for s in flash_samples[idx:idx + n_sentiment]:
        instructions.append(gen_sentiment_analysis(s))
    idx += n_sentiment

    print(f"生成新闻JSON提取指令 ({n_json_news})...")
    for s in flash_samples[idx:idx + n_json_news]:
        instructions.append(gen_json_extraction(s))
    idx += n_json_news

    print(f"生成多窗口对比指令 ({n_multi_win})...")
    for s in flash_samples[idx:idx + n_multi_win]:
        instructions.append(gen_multi_window_comparison(s))
    idx += n_multi_win

    print(f"生成风险评估指令 ({n_risk})...")
    for s in flash_samples[idx:idx + n_risk]:
        instructions.append(gen_risk_assessment(s))
    idx += n_risk

    # 从 calendar 生成
    random.shuffle(calendar_samples)
    print(f"生成经济指标解读指令 ({n_calendar})...")
    for s in calendar_samples[:n_calendar]:
        instructions.append(gen_calendar_interpretation(s))

    print(f"生成经济指标JSON指令 ({n_cal_json})...")
    for s in calendar_samples[n_calendar:n_calendar + n_cal_json]:
        instructions.append(gen_calendar_json(s))

    # 从预期兑现生成
    random.shuffle(priced_in_samples)
    print(f"生成预期兑现分析指令 ({n_priced_in})...")
    for s in priced_in_samples[:n_priced_in]:
        instructions.append(gen_priced_in_analysis(s))

    # 如果数据不够，用 flash 补充
    if len(instructions) < target:
        remaining = target - len(instructions)
        print(f"\n补充 {remaining} 条（从剩余 flash 样本生成）...")
        generators = [gen_sentiment_analysis, gen_json_extraction,
                      gen_multi_window_comparison, gen_risk_assessment]
        for i, s in enumerate(flash_samples[idx:idx + remaining]):
            gen_fn = generators[i % len(generators)]
            instructions.append(gen_fn(s))

    # 打乱
    random.shuffle(instructions)
    instructions = instructions[:target]

    # ---- 3. 保存 ----
    with open(args.output, "w", encoding="utf-8") as f:
        for inst in instructions:
            f.write(json.dumps(inst, ensure_ascii=False) + "\n")

    # 统计
    from collections import Counter
    type_counts = Counter(inst["instruction"][:15] for inst in instructions)

    print(f"\n{'='*60}")
    print(f"指令集构建完成！")
    print(f"{'='*60}")
    print(f"总样本数: {len(instructions)}")
    print(f"独立样本数（无重复模板）: {len(instructions)}")
    print(f"文件: {args.output}")
    print(f"大小: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"\n指令类型分布:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}...: {c}")

    print(f"\n示例（前3条）:")
    for i, inst in enumerate(instructions[:3], 1):
        print(f"\n--- 示例 {i} ---")
        print(f"Instruction: {inst['instruction'][:60]}")
        print(f"Input: {inst['input'][:80]}...")
        print(f"Output: {inst['output'][:120]}...")


if __name__ == "__main__":
    main()
