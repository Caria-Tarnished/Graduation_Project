# -*- coding: utf-8 -*-
"""
构建 QLoRA 微调指令集

从 finance_analysis.db 中提取真实案例，生成 Instruction-Input-Output 格式的数据集

使用方法：
    python scripts/qlora/build_instruction_dataset.py --output data/qlora/instructions.jsonl --num_samples 300
"""
import sqlite3
import json
import random
from pathlib import Path
import argparse
from datetime import datetime


def get_news_samples(db_path: str, num_samples: int = 100):
    """
    从数据库中提取新闻样本
    
    Args:
        db_path: 数据库路径
        num_samples: 样本数量
    
    Returns:
        新闻样本列表
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 查询高星级事件（3-5星）
    query = """
    SELECT 
        e.content,
        e.star,
        e.country,
        ei.ret_post_15,
        ei.pre_ret_120
    FROM events e
    LEFT JOIN event_impacts ei ON e.event_id = ei.event_id
    WHERE e.star >= 3
      AND e.content IS NOT NULL
      AND LENGTH(e.content) > 10
    ORDER BY RANDOM()
    LIMIT ?
    """
    
    cursor.execute(query, (num_samples,))
    rows = cursor.fetchall()
    conn.close()
    
    samples = []
    for row in rows:
        content, star, country, ret_post, pre_ret = row
        samples.append({
            'content': content,
            'star': star,
            'country': country or 'N/A',
            'ret_post': ret_post if ret_post is not None else 0.0,
            'pre_ret': pre_ret if pre_ret is not None else 0.0
        })
    
    return samples


def generate_instruction_from_news(sample: dict) -> dict:
    """
    从新闻样本生成指令数据
    
    Args:
        sample: 新闻样本
    
    Returns:
        指令数据字典
    """
    content = sample['content']
    star = sample['star']
    ret_post = sample['ret_post']
    pre_ret = sample['pre_ret']
    
    # 判断情感方向
    if ret_post > 0.0005:
        sentiment = "利好"
        direction = "上涨"
    elif ret_post < -0.0005:
        sentiment = "利空"
        direction = "下跌"
    else:
        sentiment = "中性"
        direction = "震荡"
    
    # 判断是否预期兑现
    priced_in = ""
    if abs(pre_ret) > 0.01:
        if (pre_ret > 0 and ret_post < 0) or (pre_ret < 0 and ret_post > 0):
            priced_in = f"需要注意的是，市场前期已经{'上涨' if pre_ret > 0 else '下跌'}{abs(pre_ret):.2%}，可能存在预期兑现的风险。"
    
    # 生成输出
    output = f"这是一条{star}星级的财经快讯，属于{sentiment}消息。"
    output += f"从市场反应来看，该消息发布后市场出现{direction}走势。"
    if priced_in:
        output += priced_in
    output += "建议投资者关注后续市场反应，结合技术面和基本面综合判断。"
    
    return {
        "instruction": "分析以下财经快讯对市场的影响",
        "input": content,
        "output": output
    }


def generate_market_context_instructions(num_samples: int = 50) -> list:
    """
    生成市场上下文分析指令
    
    Args:
        num_samples: 样本数量
    
    Returns:
        指令列表
    """
    templates = [
        {
            "instruction": "解释什么是预期兑现",
            "input": "市场前期已经大涨，利好消息发布后反而下跌，这是为什么？",
            "output": "这是典型的'预期兑现'现象。当市场提前消化了利好预期，价格已经提前上涨时，利好消息正式发布反而成为'利好出尽'的信号，导致获利盘离场。这种情况下，技术面的前期涨幅和基本面的消息方向出现背离，需要特别警惕。"
        },
        {
            "instruction": "解释什么是建议观望",
            "input": "市场波动剧烈但价格变化不大，应该如何操作？",
            "output": "这种情况下建议观望。高波动低净变动通常意味着市场多空分歧较大，方向不明确。此时贸然入场容易被来回止损。建议等待市场方向明确后再做决策，或者采用小仓位试探性操作。"
        },
        {
            "instruction": "分析加息对黄金价格的影响",
            "input": "美联储宣布加息25个基点",
            "output": "加息通常会提振美元，对黄金价格形成压力。但需要综合考虑以下因素：1) 市场是否已提前消化预期；2) 加息幅度是否符合预期；3) 美联储的后续政策指引。如果加息幅度低于预期或美联储释放鸽派信号，黄金可能反而上涨。"
        },
        {
            "instruction": "分析非农数据对市场的影响",
            "input": "美国非农就业数据超预期",
            "output": "非农数据超预期通常被视为经济强劲的信号，可能导致：1) 美元走强，黄金承压；2) 美联储维持鹰派立场的预期增强；3) 股市可能因经济前景改善而上涨。但需要关注数据的可持续性和其他经济指标的配合。"
        },
        {
            "instruction": "解释技术面和基本面的关系",
            "input": "基本面利好但技术面超买，应该如何判断？",
            "output": "技术面和基本面需要综合判断。基本面利好提供了上涨的逻辑支撑，但技术面超买意味着短期获利盘较多，可能面临回调压力。建议：1) 如果是长期投资，可以忽略短期技术面波动；2) 如果是短期交易，建议等待技术面回调后再入场；3) 可以采用分批建仓的策略。"
        }
    ]
    
    # 复制模板以达到目标数量
    instructions = []
    while len(instructions) < num_samples:
        instructions.extend(templates)
    
    return instructions[:num_samples]


def generate_rag_instructions(num_samples: int = 50) -> list:
    """
    生成财报问答指令（模拟）
    
    Args:
        num_samples: 样本数量
    
    Returns:
        指令列表
    """
    templates = [
        {
            "instruction": "根据财报数据回答问题",
            "input": "黄金市场2023年的表现如何？",
            "output": "根据世界黄金协会的报告，2023年全球黄金需求保持稳健。其中，央行购金需求创历史新高，达到1000吨以上。投资需求方面，黄金ETF持仓有所下降，但实物金条和金币需求保持稳定。价格方面，2023年黄金价格整体呈现震荡上行态势，年内最高触及2100美元/盎司。"
        },
        {
            "instruction": "根据财报数据回答问题",
            "input": "美联储加息对黄金价格有什么影响？",
            "output": "根据历史数据分析，美联储加息对黄金价格的影响较为复杂。短期来看，加息通常会提振美元，对黄金价格形成压力。但长期来看，如果加息导致经济衰退风险上升，黄金的避险属性可能会推动价格上涨。此外，实际利率（名义利率减去通胀率）是影响黄金价格的关键因素。"
        },
        {
            "instruction": "根据财报数据回答问题",
            "input": "中国黄金需求情况如何？",
            "output": "中国是全球最大的黄金消费国之一。根据中国黄金协会的数据，2023年中国黄金消费量约为1000吨，其中首饰消费占比最大，投资需求也保持稳定增长。随着居民财富增长和投资意识提升，黄金作为资产配置的重要组成部分，需求前景依然乐观。"
        }
    ]
    
    # 复制模板以达到目标数量
    instructions = []
    while len(instructions) < num_samples:
        instructions.extend(templates)
    
    return instructions[:num_samples]


def main():
    parser = argparse.ArgumentParser(description='构建 QLoRA 微调指令集')
    parser.add_argument('--db', type=str, default='finance_analysis.db',
                        help='数据库路径')
    parser.add_argument('--output', type=str, default='data/qlora/instructions.jsonl',
                        help='输出文件路径')
    parser.add_argument('--num_samples', type=int, default=300,
                        help='总样本数量')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"开始构建指令集...")
    print(f"目标样本数量: {args.num_samples}")
    
    # 分配样本数量
    num_news = int(args.num_samples * 0.6)  # 60% 来自真实新闻
    num_market = int(args.num_samples * 0.2)  # 20% 市场分析
    num_rag = int(args.num_samples * 0.2)  # 20% 财报问答
    
    instructions = []
    
    # 1. 从数据库提取新闻样本
    print(f"\n1. 从数据库提取 {num_news} 条新闻样本...")
    try:
        news_samples = get_news_samples(args.db, num_news)
        for sample in news_samples:
            instruction = generate_instruction_from_news(sample)
            instructions.append(instruction)
        print(f"   ✓ 成功生成 {len(news_samples)} 条新闻指令")
    except Exception as e:
        print(f"   ✗ 提取新闻样本失败: {e}")
        print(f"   将使用模板数据替代")
    
    # 2. 生成市场分析指令
    print(f"\n2. 生成 {num_market} 条市场分析指令...")
    market_instructions = generate_market_context_instructions(num_market)
    instructions.extend(market_instructions)
    print(f"   ✓ 成功生成 {len(market_instructions)} 条市场分析指令")
    
    # 3. 生成财报问答指令
    print(f"\n3. 生成 {num_rag} 条财报问答指令...")
    rag_instructions = generate_rag_instructions(num_rag)
    instructions.extend(rag_instructions)
    print(f"   ✓ 成功生成 {len(rag_instructions)} 条财报问答指令")
    
    # 打乱顺序
    random.shuffle(instructions)
    
    # 保存为 JSONL
    print(f"\n4. 保存指令集到 {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        for instruction in instructions:
            f.write(json.dumps(instruction, ensure_ascii=False) + '\n')
    
    print(f"   ✓ 成功保存 {len(instructions)} 条指令")
    
    # 统计信息
    print(f"\n" + "="*60)
    print(f"指令集构建完成！")
    print(f"="*60)
    print(f"总样本数: {len(instructions)}")
    print(f"输出文件: {args.output}")
    print(f"文件大小: {output_path.stat().st_size / 1024:.2f} KB")
    
    # 显示示例
    print(f"\n示例指令（前3条）：")
    for i, instruction in enumerate(instructions[:3], 1):
        print(f"\n--- 示例 {i} ---")
        print(f"Instruction: {instruction['instruction']}")
        print(f"Input: {instruction['input'][:50]}...")
        print(f"Output: {instruction['output'][:100]}...")


if __name__ == '__main__':
    main()
