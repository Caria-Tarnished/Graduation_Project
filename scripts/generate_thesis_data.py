#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
论文数据整理脚本
生成论文需要的表格和图表数据
"""

import json
import pandas as pd
from pathlib import Path


def load_json(path):
    """加载JSON文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_model_comparison_table():
    """生成模型对比表格（表6-1：情感分类模型性能对比）"""
    print("=" * 80)
    print("表6-1：情感分类模型性能对比")
    print("=" * 80)
    
    # 读取各模型结果
    baseline = load_json('models/baseline_tfidf_svm/metrics_test.json')
    bert_15min = load_json('reports/bert_3cls_enhanced_v1/metrics_test.json')
    bert_30min = load_json('reports/bert_3cls_w30_v1/metrics_test.json')
    bert_hc = load_json('reports/bert_3cls_hc_v1/metrics_test.json')
    
    # 构建表格数据
    data = {
        '模型': [
            'TF-IDF + LinearSVC (基线)',
            'BERT 3类 (15min窗口)',
            'BERT 3类 (30min窗口)',
            'BERT 3类 (HC过滤)'
        ],
        'Test Macro F1': [
            baseline['macro_f1'],
            bert_15min['eval_macro_f1'],
            bert_30min['eval_macro_f1'],
            bert_hc['eval_macro_f1']
        ],
        'Test Accuracy': [
            '-',
            bert_15min['eval_accuracy'],
            bert_30min['eval_accuracy'],
            bert_hc['eval_accuracy']
        ],
        '训练时间': [
            '~5分钟 (CPU)',
            '~1.5小时 (T4 GPU)',
            '~1.5小时 (T4 GPU)',
            '~1.5小时 (T4 GPU)'
        ]
    }
    
    df = pd.DataFrame(data)
    
    # 计算提升幅度
    baseline_f1 = baseline['macro_f1']
    df['相比基线提升'] = df['Test Macro F1'].apply(
        lambda x: f"+{(x - baseline_f1) / baseline_f1 * 100:.1f}%" if isinstance(x, float) else '-'
    )
    
    print(df.to_string(index=False))
    print()
    
    # 保存为CSV
    output_path = 'thesis_assets/tables/model_comparison.csv'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"? 已保存到: {output_path}")
    print()
    
    return df


def generate_class_performance_table():
    """生成各类别性能对比表格（表6-2：各类别F1分数对比）"""
    print("=" * 80)
    print("表6-2：各类别F1分数对比")
    print("=" * 80)
    
    # 读取分类报告
    def parse_report(path):
        """解析sklearn分类报告"""
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        results = {}
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 4 and parts[0] in ['-1', '0', '1']:
                label = int(parts[0])
                f1 = float(parts[3])
                results[label] = f1
        return results
    
    bert_15min = parse_report('reports/bert_3cls_enhanced_v1/report_test.txt')
    bert_30min = parse_report('reports/bert_3cls_w30_v1/report_test.txt')
    bert_hc = parse_report('reports/bert_3cls_hc_v1/report_test.txt')
    
    # 构建表格
    data = {
        '类别': ['Bearish (-1)', 'Neutral (0)', 'Bullish (1)', 'Macro Avg'],
        'BERT 15min': [
            bert_15min.get(-1, 0),
            bert_15min.get(0, 0),
            bert_15min.get(1, 0),
            sum(bert_15min.values()) / 3
        ],
        'BERT 30min': [
            bert_30min.get(-1, 0),
            bert_30min.get(0, 0),
            bert_30min.get(1, 0),
            sum(bert_30min.values()) / 3
        ],
        'BERT HC': [
            bert_hc.get(-1, 0),
            bert_hc.get(0, 0),
            bert_hc.get(1, 0),
            sum(bert_hc.values()) / 3
        ]
    }
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print()
    
    # 保存为CSV
    output_path = 'thesis_assets/tables/class_performance.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"? 已保存到: {output_path}")
    print()
    
    return df


def generate_rag_ablation_table():
    """生成RAG消融实验表格（表6-3：RAG检索策略消融实验）"""
    print("=" * 80)
    print("表6-3：RAG检索策略消融实验")
    print("=" * 80)
    
    # 读取RAG消融结果
    rag_results = load_json('data/reports/rag_ablation_results.json')
    
    # 构建表格
    data = {
        '配置': ['Baseline (原始)', 'Exp A (清洗+裁剪)', 'Exp B (清洗+元数据)'],
        '文档数': [633, 544, 544],
        '平均相关度': [
            rag_results['Baseline']['avg_relevance'],
            rag_results['Exp A']['avg_relevance'],
            rag_results['Exp B']['avg_relevance']
        ],
        '平均延迟 (ms)': [
            rag_results['Baseline']['avg_latency_ms'],
            rag_results['Exp A']['avg_latency_ms'],
            rag_results['Exp B']['avg_latency_ms']
        ],
        '延迟标准差 (ms)': [
            rag_results['Baseline']['std_latency_ms'],
            rag_results['Exp A']['std_latency_ms'],
            rag_results['Exp B']['std_latency_ms']
        ]
    }
    
    df = pd.DataFrame(data)
    
    # 计算相对变化
    baseline_rel = rag_results['Baseline']['avg_relevance']
    baseline_lat = rag_results['Baseline']['avg_latency_ms']
    
    df['相关度变化'] = df['平均相关度'].apply(
        lambda x: f"{(x - baseline_rel) / baseline_rel * 100:+.1f}%"
    )
    df['延迟变化'] = df['平均延迟 (ms)'].apply(
        lambda x: f"{(x - baseline_lat) / baseline_lat * 100:+.1f}%"
    )
    
    print(df.to_string(index=False))
    print()
    
    # 保存为CSV
    output_path = 'thesis_assets/tables/rag_ablation.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"? 已保存到: {output_path}")
    print()
    
    return df


def generate_system_performance_table():
    """生成系统性能优化表格（表6-4：系统性能优化效果）"""
    print("=" * 80)
    print("表6-4：系统性能优化效果")
    print("=" * 80)
    
    # 从Project_Status.md中提取的数据
    data = {
        '优化项': [
            '重复查询响应时间',
            '混合场景平均响应',
            '数据库查询时间',
            '缓存命中率',
            'BERT推理次数'
        ],
        '优化前': [
            '0.8-0.9s',
            '~0.8s',
            '155.83ms',
            '0%',
            '100%'
        ],
        '优化后': [
            '<0.001s',
            '0.276s',
            '0.20ms',
            '90%',
            '10%'
        ],
        '提升幅度': [
            '99.9%',
            '65.5%',
            '99.9%',
            '+90pp',
            '-90%'
        ]
    }
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print()
    
    # 保存为CSV
    output_path = 'thesis_assets/tables/system_performance.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"? 已保存到: {output_path}")
    print()
    
    return df


def generate_data_statistics_table():
    """生成数据统计表格（表5-1：数据集统计）"""
    print("=" * 80)
    print("表5-1：数据集统计")
    print("=" * 80)
    
    data = {
        '数据类型': [
            '分钟价数据 (XAUUSD)',
            '事件数据 (金十快讯+日历)',
            '训练样本 (3类)',
            '验证样本 (3类)',
            '测试样本 (3类)',
            'RAG文档切片',
            'QLoRA指令集'
        ],
        '数量': [
            '736,304',
            '26,182',
            '12,859',
            '2,661',
            '3,823',
            '633',
            '300'
        ],
        '时间范围/说明': [
            '2024-01-02 至 2026-01-31',
            '快讯 25,057 + 日历 1,125',
            'Bearish 30% / Neutral 40% / Bullish 30%',
            '同训练集分布',
            '同训练集分布',
            '12/15个PDF成功处理',
            '180新闻 + 60市场分析 + 60财报'
        ]
    }
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print()
    
    # 保存为CSV
    output_path = 'thesis_assets/tables/data_statistics.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"? 已保存到: {output_path}")
    print()
    
    return df


def generate_qlora_training_table():
    """生成QLoRA训练信息表格（表6-5：QLoRA微调配置与结果）"""
    print("=" * 80)
    print("表6-5：QLoRA微调配置与结果")
    print("=" * 80)
    
    training_info = load_json('scripts/qlora_output/training_info.json')
    
    data = {
        '配置项': [
            '基础模型',
            '训练数据量',
            '训练轮数',
            '批次大小',
            '学习率',
            'LoRA秩 (r)',
            'LoRA Alpha',
            '训练时间',
            'LoRA权重大小'
        ],
        '值': [
            'Deepseek-7B-Chat',
            f"{training_info['num_samples']} 条指令",
            training_info['num_epochs'],
            training_info['batch_size'],
            training_info['learning_rate'],
            training_info['lora_r'],
            training_info['lora_alpha'],
            '~19.5分钟 (T4 GPU)',
            '15.02 MB'
        ]
    }
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print()
    
    # 保存为CSV
    output_path = 'thesis_assets/tables/qlora_training.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"? 已保存到: {output_path}")
    print()
    
    return df


def main():
    """主函数"""
    print("\n")
    print("=" * 80)
    print("论文数据整理")
    print("=" * 80)
    print()
    
    # 生成所有表格
    generate_data_statistics_table()
    generate_model_comparison_table()
    generate_class_performance_table()
    generate_rag_ablation_table()
    generate_system_performance_table()
    generate_qlora_training_table()
    
    print("=" * 80)
    print("? 所有表格已生成完成")
    print("=" * 80)
    print()
    print("输出位置: thesis_assets/tables/")
    print()
    print("生成的表格:")
    print("  1. data_statistics.csv - 数据集统计")
    print("  2. model_comparison.csv - 模型性能对比")
    print("  3. class_performance.csv - 各类别F1对比")
    print("  4. rag_ablation.csv - RAG消融实验")
    print("  5. system_performance.csv - 系统性能优化")
    print("  6. qlora_training.csv - QLoRA训练配置")
    print()


if __name__ == '__main__':
    main()
