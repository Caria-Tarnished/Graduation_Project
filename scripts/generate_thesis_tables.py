#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import pandas as pd
from pathlib import Path


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_model_comparison_table():
    print("=" * 80)
    print("Model Performance Comparison")
    print("=" * 80)
    
    baseline = load_json('models/baseline_tfidf_svm/metrics_test.json')
    bert_15min = load_json('reports/bert_3cls_enhanced_v1/metrics_test.json')
    bert_30min = load_json('reports/bert_3cls_w30_v1/metrics_test.json')
    bert_hc = load_json('reports/bert_3cls_hc_v1/metrics_test.json')
    
    data = {
        'Model': [
            'TF-IDF + LinearSVC (Baseline)',
            'BERT 3-class (15min window)',
            'BERT 3-class (30min window)',
            'BERT 3-class (HC filtered)'
        ],
        'Test_Macro_F1': [
            baseline['macro_f1'],
            bert_15min['eval_macro_f1'],
            bert_30min['eval_macro_f1'],
            bert_hc['eval_macro_f1']
        ],
        'Test_Accuracy': [
            '-',
            bert_15min['eval_accuracy'],
            bert_30min['eval_accuracy'],
            bert_hc['eval_accuracy']
        ],
        'Training_Time': [
            '~5min (CPU)',
            '~1.5h (T4 GPU)',
            '~1.5h (T4 GPU)',
            '~1.5h (T4 GPU)'
        ]
    }
    
    df = pd.DataFrame(data)
    
    baseline_f1 = baseline['macro_f1']
    df['Improvement'] = df['Test_Macro_F1'].apply(
        lambda x: f"+{(x - baseline_f1) / baseline_f1 * 100:.1f}%" if isinstance(x, float) else '-'
    )
    
    print(df.to_string(index=False))
    print()
    
    output_path = 'thesis_assets/tables/model_comparison.csv'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Saved to: {output_path}")
    print()
    
    return df


def generate_class_performance_table():
    print("=" * 80)
    print("Per-Class F1 Score Comparison")
    print("=" * 80)
    
    def parse_report(path):
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
    
    data = {
        'Class': ['Bearish (-1)', 'Neutral (0)', 'Bullish (1)', 'Macro Avg'],
        'BERT_15min': [
            bert_15min.get(-1, 0),
            bert_15min.get(0, 0),
            bert_15min.get(1, 0),
            sum(bert_15min.values()) / 3
        ],
        'BERT_30min': [
            bert_30min.get(-1, 0),
            bert_30min.get(0, 0),
            bert_30min.get(1, 0),
            sum(bert_30min.values()) / 3
        ],
        'BERT_HC': [
            bert_hc.get(-1, 0),
            bert_hc.get(0, 0),
            bert_hc.get(1, 0),
            sum(bert_hc.values()) / 3
        ]
    }
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print()
    
    output_path = 'thesis_assets/tables/class_performance.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Saved to: {output_path}")
    print()
    
    return df


def generate_rag_ablation_table():
    print("=" * 80)
    print("RAG Retrieval Strategy Ablation")
    print("=" * 80)
    
    rag_results = load_json('data/reports/rag_ablation_results.json')
    
    data = {
        'Configuration': ['Baseline (Original)', 'Exp A (Cleaned+Trimmed)', 'Exp B (Cleaned+Metadata)'],
        'Num_Docs': [633, 544, 544],
        'Avg_Relevance': [
            rag_results['Baseline']['avg_relevance'],
            rag_results['Exp A']['avg_relevance'],
            rag_results['Exp B']['avg_relevance']
        ],
        'Avg_Latency_ms': [
            rag_results['Baseline']['avg_latency_ms'],
            rag_results['Exp A']['avg_latency_ms'],
            rag_results['Exp B']['avg_latency_ms']
        ],
        'Std_Latency_ms': [
            rag_results['Baseline']['std_latency_ms'],
            rag_results['Exp A']['std_latency_ms'],
            rag_results['Exp B']['std_latency_ms']
        ]
    }
    
    df = pd.DataFrame(data)
    
    baseline_rel = rag_results['Baseline']['avg_relevance']
    baseline_lat = rag_results['Baseline']['avg_latency_ms']
    
    df['Relevance_Change'] = df['Avg_Relevance'].apply(
        lambda x: f"{(x - baseline_rel) / baseline_rel * 100:+.1f}%"
    )
    df['Latency_Change'] = df['Avg_Latency_ms'].apply(
        lambda x: f"{(x - baseline_lat) / baseline_lat * 100:+.1f}%"
    )
    
    print(df.to_string(index=False))
    print()
    
    output_path = 'thesis_assets/tables/rag_ablation.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Saved to: {output_path}")
    print()
    
    return df


def generate_system_performance_table():
    print("=" * 80)
    print("System Performance Optimization")
    print("=" * 80)
    
    data = {
        'Optimization': [
            'Repeated Query Response',
            'Mixed Scenario Avg Response',
            'Database Query Time',
            'Cache Hit Rate',
            'BERT Inference Count'
        ],
        'Before': [
            '0.8-0.9s',
            '~0.8s',
            '155.83ms',
            '0%',
            '100%'
        ],
        'After': [
            '<0.001s',
            '0.276s',
            '0.20ms',
            '90%',
            '10%'
        ],
        'Improvement': [
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
    
    output_path = 'thesis_assets/tables/system_performance.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Saved to: {output_path}")
    print()
    
    return df


def generate_data_statistics_table():
    print("=" * 80)
    print("Dataset Statistics")
    print("=" * 80)
    
    data = {
        'Data_Type': [
            'Minute Price (XAUUSD)',
            'Events (Jin10 Flash+Calendar)',
            'Training Samples (3-class)',
            'Validation Samples (3-class)',
            'Test Samples (3-class)',
            'RAG Document Chunks',
            'QLoRA Instructions'
        ],
        'Count': [
            '736,304',
            '26,182',
            '12,859',
            '2,661',
            '3,823',
            '633',
            '300'
        ],
        'Time_Range_Notes': [
            '2024-01-02 to 2026-01-31',
            'Flash 25,057 + Calendar 1,125',
            'Bearish 30% / Neutral 40% / Bullish 30%',
            'Same as training',
            'Same as training',
            '12/15 PDFs processed',
            '180 news + 60 analysis + 60 reports'
        ]
    }
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print()
    
    output_path = 'thesis_assets/tables/data_statistics.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Saved to: {output_path}")
    print()
    
    return df


def generate_qlora_training_table():
    print("=" * 80)
    print("QLoRA Fine-tuning Configuration")
    print("=" * 80)
    
    training_info = load_json('scripts/qlora_output/training_info.json')
    
    data = {
        'Configuration': [
            'Base Model',
            'Training Data Size',
            'Epochs',
            'Batch Size',
            'Learning Rate',
            'LoRA Rank (r)',
            'LoRA Alpha',
            'Training Time',
            'LoRA Weights Size'
        ],
        'Value': [
            'Deepseek-7B-Chat',
            f"{training_info['num_samples']} instructions",
            training_info['num_epochs'],
            training_info['batch_size'],
            training_info['learning_rate'],
            training_info['lora_r'],
            training_info['lora_alpha'],
            '~19.5min (T4 GPU)',
            '15.02 MB'
        ]
    }
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print()
    
    output_path = 'thesis_assets/tables/qlora_training.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Saved to: {output_path}")
    print()
    
    return df


def main():
    print("\n")
    print("=" * 80)
    print("Thesis Data Generation")
    print("=" * 80)
    print()
    
    # Create output directory
    Path('thesis_assets/tables').mkdir(parents=True, exist_ok=True)
    
    generate_data_statistics_table()
    generate_model_comparison_table()
    generate_class_performance_table()
    generate_rag_ablation_table()
    generate_system_performance_table()
    generate_qlora_training_table()
    
    print("=" * 80)
    print("All tables generated successfully")
    print("=" * 80)
    print()
    print("Output location: thesis_assets/tables/")
    print()


if __name__ == '__main__':
    main()
