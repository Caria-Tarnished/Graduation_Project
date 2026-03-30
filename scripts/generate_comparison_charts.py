#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_model_f1_comparison():
    """Generate model F1 score comparison bar chart"""
    
    baseline = load_json('models/baseline_tfidf_svm/metrics_test.json')
    bert_15min = load_json('reports/bert_3cls_enhanced_v1/metrics_test.json')
    bert_30min = load_json('reports/bert_3cls_w30_v1/metrics_test.json')
    bert_hc = load_json('reports/bert_3cls_hc_v1/metrics_test.json')
    
    models = ['TF-IDF+SVM\n(Baseline)', 'BERT\n(15min)', 'BERT\n(30min)', 'BERT\n(HC)']
    f1_scores = [
        baseline['macro_f1'],
        bert_15min['eval_macro_f1'],
        bert_30min['eval_macro_f1'],
        bert_hc['eval_macro_f1']
    ]
    
    colors = ['#95a5a6', '#3498db', '#e74c3c', '#f39c12']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.axhline(y=baseline['macro_f1'], color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Baseline')
    
    ax.set_ylabel('Test Macro F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(f1_scores) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    output_path = 'thesis_assets/charts/model_f1_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated: {output_path}")


def generate_class_f1_comparison():
    """Generate per-class F1 comparison grouped bar chart"""
    
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
    
    classes = ['Bearish', 'Neutral', 'Bullish']
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width, [bert_15min[-1], bert_15min[0], bert_15min[1]], 
                   width, label='BERT 15min', color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, [bert_30min[-1], bert_30min[0], bert_30min[1]], 
                   width, label='BERT 30min', color='#e74c3c', alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + width, [bert_hc[-1], bert_hc[0], bert_hc[1]], 
                   width, label='BERT HC', color='#f39c12', alpha=0.8, edgecolor='black')
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class F1 Score Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylim(0, 0.5)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = 'thesis_assets/charts/class_f1_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated: {output_path}")


def generate_training_data_distribution():
    """Generate training data distribution pie chart"""
    
    sizes = [12859, 2661, 3823]
    labels = ['Training\n(12,859)', 'Validation\n(2,661)', 'Test\n(3,823)']
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    explode = (0.05, 0, 0)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    ax.set_title('Training Data Split', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = 'thesis_assets/charts/data_split_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated: {output_path}")


def generate_improvement_summary():
    """Generate improvement summary horizontal bar chart"""
    
    categories = [
        'Sentiment F1\n(vs Baseline)',
        'Response Time\n(Repeated Query)',
        'Database Query\n(Optimization)',
        'Cache Hit Rate\n(Improvement)'
    ]
    
    improvements = [8.9, 99.9, 99.9, 90.0]
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#e67e22']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(categories, improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, value in zip(bars, improvements):
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height()/2.,
                f'+{value:.1f}%',
                ha='left', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_title('System Improvement Summary', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 110)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = 'thesis_assets/charts/improvement_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated: {output_path}")


def main():
    print("\n" + "=" * 80)
    print("Generating Comparison Charts")
    print("=" * 80)
    print()
    
    Path('thesis_assets/charts').mkdir(parents=True, exist_ok=True)
    
    generate_model_f1_comparison()
    generate_class_f1_comparison()
    generate_training_data_distribution()
    generate_improvement_summary()
    
    print()
    print("=" * 80)
    print("All charts generated successfully")
    print("=" * 80)
    print()
    print("Output location: thesis_assets/charts/")
    print()


if __name__ == '__main__':
    main()
