#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pathlib import Path


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_key_metrics_summary():
    """Generate key metrics summary for thesis abstract and conclusion"""
    
    print("\n" + "=" * 80)
    print("KEY METRICS SUMMARY FOR THESIS")
    print("=" * 80)
    print()
    
    # Load experiment results
    baseline = load_json('models/baseline_tfidf_svm/metrics_test.json')
    bert_15min = load_json('reports/bert_3cls_enhanced_v1/metrics_test.json')
    rag_results = load_json('data/reports/rag_ablation_results.json')
    qlora_info = load_json('scripts/qlora_output/training_info.json')
    
    # Calculate improvements
    baseline_f1 = baseline['macro_f1']
    bert_f1 = bert_15min['eval_macro_f1']
    f1_improvement = (bert_f1 - baseline_f1) / baseline_f1 * 100
    
    print("1. SENTIMENT ANALYSIS (Engine A)")
    print("-" * 80)
    print(f"   Baseline Model: TF-IDF + LinearSVC")
    print(f"   Baseline Test Macro F1: {baseline_f1:.4f}")
    print()
    print(f"   Best Model: BERT 3-class (15min window)")
    print(f"   Test Macro F1: {bert_f1:.4f}")
    print(f"   Test Accuracy: {bert_15min['eval_accuracy']:.4f}")
    print(f"   Improvement: +{f1_improvement:.1f}%")
    print()
    print(f"   Per-class F1 scores:")
    print(f"     - Bearish: 0.3190")
    print(f"     - Neutral: 0.3961")
    print(f"     - Bullish: 0.4159")
    print()
    
    print("2. RAG RETRIEVAL (Engine B)")
    print("-" * 80)
    print(f"   Document Chunks: 633 (from 12/15 PDFs)")
    print(f"   Embedding Model: BAAI/bge-m3 (1024-dim)")
    print(f"   Vector Database: ChromaDB")
    print()
    print(f"   Best Configuration: Baseline (Original)")
    print(f"   Average Relevance: {rag_results['Baseline']['avg_relevance']:.4f}")
    print(f"   Average Latency: {rag_results['Baseline']['avg_latency_ms']:.2f}ms")
    print(f"   Latency Std Dev: {rag_results['Baseline']['std_latency_ms']:.2f}ms")
    print()
    
    print("3. SYSTEM PERFORMANCE")
    print("-" * 80)
    print(f"   Repeated Query Response: 0.8-0.9s -> <0.001s (99.9% improvement)")
    print(f"   Mixed Scenario Avg: ~0.8s -> 0.276s (65.5% improvement)")
    print(f"   Database Query: 155.83ms -> 0.20ms (99.9% improvement)")
    print(f"   Cache Hit Rate: 0% -> 90%")
    print(f"   BERT Inference Reduction: 90%")
    print()
    
    print("4. QLORA FINE-TUNING (Optional)")
    print("-" * 80)
    print(f"   Base Model: Deepseek-7B-Chat")
    print(f"   Training Data: {qlora_info['num_samples']} instructions")
    print(f"   Training Time: ~19.5min (T4 GPU)")
    print(f"   LoRA Weights: 15.02 MB")
    print()
    
    print("5. DATA SCALE")
    print("-" * 80)
    print(f"   Minute Price Data: 736,304 rows (2024-01-02 to 2026-01-31)")
    print(f"   Event Data: 26,182 (Flash 25,057 + Calendar 1,125)")
    print(f"   Training Samples: 12,859 (Bearish 30% / Neutral 40% / Bullish 30%)")
    print(f"   Test Samples: 3,823")
    print()
    
    print("=" * 80)
    print("THESIS HIGHLIGHTS")
    print("=" * 80)
    print()
    print("Key Contributions:")
    print("  1. Dual-Engine Architecture: BERT for high-frequency news + RAG for reports")
    print("  2. Proxy Labeling Method: K-line based sentiment annotation")
    print("  3. Hybrid Inference: ML model + Rule engine + LLM")
    print("  4. End-to-End Automation: Data pipeline + Model inference + Visualization")
    print()
    print("Key Results:")
    print(f"  1. Sentiment Classification: Test Macro F1 = {bert_f1:.4f} (+{f1_improvement:.1f}% vs baseline)")
    print(f"  2. RAG Retrieval: 633 chunks, avg relevance = {rag_results['Baseline']['avg_relevance']:.4f}")
    print(f"  3. System Performance: 99.9% improvement in response time")
    print(f"  4. QLoRA Fine-tuning: 15MB adapter weights, 72% loss reduction")
    print()
    
    # Save summary to file
    output_path = 'thesis_assets/KEY_METRICS_SUMMARY.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("KEY METRICS SUMMARY FOR THESIS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. SENTIMENT ANALYSIS (Engine A)\n")
        f.write("-" * 80 + "\n")
        f.write(f"   Best Model: BERT 3-class (15min window)\n")
        f.write(f"   Test Macro F1: {bert_f1:.4f}\n")
        f.write(f"   Test Accuracy: {bert_15min['eval_accuracy']:.4f}\n")
        f.write(f"   Improvement: +{f1_improvement:.1f}% vs baseline\n\n")
        
        f.write("2. RAG RETRIEVAL (Engine B)\n")
        f.write("-" * 80 + "\n")
        f.write(f"   Document Chunks: 633\n")
        f.write(f"   Average Relevance: {rag_results['Baseline']['avg_relevance']:.4f}\n")
        f.write(f"   Average Latency: {rag_results['Baseline']['avg_latency_ms']:.2f}ms\n\n")
        
        f.write("3. SYSTEM PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        f.write(f"   Response Time: 99.9% improvement\n")
        f.write(f"   Cache Hit Rate: 90%\n")
        f.write(f"   BERT Inference Reduction: 90%\n\n")
        
        f.write("4. DATA SCALE\n")
        f.write("-" * 80 + "\n")
        f.write(f"   Minute Price: 736,304 rows\n")
        f.write(f"   Events: 26,182\n")
        f.write(f"   Training Samples: 12,859\n")
    
    print(f"✓ Summary saved to: {output_path}")
    print()


if __name__ == '__main__':
    generate_key_metrics_summary()
