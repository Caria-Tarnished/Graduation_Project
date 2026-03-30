# Thesis Data Index

Generated: 2026-03-28

---

## 1. Tables (thesis_assets/tables/)

### Table 5-1: Dataset Statistics
- **File**: `data_statistics.csv`
- **Content**: Overview of all data
  - Minute price: 736,304 rows
  - Events: 26,182
  - Training samples: 12,859
  - RAG chunks: 633

### Table 6-1: Model Performance Comparison
- **File**: `model_comparison.csv`
- **Content**: All sentiment models
  - Baseline: F1 = 0.3461
  - BERT 15min: F1 = 0.3770 (+8.9%)
  - BERT 30min: F1 = 0.3332 (-3.7%)
  - BERT HC: F1 = 0.3483 (+0.7%)

### Table 6-2: Per-Class F1 Scores
- **File**: `class_performance.csv`
- **Content**: F1 for each class
  - Bearish, Neutral, Bullish
  - Across 15min, 30min, HC models

### Table 6-3: RAG Ablation
- **File**: `rag_ablation.csv`
- **Content**: RAG configurations
  - Baseline: 633 docs, 0.5782 relevance
  - Exp A: 544 docs, +0.5% relevance
  - Exp B: 544 docs, -8.9% latency

### Table 6-4: System Performance
- **File**: `system_performance.csv`
- **Content**: Optimization results
  - Response time: 99.9% improvement
  - Database: 99.9% improvement
  - Cache hit: 0% -> 90%

### Table 6-5: QLoRA Training
- **File**: `qlora_training.csv`
- **Content**: QLoRA parameters
  - Model: Deepseek-7B-Chat
  - Data: 300 instructions
  - Time: 19.5min (T4 GPU)
  - Weights: 15.02 MB

---

## 2. Charts (thesis_assets/charts/)

### Data & Distribution
- `source_distribution.png` - Event sources
- `label_distribution.png` - Label distribution
- `data_split_distribution.png` - Train/val/test split

### Model Performance
- `model_f1_comparison.png` - F1 comparison
- `class_f1_comparison.png` - Per-class F1
- `confusion_matrix.png` - 15min confusion matrix
- `confusion_matrix_w30.png` - 30min confusion matrix

### HC Filtering Ablation
- `hc_filter_size_comparison.png` - Dataset size
- `hc_filter_purity_improvement.png` - Label purity
- `hc_filter_ret_distribution.png` - Return distribution

### RAG Ablation
- `rag_ablation_accuracy.png` - Relevance scores
- `rag_ablation_recall.png` - Recall metrics
- `rag_ablation_latency.png` - Latency comparison

### System Optimization
- `system_latency_optimization.png` - Response times
- `system_cache_hit_ratios.png` - Cache hit rates
- `improvement_summary.png` - Overall improvements

---

## 3. Key Metrics

**Best Model**: BERT 3-class (15min window)
- Test Macro F1: 0.3770
- Test Accuracy: 0.3819
- Improvement: +8.9% vs baseline

**Per-Class F1**:
- Bearish: 0.3190
- Neutral: 0.3961
- Bullish: 0.4159

**RAG Performance**:
- Chunks: 633 (12/15 PDFs)
- Relevance: 0.5782
- Latency: 214.85ms

**System Performance**:
- Response time: 99.9% improvement
- Cache hit rate: 90%
- BERT inference: -90%

**Data Scale**:
- Price data: 736,304 rows
- Events: 26,182
- Training: 12,859 samples

---

## 4. Quick Commands

Regenerate tables:
```bash
.venv\python.exe scripts/generate_thesis_tables.py
```

Regenerate charts:
```bash
.venv\python.exe scripts/generate_comparison_charts.py
```

Generate summary:
```bash
.venv\python.exe scripts/generate_thesis_summary.py
```

---

Last Updated: 2026-03-28
