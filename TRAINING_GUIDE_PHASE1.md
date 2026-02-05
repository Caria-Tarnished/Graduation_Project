# Phase 1 Training Guide: Enhanced Dataset + Class Weights

## ? Overview

This guide covers Phase 1 improvements for Engine A (BERT sentiment classifier):
1. ? **Input Augmentation**: Add market context prefixes (COMPLETED)
2. ? **Class Weighting**: Handle severe class imbalance (NEXT STEP)

## ? Step 1: Data Augmentation (COMPLETED)

**Script**: `scripts/modeling/build_enhanced_dataset.py`

**What it does**:
- Reads `train/val/test_multi_labeled.csv`
- Adds market context prefixes based on `pre_ret` and `range_ratio`
- Generates `train/val/test_enhanced.csv` with `text_enhanced` column

**Results**:
- Train: 15,071 samples
- Val: 3,122 samples
- Test: 7,989 samples

**Prefix distribution**:
- `[Sideways]`: 65.8% (market consolidation)
- `[Mild Rally]`: 13.5% (weak uptrend)
- `[Weak Decline]`: 11.1% (weak downtrend)
- `[Sharp Decline]`: 4.9% (strong downtrend)
- `[Strong Rally]`: 4.7% (strong uptrend)
- `[High Volatility]`: 0.1% (choppy market)

## ? Step 2: Training with Class Weights (NEXT)

### Option A: Quick Start (Recommended for Colab)

**Direct command**:
```bash
python scripts/modeling/bert_finetune_cls.py \
  --train_csv data/processed/train_enhanced.csv \
  --val_csv data/processed/val_enhanced.csv \
  --test_csv data/processed/test_enhanced.csv \
  --output_dir models/bert_enhanced_weighted_v1 \
  --label_col label_multi_cls \
  --model_name hfl/chinese-roberta-wwm-ext \
  --class_weight auto \
  --epochs 5 \
  --lr 1e-5 \
  --max_length 384 \
  --train_bs 16 \
  --eval_bs 32 \
  --gradient_accumulation_steps 2 \
  --warmup_ratio 0.06 \
  --weight_decay 0.01 \
  --eval_steps 100 \
  --save_steps 100 \
  --early_stopping_patience 3
```

### Option B: Using Wrapper Script (Local testing)

```bash
python scripts/modeling/train_enhanced_with_weights.py
```

## ? Expected Improvements

### Baseline (bert_xauusd_multilabel_6cls)
- Test accuracy: 0.425
- Test macro_f1: **0.163**
- Class 3/4/5 F1: **0.000** (complete failure)

### Target (Phase 1)
- Test accuracy: 0.45-0.50
- Test macro_f1: **0.35-0.45** (2-3x improvement)
- Class 3/4 F1: **> 0.10** (at least some signal)
- Class 5 F1: **> 0.20** (reasonable performance)

## ? Colab Training Instructions

### 1. Upload Enhanced Data to Google Drive

**Local ¡ú Drive**:
```bash
# Copy these files to your Google Drive:
# /content/drive/MyDrive/Graduation_Project/data/processed/
data/processed/train_enhanced.csv
data/processed/val_enhanced.csv
data/processed/test_enhanced.csv
```

### 2. Colab Setup

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repo (if not already)
!git clone https://github.com/<your_org>/<your_repo>.git
%cd <your_repo>

# Install dependencies
!pip install -U transformers datasets evaluate accelerate
```

### 3. Run Training

```python
# Option 1: Direct command
!python scripts/modeling/bert_finetune_cls.py \
  --train_csv /content/drive/MyDrive/Graduation_Project/data/processed/train_enhanced.csv \
  --val_csv /content/drive/MyDrive/Graduation_Project/data/processed/val_enhanced.csv \
  --test_csv /content/drive/MyDrive/Graduation_Project/data/processed/test_enhanced.csv \
  --output_dir /content/drive/MyDrive/Graduation_Project/experiments/bert_enhanced_v1 \
  --label_col label_multi_cls \
  --class_weight auto \
  --epochs 5 --lr 1e-5 --max_length 384 \
  --train_bs 16 --eval_bs 32 \
  --warmup_ratio 0.06 --weight_decay 0.01 \
  --eval_steps 100 --save_steps 100 --early_stopping_patience 3
```

### 4. Monitor Training

Watch for:
- **Loss convergence**: Should decrease steadily
- **Macro F1**: Should improve on validation set
- **Early stopping**: Will trigger if no improvement for 3 eval steps

### 5. Download Results

```python
# Check results
!cat /content/drive/MyDrive/Graduation_Project/experiments/bert_enhanced_v1/metrics_test.json
!cat /content/drive/MyDrive/Graduation_Project/experiments/bert_enhanced_v1/report_test.txt
```

## ? Evaluation Checklist

After training completes, check:

1. **Overall metrics** (`metrics_test.json`):
   - [ ] accuracy > 0.45
   - [ ] macro_f1 > 0.35

2. **Per-class F1** (`report_test.txt`):
   - [ ] Class 0 (neutral): F1 > 0.55
   - [ ] Class 1 (bullish): F1 > 0.25
   - [ ] Class 2 (bearish): F1 > 0.25
   - [ ] Class 3 (bullish priced-in): F1 > 0.05 (any signal is good!)
   - [ ] Class 4 (bearish priced-in): F1 > 0.05
   - [ ] Class 5 (watch): F1 > 0.15

3. **Confusion matrix analysis**:
   - Are Class 3/4 predictions still all zeros?
   - Is Class 5 being confused with Class 0?

## ? Troubleshooting

### Issue: Class 3/4 still have F1=0

**Possible causes**:
1. Only 12 samples total (7 train, 5 test) - too few to learn
2. Class weight not aggressive enough

**Solutions**:
1. Try manual weights: `{0:0.5, 1:2.0, 2:2.0, 3:20.0, 4:20.0, 5:3.0}`
2. Consider merging Class 3/4 into a single "priced-in" class
3. Generate synthetic samples (Phase 1.5)

### Issue: Training is too slow

**Solutions**:
1. Reduce `max_length` to 256
2. Increase `gradient_accumulation_steps` to 4
3. Use smaller model: `hfl/chinese-bert-wwm-ext` (base instead of large)

### Issue: Overfitting (val loss increases)

**Solutions**:
1. Increase `weight_decay` to 0.05
2. Add dropout (requires code modification)
3. Reduce `epochs` to 3

## ? Next Steps After Phase 1

If Phase 1 achieves macro_f1 > 0.35:
- ? Proceed to Phase 2: Try financial pre-trained model (mengzi-bert-base-fin)
- ? Implement Focal Loss for better minority class handling

If Phase 1 achieves macro_f1 < 0.30:
- ?? Revisit data quality (check label correctness)
- ?? Simplify to 3-class problem (bearish/neutral/bullish only)
- ?? Consider different labeling thresholds

## ? References

- Input Augmentation theory: Financial LLM Survey Lecture 7 & 8
- Class imbalance handling: Focal Loss paper (Lin et al., 2017)
- Chinese financial BERT: Mengzi-BERT (Langboat, 2021)
