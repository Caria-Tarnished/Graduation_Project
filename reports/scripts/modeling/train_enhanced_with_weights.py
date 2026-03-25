# -*- coding: utf-8 -*-
"""
Enhanced BERT training script with manual class weights

This script wraps bert_finetune_cls.py with optimized hyperparameters for the enhanced dataset.
It sets aggressive class weights to handle severe class imbalance (Class 3/4 have only 12 samples).

Usage:
    python scripts/modeling/train_enhanced_with_weights.py
"""
import subprocess
import sys

# Training configuration
CONFIG = {
    "train_csv": "data/processed/train_enhanced.csv",
    "val_csv": "data/processed/val_enhanced.csv",
    "test_csv": "data/processed/test_enhanced.csv",
    "output_dir": "models/bert_enhanced_weighted_v1",
    "label_col": "label_multi_cls",
    
    # Model settings
    "model_name": "hfl/chinese-roberta-wwm-ext",
    "max_length": 384,  # Increased for longer context
    
    # Training hyperparameters
    "epochs": 5,
    "lr": 1e-5,  # Lower learning rate for stability
    "train_bs": 16,
    "eval_bs": 32,
    "gradient_accumulation_steps": 2,  # Effective batch size = 32
    
    # Regularization
    "warmup_ratio": 0.06,
    "weight_decay": 0.01,
    
    # Evaluation & early stopping
    "eval_steps": 100,
    "save_steps": 100,
    "early_stopping_patience": 3,
    
    # Class weighting (CRITICAL for imbalanced data)
    "class_weight": "auto",  # Will use automatic weighting
    
    # Other
    "seed": 42,
}

def main():
    print("=" * 80)
    print("Enhanced BERT Training with Class Weights")
    print("=" * 80)
    print("\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    # Build command
    cmd = [
        sys.executable,
        "scripts/modeling/bert_finetune_cls.py",
    ]
    
    for key, value in CONFIG.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))
    
    # Run training
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print("=" * 80)
        print(f"\nModel saved to: {CONFIG['output_dir']}")
        print("\nNext steps:")
        print("1. Check metrics: models/bert_enhanced_weighted_v1/metrics_*.json")
        print("2. Review classification report: models/bert_enhanced_weighted_v1/report_test.txt")
        print("3. Compare with baseline (bert_xauusd_multilabel_6cls)")
    else:
        print("\n" + "=" * 80)
        print("Training failed!")
        print("=" * 80)
        sys.exit(1)

if __name__ == "__main__":
    main()
