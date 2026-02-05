# Quick test: Train for 1 epoch with small batch to verify setup
import subprocess
import sys

cmd = [
    sys.executable,
    "scripts/modeling/bert_finetune_cls.py",
    "--train_csv", "data/processed/train_enhanced.csv",
    "--val_csv", "data/processed/val_enhanced.csv",
    "--test_csv", "data/processed/test_enhanced.csv",
    "--output_dir", "models/test_quick",
    "--label_col", "label_multi_cls",
    "--model_name", "hfl/chinese-roberta-wwm-ext",
    "--class_weight", "auto",
    "--epochs", "1",  # Just 1 epoch for testing
    "--lr", "1e-5",
    "--max_length", "128",  # Shorter for speed
    "--train_bs", "8",  # Smaller batch
    "--eval_bs", "16",
    "--eval_steps", "50",
    "--save_steps", "50",
]

print("Running quick test (1 epoch, small batch)...")
print("This will take about 5-10 minutes on CPU...")
result = subprocess.run(cmd)

if result.returncode == 0:
    print("\n? Training script works correctly!")
    print("You can now proceed to Colab for full training.")
else:
    print("\n? Training failed. Check the error above.")
    sys.exit(1)
