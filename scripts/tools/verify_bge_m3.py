# -*- coding: utf-8 -*-
"""
Verify bge-m3 model installation
"""
import os
from pathlib import Path

CACHE_DIR = Path("F:/huggingface_cache/hub")
MODEL_DIR = CACHE_DIR / "models--BAAI--bge-m3"
SNAPSHOT_DIR = MODEL_DIR / "snapshots" / "5617a9f61b028005a4858fdac845db406aefb181"

print("=" * 80)
print("Verifying bge-m3 model installation")
print("=" * 80)
print()

# Check directory
if not SNAPSHOT_DIR.exists():
    print(f"Error: Directory not found: {SNAPSHOT_DIR}")
    print("Please run: python scripts/tools/prepare_bge_m3_structure.py")
    exit(1)

print(f"Model directory: {SNAPSHOT_DIR}")
print()

# Required files
required_files = [
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "vocab.txt",
    "pytorch_model.bin",  # The large file
]

print("Checking files:")
all_ok = True
for filename in required_files:
    filepath = SNAPSHOT_DIR / filename
    if filepath.exists():
        size_mb = filepath.stat().st_size / 1024 / 1024
        if filename == "pytorch_model.bin":
            if size_mb < 2000:  # Should be ~2270 MB
                print(f"  Warning: {filename} ({size_mb:.1f} MB) - seems incomplete")
                all_ok = False
            else:
                print(f"  OK: {filename} ({size_mb:.1f} MB)")
        else:
            print(f"  OK: {filename} ({size_mb:.3f} MB)")
    else:
        print(f"  Missing: {filename}")
        all_ok = False

print()
if all_ok:
    print("=" * 80)
    print("Success! All files are present.")
    print("=" * 80)
    print()
    print("Now you can run the vectorization script:")
    print("python scripts/rag/build_vector_index.py --input data/reports/chunks.json --output data/reports/chroma_db")
    print()
else:
    print("=" * 80)
    print("Error: Some files are missing or incomplete")
    print("=" * 80)
    print()
    if not (SNAPSHOT_DIR / "pytorch_model.bin").exists():
        print("Please download pytorch_model.bin from:")
        print("https://huggingface.co/BAAI/bge-m3/resolve/5617a9f61b028005a4858fdac845db406aefb181/pytorch_model.bin")
        print()
        print(f"Save it to: {SNAPSHOT_DIR}\\pytorch_model.bin")
    print()
