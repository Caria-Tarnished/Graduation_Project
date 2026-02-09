# -*- coding: utf-8 -*-
"""
Prepare bge-m3 model directory structure and download small files

This script will:
1. Create the correct directory structure
2. Download all small config files
3. Wait for you to manually download pytorch_model.bin
"""
import os
import json
import requests
from pathlib import Path

# Model info
MODEL_ID = "BAAI/bge-m3"
COMMIT_HASH = "5617a9f61b028005a4858fdac845db406aefb181"
BASE_URL = f"https://huggingface.co/{MODEL_ID}/resolve/{COMMIT_HASH}"

# Cache directory (F drive)
CACHE_DIR = Path("F:/huggingface_cache/hub")
MODEL_DIR = CACHE_DIR / "models--BAAI--bge-m3"
SNAPSHOT_DIR = MODEL_DIR / "snapshots" / COMMIT_HASH

print("=" * 80)
print("Preparing bge-m3 model structure")
print("=" * 80)
print()

# Create directories
print("Creating directories...")
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
print(f"  Created: {SNAPSHOT_DIR}")
print()

# Small files to download
small_files = {
    "config.json": f"{BASE_URL}/config.json",
    "config_sentence_transformers.json": f"{BASE_URL}/config_sentence_transformers.json",
    "modules.json": f"{BASE_URL}/modules.json",
    "sentence_bert_config.json": f"{BASE_URL}/sentence_bert_config.json",
    "tokenizer_config.json": f"{BASE_URL}/tokenizer_config.json",
    "tokenizer.json": f"{BASE_URL}/tokenizer.json",
    "special_tokens_map.json": f"{BASE_URL}/special_tokens_map.json",
    "vocab.txt": f"{BASE_URL}/vocab.txt",
    "README.md": f"{BASE_URL}/README.md",
}

print("Downloading small config files...")
for filename, url in small_files.items():
    filepath = SNAPSHOT_DIR / filename
    if filepath.exists():
        print(f"  Skip (exists): {filename}")
        continue
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        filepath.write_bytes(response.content)
        print(f"  Downloaded: {filename} ({len(response.content)} bytes)")
    except Exception as e:
        print(f"  Failed: {filename} - {e}")

print()
print("=" * 80)
print("Next step: Download pytorch_model.bin manually")
print("=" * 80)
print()
print("1. Open this URL in your browser:")
print(f"   {BASE_URL}/pytorch_model.bin")
print()
print("2. Save the file to:")
print(f"   {SNAPSHOT_DIR}\\pytorch_model.bin")
print()
print("3. After download completes, run:")
print("   python scripts/tools/verify_bge_m3.py")
print()
print("=" * 80)
print()

# Create a refs file
refs_dir = MODEL_DIR / "refs"
refs_dir.mkdir(exist_ok=True)
(refs_dir / "main").write_text(COMMIT_HASH)
print(f"Created refs file: {refs_dir / 'main'}")
print()
