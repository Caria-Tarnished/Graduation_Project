#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("RAG Experiment Environment Check")
print("=" * 70)

# 1. Check Python dependencies
print("\n1. Checking Python dependencies...")
required_packages = [
    'chromadb',
    'sentence_transformers',
    'matplotlib',
    'seaborn',
    'pandas',
    'numpy',
    'pdfplumber',
    'langchain'
]

missing_packages = []
for pkg in required_packages:
    try:
        __import__(pkg)
        print(f"  OK {pkg}")
    except ImportError:
        print(f"  MISSING {pkg}")
        missing_packages.append(pkg)

if missing_packages:
    print(f"\nMissing packages: {', '.join(missing_packages)}")
    print(f"Run: pip install {' '.join(missing_packages)}")
else:
    print("\n  All dependencies installed!")

# 2. Check evaluation dataset
print("\n2. Checking evaluation dataset...")
eval_path = project_root / "data" / "reports" / "rag_eval_dataset.json"
if eval_path.exists():
    import json
    with open(eval_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    print(f"  OK Evaluation dataset exists: {len(queries)} queries")
else:
    print(f"  MISSING Evaluation dataset: {eval_path}")

# 3. Check vector databases
print("\n3. Checking vector databases...")
db_paths = [
    ("Main DB", project_root / "data" / "reports" / "chroma_db", "finance_docs"),
    ("Baseline DB", project_root / "data" / "reports" / "chroma_db_baseline", None),
    ("Exp A DB", project_root / "data" / "reports" / "chroma_db_exp_a", None),
    ("Exp B DB", project_root / "data" / "reports" / "chroma_db_exp_b", None)
]

for name, db_path, collection_name in db_paths:
    if db_path.exists():
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(db_path))
            collections = client.list_collections()
            if collections:
                for col in collections:
                    print(f"  OK {name}: {col.name} ({col.count()} docs)")
            else:
                print(f"  WARNING {name}: exists but no collections")
        except Exception as e:
            print(f"  WARNING {name}: exists but cannot read ({e})")
    else:
        print(f"  MISSING {name}: {db_path}")

# 4. Check PDF source files
print("\n4. Checking PDF source files...")
pdf_dir = project_root / "data" / "raw" / "reports" / "research_reports"
if pdf_dir.exists():
    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"  OK PDF directory exists: {len(pdf_files)} files")
else:
    print(f"  MISSING PDF directory: {pdf_dir}")

# 5. Check output directory
print("\n5. Checking output directory...")
output_dir = project_root / "thesis_assets" / "charts"
if output_dir.exists():
    chart_files = list(output_dir.glob("rag_ablation_*.png"))
    print(f"  OK Output directory exists: {len(chart_files)} RAG charts")
else:
    print(f"  WARNING Output directory does not exist, will be created")

print("\n" + "=" * 70)
print("Check complete!")
print("=" * 70)
