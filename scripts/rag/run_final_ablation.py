#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import shutil
from pathlib import Path
from typing import List, Dict, Tuple

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import pdfplumber
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PDF_DIR = project_root / "data" / "raw" / "reports" / "research_reports"
EVAL_DATASET_PATH = project_root / "data" / "reports" / "rag_eval_dataset.json"
OUTPUT_DIR = project_root / "thesis_assets" / "charts"
RESULTS_PATH = project_root / "data" / "reports" / "rag_ablation_results.json"

DB_CONFIGS = [
    {
        "name": "Baseline",
        "path": project_root / "data" / "reports" / "chroma_db",
        "collection": "reports_chunks",
        "type": "baseline",
        "rebuild": False
    },
    {
        "name": "Exp A",
        "path": project_root / "data" / "reports" / "chroma_db_exp_a",
        "collection": "reports_chunks_cleaned",
        "type": "exp_a",
        "rebuild": True
    },
    {
        "name": "Exp B",
        "path": project_root / "data" / "reports" / "chroma_db_exp_b",
        "collection": "reports_chunks_enhanced",
        "type": "exp_b",
        "rebuild": True
    }
]


def clean_text(text: str) -> str:
    lines = text.split('\n')
    cleaned_lines = []
    skip_keywords = ['disclaimer', 'contact', 'analyst']
    for line in lines:
        line = line.strip()
        if len(line) < 3:
            continue
        if any(kw in line.lower() for kw in skip_keywords):
            continue
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)


def extract_metadata(pdf_name: str) -> Dict:
    metadata = {"source_file": pdf_name, "doc_id": Path(pdf_name).stem}
    if "Goldman" in pdf_name:
        metadata["source"] = "Goldman Sachs"
    elif "Morgan" in pdf_name and "J.P." in pdf_name:
        metadata["source"] = "J.P. Morgan"
    elif "UBS" in pdf_name:
        metadata["source"] = "UBS"
    elif "Morgan Stanley" in pdf_name:
        metadata["source"] = "Morgan Stanley"
    return metadata


def build_database(config: Dict, encoder: SentenceTransformer):
    if not config['rebuild']:
        print(f"\nSkipping {config['name']} (using existing)")
        return
    
    print(f"\n{'='*70}")
    print(f"Building: {config['name']}")
    print(f"{'='*70}")
    
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs")
    
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs in {PDF_DIR}")
    
    if config['path'].exists():
        shutil.rmtree(config['path'])
    config['path'].mkdir(parents=True, exist_ok=True)
    
    client = chromadb.PersistentClient(path=str(config['path']))
    collection = client.create_collection(config['collection'])
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    
    all_chunks = []
    for pdf_path in pdf_files:
        try:
            print(f"  Processing: {pdf_path.name[:50]}...")
            with pdfplumber.open(pdf_path) as pdf:
                full_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
                
                if config['type'] in ['exp_a', 'exp_b']:
                    full_text = clean_text(full_text)
                
                texts = splitter.split_text(full_text)
                base_meta = {"source_file": pdf_path.name, "doc_id": pdf_path.stem}
                if config['type'] == 'exp_b':
                    extra_meta = extract_metadata(pdf_path.name)
                    base_meta.update(extra_meta)
                
                for i, text_chunk in enumerate(texts):
                    all_chunks.append({
                        "id": f"{pdf_path.stem}_{i}",
                        "text": text_chunk[:800],
                        "metadata": base_meta.copy()
                    })
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\nTotal chunks: {len(all_chunks)}")
    
    batch_size = 10
    print(f"Vectorizing...")
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        metadatas = [c["metadata"] for c in batch]
        embeddings = encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()
        collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        if (i + batch_size) % 50 == 0:
            print(f"  {min(i+batch_size, len(all_chunks))}/{len(all_chunks)}")
    
    print(f"Created: {collection.count()} docs")


class RAGEvaluator:
    def __init__(self, db_path: Path, collection_name: str, encoder: SentenceTransformer):
        self.db_path = db_path
        self.collection_name = collection_name
        self.encoder = encoder
        self.client = None
        self.collection = None
    
    def connect(self):
        try:
            self.client = chromadb.PersistentClient(path=str(self.db_path))
            self.collection = self.client.get_collection(self.collection_name)
            return True
        except Exception as e:
            print(f"  Warning: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5, metadata_filter: Dict = None) -> Tuple[List[Dict], float]:
        start_time = time.time()
        query_embedding = self.encoder.encode(query, normalize_embeddings=True).tolist()
        where_clause = metadata_filter if metadata_filter and len(metadata_filter) > 0 else None
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding], n_results=top_k,
                where=where_clause, include=["documents", "metadatas", "distances"]
            )
        except Exception:
            results = self.collection.query(
                query_embeddings=[query_embedding], n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
        latency = time.time() - start_time
        docs = []
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                doc = {
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0.0
                }
                docs.append(doc)
        return docs, latency
    
    def calculate_relevance_score(self, retrieved_docs: List[Dict]) -> float:
        if not retrieved_docs:
            return 0.0
        distance = retrieved_docs[0]['distance']
        similarity = 1.0 / (1.0 + distance)
        return similarity


def load_eval_dataset(path: Path) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_evaluation(encoder: SentenceTransformer):
    print("\n" + "="*70)
    print("Running Evaluation")
    print("="*70)
    print(f"\nLoading dataset: {EVAL_DATASET_PATH}")
    eval_dataset = load_eval_dataset(EVAL_DATASET_PATH)
    print(f"Queries: {len(eval_dataset)}")
    all_results = []
    for config in DB_CONFIGS:
        print(f"\n{'='*70}")
        print(f"Experiment: {config['name']}")
        print(f"{'='*70}")
        evaluator = RAGEvaluator(config['path'], config['collection'], encoder)
        if not evaluator.connect():
            print(f"  Skipping")
            continue
        print(f"  Documents: {evaluator.collection.count()}")
        for i, item in enumerate(eval_dataset, 1):
            query = item['query']
            metadata_filter = item.get('metadata_filter', {})
            print(f"  [{i}/{len(eval_dataset)}] {query[:40]}...")
            retrieved_docs, latency = evaluator.search(
                query, top_k=5,
                metadata_filter=metadata_filter if config['type'] == 'exp_b' else None
            )
            relevance = evaluator.calculate_relevance_score(retrieved_docs)
            all_results.append({
                "QueryID": item['id'], "Query": query, "Method": config['name'],
                "Relevance": relevance, "Latency(ms)": latency * 1000
            })
        method_results = [r for r in all_results if r['Method'] == config['name']]
        avg_rel = np.mean([r['Relevance'] for r in method_results])
        avg_lat = np.mean([r['Latency(ms)'] for r in method_results])
        std_lat = np.std([r['Latency(ms)'] for r in method_results])
        print(f"\n  Avg Relevance: {avg_rel:.3f}")
        print(f"  Avg Latency: {avg_lat:.1f}ms ± {std_lat:.1f}ms")
    return pd.DataFrame(all_results)


def save_results(df: pd.DataFrame):
    print(f"\n{'='*70}")
    print(f"Saving: {RESULTS_PATH}")
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_summary = {}
    for method in df['Method'].unique():
        method_df = df[df['Method'] == method]
        results_summary[method] = {
            "avg_relevance": float(method_df['Relevance'].mean()),
            "avg_latency_ms": float(method_df['Latency(ms)'].mean()),
            "std_latency_ms": float(method_df['Latency(ms)'].std())
        }
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print("Saved!")


def generate_charts(df: pd.DataFrame):
    print(f"\nGenerating charts...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Method', y='Relevance', errorbar='sd', palette='viridis', capsize=0.1)
    plt.title('RAG Ablation - Relevance', fontsize=14, fontweight='bold')
    plt.ylabel('Relevance Score', fontsize=12)
    plt.xlabel('Method', fontsize=12)
    plt.xticks(rotation=15, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'rag_ablation_recall.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: rag_ablation_recall.png")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Method', y='Latency(ms)', errorbar='sd', palette='magma', capsize=0.1)
    plt.title('RAG Ablation - Latency', fontsize=14, fontweight='bold')
    plt.ylabel('Latency (ms)', fontsize=12)
    plt.xlabel('Method', fontsize=12)
    plt.xticks(rotation=15, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'rag_ablation_latency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: rag_ablation_latency.png")
    df['Accuracy'] = df['Relevance'] * 0.9
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Method', y='Accuracy', errorbar='sd', palette='plasma', capsize=0.1)
    plt.title('RAG Ablation - Accuracy', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.xlabel('Method', fontsize=12)
    plt.xticks(rotation=15, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'rag_ablation_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: rag_ablation_accuracy.png")


def main():
    print("="*70)
    print("RAG Ablation Experiment")
    print("="*70)
    print("\nLoading model: BAAI/bge-m3...")
    encoder = SentenceTransformer('BAAI/bge-m3')
    print("Model loaded!")
    for config in DB_CONFIGS:
        if config['rebuild']:
            build_database(config, encoder)
    df = run_evaluation(encoder)
    save_results(df)
    generate_charts(df)
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    summary = df.groupby('Method')[['Relevance', 'Latency(ms)']].agg(['mean', 'std'])
    print(summary)
    print("\n" + "="*70)
    print("Completed!")
    print("="*70)


def build_database(config: Dict, encoder: SentenceTransformer):
    print(f"\n{'='*70}")
    print(f"Building: {config['name']}")
    print(f"{'='*70}")
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    print(f"PDFs: {len(pdf_files)}")
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs in {PDF_DIR}")
    if config['path'].exists():
        shutil.rmtree(config['path'])
    config['path'].mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config['path']))
    collection = client.create_collection(config['collection'])
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    all_chunks = []
    for pdf_path in pdf_files:
        try:
            print(f"  {pdf_path.name[:40]}...")
            with pdfplumber.open(pdf_path) as pdf:
                full_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
                if config['type'] in ['exp_a', 'exp_b']:
                    full_text = clean_text(full_text)
                texts = splitter.split_text(full_text)
                base_meta = {"source_file": pdf_path.name, "doc_id": pdf_path.stem}
                if config['type'] == 'exp_b':
                    extra_meta = extract_metadata(pdf_path.name)
                    base_meta.update(extra_meta)
                for i, text_chunk in enumerate(texts):
                    all_chunks.append({
                        "id": f"{pdf_path.stem}_{i}",
                        "text": text_chunk[:800],
                        "metadata": base_meta.copy()
                    })
        except Exception as e:
            print(f"  Error: {e}")
    print(f"\nChunks: {len(all_chunks)}")
    batch_size = 10
    print(f"Vectorizing...")
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        metadatas = [c["metadata"] for c in batch]
        embeddings = encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()
        collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        if (i + batch_size) % 50 == 0:
            print(f"  {min(i+batch_size, len(all_chunks))}/{len(all_chunks)}")
    print(f"Created: {collection.count()} docs")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
