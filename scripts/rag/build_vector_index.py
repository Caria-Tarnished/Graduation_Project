# -*- coding: utf-8 -*-
"""
向量化与索引构建脚本

功能：
1. 加载切片数据（chunks.json）
2. 使用 bge-m3 嵌入模型向量化
3. 构建 Chroma 向量库
4. 支持批量处理和进度显示

使用方法：
    python scripts/rag/build_vector_index.py
"""
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers not installed")
    print("Run: pip install sentence-transformers")
    exit(1)

try:
    import chromadb
except ImportError:
    print("Error: chromadb not installed")
    print("Run: pip install chromadb")
    exit(1)


def load_chunks(chunks_path: str) -> List[Dict]:
    """Load chunks from JSON file"""
    print(f"Loading chunks: {chunks_path}")
    
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"  Loaded {len(chunks)} chunks")
    return chunks


def build_vector_index(
    chunks: List[Dict],
    output_path: str,
    model_name: str = "BAAI/bge-m3",
    batch_size: int = 32,
    collection_name: str = "reports_chunks"
):
    """Build vector index"""
    # 1. Load embedding model
    print(f"\nLoading embedding model: {model_name}")
    print("  (First run will download model, ~2GB)")
    
    model = SentenceTransformer(model_name)
    print("  Model loaded")
    
    # 2. Initialize Chroma client
    print(f"\nInitializing Chroma database: {output_path}")
    
    # Remove old database if exists
    if os.path.exists(output_path):
        print("  Removing old database")
        import shutil
        shutil.rmtree(output_path)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Create Chroma client
    client = chromadb.PersistentClient(path=output_path)
    
    # Create collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "Financial reports chunks"}
    )
    print(f"  Created collection: {collection_name}")
    
    # 3. Batch embedding and insertion
    print(f"\nStarting vectorization (batch size: {batch_size})")
    
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        print(f"  Processing batch {batch_num}/{total_batches}")
        
        # Extract texts
        texts = [chunk['text'] for chunk in batch]
        
        # Vectorize
        embeddings = model.encode(texts, show_progress_bar=False)
        
        # Prepare data
        ids = [chunk['chunk_id'] for chunk in batch]
        metadatas = [chunk['metadata'] for chunk in batch]
        
        # Insert into Chroma
        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )
    
    print(f"\nVectorization complete")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Vector dimension: {embeddings.shape[1]}")
    print(f"  Database path: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build vector index")
    parser.add_argument(
        "--input",
        type=str,
        default="data/reports/chunks.json",
        help="Input chunks file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/reports/chroma_db",
        help="Output Chroma database directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-m3",
        help="Embedding model name"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="reports_chunks",
        help="Collection name"
    )
    
    args = parser.parse_args()
    
    # Check input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    print("=" * 80)
    print("Vector Index Builder")
    print("=" * 80)
    
    # 1. Load chunks
    chunks = load_chunks(args.input)
    
    # 2. Build vector index
    build_vector_index(
        chunks=chunks,
        output_path=args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        collection_name=args.collection
    )
    
    print("\n" + "=" * 80)
    print("Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
