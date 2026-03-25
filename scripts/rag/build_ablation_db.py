# -*- coding: utf-8 -*-
import os
import json
import pdfplumber
import time
from pathlib import Path

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

def build_baseline():
    input_dir = "e:/Projects/Graduation_Project/data/raw/reports/research_reports"
    pdf_files = list(Path(input_dir).glob("*.pdf"))[:5] # 仅使用前5个PDF以防内存不足
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
    )
    
    chunks = []
    print("Building baseline chunks (no cleaning, no cropping)...")
    for pdf_path in pdf_files:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                full_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
                
                texts = splitter.split_text(full_text)
                for i, t in enumerate(texts):
                    chunks.append({
                        "id": f"{pdf_path.name}_{i}_baseline",
                        "text": t,
                        "metadata": {"source_file": pdf_path.name, "chunk_index": i}
                    })
        except Exception as e:
            print(f"Error parsing {pdf_path.name}: {e}")
            
    print(f"Total baseline chunks: {len(chunks)}")
    
    db_path = "e:/Projects/Graduation_Project/data/reports/chroma_db_baseline"
    if os.path.exists(db_path):
        import shutil
        shutil.rmtree(db_path)
    os.makedirs(db_path, exist_ok=True)
    
    client = chromadb.PersistentClient(path=db_path)
    collection = client.create_collection("reports_chunks_baseline")
    
    print("Loading embedding model BAAI/bge-m3...")
    model = SentenceTransformer("BAAI/bge-m3")
    
    batch_size = 1
    print("Vectorizing baseline chunks...")
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [c["text"][:800] for c in batch] # 强行截断避免单块过大导致OOM
        ids = [c["id"] for c in batch]
        metadatas = [c["metadata"] for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        print(f"Processed {min(i+batch_size, len(chunks))}/{len(chunks)}")
    
    print("Baseline DB created at", db_path)

if __name__ == "__main__":
    build_baseline()
