"""
改进版RAG消融实验 v2
改进点:
1. 扩展元数据: 从文件名提取语言(zh/en)和日期
2. 为更多问题设置有意义的过滤条件(语言、机构)
3. 重建Exp B数据库, 包含更丰富的元数据
"""

import sys
import os
import json
import time
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
from sentence_transformers import SentenceTransformer
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        # 简易实现
        class RecursiveCharacterTextSplitter:
            def __init__(self, chunk_size=500, chunk_overlap=50, **kwargs):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
            def split_text(self, text):
                chunks = []
                start = 0
                while start < len(text):
                    end = start + self.chunk_size
                    chunks.append(text[start:end])
                    start = end - self.chunk_overlap
                return [c for c in chunks if len(c.strip()) > 20]

# ── 路径 ──
RAW_REPORTS_DIR = PROJECT_ROOT / "data" / "raw" / "reports" / "research_reports"
DB_BASELINE = PROJECT_ROOT / "data" / "reports" / "chroma_db_v2_baseline"
DB_EXP_A = PROJECT_ROOT / "data" / "reports" / "chroma_db_v2_exp_a"
DB_EXP_B = PROJECT_ROOT / "data" / "reports" / "chroma_db_v2_exp_b"
RESULTS_PATH = PROJECT_ROOT / "data" / "reports" / "rag_ablation_results_v2.json"


# ═══════════════════════════════════════════
# 元数据提取（改进版）
# ═══════════════════════════════════════════
def extract_metadata_v2(pdf_name: str) -> dict:
    """从文件名提取丰富的元数据"""
    meta = {"source_file": pdf_name, "doc_id": Path(pdf_name).stem}

    # 机构
    if "Goldman" in pdf_name:
        meta["source"] = "Goldman Sachs"
    elif "J.P. Morgan" in pdf_name or "J.P.Morgan" in pdf_name:
        meta["source"] = "J.P. Morgan"
    elif "Morgan Stanley" in pdf_name:
        meta["source"] = "Morgan Stanley"
    elif "UBS" in pdf_name:
        meta["source"] = "UBS"
    else:
        meta["source"] = "other"

    # 语言: H3开头或中文文件名为中文, 其余为英文
    if pdf_name.startswith("H3_") or pdf_name.startswith("4f20"):
        meta["language"] = "zh"
    else:
        meta["language"] = "en"

    # 日期: 尝试从文件名提取
    # 格式1: -260123 (YYMMDD)
    m = re.search(r'-(\d{6})\.pdf$', pdf_name)
    if m:
        yy, mm, dd = m.group(1)[:2], m.group(1)[2:4], m.group(1)[4:6]
        meta["date"] = f"20{yy}-{mm}-{dd}"
    # 格式2: 20250406
    m2 = re.search(r'(20\d{6})', pdf_name)
    if m2:
        d = m2.group(1)
        meta["date"] = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
    # 格式3: H3_AP2025MMDD...
    m3 = re.search(r'H3_AP(\d{8})', pdf_name)
    if m3:
        d = m3.group(1)
        meta["date"] = f"{d[:4]}-{d[4:6]}-{d[6:8]}"

    if "date" not in meta:
        meta["date"] = "unknown"

    return meta


def clean_text(text: str) -> str:
    """清洗文本, 去除噪声"""
    lines = text.split('\n')
    cleaned = []
    skip_kw = ['disclaimer', 'contact', 'analyst', '免责声明', '风险提示',
                '联系方式', '分析师', '@']
    for line in lines:
        line = line.strip()
        if len(line) < 3:
            continue
        if any(kw in line.lower() for kw in skip_kw):
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)


# ═══════════════════════════════════════════
# 构建数据库
# ═══════════════════════════════════════════
def build_databases(encoder):
    """构建三个对比数据库"""
    import pdfplumber

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )

    pdf_files = sorted(RAW_REPORTS_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs")

    # 收集所有文档数据
    all_docs = []  # [(pdf_name, raw_text, cleaned_text, metadata)]

    for pdf_path in pdf_files:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                raw_text = ""
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        raw_text += t + "\n"

            if len(raw_text.strip()) < 50:
                print(f"  Skip (too short): {pdf_path.name}")
                continue

            cleaned = clean_text(raw_text)
            meta = extract_metadata_v2(pdf_path.name)
            all_docs.append((pdf_path.name, raw_text, cleaned, meta))
            print(f"  OK: {pdf_path.name} -> {meta.get('source')}, {meta.get('language')}, {meta.get('date')}")

        except Exception as e:
            print(f"  Error: {pdf_path.name}: {e}")

    print(f"\nProcessed {len(all_docs)} PDFs successfully")

    # 构建 Baseline: 原始文本, 无元数据过滤
    print("\n--- Building Baseline ---")
    build_single_db(DB_BASELINE, "reports_chunks_baseline",
                    all_docs, encoder, splitter,
                    use_cleaning=False, use_metadata=False)

    # 构建 Exp A: 清洗文本, 无元数据
    print("\n--- Building Exp A ---")
    build_single_db(DB_EXP_A, "reports_chunks_cleaned",
                    all_docs, encoder, splitter,
                    use_cleaning=True, use_metadata=False)

    # 构建 Exp B: 清洗文本 + 元数据
    print("\n--- Building Exp B ---")
    build_single_db(DB_EXP_B, "reports_chunks_enhanced",
                    all_docs, encoder, splitter,
                    use_cleaning=True, use_metadata=True)


def build_single_db(db_path, collection_name, all_docs, encoder, splitter,
                    use_cleaning, use_metadata):
    """构建单个Chroma数据库"""
    import shutil
    if db_path.exists():
        shutil.rmtree(db_path)

    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(name=collection_name)

    total_chunks = 0
    for pdf_name, raw_text, cleaned_text, meta in all_docs:
        text = cleaned_text if use_cleaning else raw_text
        chunks = splitter.split_text(text)

        if not chunks:
            continue

        ids = [f"{meta['doc_id']}_{i}" for i in range(len(chunks))]
        embeddings = encoder.encode(chunks, normalize_embeddings=True).tolist()

        metadatas = []
        for i in range(len(chunks)):
            m = {"source_file": pdf_name, "chunk_index": i}
            if use_metadata:
                m["source"] = meta.get("source", "other")
                m["language"] = meta.get("language", "unknown")
                m["date"] = meta.get("date", "unknown")
            metadatas.append(m)

        collection.add(ids=ids, embeddings=embeddings,
                       documents=chunks, metadatas=metadatas)
        total_chunks += len(chunks)

    print(f"  Total chunks: {total_chunks}")


# ═══════════════════════════════════════════
# 改进版测试问题
# ═══════════════════════════════════════════
EVAL_QUESTIONS = [
    # --- 机构定向查询 (英文问英文研报) ---
    {"id": "Q1", "query": "What is Goldman Sachs' gold price forecast for 2026?",
     "metadata_filter": {"source": "Goldman Sachs"}},
    {"id": "Q2", "query": "What is Morgan Stanley's analysis framework for the gold bull case?",
     "metadata_filter": {"source": "Morgan Stanley"}},
    {"id": "Q3", "query": "Does UBS think the gold rally is over?",
     "metadata_filter": {"source": "UBS"}},
    {"id": "Q4", "query": "What is J.P. Morgan's investment recommendation on gold and silver?",
     "metadata_filter": {"source": "J.P. Morgan"}},

    # --- 中文研报问题 (语言过滤) ---
    {"id": "Q5", "query": "国内机构对贵金属市场的投资策略是什么？",
     "metadata_filter": {"language": "zh"}},
    {"id": "Q6", "query": "中国黄金市场的供需分析报告要点有哪些？",
     "metadata_filter": {"language": "zh"}},
    {"id": "Q7", "query": "近期贵金属行业研究的核心观点是什么？",
     "metadata_filter": {"language": "zh"}},

    # --- 英文研报问题 (语言过滤) ---
    {"id": "Q8", "query": "What are the key drivers of the current gold bull market?",
     "metadata_filter": {"language": "en"}},
    {"id": "Q9", "query": "How do international investment banks view the gold price outlook for 2026?",
     "metadata_filter": {"language": "en"}},
    {"id": "Q10", "query": "What impact do US tariff policies have on precious metals?",
     "metadata_filter": {"language": "en"}},

    # --- 通用问题 (无过滤, 作为基准) ---
    {"id": "Q11", "query": "影响黄金价格的主要因素有哪些？",
     "metadata_filter": {}},
    {"id": "Q12", "query": "地缘政治冲突是如何影响贵金属市场的？",
     "metadata_filter": {}},
    {"id": "Q13", "query": "美债收益率走势对非孳息资产的影响机制是什么？",
     "metadata_filter": {}},
    {"id": "Q14", "query": "为什么通货膨胀预期上升通常利好黄金？",
     "metadata_filter": {}},
    {"id": "Q15", "query": "黄金和白银的走势有什么区别？",
     "metadata_filter": {}},
    {"id": "Q16", "query": "现阶段投资黄金ETF和投资实物黄金各有什么优劣势？",
     "metadata_filter": {}},
    {"id": "Q17", "query": "相比2011年的高点, 当前的黄金牛市有哪些不同的驱动因素？",
     "metadata_filter": {}},
    {"id": "Q18", "query": "对于抗通胀需求, 黄金和比特币的避险属性有什么差异？",
     "metadata_filter": {}},
    {"id": "Q19", "query": "未来一个月金价走势如何？",
     "metadata_filter": {}},
    {"id": "Q20", "query": "在降息周期下, 贵金属板块的整体投资建议是什么？",
     "metadata_filter": {}},
]


# ═══════════════════════════════════════════
# 评估
# ═══════════════════════════════════════════
def evaluate(encoder):
    """在三个数据库上评估"""

    configs = [
        {"name": "Baseline", "path": DB_BASELINE,
         "collection": "reports_chunks_baseline", "use_filter": False},
        {"name": "Exp A", "path": DB_EXP_A,
         "collection": "reports_chunks_cleaned", "use_filter": False},
        {"name": "Exp B", "path": DB_EXP_B,
         "collection": "reports_chunks_enhanced", "use_filter": True},
    ]

    all_results = {}

    for cfg in configs:
        print(f"\n=== Evaluating: {cfg['name']} ===")

        client = chromadb.PersistentClient(path=str(cfg["path"]))
        collection = client.get_collection(name=cfg["collection"])
        print(f"  Chunks in DB: {collection.count()}")

        relevances = []
        latencies = []

        for q in EVAL_QUESTIONS:
            query = q["query"]
            meta_filter = q.get("metadata_filter", {})

            # 编码查询
            query_emb = encoder.encode(query, normalize_embeddings=True).tolist()

            # 构建where子句 (仅Exp B使用)
            where = None
            if cfg["use_filter"] and meta_filter:
                # ChromaDB where格式
                if len(meta_filter) == 1:
                    key, val = list(meta_filter.items())[0]
                    where = {key: {"$eq": val}}
                elif len(meta_filter) > 1:
                    where = {"$and": [{k: {"$eq": v}} for k, v in meta_filter.items()]}

            start = time.time()
            try:
                results = collection.query(
                    query_embeddings=[query_emb],
                    n_results=5,
                    where=where,
                    include=["documents", "metadatas", "distances"]
                )
            except Exception as e:
                # 过滤失败时降级为无过滤
                results = collection.query(
                    query_embeddings=[query_emb],
                    n_results=5,
                    include=["documents", "metadatas", "distances"]
                )
            latency = (time.time() - start) * 1000  # ms

            # 计算相关性 (Top-1 similarity)
            dists = results["distances"][0] if results["distances"] else []
            if dists:
                similarity = 1.0 / (1.0 + dists[0])
            else:
                similarity = 0.0

            relevances.append(similarity)
            latencies.append(latency)

        avg_rel = sum(relevances) / len(relevances)
        avg_lat = sum(latencies) / len(latencies)
        std_lat = (sum((l - avg_lat) ** 2 for l in latencies) / len(latencies)) ** 0.5

        # 分组统计: 有过滤的问题 vs 无过滤的问题
        filtered_rels = [relevances[i] for i, q in enumerate(EVAL_QUESTIONS) if q["metadata_filter"]]
        unfiltered_rels = [relevances[i] for i, q in enumerate(EVAL_QUESTIONS) if not q["metadata_filter"]]

        print(f"  Avg Relevance: {avg_rel:.4f}")
        print(f"  Avg Latency: {avg_lat:.2f}ms (std: {std_lat:.2f}ms)")
        if filtered_rels:
            print(f"  Filtered queries relevance: {sum(filtered_rels)/len(filtered_rels):.4f} (n={len(filtered_rels)})")
        if unfiltered_rels:
            print(f"  Unfiltered queries relevance: {sum(unfiltered_rels)/len(unfiltered_rels):.4f} (n={len(unfiltered_rels)})")

        all_results[cfg["name"]] = {
            "num_docs": collection.count(),
            "avg_relevance": round(avg_rel, 6),
            "avg_latency_ms": round(avg_lat, 2),
            "std_latency_ms": round(std_lat, 2),
            "filtered_relevance": round(sum(filtered_rels)/len(filtered_rels), 6) if filtered_rels else None,
            "unfiltered_relevance": round(sum(unfiltered_rels)/len(unfiltered_rels), 6) if unfiltered_rels else None,
        }

    # 计算变化百分比
    base = all_results["Baseline"]
    for name in ["Exp A", "Exp B"]:
        r = all_results[name]
        r["relevance_change"] = f"{(r['avg_relevance'] - base['avg_relevance']) / base['avg_relevance'] * 100:+.1f}%"
        r["latency_change"] = f"{(r['avg_latency_ms'] - base['avg_latency_ms']) / base['avg_latency_ms'] * 100:+.1f}%"

    # 保存
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    return all_results


# ═══════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("RAG消融实验 v2 (改进版)")
    print("=" * 60)

    print("\nLoading BAAI/bge-m3 model...")
    encoder = SentenceTransformer("BAAI/bge-m3")
    print("Model loaded.")

    # 构建数据库
    print("\n--- Phase 1: Building databases ---")
    build_databases(encoder)

    # 评估
    print("\n--- Phase 2: Evaluation ---")
    results = evaluate(encoder)

    # 总结
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    for name, r in results.items():
        print(f"\n{name}:")
        print(f"  Docs: {r['num_docs']}")
        print(f"  Relevance: {r['avg_relevance']:.4f}", end="")
        if "relevance_change" in r:
            print(f" ({r['relevance_change']})", end="")
        print(f"\n  Latency: {r['avg_latency_ms']:.2f}ms", end="")
        if "latency_change" in r:
            print(f" ({r['latency_change']})", end="")
        print()
