# -*- coding: utf-8 -*-
"""
RAG Engine - 检索增强生成引擎

功能：
1. 加载 Chroma 向量库
2. 加载嵌入模型
3. 提供检索接口（query -> top_k citations）
4. 支持元数据过滤（日期、来源等）

使用示例：
    engine = RagEngine(
        chroma_path="data/reports/chroma_db",
        model_name="BAAI/bge-m3"
    )
    
    citations = engine.retrieve(
        query="贵州茅台2023年营收情况",
        top_k=5
    )
"""
import os
from typing import List, Optional, Dict
from dataclasses import dataclass

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("需要安装 sentence-transformers: pip install sentence-transformers")

try:
    import chromadb
except ImportError:
    raise ImportError("需要安装 chromadb: pip install chromadb")


@dataclass
class Citation:
    """引用片段"""
    text: str                    # 文本内容
    score: float                 # 相似度分数
    source_file: str             # 来源文件
    chunk_index: int             # 切片索引
    metadata: Dict               # 元数据（日期、来源、语言等）


class RagEngine:
    """RAG 检索引擎"""
    
    def __init__(
        self,
        chroma_path: str,
        model_name: str = "BAAI/bge-m3",
        collection_name: str = "reports_chunks"
    ):
        """
        初始化 RAG 引擎
        
        Args:
            chroma_path: Chroma 数据库路径
            model_name: 嵌入模型名称
            collection_name: 集合名称
        """
        self.chroma_path = chroma_path
        self.model_name = model_name
        self.collection_name = collection_name
        
        # 检查 Chroma 数据库是否存在
        if not os.path.exists(chroma_path):
            raise FileNotFoundError(
                f"Chroma 数据库不存在: {chroma_path}\n"
                f"请先运行: python scripts/rag/build_vector_index.py"
            )
        
        # 加载嵌入模型
        print(f"加载嵌入模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # 加载 Chroma 客户端
        print(f"加载 Chroma 数据库: {chroma_path}")
        self.client = chromadb.PersistentClient(path=chroma_path)
        
        # 获取集合
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"✓ RAG 引擎初始化完成（集合: {collection_name}）")
        except Exception as e:
            raise ValueError(
                f"无法加载集合 '{collection_name}': {e}\n"
                f"请检查 Chroma 数据库是否正确构建"
            )
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Citation]:
        """
        检索相关片段
        
        Args:
            query: 查询文本
            top_k: 返回前 k 个结果
            filter_metadata: 元数据过滤条件（如 {"language": "zh"}）
        
        Returns:
            引用片段列表
        """
        # 1. 查询向量化
        query_embedding = self.model.encode(query).tolist()
        
        # 2. Chroma 检索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata  # 元数据过滤
        )
        
        # 3. 构建 Citation 列表
        citations = []
        
        if results['documents'] and len(results['documents']) > 0:
            documents = results['documents'][0]
            distances = results['distances'][0]
            metadatas = results['metadatas'][0]
            
            for i in range(len(documents)):
                # 计算相似度分数（距离越小，相似度越高）
                # 使用 1 / (1 + distance) 转换为 [0, 1] 区间
                score = 1.0 / (1.0 + distances[i])
                
                citation = Citation(
                    text=documents[i],
                    score=score,
                    source_file=metadatas[i].get('source_file', 'Unknown'),
                    chunk_index=metadatas[i].get('chunk_index', -1),
                    metadata=metadatas[i]
                )
                citations.append(citation)
        
        return citations
    
    def retrieve_by_date_range(
        self,
        query: str,
        start_date: str,
        end_date: str,
        top_k: int = 5
    ) -> List[Citation]:
        """
        按日期范围检索
        
        Args:
            query: 查询文本
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
            top_k: 返回前 k 个结果
        
        Returns:
            引用片段列表
        """
        # 注意：Chroma 的 where 过滤不支持范围查询
        # 这里先检索所有结果，然后在内存中过滤
        
        # 检索更多结果
        all_citations = self.retrieve(query, top_k=top_k * 3)
        
        # 过滤日期范围
        filtered = []
        for citation in all_citations:
            date = citation.metadata.get('date', 'Unknown')
            if date != 'Unknown' and start_date <= date <= end_date:
                filtered.append(citation)
        
        # 返回前 top_k 个
        return filtered[:top_k]
    
    def get_stats(self) -> Dict:
        """
        获取向量库统计信息
        
        Returns:
            统计信息字典
        """
        count = self.collection.count()
        
        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "model_name": self.model_name,
            "chroma_path": self.chroma_path
        }


# 测试代码
if __name__ == "__main__":
    # 测试 RAG 引擎
    try:
        engine = RagEngine(
            chroma_path="data/reports/chroma_db",
            model_name="BAAI/bge-m3"
        )
        
        # 获取统计信息
        stats = engine.get_stats()
        print("\n向量库统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 测试检索
        print("\n测试检索:")
        query = "黄金价格走势"
        citations = engine.retrieve(query, top_k=3)
        
        print(f"\n查询: {query}")
        print(f"结果数: {len(citations)}")
        
        for i, citation in enumerate(citations, 1):
            print(f"\n[{i}] 相似度: {citation.score:.4f}")
            print(f"    来源: {citation.source_file}")
            print(f"    内容: {citation.text[:100]}...")
    
    except Exception as e:
        print(f"错误: {e}")
