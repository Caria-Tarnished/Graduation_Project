# -*- coding: utf-8 -*-
"""
RAG Engine 测试脚本

功能：
1. 测试 RAG Engine 能否正常加载 Chroma 向量库
2. 测试检索功能是否正常工作
3. 验证返回的 Citation 对象是否包含正确的元数据
4. 测试不同查询场景

使用方法：
    python scripts/rag/test_rag_engine.py
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.core.engines.rag_engine import RagEngine


def test_initialization():
    """测试 1: RAG Engine 初始化"""
    print("\n" + "=" * 80)
    print("测试 1: RAG Engine 初始化")
    print("=" * 80)
    
    try:
        engine = RagEngine(
            chroma_path="data/reports/chroma_db",
            model_name="BAAI/bge-m3"
        )
        print("✓ RAG Engine 初始化成功")
        return engine
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        return None


def test_stats(engine):
    """测试 2: 获取向量库统计信息"""
    print("\n" + "=" * 80)
    print("测试 2: 向量库统计信息")
    print("=" * 80)
    
    try:
        stats = engine.get_stats()
        print(f"✓ 统计信息获取成功:")
        print(f"  - 集合名称: {stats['collection_name']}")
        print(f"  - 切片总数: {stats['total_chunks']}")
        print(f"  - 嵌入模型: {stats['model_name']}")
        print(f"  - 存储路径: {stats['chroma_path']}")
        return True
    except Exception as e:
        print(f"✗ 获取统计信息失败: {e}")
        return False


def test_basic_retrieval(engine):
    """测试 3: 基础检索功能"""
    print("\n" + "=" * 80)
    print("测试 3: 基础检索功能")
    print("=" * 80)
    
    test_queries = [
        "黄金价格走势",
        "gold price trend",
        "美联储加息",
        "Federal Reserve interest rate"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        try:
            citations = engine.retrieve(query, top_k=3)
            print(f"✓ 检索成功，返回 {len(citations)} 个结果")
            
            for i, citation in enumerate(citations, 1):
                print(f"\n  [{i}] 相似度: {citation.score:.4f}")
                print(f"      来源: {citation.source_file}")
                print(f"      切片索引: {citation.chunk_index}")
                print(f"      内容预览: {citation.text[:100]}...")
                
                # 验证元数据
                if citation.metadata:
                    print(f"      元数据: {list(citation.metadata.keys())}")
        
        except Exception as e:
            print(f"✗ 检索失败: {e}")
            return False
    
    return True


def test_metadata_filtering(engine):
    """测试 4: 元数据过滤"""
    print("\n" + "=" * 80)
    print("测试 4: 元数据过滤（按语言）")
    print("=" * 80)
    
    query = "gold market analysis"
    
    # 测试中文过滤
    print(f"\n查询: {query} (仅中文)")
    try:
        citations_zh = engine.retrieve(
            query, 
            top_k=3, 
            filter_metadata={"language": "zh"}
        )
        print(f"✓ 中文结果: {len(citations_zh)} 个")
        for i, citation in enumerate(citations_zh, 1):
            print(f"  [{i}] {citation.source_file} (相似度: {citation.score:.4f})")
    except Exception as e:
        print(f"✗ 中文过滤失败: {e}")
    
    # 测试英文过滤
    print(f"\n查询: {query} (仅英文)")
    try:
        citations_en = engine.retrieve(
            query, 
            top_k=3, 
            filter_metadata={"language": "en"}
        )
        print(f"✓ 英文结果: {len(citations_en)} 个")
        for i, citation in enumerate(citations_en, 1):
            print(f"  [{i}] {citation.source_file} (相似度: {citation.score:.4f})")
    except Exception as e:
        print(f"✗ 英文过滤失败: {e}")
    
    return True


def test_edge_cases(engine):
    """测试 5: 边界情况"""
    print("\n" + "=" * 80)
    print("测试 5: 边界情况")
    print("=" * 80)
    
    # 测试空查询
    print("\n5.1 空查询")
    try:
        citations = engine.retrieve("", top_k=3)
        print(f"✓ 空查询返回 {len(citations)} 个结果")
    except Exception as e:
        print(f"✗ 空查询失败: {e}")
    
    # 测试超长查询
    print("\n5.2 超长查询")
    long_query = "黄金价格 " * 100
    try:
        citations = engine.retrieve(long_query, top_k=3)
        print(f"✓ 超长查询返回 {len(citations)} 个结果")
    except Exception as e:
        print(f"✗ 超长查询失败: {e}")
    
    # 测试 top_k=0
    print("\n5.3 top_k=0")
    try:
        citations = engine.retrieve("黄金", top_k=0)
        print(f"✓ top_k=0 返回 {len(citations)} 个结果")
    except Exception as e:
        print(f"✗ top_k=0 失败: {e}")
    
    # 测试 top_k 超大
    print("\n5.4 top_k=1000")
    try:
        citations = engine.retrieve("黄金", top_k=1000)
        print(f"✓ top_k=1000 返回 {len(citations)} 个结果")
    except Exception as e:
        print(f"✗ top_k=1000 失败: {e}")
    
    return True


def test_citation_structure(engine):
    """测试 6: Citation 对象结构验证"""
    print("\n" + "=" * 80)
    print("测试 6: Citation 对象结构验证")
    print("=" * 80)
    
    query = "黄金市场分析"
    try:
        citations = engine.retrieve(query, top_k=1)
        
        if len(citations) == 0:
            print("✗ 没有返回结果")
            return False
        
        citation = citations[0]
        
        # 验证必需字段
        print("\n验证必需字段:")
        assert hasattr(citation, 'text'), "缺少 text 字段"
        assert hasattr(citation, 'score'), "缺少 score 字段"
        assert hasattr(citation, 'source_file'), "缺少 source_file 字段"
        assert hasattr(citation, 'chunk_index'), "缺少 chunk_index 字段"
        assert hasattr(citation, 'metadata'), "缺少 metadata 字段"
        print("✓ 所有必需字段存在")
        
        # 验证字段类型
        print("\n验证字段类型:")
        assert isinstance(citation.text, str), "text 应为 str"
        assert isinstance(citation.score, float), "score 应为 float"
        assert isinstance(citation.source_file, str), "source_file 应为 str"
        assert isinstance(citation.chunk_index, int), "chunk_index 应为 int"
        assert isinstance(citation.metadata, dict), "metadata 应为 dict"
        print("✓ 所有字段类型正确")
        
        # 验证字段值范围
        print("\n验证字段值范围:")
        assert 0 <= citation.score <= 1, f"score 应在 [0, 1] 范围内，实际: {citation.score}"
        assert len(citation.text) > 0, "text 不应为空"
        assert citation.chunk_index >= 0, f"chunk_index 应 >= 0，实际: {citation.chunk_index}"
        print("✓ 所有字段值范围正确")
        
        # 打印示例
        print("\nCitation 示例:")
        print(f"  text: {citation.text[:100]}...")
        print(f"  score: {citation.score:.4f}")
        print(f"  source_file: {citation.source_file}")
        print(f"  chunk_index: {citation.chunk_index}")
        print(f"  metadata: {citation.metadata}")
        
        return True
    
    except AssertionError as e:
        print(f"✗ 验证失败: {e}")
        return False
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


def main():
    """主测试流程"""
    print("\n" + "=" * 80)
    print("RAG Engine 测试套件")
    print("=" * 80)
    
    # 测试 1: 初始化
    engine = test_initialization()
    if engine is None:
        print("\n✗ 初始化失败，终止测试")
        return
    
    # 测试 2: 统计信息
    if not test_stats(engine):
        print("\n⚠ 统计信息测试失败，但继续其他测试")
    
    # 测试 3: 基础检索
    if not test_basic_retrieval(engine):
        print("\n✗ 基础检索测试失败，终止测试")
        return
    
    # 测试 4: 元数据过滤
    if not test_metadata_filtering(engine):
        print("\n⚠ 元数据过滤测试失败，但继续其他测试")
    
    # 测试 5: 边界情况
    if not test_edge_cases(engine):
        print("\n⚠ 边界情况测试失败，但继续其他测试")
    
    # 测试 6: Citation 结构验证
    if not test_citation_structure(engine):
        print("\n✗ Citation 结构验证失败")
        return
    
    # 总结
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    print("\n✓ RAG Engine 工作正常")
    print("✓ 检索功能正常")
    print("✓ Citation 对象结构正确")
    print("\n下一步:")
    print("  1. 更新 Project_Status.md 标记任务 2.3 和 2.4 完成")
    print("  2. 开始阶段 3（Agent 编排与工具集成）")


if __name__ == "__main__":
    main()
