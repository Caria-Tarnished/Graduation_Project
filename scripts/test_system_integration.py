# -*- coding: utf-8 -*-
"""
系统集成测试脚本

功能：
1. 测试所有核心组件的加载和初始化
2. 测试快讯分析功能（Engine A + 规则引擎）
3. 测试财报检索功能（Engine B + RAG）
4. 测试 Agent 编排器
5. 生成测试报告

使用方法：
    python scripts/test_system_integration.py
"""
import sys
import os
from pathlib import Path
import time
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_section(title: str):
    """打印章节标题"""
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}\n")


def print_result(test_name: str, passed: bool, message: str = ""):
    """打印测试结果"""
    status = "✓ 通过" if passed else "✗ 失败"
    print(f"{status} | {test_name}")
    if message:
        print(f"     {message}")


def test_environment():
    """测试环境配置"""
    print_section("1. 环境配置测试")
    
    results = []
    
    # 测试 .env 文件
    env_exists = Path(".env").exists()
    print_result(".env 文件", env_exists)
    results.append(("环境变量文件", env_exists))
    
    # 测试数据库
    db_exists = Path("finance_analysis.db").exists()
    print_result("数据库文件", db_exists)
    results.append(("数据库", db_exists))
    
    # 测试 Chroma 向量库
    chroma_exists = Path("data/reports/chroma_db").exists()
    print_result("Chroma 向量库", chroma_exists)
    results.append(("向量库", chroma_exists))
    
    # 测试 BERT 模型（可选）
    bert_exists = Path("models/bert_3cls/best").exists()
    print_result("BERT 模型（可选）", bert_exists, 
                 "未找到，将使用降级模式" if not bert_exists else "")
    results.append(("BERT 模型", bert_exists))
    
    return results


def test_imports():
    """测试模块导入"""
    print_section("2. 模块导入测试")
    
    results = []
    
    # 测试核心依赖
    try:
        import torch
        print_result("PyTorch", True, f"版本: {torch.__version__}")
        results.append(("PyTorch", True))
    except ImportError as e:
        print_result("PyTorch", False, str(e))
        results.append(("PyTorch", False))
    
    try:
        import transformers
        print_result("Transformers", True, f"版本: {transformers.__version__}")
        results.append(("Transformers", True))
    except ImportError as e:
        print_result("Transformers", False, str(e))
        results.append(("Transformers", False))
    
    try:
        import chromadb
        print_result("ChromaDB", True, f"版本: {chromadb.__version__}")
        results.append(("ChromaDB", True))
    except ImportError as e:
        print_result("ChromaDB", False, str(e))
        results.append(("ChromaDB", False))
    
    try:
        import streamlit
        print_result("Streamlit", True, f"版本: {streamlit.__version__}")
        results.append(("Streamlit", True))
    except ImportError as e:
        print_result("Streamlit", False, str(e))
        results.append(("Streamlit", False))
    
    return results


def test_database_connection():
    """测试数据库连接"""
    print_section("3. 数据库连接测试")
    
    results = []
    
    try:
        import sqlite3
        conn = sqlite3.connect("finance_analysis.db")
        cursor = conn.cursor()
        
        # 测试 prices_m1 表
        cursor.execute("SELECT COUNT(*) FROM prices_m1")
        price_count = cursor.fetchone()[0]
        print_result("prices_m1 表", True, f"记录数: {price_count:,}")
        results.append(("prices_m1", True))
        
        # 测试 events 表
        cursor.execute("SELECT COUNT(*) FROM events")
        event_count = cursor.fetchone()[0]
        print_result("events 表", True, f"记录数: {event_count:,}")
        results.append(("events", True))
        
        # 测试 event_impacts 表
        cursor.execute("SELECT COUNT(*) FROM event_impacts")
        impact_count = cursor.fetchone()[0]
        print_result("event_impacts 表", True, f"记录数: {impact_count:,}")
        results.append(("event_impacts", True))
        
        conn.close()
        
    except Exception as e:
        print_result("数据库连接", False, str(e))
        results.append(("数据库连接", False))
    
    return results


def test_rag_engine():
    """测试 RAG 引擎"""
    print_section("4. RAG 引擎测试")
    
    results = []
    
    try:
        # 动态导入 RAG 引擎
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "rag_engine",
            "app/core/engines/rag_engine.py"
        )
        rag_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rag_module)
        
        RagEngine = rag_module.RagEngine
        
        # 初始化引擎
        start = time.time()
        engine = RagEngine(
            chroma_path="data/reports/chroma_db",
            model_name="BAAI/bge-m3"
        )
        init_time = time.time() - start
        print_result("RAG 引擎初始化", True, f"耗时: {init_time:.2f}秒")
        results.append(("RAG 初始化", True))
        
        # 测试检索
        start = time.time()
        citations = engine.retrieve("黄金价格走势", top_k=3)
        retrieve_time = time.time() - start
        print_result("RAG 检索", True, 
                     f"找到 {len(citations)} 条结果，耗时: {retrieve_time:.2f}秒")
        results.append(("RAG 检索", True))
        
        # 显示第一条结果
        if citations:
            print(f"\n     示例结果:")
            print(f"     相似度: {citations[0].score:.4f}")
            print(f"     来源: {citations[0].source_file}")
            print(f"     内容: {citations[0].text[:100]}...")
        
    except Exception as e:
        print_result("RAG 引擎", False, str(e))
        results.append(("RAG 引擎", False))
    
    return results


def test_sentiment_analyzer():
    """测试情感分析器"""
    print_section("5. 情感分析器测试（可选）")
    
    results = []
    
    # 检查模型是否存在
    if not Path("models/bert_3cls/best").exists():
        print_result("情感分析器", False, "BERT 模型不存在，跳过测试")
        results.append(("情感分析器", False))
        return results
    
    try:
        # 动态导入情感分析器
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "sentiment_analyzer",
            "app/services/sentiment_analyzer.py"
        )
        sa_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sa_module)
        
        SentimentAnalyzer = sa_module.SentimentAnalyzer
        
        # 初始化分析器
        start = time.time()
        analyzer = SentimentAnalyzer(model_path="models/bert_3cls/best")
        init_time = time.time() - start
        print_result("情感分析器初始化", True, f"耗时: {init_time:.2f}秒")
        results.append(("情感分析器初始化", True))
        
        # 测试分析
        test_text = "美联储宣布降息25个基点 符合市场预期"
        start = time.time()
        result = analyzer.analyze(
            text=test_text,
            pre_ret=0.002,
            range_ratio=0.005
        )
        analyze_time = time.time() - start
        print_result("情感分析", True, f"耗时: {analyze_time:.2f}秒")
        results.append(("情感分析", True))
        
        # 显示结果
        print(f"\n     测试文本: {test_text}")
        print(f"     基础情感: {result['base_sentiment']}")
        print(f"     最终情感: {result['final_sentiment']}")
        print(f"     解释: {result['explanation']}")
        
    except Exception as e:
        print_result("情感分析器", False, str(e))
        results.append(("情感分析器", False))
    
    return results


def test_agent():
    """测试 Agent 编排器"""
    print_section("6. Agent 编排器测试")
    
    results = []
    
    try:
        # 动态导入 Agent
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "agent",
            "app/core/orchestrator/agent.py"
        )
        agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_module)
        
        Agent = agent_module.Agent
        
        # 初始化 Agent（无引擎）
        agent = Agent()
        print_result("Agent 初始化", True, "使用降级模式")
        results.append(("Agent 初始化", True))
        
        # 测试查询类型检测
        query1 = "美联储宣布加息 25 个基点"
        query_type1 = agent._detect_query_type(query1)
        print_result("查询类型检测（快讯）", query_type1 == "news_analysis",
                     f"检测结果: {query_type1}")
        results.append(("查询类型检测（快讯）", query_type1 == "news_analysis"))
        
        query2 = "贵州茅台 2023 年营收情况如何？"
        query_type2 = agent._detect_query_type(query2)
        print_result("查询类型检测（财报）", query_type2 == "report_qa",
                     f"检测结果: {query_type2}")
        results.append(("查询类型检测（财报）", query_type2 == "report_qa"))
        
        # 测试完整查询（降级模式）
        start = time.time()
        answer = agent.process_query(query1)
        process_time = time.time() - start
        print_result("Agent 查询处理", True, f"耗时: {process_time:.2f}秒")
        results.append(("Agent 查询处理", True))
        
        # 显示结果
        print(f"\n     查询: {query1}")
        print(f"     查询类型: {answer.query_type}")
        print(f"     工具调用: {len(answer.tool_trace)} 个")
        for trace in answer.tool_trace:
            status = "✓" if trace.ok else "✗"
            print(f"       {status} {trace.name} ({trace.elapsed_ms}ms)")
        
    except Exception as e:
        print_result("Agent 编排器", False, str(e))
        results.append(("Agent 编排器", False))
    
    return results


def generate_report(all_results):
    """生成测试报告"""
    print_section("测试报告")
    
    total_tests = sum(len(results) for results in all_results.values())
    passed_tests = sum(
        sum(1 for _, passed in results if passed)
        for results in all_results.values()
    )
    
    print(f"总测试数: {total_tests}")
    print(f"通过: {passed_tests}")
    print(f"失败: {total_tests - passed_tests}")
    print(f"通过率: {passed_tests / total_tests * 100:.1f}%")
    
    print(f"\n详细结果:")
    for category, results in all_results.items():
        print(f"\n{category}:")
        for test_name, passed in results:
            status = "✓" if passed else "✗"
            print(f"  {status} {test_name}")
    
    # 保存报告到文件
    report_path = "test_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"系统集成测试报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"总测试数: {total_tests}\n")
        f.write(f"通过: {passed_tests}\n")
        f.write(f"失败: {total_tests - passed_tests}\n")
        f.write(f"通过率: {passed_tests / total_tests * 100:.1f}%\n\n")
        f.write(f"详细结果:\n")
        for category, results in all_results.items():
            f.write(f"\n{category}:\n")
            for test_name, passed in results:
                status = "✓" if passed else "✗"
                f.write(f"  {status} {test_name}\n")
    
    print(f"\n报告已保存到: {report_path}")


def main():
    """主测试函数"""
    print("=" * 80)
    print("系统集成测试")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = {}
    
    # 执行所有测试
    all_results["环境配置"] = test_environment()
    all_results["模块导入"] = test_imports()
    all_results["数据库连接"] = test_database_connection()
    all_results["RAG 引擎"] = test_rag_engine()
    all_results["情感分析器"] = test_sentiment_analyzer()
    all_results["Agent 编排器"] = test_agent()
    
    # 生成报告
    generate_report(all_results)
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
