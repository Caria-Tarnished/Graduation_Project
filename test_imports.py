# -*- coding: utf-8 -*-
"""
模块导入测试脚本

用途：测试所有关键模块是否能正确导入

运行命令：
    python test_imports.py
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("=" * 60)
print("模块导入测试")
print("=" * 60)
print()

# 测试列表
tests = []

# 1. 测试 DTO 模块
print("1. 测试 DTO 模块...")
try:
    from app.core.dto import AgentAnswer, ToolTraceItem, NewsItem, SentimentResult
    print("   ✓ app.core.dto 导入成功")
    tests.append(("DTO", True, None))
except Exception as e:
    print(f"   ❌ app.core.dto 导入失败: {e}")
    tests.append(("DTO", False, str(e)))

print()

# 2. 测试 Agent 模块
print("2. 测试 Agent 模块...")
try:
    from app.core.orchestrator.agent import Agent
    print("   ✓ app.core.orchestrator.agent 导入成功")
    tests.append(("Agent", True, None))
except Exception as e:
    print(f"   ❌ app.core.orchestrator.agent 导入失败: {e}")
    tests.append(("Agent", False, str(e)))

print()

# 3. 测试情感分析引擎
print("3. 测试情感分析引擎...")
try:
    from app.services.sentiment_analyzer import SentimentAnalyzer
    print("   ✓ app.services.sentiment_analyzer 导入成功")
    tests.append(("SentimentAnalyzer", True, None))
except Exception as e:
    print(f"   ❌ app.services.sentiment_analyzer 导入失败: {e}")
    tests.append(("SentimentAnalyzer", False, str(e)))

print()

# 4. 测试 RAG 引擎
print("4. 测试 RAG 引擎...")
try:
    from app.core.engines.rag_engine import RagEngine
    print("   ✓ app.core.engines.rag_engine 导入成功")
    tests.append(("RagEngine", True, None))
except Exception as e:
    print(f"   ❌ app.core.engines.rag_engine 导入失败: {e}")
    tests.append(("RagEngine", False, str(e)))

print()

# 5. 测试 LLM 客户端
print("5. 测试 LLM 客户端...")
try:
    from app.adapters.llm.deepseek_client import DeepseekClient
    print("   ✓ app.adapters.llm.deepseek_client 导入成功")
    tests.append(("DeepseekClient", True, None))
except Exception as e:
    print(f"   ❌ app.adapters.llm.deepseek_client 导入失败: {e}")
    tests.append(("DeepseekClient", False, str(e)))

print()

# 6. 测试工具模块
print("6. 测试工具模块...")
try:
    from app.core.orchestrator.tools import get_market_context, analyze_sentiment, search_reports
    print("   ✓ app.core.orchestrator.tools 导入成功")
    tests.append(("Tools", True, None))
except Exception as e:
    print(f"   ❌ app.core.orchestrator.tools 导入失败: {e}")
    tests.append(("Tools", False, str(e)))

print()

# 7. 测试缓存模块
print("7. 测试缓存模块...")
try:
    from app.core.utils.cache import get_cache
    print("   ✓ app.core.utils.cache 导入成功")
    tests.append(("Cache", True, None))
except Exception as e:
    print(f"   ❌ app.core.utils.cache 导入失败: {e}")
    tests.append(("Cache", False, str(e)))

print()

# 8. 测试环境变量加载
print("8. 测试环境变量加载...")
try:
    import os
    from dotenv import load_dotenv
    
    env_path = project_root / ".env"
    load_dotenv(env_path)
    
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_key:
        print(f"   ✓ DEEPSEEK_API_KEY 已加载")
        tests.append(("EnvVars", True, None))
    else:
        print(f"   ⚠ DEEPSEEK_API_KEY 未配置")
        tests.append(("EnvVars", False, "DEEPSEEK_API_KEY not set"))
except Exception as e:
    print(f"   ❌ 环境变量加载失败: {e}")
    tests.append(("EnvVars", False, str(e)))

print()

# 总结
print("=" * 60)
print("测试总结")
print("=" * 60)
print()

success_count = sum(1 for _, success, _ in tests if success)
total_count = len(tests)

print(f"通过: {success_count}/{total_count}")
print()

if success_count == total_count:
    print("✓ 所有模块导入成功！")
    print()
    print("现在可以启动 Streamlit 应用了：")
    print("  双击 start_streamlit.cmd")
else:
    print("⚠ 部分模块导入失败：")
    print()
    for name, success, error in tests:
        if not success:
            print(f"  - {name}: {error}")
    print()
    print("请检查上述错误并修复后再启动 Streamlit")

print()
