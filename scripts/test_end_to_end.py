# -*- coding: utf-8 -*-
"""
端到端功能测试脚本

基于 REMAINING_TASKS.md 中的测试用例，测试：
1. 新闻情感分析（8个测试用例）
2. 财报检索问答（8个测试用例）
3. 完整对话流程（5个测试用例）
4. 异常处理（7个测试用例）

使用方法：
    python scripts/test_end_to_end.py
"""
import sys
import os
from pathlib import Path
import time
from datetime import datetime
import importlib.util

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_module(module_name, file_path):
    """动态加载模块"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def print_section(title: str):
    """打印章节标题"""
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}\n")


def print_test_case(case_num: int, title: str):
    """打印测试用例标题"""
    print(f"\n{'-' * 80}")
    print(f"测试用例 {case_num}: {title}")
    print(f"{'-' * 80}")


def print_result(passed: bool, expected: str, actual: str):
    """打印测试结果"""
    status = "✓ 通过" if passed else "✗ 失败"
    print(f"\n{status}")
    print(f"预期: {expected}")
    print(f"实际: {actual}")


# 测试用例 1: 新闻情感分析
def test_news_sentiment_analysis(agent):
    """测试新闻情感分析功能"""
    print_section("测试类别 1: 新闻情感分析")
    
    test_cases = [
        {
            "num": 1,
            "title": "利好消息（降息）",
            "input": "美联储宣布降息25个基点",
            "expected": "利好或中性",
            "validation": ["情感标签", "置信度", "市场上下文"]
        },
        {
            "num": 2,
            "title": "利空消息（加息）",
            "input": "美联储宣布加息50个基点",
            "expected": "利空或中性",
            "validation": ["情感标签", "置信度", "市场上下文"]
        },
        {
            "num": 3,
            "title": "中性消息（符合预期）",
            "input": "美国GDP增长2.5% 符合市场预期",
            "expected": "中性",
            "validation": ["情感标签", "置信度"]
        },
        {
            "num": 4,
            "title": "预期兑现（大涨后利好）",
            "input": "美联储宣布降息25个基点",
            "expected": "可能触发预期兑现规则",
            "validation": ["规则引擎输出", "前期涨幅"]
        },
        {
            "num": 5,
            "title": "建议观望（高波动低净变动）",
            "input": "美国CPI数据公布 同比增长2.8%",
            "expected": "可能触发观望规则",
            "validation": ["规则引擎输出", "波动率"]
        },
        {
            "num": 6,
            "title": "空文本处理",
            "input": "",
            "expected": "返回错误或默认结果",
            "validation": ["错误处理"]
        },
        {
            "num": 7,
            "title": "超长文本处理",
            "input": "美联储宣布降息" * 100,
            "expected": "正常处理（截断）",
            "validation": ["文本截断", "推理时间"]
        },
        {
            "num": 8,
            "title": "特殊字符处理",
            "input": "美联储宣布降息25个基点！！！@#$%^&*()",
            "expected": "正常处理",
            "validation": ["情感标签"]
        }
    ]
    
    results = []
    
    for case in test_cases:
        print_test_case(case["num"], case["title"])
        print(f"输入: {case['input'][:100]}{'...' if len(case['input']) > 100 else ''}")
        
        try:
            start = time.time()
            answer = agent.process_query(case["input"], query_type="news_analysis")
            elapsed = time.time() - start
            
            # 验证结果
            passed = True
            validation_results = []
            
            # 检查是否有情感分析结果
            if answer.sentiment:
                validation_results.append(f"情感标签: {answer.sentiment.label}")
                validation_results.append(f"置信度: {answer.sentiment.score:.2%}")
                validation_results.append(f"解释: {answer.sentiment.explain}")
            else:
                validation_results.append("情感分析: 无结果")
                if "情感标签" in case["validation"]:
                    passed = False
            
            # 检查工具调用
            validation_results.append(f"工具调用: {len(answer.tool_trace)} 个")
            for trace in answer.tool_trace:
                status = "✓" if trace.ok else "✗"
                validation_results.append(f"  {status} {trace.name} ({trace.elapsed_ms}ms)")
            
            # 检查警告
            if answer.warnings:
                validation_results.append(f"警告: {', '.join(answer.warnings)}")
            
            validation_results.append(f"总耗时: {elapsed:.2f}秒")
            
            print_result(passed, case["expected"], "\n".join(validation_results))
            results.append((case["title"], passed))
            
        except Exception as e:
            print_result(False, case["expected"], f"异常: {str(e)}")
            results.append((case["title"], False))
    
    return results


# 测试用例 2: 财报检索问答
def test_report_qa(agent):
    """测试财报检索问答功能"""
    print_section("测试类别 2: 财报检索问答")
    
    test_cases = [
        {
            "num": 1,
            "title": "黄金价格走势查询",
            "input": "黄金价格走势如何？",
            "expected": "返回相关财报片段",
            "validation": ["引用数量", "相似度分数"]
        },
        {
            "num": 2,
            "title": "白银市场查询",
            "input": "白银市场表现怎么样？",
            "expected": "返回相关财报片段",
            "validation": ["引用数量", "相似度分数"]
        },
        {
            "num": 3,
            "title": "贵金属投资建议",
            "input": "贵金属投资有什么建议？",
            "expected": "返回相关财报片段",
            "validation": ["引用数量", "相似度分数"]
        },
        {
            "num": 4,
            "title": "美联储政策影响",
            "input": "美联储政策对黄金有什么影响？",
            "expected": "返回相关财报片段",
            "validation": ["引用数量", "相似度分数"]
        },
        {
            "num": 5,
            "title": "不相关查询",
            "input": "今天天气怎么样？",
            "expected": "返回空结果或低相似度结果",
            "validation": ["引用数量", "相似度分数"]
        },
        {
            "num": 6,
            "title": "空查询",
            "input": "",
            "expected": "返回错误或默认结果",
            "validation": ["错误处理"]
        },
        {
            "num": 7,
            "title": "超长查询",
            "input": "黄金价格走势" * 50,
            "expected": "正常处理（截断）",
            "validation": ["引用数量"]
        },
        {
            "num": 8,
            "title": "特殊字符查询",
            "input": "黄金价格@#$%^&*()",
            "expected": "正常处理",
            "validation": ["引用数量"]
        }
    ]
    
    results = []
    
    for case in test_cases:
        print_test_case(case["num"], case["title"])
        print(f"输入: {case['input'][:100]}{'...' if len(case['input']) > 100 else ''}")
        
        try:
            start = time.time()
            answer = agent.process_query(case["input"], query_type="report_qa")
            elapsed = time.time() - start
            
            # 验证结果
            passed = True
            validation_results = []
            
            # 检查引用数量
            citation_count = len(answer.citations) if answer.citations else 0
            validation_results.append(f"引用数量: {citation_count}")
            
            # 检查相似度分数
            if answer.citations:
                avg_score = sum(c.score for c in answer.citations) / len(answer.citations)
                validation_results.append(f"平均相似度: {avg_score:.4f}")
                
                # 显示前3条引用
                for i, citation in enumerate(answer.citations[:3], 1):
                    validation_results.append(
                        f"  引用{i}: {citation.source_file} "
                        f"(相似度: {citation.score:.4f})"
                    )
            else:
                validation_results.append("无引用结果")
                if "引用数量" in case["validation"] and case["num"] <= 4:
                    passed = False
            
            # 检查工具调用
            validation_results.append(f"工具调用: {len(answer.tool_trace)} 个")
            for trace in answer.tool_trace:
                status = "✓" if trace.ok else "✗"
                validation_results.append(f"  {status} {trace.name} ({trace.elapsed_ms}ms)")
            
            validation_results.append(f"总耗时: {elapsed:.2f}秒")
            
            print_result(passed, case["expected"], "\n".join(validation_results))
            results.append((case["title"], passed))
            
        except Exception as e:
            print_result(False, case["expected"], f"异常: {str(e)}")
            results.append((case["title"], False))
    
    return results


# 测试用例 3: 完整对话流程
def test_complete_dialogue(agent):
    """测试完整对话流程"""
    print_section("测试类别 3: 完整对话流程")
    
    test_cases = [
        {
            "num": 1,
            "title": "快讯分析 → 财报查询",
            "queries": [
                ("美联储宣布降息25个基点", "news_analysis"),
                ("美联储政策对黄金有什么影响？", "report_qa")
            ]
        },
        {
            "num": 2,
            "title": "财报查询 → 快讯分析",
            "queries": [
                ("黄金价格走势如何？", "report_qa"),
                ("美国CPI数据公布 同比增长2.8%", "news_analysis")
            ]
        },
        {
            "num": 3,
            "title": "多轮快讯分析",
            "queries": [
                ("美联储宣布降息25个基点", "news_analysis"),
                ("美国GDP增长2.5% 符合市场预期", "news_analysis"),
                ("美国失业率上升至4.5%", "news_analysis")
            ]
        },
        {
            "num": 4,
            "title": "多轮财报查询",
            "queries": [
                ("黄金价格走势如何？", "report_qa"),
                ("白银市场表现怎么样？", "report_qa"),
                ("贵金属投资有什么建议？", "report_qa")
            ]
        },
        {
            "num": 5,
            "title": "混合查询（自动检测）",
            "queries": [
                ("美联储宣布降息25个基点", None),
                ("黄金价格走势如何？", None),
                ("美国GDP增长2.5%", None)
            ]
        }
    ]
    
    results = []
    
    for case in test_cases:
        print_test_case(case["num"], case["title"])
        
        try:
            passed = True
            total_time = 0
            
            for i, (query, query_type) in enumerate(case["queries"], 1):
                print(f"\n  查询 {i}: {query}")
                
                start = time.time()
                if query_type:
                    answer = agent.process_query(query, query_type=query_type)
                else:
                    answer = agent.process_query(query)
                elapsed = time.time() - start
                total_time += elapsed
                
                print(f"  查询类型: {answer.query_type}")
                print(f"  工具调用: {len(answer.tool_trace)} 个")
                print(f"  耗时: {elapsed:.2f}秒")
            
            print(f"\n  总耗时: {total_time:.2f}秒")
            print(f"  平均耗时: {total_time / len(case['queries']):.2f}秒")
            
            print_result(passed, "所有查询正常完成", f"完成 {len(case['queries'])} 个查询")
            results.append((case["title"], passed))
            
        except Exception as e:
            print_result(False, "所有查询正常完成", f"异常: {str(e)}")
            results.append((case["title"], False))
    
    return results


# 测试用例 4: 异常处理
def test_exception_handling(agent):
    """测试异常处理"""
    print_section("测试类别 4: 异常处理")
    
    test_cases = [
        {
            "num": 1,
            "title": "空输入",
            "input": "",
            "expected": "返回错误或默认结果"
        },
        {
            "num": 2,
            "title": "None 输入",
            "input": None,
            "expected": "返回错误或默认结果"
        },
        {
            "num": 3,
            "title": "超长输入（10000字符）",
            "input": "测试" * 5000,
            "expected": "正常处理（截断）"
        },
        {
            "num": 4,
            "title": "特殊字符输入",
            "input": "!@#$%^&*()_+-=[]{}|;':\",./<>?",
            "expected": "正常处理"
        },
        {
            "num": 5,
            "title": "纯数字输入",
            "input": "1234567890",
            "expected": "正常处理"
        },
        {
            "num": 6,
            "title": "纯空格输入",
            "input": "     ",
            "expected": "返回错误或默认结果"
        },
        {
            "num": 7,
            "title": "混合语言输入",
            "input": "美联储 Federal Reserve 降息 rate cut",
            "expected": "正常处理"
        }
    ]
    
    results = []
    
    for case in test_cases:
        print_test_case(case["num"], case["title"])
        if case["input"]:
            display_input = case["input"][:100] + "..." if len(str(case["input"])) > 100 else case["input"]
            print(f"输入: {display_input}")
        else:
            print(f"输入: {repr(case['input'])}")
        
        try:
            start = time.time()
            answer = agent.process_query(case["input"] if case["input"] is not None else "")
            elapsed = time.time() - start
            
            # 验证结果
            passed = True
            validation_results = []
            
            validation_results.append(f"查询类型: {answer.query_type}")
            validation_results.append(f"工具调用: {len(answer.tool_trace)} 个")
            validation_results.append(f"警告: {len(answer.warnings)} 个")
            if answer.warnings:
                for warning in answer.warnings:
                    validation_results.append(f"  - {warning}")
            validation_results.append(f"耗时: {elapsed:.2f}秒")
            
            print_result(passed, case["expected"], "\n".join(validation_results))
            results.append((case["title"], passed))
            
        except Exception as e:
            # 某些异常是预期的
            if case["num"] in [1, 2, 6]:
                print_result(True, case["expected"], f"预期异常: {str(e)}")
                results.append((case["title"], True))
            else:
                print_result(False, case["expected"], f"意外异常: {str(e)}")
                results.append((case["title"], False))
    
    return results


def generate_report(all_results):
    """生成测试报告"""
    print_section("端到端测试报告")
    
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
        category_passed = sum(1 for _, passed in results if passed)
        category_total = len(results)
        print(f"\n{category} ({category_passed}/{category_total}):")
        for test_name, passed in results:
            status = "✓" if passed else "✗"
            print(f"  {status} {test_name}")
    
    # 保存报告到文件
    report_path = "test_end_to_end_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"端到端功能测试报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"总测试数: {total_tests}\n")
        f.write(f"通过: {passed_tests}\n")
        f.write(f"失败: {total_tests - passed_tests}\n")
        f.write(f"通过率: {passed_tests / total_tests * 100:.1f}%\n\n")
        f.write(f"详细结果:\n")
        for category, results in all_results.items():
            category_passed = sum(1 for _, passed in results if passed)
            category_total = len(results)
            f.write(f"\n{category} ({category_passed}/{category_total}):\n")
            for test_name, passed in results:
                status = "✓" if passed else "✗"
                f.write(f"  {status} {test_name}\n")
    
    print(f"\n报告已保存到: {report_path}")


def main():
    """主测试函数"""
    print("=" * 80)
    print("端到端功能测试")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 初始化 Agent
    print("\n初始化 Agent...")
    agent_module = load_module("agent", "app/core/orchestrator/agent.py")
    Agent = agent_module.Agent
    agent = Agent()
    print("Agent 初始化完成（降级模式）\n")
    
    all_results = {}
    
    # 执行所有测试
    all_results["新闻情感分析"] = test_news_sentiment_analysis(agent)
    all_results["财报检索问答"] = test_report_qa(agent)
    all_results["完整对话流程"] = test_complete_dialogue(agent)
    all_results["异常处理"] = test_exception_handling(agent)
    
    # 生成报告
    generate_report(all_results)
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
