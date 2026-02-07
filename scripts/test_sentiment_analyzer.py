# -*- coding: utf-8 -*-
"""
测试情感分析器 Engine A

用途:
1. 验证 BERT 模型加载是否正常
2. 测试规则引擎逻辑是否正确
3. 评估不同场景下的分析结果

使用方法:
    python scripts/test_sentiment_analyzer.py
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.sentiment_analyzer import SentimentAnalyzer


def print_result(title: str, result: dict):
    """打印分析结果"""
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")
    print(f"基础情感: {result['base_sentiment']} (置信度: {result['base_confidence']:.2%})")
    print(f"最终情感: {result['final_sentiment']}")
    print(f"解释: {result['explanation']}")
    if result['recommendation']:
        print(f"建议: {result['recommendation']}")
    print(f"\n市场上下文:")
    print(f"  前期收益率: {result['market_context']['pre_ret']:.2%}")
    print(f"  波动率: {result['market_context']['range_ratio']:.2%}")
    print(f"  增强文本: {result['market_context']['enhanced_text'][:100]}...")


def main():
    """主测试函数"""
    print("=" * 80)
    print("情感分析器测试")
    print("=" * 80)
    
    # 检查模型路径
    model_path = "models/bert_3cls/best"
    if not Path(model_path).exists():
        print(f"\n错误: 模型路径不存在: {model_path}")
        print("\n请先完成以下步骤:")
        print("1. 确保已在 Colab 上完成 BERT 3类模型训练")
        print("2. 使用 sync_results.py 同步训练结果到本地 reports/")
        print("3. 将 reports/bert_3cls_enhanced_v1/best/ 复制到 models/bert_3cls/best/")
        print("\n复制命令示例 PowerShell:")
        print("  python scripts/tools/copy_model_weights.py --src reports/bert_3cls_enhanced_v1/best --dst models/bert_3cls/best --verbose")
        return 1
    
    # 创建分析器
    try:
        print(f"\n正在加载模型: {model_path}")
        analyzer = SentimentAnalyzer(model_path=model_path)
        print("模型加载成功!\n")
    except Exception as e:
        print(f"\n模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 测试案例
    test_cases = [
        {
            "title": "案例1: 普通利好消息 横盘后降息",
            "text": "美联储宣布降息25个基点 符合市场预期",
            "pre_ret": 0.002,
            "range_ratio": 0.005
        },
        {
            "title": "案例2: 利好预期兑现 大涨后降息",
            "text": "美联储宣布降息25个基点 符合市场预期",
            "pre_ret": 0.015,
            "range_ratio": 0.008
        },
        {
            "title": "案例3: 利空预期兑现 大跌后加息",
            "text": "美联储宣布加息50个基点 超出市场预期",
            "pre_ret": -0.018,
            "range_ratio": 0.012
        },
        {
            "title": "案例4: 建议观望 高波动低净变动",
            "text": "美国CPI数据公布 同比增长2.8% 符合预期",
            "pre_ret": 0.001,
            "range_ratio": 0.020
        },
        {
            "title": "案例5: 普通利空消息 横盘后负面数据",
            "text": "美国失业率上升至4.5% 高于预期的4.2%",
            "pre_ret": -0.001,
            "range_ratio": 0.006
        },
        {
            "title": "案例6: 中性消息 横盘后中性数据",
            "text": "美国GDP增长2.5% 符合市场预期",
            "pre_ret": 0.0005,
            "range_ratio": 0.004
        }
    ]
    
    # 执行测试
    for case in test_cases:
        try:
            result = analyzer.analyze(
                text=case["text"],
                pre_ret=case["pre_ret"],
                range_ratio=case["range_ratio"]
            )
            print_result(case["title"], result)
        except Exception as e:
            print(f"\n测试失败: {case['title']}")
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 总结
    print(f"\n{'=' * 80}")
    print("测试完成")
    print(f"{'=' * 80}")
    print("\n规则引擎阈值设置:")
    print(f"  预期兑现阈值: {analyzer.PRICED_IN_THRESHOLD:.2%}")
    print(f"  高波动阈值: {analyzer.HIGH_VOLATILITY_THRESHOLD:.2%}")
    print(f"  低净变动阈值: {analyzer.LOW_NET_CHANGE_THRESHOLD:.2%}")
    print("\n如需调整阈值 请修改 app/services/sentiment_analyzer.py 中的类常量")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
