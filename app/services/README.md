# 情感分析服务（Engine A）

## 概述

情感分析服务是本项目的核心组件之一，采用"BERT 3类分类 + 后处理规则引擎"的混合架构。

### 架构设计

- **ML 模型（BERT）**：专注于可学习的 3 类基础方向
  - Bearish（利空/看跌）
  - Neutral（中性）
  - Bullish（利好/看涨）

- **规则引擎**：处理复杂的金融逻辑
  - 预期兑现（Priced-in）
  - 建议观望（Watch）

### 优势

- **可解释性**：规则引擎的逻辑清晰透明
- **可维护性**：规则阈值可灵活调整
- **可调试性**：问题定位更容易
- **符合金融系统设计模式**：ML + 规则引擎的混合架构

## 快速开始

### 1. 准备模型权重

首先需要将训练好的模型权重复制到标准模型目录：

```powershell
# 方法1：使用复制工具（推荐）
python scripts/tools/copy_model_weights.py `
  --src reports/bert_3cls_enhanced_v1/best `
  --dst models/bert_3cls/best `
  --verbose

# 方法2：手动复制（PowerShell）
Copy-Item -Recurse -Force reports/bert_3cls_enhanced_v1/best models/bert_3cls/
```

### 2. 测试情感分析器

运行测试脚本验证功能：

```powershell
python scripts/test_sentiment_analyzer.py
```

测试脚本包含 6 个测试案例：
1. 普通利好消息（横盘后降息）
2. 利好预期兑现（大涨后降息）
3. 利空预期兑现（大跌后加息）
4. 建议观望（高波动低净变动）
5. 普通利空消息（横盘后负面数据）
6. 中性消息（横盘后中性数据）

### 3. 在代码中使用

```python
from app.services.sentiment_analyzer import SentimentAnalyzer

# 创建分析器
analyzer = SentimentAnalyzer(model_path="models/bert_3cls/best")

# 分析新闻
result = analyzer.analyze(
    text="美联储宣布降息25个基点",
    pre_ret=0.015,  # 前120分钟涨幅1.5%
    range_ratio=0.008  # 波动率0.8%
)

print(f"基础情感: {result['base_sentiment']}")
print(f"最终情感: {result['final_sentiment']}")
print(f"解释: {result['explanation']}")
print(f"建议: {result['recommendation']}")
```

## 规则引擎详解

### 规则1：预期兑现（Priced-in）

**触发条件**：
- 利好预期兑现：`BERT输出=利好 AND 前120分钟涨幅 > 1%`
- 利空预期兑现：`BERT输出=利空 AND 前120分钟跌幅 > 1%`

**逻辑说明**：
当市场在消息发布前已经大幅上涨/下跌，说明市场可能已经提前反应了预期。此时即使消息本身是利好/利空，也可能出现"利好出尽"或"利空出尽"的情况。

**示例**：
```
输入：
  text: "美联储宣布降息25个基点"
  pre_ret: 0.015 (前期涨1.5%)

输出：
  base_sentiment: "bullish"
  final_sentiment: "bullish_priced_in"
  explanation: "虽然消息利好，但前期已大涨1.50%，可能是利好预期兑现"
  recommendation: "建议谨慎追多"
```

### 规则2：建议观望（Watch）

**触发条件**：
- `高波动(range_ratio > 1.5%) AND 低净变动(abs_ret < 0.2%)`

**逻辑说明**：
当市场波动率很高但净变动很小时，说明多空分歧较大，方向不明确，此时建议观望等待方向明确。

**示例**：
```
输入：
  text: "美国CPI数据公布，符合预期"
  pre_ret: 0.001 (前期几乎无变动)
  range_ratio: 0.020 (高波动2.0%)

输出：
  base_sentiment: "neutral"
  final_sentiment: "watch"
  explanation: "市场波动率较高（2.00%），但净变动较小（0.10%），多空分歧较大"
  recommendation: "建议观望，等待方向明确"
```

### 规则优先级

规则按以下优先级应用：
1. **建议观望**（最高优先级）
2. **预期兑现**
3. **基础情感**（BERT 输出）

## 阈值调整

规则引擎的阈值可以根据实际情况调整。在 `sentiment_analyzer.py` 中修改类常量：

```python
class SentimentAnalyzer:
    # 规则引擎阈值（可根据实际情况调整）
    PRICED_IN_THRESHOLD = 0.01  # 预期兑现阈值（1%）
    HIGH_VOLATILITY_THRESHOLD = 0.015  # 高波动阈值（1.5%）
    LOW_NET_CHANGE_THRESHOLD = 0.002  # 低净变动阈值（0.2%）
```

### 调整建议

- **预期兑现阈值**：
  - 提高（如 1.5%）：更严格，减少误判
  - 降低（如 0.5%）：更宽松，捕获更多案例

- **高波动阈值**：
  - 提高（如 2.0%）：只在极端波动时触发
  - 降低（如 1.0%）：更容易触发观望建议

- **低净变动阈值**：
  - 提高（如 0.5%）：允许更大的净变动
  - 降低（如 0.1%）：要求更小的净变动

## 输入增强

情感分析器会自动为输入文本添加市场上下文前缀：

| 市场状态 | 判定条件 | 前缀 |
|:---------|:---------|:-----|
| 强势上涨 | `pre_ret > 1.0%` | `[Strong Rally]` |
| 急剧下跌 | `pre_ret < -1.0%` | `[Sharp Decline]` |
| 温和上涨 | `0.3% < pre_ret <= 1.0%` | `[Mild Rally]` |
| 弱势下跌 | `-1.0% <= pre_ret < -0.3%` | `[Weak Decline]` |
| 高波动 | `abs(pre_ret) < 0.3%` AND `range_ratio > 1.5%` | `[High Volatility]` |
| 横盘震荡 | 其他情况 | `[Sideways]` |

这些前缀帮助 BERT 模型理解当前的市场环境。

## API 参考

### SentimentAnalyzer

#### `__init__(model_path, device=None, max_length=384)`

初始化情感分析器。

**参数**：
- `model_path` (str): BERT 模型路径
- `device` (str, optional): 设备（"cpu" 或 "cuda"），默认自动检测
- `max_length` (int): 最大序列长度，默认 384

#### `analyze(text, pre_ret=0.0, range_ratio=0.0, actual=None, consensus=None)`

分析新闻情感。

**参数**：
- `text` (str): 新闻文本
- `pre_ret` (float): 前120分钟收益率，默认 0
- `range_ratio` (float): 波动率，默认 0
- `actual` (float, optional): 实际值（宏观数据）
- `consensus` (float, optional): 预期值（宏观数据）

**返回**：
```python
{
    "base_sentiment": "bearish/neutral/bullish",
    "base_confidence": 0.65,
    "final_sentiment": "bearish/neutral/bullish/bullish_priced_in/bearish_priced_in/watch",
    "explanation": "解释文本",
    "recommendation": "建议文本",
    "market_context": {
        "pre_ret": 0.015,
        "range_ratio": 0.008,
        "enhanced_text": "[Strong Rally] 美联储宣布降息..."
    }
}
```

### 便捷函数

#### `get_analyzer(model_path="models/bert_3cls/best", force_reload=False)`

获取全局情感分析器实例（单例模式）。

**参数**：
- `model_path` (str): 模型路径
- `force_reload` (bool): 是否强制重新加载模型

**返回**：
- `SentimentAnalyzer` 实例

**示例**：
```python
from app.services.sentiment_analyzer import get_analyzer

# 第一次调用会加载模型
analyzer = get_analyzer()

# 后续调用返回同一实例（不重新加载）
analyzer = get_analyzer()
```

## 性能优化

### CPU 推理

在 CPU 上推理速度约为 50-100 样本/秒（取决于 CPU 性能）。

### GPU 推理

如果有 GPU，模型会自动使用 GPU 加速，速度可提升 5-10 倍。

### 批量推理

如需批量处理大量新闻，可以修改代码支持批量推理：

```python
# TODO: 实现批量推理接口
# analyzer.analyze_batch(texts, pre_rets, range_ratios)
```

## 故障排查

### 问题1：模型路径不存在

**错误信息**：
```
FileNotFoundError: 模型路径不存在: models/bert_3cls/best
```

**解决方法**：
1. 确保已在 Colab 上完成训练
2. 使用 `sync_results.py` 同步结果到本地
3. 使用 `copy_model_weights.py` 复制模型权重

### 问题2：标签映射错误

**症状**：预测结果与预期不符

**解决方法**：
检查 `_predict_base_sentiment` 方法中的标签映射是否正确：
```python
# 将模型输出（0/1/2）映射回标签（-1/0/1）
label_mapping = {0: -1, 1: 0, 2: 1}
```

### 问题3：内存不足

**症状**：加载模型时内存溢出

**解决方法**：
- 使用更小的 `max_length`（如 256）
- 关闭其他占用内存的程序
- 考虑使用模型量化（TODO）

## 下一步

- [ ] 实现批量推理接口
- [ ] 添加模型量化支持（减少内存占用）
- [ ] 实现缓存机制（避免重复推理）
- [ ] 添加更多规则（如"超预期"、"不及预期"）
- [ ] 集成到 Agent 系统（Engine B 调用）
- [ ] 添加 API 接口（FastAPI/Flask）
