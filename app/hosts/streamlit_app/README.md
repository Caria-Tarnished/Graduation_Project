# Streamlit UI - 财经分析 Agent

财经分析 Agent 系统的 Web 界面（答辩演示版）。

## 功能

### 1. 💬 聊天页面（主页）
- 快讯情感分析：输入财经新闻，分析其对市场的影响
- 财报问答：询问财报相关问题，从财报中检索答案
- 显示工具调用追踪和分析详情

### 2. 📈 K 线图表页面
- 显示 K 线图（使用 Plotly）
- 在图表上标注重要事件
- 点击事件点查看情感分析结果
- 支持时间范围和星级筛选

### 3. 📄 财报检索页面
- 输入问题，从财报中检索相关内容
- 显示 Top-K 引用片段（包含页码和相似度分数）
- LLM 生成的答案总结
- 支持语言筛选和显示选项配置

## 启动方法

### 前置条件

1. 确保已安装所有依赖：
```powershell
pip install -r requirements.txt
```

2. 配置环境变量（创建 `.env` 文件）：
```env
DEEPSEEK_API_KEY=your_api_key_here
HF_HOME=F:\huggingface_cache
```

3. 确保以下文件/目录存在：
- `models/bert_3cls/best/` - BERT 模型权重
- `data/reports/chroma_db/` - Chroma 向量库
- `finance_analysis.db` - 数据库

### 启动命令

```powershell
streamlit run app/hosts/streamlit_app/app.py
```

启动后，浏览器会自动打开 `http://localhost:8501`。

## 目录结构

```
app/hosts/streamlit_app/
├── app.py                  # 主入口（聊天页面）
├── pages/
│   ├── 2_Charts.py         # K 线图表页面
│   └── 3_Reports.py        # 财报检索页面
└── README.md               # 本文件
```

## 使用说明

### 聊天页面

1. 在输入框中输入问题
2. 系统自动判断查询类型（快讯分析 vs 财报问答）
3. 查看分析结果和详细信息

**示例问题**：
- 快讯分析：`美联储宣布加息 25 个基点`
- 财报问答：`黄金市场 2023 年的表现如何？`

### K 线图表页面

1. 在侧边栏选择标的、时间范围和最低星级
2. 点击"加载数据"按钮
3. 查看 K 线图和事件标注
4. 在下拉框中选择事件查看详情

### 财报检索页面

1. 在搜索框中输入问题
2. 配置检索参数（Top-K、语言筛选等）
3. 点击"搜索"按钮
4. 查看 AI 总结和引用片段

## 系统状态检查

侧边栏会显示各个引擎的加载状态：
- ✓ 情感分析引擎：BERT 模型已加载
- ✓ RAG 检索引擎：Chroma 向量库已加载
- ✓ LLM 客户端：Deepseek API 已配置

如果某个引擎未加载，会显示警告信息。

## 常见问题

### 1. Agent 未初始化

**原因**：引擎加载失败

**解决方法**：
- 检查 BERT 模型路径是否正确
- 检查 Chroma 向量库是否存在
- 检查 DEEPSEEK_API_KEY 是否配置

### 2. 数据库不存在

**原因**：`finance_analysis.db` 文件不存在

**解决方法**：
- 运行 `scripts/build_finance_analysis.py` 构建数据库
- 确保数据库文件在项目根目录

### 3. 向量库未找到

**原因**：Chroma 向量库未构建

**解决方法**：
- 运行 `scripts/rag/build_vector_index.py` 构建向量库
- 确保 `data/reports/chroma_db/` 目录存在

### 4. LLM 调用失败

**原因**：Deepseek API Key 未配置或无效

**解决方法**：
- 检查 `.env` 文件中的 `DEEPSEEK_API_KEY`
- 确保 API Key 有效且有余额

## 性能优化

### 缓存机制

- Agent 初始化使用 `@st.cache_resource` 缓存，只执行一次
- 避免重复加载模型和向量库

### 超时控制

- LLM 调用默认超时 10 秒
- 可在 `app/core/dto.py` 的 `EngineConfig` 中调整

### 降级策略

- 如果某个引擎未加载，系统会使用降级策略
- 例如：无 LLM 时返回简单总结

## 答辩演示建议

### 演示流程

1. **开场**（1 分钟）
   - 介绍系统架构（双引擎 + Agent）
   - 展示主界面

2. **核心演示**（5 分钟）
   - **场景 1**：快讯情感分析
     - 输入：`美联储宣布加息 25 个基点`
     - 展示：情感分析结果 + 市场上下文 + 规则引擎输出
   
   - **场景 2**：K 线联动
     - 展示：K 线图 + 事件标注
     - 操作：点击事件点，触发情感分析
   
   - **场景 3**：财报检索
     - 输入：`黄金市场 2023 年的表现如何？`
     - 展示：Top-5 引用片段 + LLM 总结

3. **技术亮点**（2 分钟）
   - 代理标注：利用 K 线走势反向标注情感
   - 混合架构：ML 模型 + 规则引擎
   - 工具追踪：完整的 tool trace

4. **结尾**（1 分钟）
   - 总结成果：Test Macro F1 达到 0.37+
   - 未来展望：集成到 QuantSway 交易平台

### 备用方案

如果演示时网络故障或 API 失败：
- 提前录制演示视频
- 准备离线模式（不使用 LLM）
- 准备静态截图

## 技术栈

- **UI 框架**：Streamlit 1.30+
- **图表库**：Plotly 5.18+
- **ML 模型**：BERT（hfl/chinese-roberta-wwm-ext）
- **向量检索**：Chroma + bge-m3
- **LLM**：Deepseek API
- **数据库**：SQLite

## 开发规范

- 文件编码：UTF-8
- 代码注释：中文
- 避免使用 emoji（可能显示为"?"）
- 遵循 PEP8 代码风格

## 联系方式

如有问题，请参考：
1. 项目文档（`PLAN.md`、`Project_Status.md`、`REMAINING_TASKS.md`）
2. 代码注释（关键函数都有中文注释）
3. 提交 Issue 到 GitHub 仓库

---

**祝答辩顺利！** 🎓
