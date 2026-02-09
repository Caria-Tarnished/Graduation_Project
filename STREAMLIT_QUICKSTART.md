# Streamlit UI 快速启动指南

**更新时间**: 2026-02-09  
**目标**: 快速启动 Streamlit UI 进行答辩演示

---

## 1. 前置条件检查

### 1.1 必需文件

在启动前，请确认以下文件存在：

```
Graduation_Project/
├── models/bert_3cls/best/          # BERT 模型权重（必需）
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   └── vocab.txt
│
├── data/reports/chroma_db/         # Chroma 向量库（必需）
│   └── chroma.sqlite3
│
├── finance_analysis.db             # 数据库（必需）
│
└── .env                            # 环境变量（可选）
    └── DEEPSEEK_API_KEY=xxx        # Deepseek API Key（可选）
```

### 1.2 依赖安装

确保已安装所有依赖：

```powershell
pip install -r requirements.txt
```

关键依赖：
- `streamlit>=1.30.0`
- `plotly>=5.18.0`
- `torch`
- `transformers`
- `chromadb>=0.4.0`
- `sentence-transformers>=2.2.0`

---

## 2. 启动步骤

### 2.1 基础启动（无 LLM）

如果没有配置 Deepseek API Key，系统仍可运行（降级模式）：

```powershell
# 进入项目根目录
cd E:\Projects\Graduation_Project

# 启动 Streamlit
streamlit run app/hosts/streamlit_app/app.py
```

**降级模式说明**：
- ? 情感分析功能正常（BERT 本地推理）
- ? 财报检索功能正常（Chroma 向量检索）
- ?? LLM 总结功能不可用（返回默认文本）

### 2.2 完整启动（带 LLM）

如果已配置 Deepseek API Key：

```powershell
# 1. 创建 .env 文件（如果不存在）
echo DEEPSEEK_API_KEY=your_api_key_here > .env

# 2. 启动 Streamlit
streamlit run app/hosts/streamlit_app/app.py
```

**完整模式说明**：
- ? 所有功能正常
- ? LLM 总结功能可用

---

## 3. 访问界面

启动成功后，浏览器会自动打开：

```
http://localhost:8501
```

如果没有自动打开，请手动访问上述地址。

---

## 4. 功能测试

### 4.1 聊天页面（主页）

**测试快讯分析**：
```
输入: 美联储宣布加息 25 个基点
预期: 显示情感分析结果 + 市场上下文 + 规则引擎输出
```

**测试财报问答**：
```
输入: 黄金市场 2023 年的表现如何？
预期: 显示 RAG 检索结果 + LLM 总结
```

### 4.2 K 线图表页面

1. 在侧边栏选择标的（如 XAUUSD）
2. 选择时间范围（如最近 7 天）
3. 设置最低星级（如 3 星）
4. 点击"加载数据"
5. 查看 K 线图 + 事件标注
6. 选择事件查看详情

### 4.3 财报检索页面

1. 输入问题（如"黄金市场 2023 年的表现如何？"）
2. 点击"搜索"
3. 查看检索结果（Top-K 引用片段）
4. 查看 LLM 总结（如果已配置 API Key）

---

## 5. 常见问题

### 5.1 启动失败

**问题**: `ModuleNotFoundError: No module named 'streamlit'`

**解决**:
```powershell
pip install streamlit
```

---

### 5.2 BERT 模型未找到

**问题**: 侧边栏显示"? 情感分析引擎未加载"

**解决**:
1. 检查 `models/bert_3cls/best/` 目录是否存在
2. 如果不存在，需要先完成 BERT 训练（参考 `colab_3cls_training_cells.txt`）
3. 或者使用 Baseline 模型（TF-IDF + SVM）

---

### 5.3 Chroma 向量库未找到

**问题**: 侧边栏显示"? RAG 检索引擎未加载"

**解决**:
1. 检查 `data/reports/chroma_db/` 目录是否存在
2. 如果不存在，需要先运行 RAG 构建脚本：
   ```powershell
   python scripts/rag/build_chunks.py
   python scripts/rag/build_vector_index.py
   ```

---

### 5.4 Deepseek API Key 未配置

**问题**: 侧边栏显示"? LLM 客户端未配置"

**解决**:
1. 创建 `.env` 文件：
   ```
   DEEPSEEK_API_KEY=your_api_key_here
   ```
2. 或者在系统环境变量中设置 `DEEPSEEK_API_KEY`

**注意**: 没有 API Key 也可以运行，只是 LLM 总结功能不可用。

---

### 5.5 数据库未找到

**问题**: 无法加载价格数据或事件数据

**解决**:
1. 检查 `finance_analysis.db` 是否存在
2. 如果不存在，需要先运行数据构建脚本：
   ```powershell
   python scripts/build_finance_analysis.py
   ```

---

## 6. 性能优化

### 6.1 首次加载慢

**原因**: 需要加载 BERT 模型（约 400MB）和嵌入模型（约 2.27GB）

**优化**:
- 使用 `@st.cache_resource` 缓存模型（已实现）
- 首次加载后，后续访问会很快

### 6.2 推理速度慢

**原因**: CPU 推理速度较慢

**优化**:
- 使用批处理（如果有多条文本）
- 减少 `max_length`（当前 384，可降至 256）
- 考虑使用量化模型（INT8）

---

## 7. 答辩演示建议

### 7.1 演示流程（5-8 分钟）

1. **开场**（1 分钟）
   - 介绍系统架构（双引擎 + Agent）
   - 展示侧边栏的系统状态

2. **快讯分析演示**（2 分钟）
   - 输入典型快讯（如"美联储加息"）
   - 展示情感分析结果
   - 展示市场上下文和规则引擎输出
   - 展示工具调用追踪

3. **K 线联动演示**（2 分钟）
   - 切换到 K 线图表页面
   - 展示事件标注
   - 点击事件触发分析

4. **财报检索演示**（2 分钟）
   - 切换到财报检索页面
   - 输入问题
   - 展示检索结果和引用
   - 展示 LLM 总结

5. **技术亮点总结**（1 分钟）
   - 代理标注方法
   - 混合架构（ML + 规则）
   - 工具追踪和可解释性

### 7.2 备用方案

**如果网络故障**：
- 提前录制演示视频
- 准备截图和 PPT

**如果 API 失败**：
- 降级模式仍可演示（无 LLM 总结）
- 强调系统的降级策略设计

---

## 8. 关闭系统

在终端按 `Ctrl+C` 停止 Streamlit 服务。

---

## 9. 下一步

完成 UI 测试后，可以：

1. **优化性能**：调整模型参数、添加缓存
2. **准备答辩**：制作 PPT、录制演示视频
3. **扩展功能**：添加更多标的、更多财报来源

---

**祝答辩顺利！** ?
