# 毕业论文 Mermaid 逻辑结构图集

在此文件中包含了您论文第四章和第五章所需的核心系统逻辑图与架构图。

> **渲染说明**：建议复制到 [mermaid.live](https://mermaid.live) 选择 v9+ 版本渲染后导出高清 SVG/PNG。Typora 用户请升级到最新版本以确保兼容性。

---

## 1. 总体系统微服务架构图 (第四章 4.1)

```mermaid
graph TD
    classDef ui fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef agent fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef engine fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef data fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px;

    subgraph HOST["宿主层 Host Layer"]
        UI["Streamlit Web UI"]:::ui
    end

    subgraph APP["用例层 Application Layer"]
        App1["快讯情感分析用例"]:::ui
        App2["财报智能问答用例"]:::ui
    end

    subgraph CORE["核心层 Core Layer"]
        Agent["Agent 大脑编排器"]:::agent
        Rule["规则引擎"]:::engine
        EngineA["Engine A - BERT 文本分类"]:::engine
        EngineB["Engine B - RAG 混合检索"]:::engine
    end

    subgraph ADAPTER["适配器层 Adapters Layer"]
        API["DeepSeek LLM API"]:::data
        SQLite[("SQLite 关系型数据库")]:::data
        Chroma[("ChromaDB 向量库")]:::data
    end

    UI --> App1
    UI --> App2
    App1 --> Agent
    App2 --> Agent

    Agent -->|工具调用| EngineA
    Agent -->|工具调用| EngineB
    Agent -->|市场上下文| SQLite
    Agent <-->|语义推理| API

    EngineA -->|后处理修正| Rule
    EngineA -->|获取近期K线| SQLite
    EngineB -->|向量检索| Chroma
```

---

## 2. Engine A 快讯情感分析算法流与代理标注 (第四章 4.2)

```mermaid
flowchart TD
    subgraph TRAIN["训练阶段 - 走势代理标注"]
        Raw["原始财经快讯"] --> Align["对应事件发生时间 T0"]
        Align --> Window["提取 T0 到 T0+15min 价格窗口"]
        Window --> Calc["计算窗口区间收益率"]
        Calc -->|高于70%分位| L1["利多 +1"]
        Calc -->|低于30%分位| L2["利空 -1"]
        Calc -->|居中| L3["中立 0"]
    end

    subgraph INFER["推理阶段 - 时序特征输入增强"]
        Input["新快讯输入"] --> Context["查询前120分钟市场上下文"]
        Context --> Augment["生成前缀特征 如 Sharp Decline"]
        Input -.-> Concat["拼接: 前缀 + 快讯文本"]
        Augment -.-> Concat

        Concat --> BERT["BERT 文本分类器"]
        BERT --> InitPred["初步情感标签"]

        InitPred --> RuleEngine{"规则引擎检查"}
        RuleEngine -->|前期大涨且当前利好| R1["预期兑现 输出中立或利空"]
        RuleEngine -->|波动率极高且涨幅低| R2["建议观望 输出中立"]
        RuleEngine -->|无冲突| R3["保持原标签"]

        R1 --> Final["最终情感标签输出"]
        R2 --> Final
        R3 --> Final
    end
```

---

## 3. Engine B 财报 RAG 清洗流水线图 (第四章 4.3)

```mermaid
flowchart LR
    PDF["券商研报 PDF"] --> PDFPlumber["PDFPlumber 解析"]

    subgraph PIPELINE["数据清洗与治理流水线"]
        PDFPlumber --> ROI["ROI 裁剪 - 去除上下10%页眉页脚"]
        ROI --> TableDetect{"智能表格处理"}
        TableDetect -->|简单表格| Markdown["转换为 Markdown 格式"]
        TableDetect -->|复杂表格| Desc["生成表格行列描述"]
        ROI --> Regex["正则清洗 - 免责声明和联系方式"]
    end

    Markdown --> Merge["图文表拼接"]
    Desc --> Merge
    Regex --> Merge

    Merge --> LangDetect["语言检测和机构提取"]
    LangDetect --> Splitter["文本切片 chunk=500 overlap=50"]
    Splitter --> Meta["附加元数据: 时间/页码/语言/机构"]
    Meta --> BGE["BAAI/bge-m3 向量化"]
    BGE --> Chroma[("ChromaDB 向量库")]
```

---

## 4. 基于规则路由的 Agent 编排设计图 (第四章 4.4)

```mermaid
sequenceDiagram
    participant User as "用户"
    participant UI as "Streamlit界面"
    participant Agent as "Agent编排中枢"
    participant EngineA as "Engine A"
    participant EngineB as "Engine B"
    participant LLM as "DeepSeek LLM"

    User->>UI: 提交查询问题
    UI->>Agent: 转发查询请求

    Agent->>LLM: 意图识别与工具规划
    LLM-->>Agent: 返回工具调用计划

    par 引擎A分析
        Agent->>EngineA: 调用快讯情感分析
        EngineA-->>Agent: 返回情感标签与市场上下文
    and 引擎B检索
        Agent->>EngineB: 调用RAG检索接口
        EngineB-->>Agent: 返回Top-K研报片段
    end

    Agent->>LLM: 注入工具结果作为上下文
    LLM-->>Agent: 生成融合分析答案
    Agent-->>UI: 返回答案与工具追踪日志
    UI-->>User: 可视化展示结果
```

---

## 5. 核心数据库 E-R 图 (第五章 5.1)

```mermaid
erDiagram
    prices_m1 {
        string ticker PK "商品代码 如 XAUUSD"
        string ts PK "时间戳"
        float open "开盘价"
        float high "最高价"
        float low "最低价"
        float close "收盘价"
        int volume "交易量"
    }

    events {
        string id PK "快讯唯一ID"
        string ts "事件发布时间"
        string source "数据来源 flash或calendar"
        string title "标题"
        string content "快讯正文"
        int star "重要程度星级"
        string country "相关国家"
        string affect "影响商品类别"
    }

    event_impacts {
        string event_id FK "关联事件ID"
        string window PK "评估窗口 如 m15"
        float price_event "事件发生时价格"
        float price_future "事件结束后价格"
        float ret "区间收益率"
    }

    events ||--o{ event_impacts : "拥有"
    prices_m1 }o..o{ events : "时间对齐联动"
```

---
**使用说明**：将各代码块复制到 [mermaid.live](https://mermaid.live) 即可在线渲染并导出高清图片。建议选择 Mermaid **v9 或以上版本**，以避免旧版语法限制。
