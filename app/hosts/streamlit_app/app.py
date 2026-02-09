# -*- coding: utf-8 -*-
"""
Streamlit 主入口

财经分析 Agent 系统 - 答辩演示版

功能：
- 聊天页面：快讯分析和财报问答
- K 线图表页面：可视化 + 事件标注
- 财报检索页面：RAG 检索展示

启动命令：
    streamlit run app/hosts/streamlit_app/app.py
"""
import streamlit as st
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


# 页面配置
st.set_page_config(
    page_title="财经分析 Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """主函数"""
    
    # 侧边栏
    with st.sidebar:
        st.title("📊 财经分析 Agent")
        st.markdown("---")
        
        st.markdown("""
        ### 功能导航
        
        - **💬 聊天**: 快讯分析和财报问答
        - **📈 K 线图表**: 可视化 + 事件标注
        - **📄 财报检索**: RAG 检索展示
        
        ### 系统状态
        """)
        
        # 检查引擎状态
        engine_status = check_engine_status()
        
        if engine_status['sentiment_engine']:
            st.success("✓ 情感分析引擎")
        else:
            st.warning("⚠ 情感分析引擎未加载")
        
        if engine_status['rag_engine']:
            st.success("✓ RAG 检索引擎")
        else:
            st.warning("⚠ RAG 检索引擎未加载")
        
        if engine_status['llm_client']:
            st.success("✓ LLM 客户端")
        else:
            st.warning("⚠ LLM 客户端未配置")
        
        st.markdown("---")
        st.markdown("""
        ### 关于
        
        **财经分析 Agent 系统**
        
        基于混合 NLP 模型的财经分析系统，采用双引擎架构：
        - Engine A: 情感分类（BERT + 规则引擎）
        - Engine B: RAG 检索（财报问答）
        
        **技术栈**
        - BERT: 情感分类
        - Chroma: 向量检索
        - Deepseek: LLM 总结
        - Streamlit: UI 界面
        """)
    
    # 主页面
    st.title("💬 财经分析 Agent - 聊天界面")
    
    st.markdown("""
    欢迎使用财经分析 Agent！我可以帮助您：
    
    1. **分析财经快讯**：输入新闻内容，我会分析其对市场的影响
    2. **回答财报问题**：询问财报相关问题，我会从财报中检索答案
    
    请在下方输入您的问题：
    """)
    
    # 初始化 Agent（使用缓存）
    agent = initialize_agent()
    
    # 聊天界面
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 显示历史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 用户输入
    if prompt := st.chat_input("请输入您的问题..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 生成回复
        with st.chat_message("assistant"):
            with st.spinner("正在分析..."):
                response = process_user_query(prompt, agent)
                st.markdown(response["summary"])
                
                # 显示详细信息
                with st.expander("📊 分析详情"):
                    if response.get("sentiment"):
                        st.markdown(f"**情感分析**: {response['sentiment']}")
                    
                    if response.get("citations"):
                        st.markdown(f"**引用数量**: {len(response['citations'])} 条")
                    
                    if response.get("tool_trace"):
                        st.markdown("**工具调用追踪**:")
                        for trace in response["tool_trace"]:
                            status = "✓" if trace["ok"] else "✗"
                            st.text(f"{status} {trace['name']} ({trace['elapsed_ms']}ms)")
        
        # 添加助手消息
        st.session_state.messages.append({"role": "assistant", "content": response["summary"]})


@st.cache_resource
def initialize_agent():
    """
    初始化 Agent 系统（使用缓存，只执行一次）
    
    Returns:
        Agent 实例或 None
    """
    try:
        import os
        
        # 动态导入模块（避免路径问题）
        import importlib.util
        
        # 加载 Agent 模块
        agent_module_path = project_root / "app" / "core" / "orchestrator" / "agent.py"
        spec = importlib.util.spec_from_file_location("agent", agent_module_path)
        agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_module)
        Agent = agent_module.Agent
        
        # 尝试加载所有引擎
        sentiment_engine = None
        rag_engine = None
        rule_engine = None
        llm_client = None
        
        # 1. 加载情感分析引擎（Engine A）
        try:
            sentiment_module_path = project_root / "app" / "services" / "sentiment_analyzer.py"
            spec = importlib.util.spec_from_file_location("sentiment_analyzer", sentiment_module_path)
            sentiment_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sentiment_module)
            SentimentAnalyzer = sentiment_module.SentimentAnalyzer
            
            bert_path = project_root / "models" / "bert_3cls" / "best"
            if bert_path.exists():
                sentiment_engine = SentimentAnalyzer(model_path=str(bert_path))
                st.success("✓ 情感分析引擎加载成功")
            else:
                st.warning(f"⚠ BERT 模型未找到: {bert_path}")
        except Exception as e:
            st.warning(f"⚠ 情感分析引擎加载失败: {e}")
        
        # 2. 加载 RAG 引擎（Engine B）
        try:
            rag_module_path = project_root / "app" / "core" / "engines" / "rag_engine.py"
            spec = importlib.util.spec_from_file_location("rag_engine", rag_module_path)
            rag_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(rag_module)
            RagEngine = rag_module.RagEngine
            
            chroma_path = project_root / "data" / "reports" / "chroma_db"
            if chroma_path.exists():
                rag_engine = RagEngine(
                    chroma_path=str(chroma_path),
                    model_name="BAAI/bge-m3"
                )
                st.success("✓ RAG 检索引擎加载成功")
            else:
                st.warning(f"⚠ Chroma 向量库未找到: {chroma_path}")
        except Exception as e:
            st.warning(f"⚠ RAG 引擎加载失败: {e}")
        
        # 3. 加载规则引擎（已集成在 SentimentAnalyzer 中）
        # 规则引擎不需要单独加载
        
        # 4. 加载 LLM 客户端
        try:
            llm_module_path = project_root / "app" / "adapters" / "llm" / "deepseek_client.py"
            spec = importlib.util.spec_from_file_location("deepseek_client", llm_module_path)
            llm_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(llm_module)
            DeepseekClient = llm_module.DeepseekClient
            
            if os.getenv("DEEPSEEK_API_KEY"):
                llm_client = DeepseekClient()
                st.success("✓ LLM 客户端初始化成功")
            else:
                st.warning("⚠ DEEPSEEK_API_KEY 未配置")
        except Exception as e:
            st.warning(f"⚠ LLM 客户端初始化失败: {e}")
        
        # 5. 创建 Agent
        db_path = project_root / "finance_analysis.db"
        agent = Agent(
            sentiment_engine=sentiment_engine,
            rag_engine=rag_engine,
            rule_engine=None,  # 规则引擎已集成在 sentiment_engine 中
            llm_client=llm_client,
            db_path=str(db_path)
        )
        
        return agent
    
    except Exception as e:
        st.error(f"初始化 Agent 失败: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None


def check_engine_status():
    """
    检查引擎状态
    
    Returns:
        状态字典
    """
    import os
    
    status = {
        "sentiment_engine": False,
        "rag_engine": False,
        "llm_client": False
    }
    
    # 检查 BERT 模型
    bert_path = project_root / "models" / "bert_3cls" / "best"
    if bert_path.exists():
        status["sentiment_engine"] = True
    
    # 检查 Chroma 向量库
    chroma_path = project_root / "data" / "reports" / "chroma_db"
    if chroma_path.exists():
        status["rag_engine"] = True
    
    # 检查 Deepseek API Key
    if os.getenv("DEEPSEEK_API_KEY"):
        status["llm_client"] = True
    
    return status


def process_user_query(query: str, agent):
    """
    处理用户查询
    
    Args:
        query: 用户查询
        agent: Agent 实例
    
    Returns:
        响应字典
    """
    if agent is None:
        return {
            "summary": "抱歉，Agent 未初始化，无法处理您的请求。",
            "sentiment": None,
            "citations": [],
            "tool_trace": []
        }
    
    try:
        # 调用 Agent 处理
        answer = agent.process_query(query)
        
        # 转换为字典
        response = {
            "summary": answer.summary,
            "sentiment": None,
            "citations": [],
            "tool_trace": []
        }
        
        # 添加情感分析结果
        if answer.sentiment:
            from app.core.dto import sentiment_label_to_text
            label_text = sentiment_label_to_text(answer.sentiment.label)
            response["sentiment"] = f"{label_text}（置信度 {answer.sentiment.score:.2%}）"
        
        # 添加引用
        if answer.citations:
            response["citations"] = [
                {
                    "text": c.text[:100] + "...",
                    "source": c.source_file,
                    "score": c.score
                }
                for c in answer.citations
            ]
        
        # 添加工具追踪
        if answer.tool_trace:
            response["tool_trace"] = [
                {
                    "name": t.name,
                    "elapsed_ms": t.elapsed_ms,
                    "ok": t.ok
                }
                for t in answer.tool_trace
            ]
        
        return response
    
    except Exception as e:
        return {
            "summary": f"处理查询时出错: {str(e)}",
            "sentiment": None,
            "citations": [],
            "tool_trace": []
        }


if __name__ == "__main__":
    main()
