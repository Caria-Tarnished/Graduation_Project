# -*- coding: utf-8 -*-
"""
财报检索页面

功能：
- 输入问题
- 显示 Top-K 引用片段
- 显示页码和相似度分数
- LLM 生成的答案

启动命令：
    streamlit run app/hosts/streamlit_app/app.py
"""
import streamlit as st
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


st.set_page_config(
    page_title="财报检索 - 财经分析 Agent",
    page_icon="📄",
    layout="wide"
)


def main():
    """主函数"""
    st.title("📄 财报检索")
    
    st.markdown("""
    输入您的问题，系统将从财报中检索相关内容并生成答案。
    """)
    
    # 侧边栏：参数配置
    with st.sidebar:
        st.header("检索参数")
        
        # Top-K 设置
        top_k = st.slider(
            "返回结果数量",
            min_value=1,
            max_value=10,
            value=5,
            help="返回相似度最高的前 K 个片段"
        )
        
        # 语言筛选
        language_filter = st.selectbox(
            "语言筛选",
            ["全部", "中文", "英文"],
            index=0
        )
        
        # 显示选项
        st.header("显示选项")
        show_metadata = st.checkbox("显示元数据", value=True)
        show_full_text = st.checkbox("显示完整文本", value=False)
    
    # 主区域：搜索框
    question = st.text_input(
        "请输入您的问题",
        placeholder="例如：贵州茅台 2023 年营收情况如何？",
        key="question_input"
    )
    
    # 搜索按钮
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        search_button = st.button("🔍 搜索", type="primary")
    with col2:
        clear_button = st.button("🗑️ 清空")
    
    if clear_button:
        st.session_state.question_input = ""
        st.rerun()
    
    # 执行搜索
    if search_button and question:
        with st.spinner("正在检索..."):
            # 初始化 Agent（使用动态导入）
            import importlib.util
            
            app_module_path = project_root / "app" / "hosts" / "streamlit_app" / "app.py"
            spec = importlib.util.spec_from_file_location("streamlit_app", app_module_path)
            app_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(app_module)
            
            agent = app_module.initialize_agent()
            
            if agent is None:
                st.error("Agent 未初始化")
                return
            
            # 调用 Agent 检索
            try:
                answer = agent.process_query(
                    user_query=question,
                    query_type="report_qa"
                )
                
                # 显示 LLM 总结
                st.subheader("💡 AI 总结")
                st.markdown(answer.summary)
                
                # 显示引用片段
                if answer.citations and len(answer.citations) > 0:
                    st.subheader(f"📚 检索结果（共 {len(answer.citations)} 条）")
                    
                    for i, citation in enumerate(answer.citations, 1):
                        with st.expander(
                            f"引用 {i} - {citation.source_file} (相似度: {citation.score:.2%})",
                            expanded=(i <= 3)  # 默认展开前 3 个
                        ):
                            # 显示文本
                            if show_full_text:
                                st.markdown(citation.text)
                            else:
                                # 只显示前 200 字符
                                preview = citation.text[:200]
                                if len(citation.text) > 200:
                                    preview += "..."
                                st.markdown(preview)
                            
                            # 显示元数据
                            if show_metadata:
                                st.markdown("---")
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("页码", citation.page_idx + 1)
                                
                                with col2:
                                    st.metric("相似度", f"{citation.score:.2%}")
                                
                                with col3:
                                    st.metric("来源", citation.source_file.split('/')[-1])
                                
                                # 显示额外元数据
                                if citation.metadata:
                                    st.json(citation.metadata)
                else:
                    st.warning("未找到相关内容")
                
                # 显示工具追踪
                if answer.tool_trace:
                    with st.expander("🔧 工具调用追踪"):
                        for trace in answer.tool_trace:
                            status = "✓" if trace.ok else "✗"
                            st.text(f"{status} {trace.name} ({trace.elapsed_ms}ms)")
                            if trace.error:
                                st.error(f"错误: {trace.error}")
                
                # 显示警告
                if answer.warnings:
                    st.warning("⚠️ " + " | ".join(answer.warnings))
            
            except Exception as e:
                st.error(f"检索失败: {e}")
                import traceback
                with st.expander("错误详情"):
                    st.code(traceback.format_exc())
    
    elif search_button and not question:
        st.warning("请输入问题")
    
    # 示例问题
    st.markdown("---")
    st.subheader("💡 示例问题")
    
    example_questions = [
        "黄金市场 2023 年的表现如何？",
        "美联储加息对黄金价格有什么影响？",
        "2024 年黄金价格走势预测",
        "中国黄金需求情况",
        "全球黄金供需平衡"
    ]
    
    cols = st.columns(len(example_questions))
    for i, (col, example) in enumerate(zip(cols, example_questions)):
        with col:
            if st.button(example, key=f"example_{i}"):
                st.session_state.question_input = example
                st.rerun()


if __name__ == "__main__":
    main()
