# -*- coding: utf-8 -*-
"""
K 线图表页面

功能：
- 显示 K 线图（使用 Plotly）
- 在图表上标注事件点
- 点击事件点触发情感分析

启动命令：
    streamlit run app/hosts/streamlit_app/app.py
"""
import streamlit as st
import sys
import os
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 添加项目根目录到路径（使用绝对路径）
project_root = Path(__file__).parent.parent.parent.parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 确保当前工作目录是项目根目录
os.chdir(str(project_root))


st.set_page_config(
    page_title="K 线图表 - 财经分析 Agent",
    page_icon="📈",
    layout="wide"
)


def main():
    """主函数"""
    st.title("📈 K 线图表 + 事件标注")
    
    st.markdown("""
    本页面展示 K 线图并标注重要事件，点击事件点可查看情感分析结果。
    """)
    
    # 侧边栏：参数配置
    with st.sidebar:
        st.header("参数配置")
        
        # 标的选择
        ticker = st.selectbox(
            "标的",
            ["XAUUSD", "000001.SH", "300750.SZ", "NVDA"],
            index=0
        )
        
        # 时间范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        date_range = st.date_input(
            "时间范围",
            value=(start_date, end_date),
            max_value=end_date
        )
        
        # 事件筛选
        min_star = st.slider(
            "最低星级",
            min_value=1,
            max_value=5,
            value=3,
            help="只显示星级 >= 该值的事件"
        )
        
        # 加载按钮
        load_button = st.button("加载数据", type="primary")
    
    # 主区域：图表
    if load_button:
        with st.spinner("正在加载数据..."):
            # 加载价格数据
            prices_df = load_price_data(ticker, date_range)
            
            # 加载事件数据
            events_df = load_event_data(ticker, date_range, min_star)
            
            if prices_df is None or len(prices_df) == 0:
                st.error("未找到价格数据")
                return
            
            # 绘制 K 线图
            fig, config = plot_kline_with_events(prices_df, events_df, ticker)
            st.plotly_chart(fig, width="stretch", config=config)
            
            # 显示事件列表
            if events_df is not None and len(events_df) > 0:
                st.subheader(f"事件列表（共 {len(events_df)} 条）")
                
                # 选择事件进行分析
                selected_event = st.selectbox(
                    "选择事件查看详情",
                    options=range(len(events_df)),
                    format_func=lambda i: f"{events_df.iloc[i]['ts_local']} - {events_df.iloc[i]['content'][:50]}..."
                )
                
                if selected_event is not None:
                    event = events_df.iloc[selected_event]
                    show_event_analysis(event, ticker)
            else:
                st.info("该时间范围内没有符合条件的事件")
    else:
        st.info("请在侧边栏配置参数并点击\"加载数据\"")


def load_price_data(ticker: str, date_range) -> pd.DataFrame:
    """
    加载价格数据
    
    Args:
        ticker: 标的代码
        date_range: 时间范围
    
    Returns:
        价格数据 DataFrame
    """
    try:
        import sqlite3
        
        db_path = project_root / "finance_analysis.db"
        if not db_path.exists():
            st.warning(f"数据库不存在: {db_path}")
            return None
        
        conn = sqlite3.connect(str(db_path))
        
        # 查询价格数据
        query = """
        SELECT 
            ts_local,
            open,
            high,
            low,
            close,
            volume
        FROM prices_m1
        WHERE ticker = ?
          AND ts_local >= ?
          AND ts_local <= ?
        ORDER BY ts_local ASC
        """
        
        start_str = date_range[0].strftime("%Y-%m-%d 00:00:00")
        end_str = date_range[1].strftime("%Y-%m-%d 23:59:59") if len(date_range) > 1 else start_str
        
        df = pd.read_sql_query(query, conn, params=(ticker, start_str, end_str))
        conn.close()
        
        if len(df) == 0:
            return None
        
        # 转换时间列
        df['ts_local'] = pd.to_datetime(df['ts_local'])
        
        return df
    
    except Exception as e:
        st.error(f"加载价格数据失败: {e}")
        return None


def load_event_data(ticker: str, date_range, min_star: int) -> pd.DataFrame:
    """
    加载事件数据
    
    Args:
        ticker: 标的代码
        date_range: 时间范围
        min_star: 最低星级
    
    Returns:
        事件数据 DataFrame
    """
    try:
        import sqlite3
        
        db_path = project_root / "finance_analysis.db"
        if not db_path.exists():
            return None
        
        conn = sqlite3.connect(str(db_path))
        
        # 查询事件数据
        query = """
        SELECT 
            e.event_id,
            e.ts_local,
            e.source,
            e.content,
            e.name,
            e.star,
            e.country,
            ei.price_event
        FROM events e
        LEFT JOIN event_impacts ei ON e.event_id = ei.event_id AND ei.ticker = ?
        WHERE e.ts_local >= ?
          AND e.ts_local <= ?
          AND e.star >= ?
        ORDER BY e.ts_local ASC
        """
        
        start_str = date_range[0].strftime("%Y-%m-%d 00:00:00")
        end_str = date_range[1].strftime("%Y-%m-%d 23:59:59") if len(date_range) > 1 else start_str
        
        df = pd.read_sql_query(query, conn, params=(ticker, start_str, end_str, min_star))
        conn.close()
        
        if len(df) == 0:
            return None
        
        # 转换时间列
        df['ts_local'] = pd.to_datetime(df['ts_local'])
        
        # 填充内容
        df['content'] = df['content'].fillna(df['name'])
        
        return df
    
    except Exception as e:
        st.error(f"加载事件数据失败: {e}")
        return None


def plot_kline_with_events(prices_df: pd.DataFrame, events_df: pd.DataFrame, ticker: str):
    """
    绘制 K 线图并标注事件
    
    Args:
        prices_df: 价格数据
        events_df: 事件数据
        ticker: 标的代码
    
    Returns:
        Plotly Figure 对象
    """
    # 创建 K 线图
    fig = go.Figure()
    
    # 添加 K 线
    fig.add_trace(go.Candlestick(
        x=prices_df['ts_local'],
        open=prices_df['open'],
        high=prices_df['high'],
        low=prices_df['low'],
        close=prices_df['close'],
        name='K线'
    ))
    
    # 添加事件标注
    if events_df is not None and len(events_df) > 0:
        # 计算价格范围，用于调整标注位置
        price_range = prices_df['high'].max() - prices_df['low'].min()
        annotation_offset = price_range * 0.05  # 标注偏移量为价格范围的 5%
        
        for idx, event in events_df.iterrows():
            # 获取事件时间对应的价格
            price = event['price_event']
            if pd.isna(price):
                # 如果没有价格，使用最近的收盘价
                nearest_price = prices_df[prices_df['ts_local'] <= event['ts_local']]['close'].iloc[-1] if len(prices_df[prices_df['ts_local'] <= event['ts_local']]) > 0 else prices_df['close'].iloc[0]
                price = nearest_price
            
            # 交替显示标注在上方和下方，避免重叠
            ay_offset = -50 if idx % 2 == 0 else 50
            
            # 添加标注
            fig.add_annotation(
                x=event['ts_local'],
                y=price,
                text=f"★{event['star']} {event['content'][:15]}...",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="red" if idx % 2 == 0 else "blue",
                ax=0,
                ay=ay_offset,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="red" if idx % 2 == 0 else "blue",
                borderwidth=1,
                font=dict(size=10)
            )
    
    # 更新布局
    fig.update_layout(
        title=f"{ticker} K 线图 + 事件标注",
        xaxis_title="时间",
        yaxis_title="价格",
        xaxis_rangeslider_visible=False,
        height=600,
        hovermode='x unified',
        # 添加更好的交互配置
        dragmode='zoom',  # 默认拖拽模式为缩放
    )
    
    # 配置交互工具
    config = {
        'scrollZoom': True,  # 启用鼠标滚轮缩放
        'displayModeBar': True,  # 显示工具栏
        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
        'modeBarButtonsToRemove': [],
        'displaylogo': False,  # 隐藏 Plotly logo
    }
    
    return fig, config


def show_event_analysis(event, ticker: str):
    """
    显示事件分析结果
    
    Args:
        event: 事件数据（Series）
        ticker: 标的代码
    """
    st.subheader("事件详情")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**时间**: {event['ts_local']}")
        st.markdown(f"**来源**: {event['source']}")
        st.markdown(f"**星级**: {'★' * event['star']}")
    
    with col2:
        st.markdown(f"**国家**: {event.get('country', 'N/A')}")
        st.markdown(f"**事件 ID**: {event['event_id']}")
    
    st.markdown(f"**内容**: {event['content']}")
    
    # 情感分析
    st.subheader("情感分析")
    
    with st.spinner("正在分析..."):
        # 直接导入模块
        try:
            from app.core.orchestrator.agent import Agent
            from app.services.sentiment_analyzer import SentimentAnalyzer
            from app.core.engines.rag_engine import RagEngine
            from app.adapters.llm.deepseek_client import DeepseekClient
            from app.core.dto import sentiment_label_to_text
            import os
            
            # 初始化引擎
            sentiment_engine = None
            rag_engine = None
            llm_client = None
            
            # 加载情感分析引擎
            bert_path = project_root / "models" / "bert_3cls" / "best"
            if bert_path.exists():
                sentiment_engine = SentimentAnalyzer(model_path=str(bert_path))
            
            # 加载 RAG 引擎
            chroma_path = project_root / "data" / "reports" / "chroma_db"
            if chroma_path.exists():
                rag_engine = RagEngine(
                    chroma_path=str(chroma_path),
                    model_name="BAAI/bge-m3"
                )
            
            # 加载 LLM 客户端
            if os.getenv("DEEPSEEK_API_KEY"):
                llm_client = DeepseekClient()
            
            # 创建 Agent
            db_path = project_root / "finance_analysis.db"
            agent = Agent(
                sentiment_engine=sentiment_engine,
                rag_engine=rag_engine,
                rule_engine=None,
                llm_client=llm_client,
                db_path=str(db_path)
            )
            
            # 调用 Agent 分析
            answer = agent.process_query(
                user_query=event['content'],
                ticker=ticker,
                query_type="news_analysis"
            )
            
            # 显示结果
            st.markdown(f"**总结**: {answer.summary}")
            
            if answer.sentiment:
                label_text = sentiment_label_to_text(answer.sentiment.label)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("情感", label_text)
                with col2:
                    st.metric("置信度", f"{answer.sentiment.score:.2%}")
                
                st.markdown(f"**解释**: {answer.sentiment.explain}")
            
            # 工具追踪
            if answer.tool_trace:
                with st.expander("工具调用追踪"):
                    for trace in answer.tool_trace:
                        status = "✓" if trace.ok else "✗"
                        st.text(f"{status} {trace.name} ({trace.elapsed_ms}ms)")
        
        except Exception as e:
            import traceback
            st.error(f"分析失败: {e}")
            st.error(traceback.format_exc())


if __name__ == "__main__":
    main()
