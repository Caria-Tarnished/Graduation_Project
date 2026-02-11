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
    
    # 显示 Deepseek 配置状态
    with st.expander("🔧 系统配置状态", expanded=False):
        import os
        from dotenv import load_dotenv
        
        # 加载环境变量
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        
        # 检查 Deepseek API Key
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if api_key:
            masked_key = api_key[:8] + "..." + api_key[-8:] if len(api_key) > 16 else "***"
            st.success(f"✓ DEEPSEEK_API_KEY 已配置: {masked_key}")
        else:
            st.error("✗ DEEPSEEK_API_KEY 未配置")
        
        # 检查 BERT 模型
        bert_path = project_root / "models" / "bert_3cls" / "best"
        if bert_path.exists():
            st.success(f"✓ BERT 模型已加载: {bert_path}")
        else:
            st.warning(f"⚠ BERT 模型未找到: {bert_path}")
        
        # 检查数据库
        db_path = project_root / "finance_analysis.db"
        if db_path.exists():
            st.success(f"✓ 数据库已连接: {db_path}")
        else:
            st.warning(f"⚠ 数据库未找到: {db_path}")
    
    # 初始化 session_state
    if 'chart_loaded' not in st.session_state:
        st.session_state.chart_loaded = False
    if 'prices_df' not in st.session_state:
        st.session_state.prices_df = None
    if 'events_df' not in st.session_state:
        st.session_state.events_df = None
    if 'ticker' not in st.session_state:
        st.session_state.ticker = "XAUUSD"
    if 'date_range' not in st.session_state:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        st.session_state.date_range = (start_date, end_date)
    if 'show_flash' not in st.session_state:
        st.session_state.show_flash = True  # 默认显示快讯
    if 'min_calendar_star' not in st.session_state:
        st.session_state.min_calendar_star = 3  # 默认显示 3 星及以上的日历事件
    if 'show_neutral' not in st.session_state:
        st.session_state.show_neutral = True  # 默认显示中性事件
    
    # 侧边栏：参数配置
    with st.sidebar:
        st.header("参数配置")
        
        # 标的选择
        ticker = st.selectbox(
            "标的",
            ["XAUUSD", "000001.SH", "300750.SZ", "NVDA"],
            index=0,
            key="ticker_select"
        )
        
        # 时间范围（使用容器避免遮挡）
        st.markdown("**时间范围**")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        # 使用两个独立的日期选择器，避免弹出窗口被遮挡
        col1, col2 = st.columns(2)
        with col1:
            start_date_input = st.date_input(
                "开始日期",
                value=start_date,
                max_value=end_date,
                key="start_date"
            )
        with col2:
            end_date_input = st.date_input(
                "结束日期",
                value=end_date,
                max_value=end_date,
                key="end_date"
            )
        
        date_range = (start_date_input, end_date_input)
        
        # 事件筛选
        st.markdown("**事件筛选**")
        
        # 快讯类事件开关
        show_flash = st.checkbox(
            "显示快讯类事件",
            value=True,
            help="快讯类事件大多无星级，内容为文字描述，需要 BERT 模型分析"
        )
        
        # 日历类事件星级筛选
        min_calendar_star = st.slider(
            "日历事件最低星级",
            min_value=3,
            max_value=5,
            value=3,
            help="只显示星级 >= 该值的日历事件（日历事件全部有星级和 affect 标签）"
        )
        
        # 中性事件筛选
        show_neutral = st.checkbox(
            "显示中性事件",
            value=True,
            help="取消勾选后，K线图上只显示利多/利空事件，不显示中性事件"
        )
        
        # 加载按钮
        load_button = st.button("加载数据", type="primary")
    
    # 主区域：图表
    if load_button:
        with st.spinner("正在加载数据..."):
            # 加载价格数据
            prices_df = load_price_data(ticker, date_range)
            
            # 加载事件数据
            events_df = load_event_data(ticker, date_range, show_flash, min_calendar_star)
            
            if prices_df is None or len(prices_df) == 0:
                st.error("未找到价格数据")
                return
            
            # 保存到 session_state
            st.session_state.chart_loaded = True
            st.session_state.prices_df = prices_df
            st.session_state.events_df = events_df
            st.session_state.ticker = ticker
            st.session_state.date_range = date_range
            st.session_state.show_flash = show_flash
            st.session_state.min_calendar_star = min_calendar_star
            st.session_state.show_neutral = show_neutral
    
    # 显示图表（如果已加载）
    if st.session_state.chart_loaded and st.session_state.prices_df is not None:
        # 检查筛选条件是否改变
        filter_changed = (
            show_neutral != st.session_state.show_neutral
        )
        
        # 如果筛选条件改变，更新 session_state 并重新绘图
        if filter_changed:
            st.session_state.show_neutral = show_neutral
            st.info("筛选条件已更新，图表已刷新")
        
        # 绘制 K 线图
        fig, config = plot_kline_with_events(
            st.session_state.prices_df, 
            st.session_state.events_df, 
            st.session_state.ticker,
            st.session_state.show_neutral  # 传递中性事件筛选参数
        )
        st.plotly_chart(fig, use_container_width=True, config=config)
        
        # 显示事件列表
        if st.session_state.events_df is not None and len(st.session_state.events_df) > 0:
            # 重置索引，确保索引是连续的整数
            events_df_reset = st.session_state.events_df.reset_index(drop=True)
            
            # 确保 content 字段是字符串类型
            events_df_reset['content'] = events_df_reset['content'].fillna(events_df_reset['name'])
            events_df_reset['content'] = events_df_reset['content'].astype(str)
            
            st.subheader(f"事件列表（共 {len(events_df_reset)} 条）")
            
            # 如果事件太多，添加分页或限制显示数量
            max_display = 1000  # 最多显示 1000 条
            if len(events_df_reset) > max_display:
                st.warning(f"事件数量过多（{len(events_df_reset)} 条），仅显示最近的 {max_display} 条")
                events_df_display = events_df_reset.tail(max_display).reset_index(drop=True)
            else:
                events_df_display = events_df_reset
            
            # 预先生成格式化字符串列表（避免 format_func 中的类型错误）
            event_options = []
            for i in range(len(events_df_display)):
                try:
                    event = events_df_display.iloc[i]
                    ts = event['ts_local']
                    content = str(event['content'])[:50]
                    event_options.append(f"{ts} - {content}...")
                except Exception as e:
                    event_options.append(f"事件 {i} (解析失败)")
            
            # 选择事件进行分析
            selected_event = st.selectbox(
                "选择事件查看详情",
                options=range(len(events_df_display)),
                format_func=lambda i: event_options[int(i)],
                key="event_selector"
            )
            
            if selected_event is not None:
                event = events_df_display.iloc[int(selected_event)]
                show_event_analysis(event, st.session_state.ticker)
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


def load_event_data(ticker: str, date_range, show_flash: bool, min_calendar_star: int) -> pd.DataFrame:
    """
    加载事件数据
    
    Args:
        ticker: 标的代码
        date_range: 时间范围
        show_flash: 是否显示快讯类事件
        min_calendar_star: 日历事件最低星级
    
    Returns:
        事件数据 DataFrame
    """
    try:
        import sqlite3
        
        db_path = project_root / "finance_analysis.db"
        if not db_path.exists():
            return None
        
        conn = sqlite3.connect(str(db_path))
        
        # 构建查询条件
        # 1. 快讯类事件：source='flash'
        # 2. 日历类事件：source='calendar' AND star >= min_calendar_star
        
        if show_flash:
            # 显示快讯 + 符合星级的日历事件
            query = """
            SELECT 
                e.event_id,
                e.ts_local,
                e.source,
                e.content,
                e.name,
                e.star,
                e.country,
                e.affect,
                ei.price_event
            FROM events e
            LEFT JOIN event_impacts ei ON e.event_id = ei.event_id AND ei.ticker = ?
            WHERE e.ts_local >= ?
              AND e.ts_local <= ?
              AND (
                  e.source = 'flash'
                  OR (e.source = 'calendar' AND e.star >= ?)
              )
            ORDER BY e.ts_local ASC
            """
            params = (ticker, date_range[0].strftime("%Y-%m-%d 00:00:00"), 
                     date_range[1].strftime("%Y-%m-%d 23:59:59") if len(date_range) > 1 else date_range[0].strftime("%Y-%m-%d 23:59:59"),
                     min_calendar_star)
        else:
            # 只显示符合星级的日历事件
            query = """
            SELECT 
                e.event_id,
                e.ts_local,
                e.source,
                e.content,
                e.name,
                e.star,
                e.country,
                e.affect,
                ei.price_event
            FROM events e
            LEFT JOIN event_impacts ei ON e.event_id = ei.event_id AND ei.ticker = ?
            WHERE e.ts_local >= ?
              AND e.ts_local <= ?
              AND e.source = 'calendar'
              AND e.star >= ?
            ORDER BY e.ts_local ASC
            """
            params = (ticker, date_range[0].strftime("%Y-%m-%d 00:00:00"),
                     date_range[1].strftime("%Y-%m-%d 23:59:59") if len(date_range) > 1 else date_range[0].strftime("%Y-%m-%d 23:59:59"),
                     min_calendar_star)
        
        df = pd.read_sql_query(query, conn, params=params)
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


def plot_kline_with_events(prices_df: pd.DataFrame, events_df: pd.DataFrame, ticker: str, show_neutral: bool = True):
    """
    绘制 K 线图并标注事件
    
    使用箭头标注：
    - 利好事件：绿色向上箭头，标注在 K 线下方
    - 利空事件：红色向下箭头，标注在 K 线上方
    - 中性事件：灰色圆点，标注在 K 线中间
    - 星级：通过颜色深浅表示（星级越高，颜色越深）
    
    Args:
        prices_df: 价格数据
        events_df: 事件数据
        ticker: 标的代码
        show_neutral: 是否显示中性事件（默认 True）
    
    Returns:
        Plotly Figure 对象和配置
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
        name='K线',
        showlegend=True
    ))
    
    # 添加事件标注（使用箭头）
    if events_df is not None and len(events_df) > 0:
        # 首先需要对每个事件进行情感分析，以确定箭头方向
        # 为了性能，我们先使用简单的规则：根据事件内容关键词判断
        
        # 定义颜色映射（根据星级和情感）
        # 利好：绿色系（星级越高越深）
        bullish_colors = {
            1: 'rgba(144, 238, 144, 0.6)',  # 浅绿
            2: 'rgba(60, 179, 113, 0.7)',   # 中绿
            3: 'rgba(34, 139, 34, 0.8)',    # 深绿
            4: 'rgba(0, 128, 0, 0.9)',      # 更深绿
            5: 'rgba(0, 100, 0, 1.0)'       # 最深绿
        }
        
        # 利空：红色系（星级越高越深）
        bearish_colors = {
            1: 'rgba(255, 182, 193, 0.6)',  # 浅红
            2: 'rgba(255, 99, 71, 0.7)',    # 中红
            3: 'rgba(220, 20, 60, 0.8)',    # 深红
            4: 'rgba(178, 34, 34, 0.9)',    # 更深红
            5: 'rgba(139, 0, 0, 1.0)'       # 最深红
        }
        
        # 中性：灰色系
        neutral_colors = {
            1: 'rgba(211, 211, 211, 0.6)',  # 浅灰
            2: 'rgba(169, 169, 169, 0.7)',  # 中灰
            3: 'rgba(128, 128, 128, 0.8)',  # 深灰
            4: 'rgba(105, 105, 105, 0.9)',  # 更深灰
            5: 'rgba(64, 64, 64, 1.0)'      # 最深灰
        }
        
        # 简单的情感判断（优先使用 affect 标签，然后是数值比较，最后是关键词）
        def simple_sentiment(content: str, affect: str = None) -> str:
            """
            简单的情感判断（优先使用 affect 标签）
            
            处理三种类型的事件：
            1. 有 affect 标签的事件：直接使用标签
            2. 数值型事件：比较实际值与预期值
            3. 文本型事件：使用关键词匹配
            
            注意：这是一个简化版本，仅用于图表标注的视觉区分
            真正的情感分析在点击事件详情时由 BERT 模型完成
            """
            # 方法 1: 优先使用 affect 标签
            if affect and not pd.isna(affect):
                affect_lower = str(affect).lower()
                if '利多' in affect_lower or '利好' in affect_lower:
                    return 'bullish'
                elif '利空' in affect_lower:
                    return 'bearish'
                elif '影响较小' in affect_lower or '未公布' in affect_lower:
                    return 'neutral'
            
            if pd.isna(content):
                return 'neutral'
            
            content_lower = content.lower()
            
            # 方法 2: 尝试解析数值型事件（如 "前值:52.1 预期:52.1 公布:51.4"）
            import re
            
            # 匹配模式：前值:X 预期:Y 公布:Z
            pattern = r'预期[：:]\s*([-\d.]+).*?公布[：:]\s*([-\d.]+)'
            match = re.search(pattern, content)
            
            if match:
                try:
                    expected = float(match.group(1))
                    actual = float(match.group(2))
                    
                    # 判断是否超预期
                    diff = actual - expected
                    
                    # 判断指标类型（失业率、通胀等是负向指标）
                    negative_indicators = ['失业', 'unemployment', 'cpi', '通胀', 'inflation']
                    is_negative_indicator = any(ind in content_lower for ind in negative_indicators)
                    
                    # 阈值：至少有 0.05 的差异才算有意义
                    threshold = 0.05
                    
                    if abs(diff) < threshold:
                        return 'neutral'
                    
                    if is_negative_indicator:
                        # 负向指标：实际值高于预期 = 利空
                        return 'bearish' if diff > 0 else 'bullish'
                    else:
                        # 正向指标：实际值高于预期 = 利好
                        return 'bullish' if diff > 0 else 'bearish'
                
                except (ValueError, IndexError):
                    pass  # 解析失败，继续使用关键词匹配
            
            # 方法 3: 关键词匹配（用于文本型事件）
            # 利好关键词（扩展版）
            bullish_keywords = [
                # 中文
                '上涨', '增长', '超预期', '好于预期', '利好', '上调', '提高', '增加', 
                '扩张', '改善', '复苏', '强劲', '乐观', '积极', '升', '涨', '高于',
                '加速', '反弹', '突破', '创新高', '大幅增长', '显著增长',
                # 英文
                'beat', 'rise', 'increase', 'growth', 'surge', 'rally', 'gain',
                'improve', 'strong', 'robust', 'positive', 'optimistic', 'exceed',
                'outperform', 'bullish', 'up', 'higher', 'above'
            ]
            
            # 利空关键词（扩展版）
            bearish_keywords = [
                # 中文
                '下跌', '下降', '低于预期', '不及预期', '利空', '下调', '降低', '减少',
                '收缩', '恶化', '衰退', '疲软', '悲观', '消极', '降', '跌', '低于',
                '放缓', '下滑', '跌破', '创新低', '大幅下降', '显著下降',
                # 英文
                'miss', 'fall', 'decrease', 'decline', 'drop', 'plunge', 'slump',
                'weaken', 'weak', 'soft', 'negative', 'pessimistic', 'below',
                'underperform', 'bearish', 'down', 'lower'
            ]
            
            # 计算关键词出现次数
            bullish_count = sum(1 for kw in bullish_keywords if kw in content_lower)
            bearish_count = sum(1 for kw in bearish_keywords if kw in content_lower)
            
            # 判断逻辑：需要明显的倾向性
            if bullish_count > bearish_count and bullish_count >= 1:
                return 'bullish'
            elif bearish_count > bullish_count and bearish_count >= 1:
                return 'bearish'
            else:
                return 'neutral'
        
        # 按情感分组（使用 affect 标签）
        events_df = events_df.copy()
        events_df['sentiment'] = events_df.apply(
            lambda row: simple_sentiment(row['content'], row.get('affect')), 
            axis=1
        )
        
        # 分别为每种情感类型添加散点
        for sentiment_type in ['bullish', 'bearish', 'neutral']:
            # 如果不显示中性事件，跳过中性类型
            if sentiment_type == 'neutral' and not show_neutral:
                continue
            
            sentiment_events = events_df[events_df['sentiment'] == sentiment_type].copy()
            
            if len(sentiment_events) == 0:
                continue
            
            # 按星级分组
            for star_level in sorted(sentiment_events['star'].unique()):
                star_events = sentiment_events[sentiment_events['star'] == star_level].copy()
                
                # 确保星级是整数
                star_level_int = int(star_level) if not pd.isna(star_level) else 1
                star_level_int = max(1, min(5, star_level_int))  # 限制在 1-5 范围
                
                # 获取事件对应的价格
                event_prices = []
                hover_texts = []
                
                for idx, event in star_events.iterrows():
                    # 获取事件时间对应的价格
                    price = event['price_event']
                    if pd.isna(price):
                        # 如果没有价格，使用最近的价格
                        nearest_prices = prices_df[prices_df['ts_local'] <= event['ts_local']]
                        if len(nearest_prices) > 0:
                            if sentiment_type == 'bullish':
                                price = nearest_prices['low'].iloc[-1]  # 利好标注在下方
                            elif sentiment_type == 'bearish':
                                price = nearest_prices['high'].iloc[-1]  # 利空标注在上方
                            else:
                                price = nearest_prices['close'].iloc[-1]  # 中性标注在中间
                        else:
                            price = prices_df['close'].iloc[0]
                    else:
                        # 根据情感调整价格位置
                        if sentiment_type == 'bullish':
                            # 利好事件标注在 K 线下方
                            price = price * 0.998  # 稍微低一点
                        elif sentiment_type == 'bearish':
                            # 利空事件标注在 K 线上方
                            price = price * 1.002  # 稍微高一点
                    
                    event_prices.append(price)
                    
                    # 构建悬停文本
                    star_text = '★' * star_level_int
                    sentiment_text = {
                        'bullish': '利好',
                        'bearish': '利空',
                        'neutral': '中性'
                    }[sentiment_type]
                    
                    hover_text = (
                        f"<b>{star_text} {sentiment_text}</b><br>"
                        f"<b>{event['content'][:60]}</b><br>"
                        f"时间: {event['ts_local']}<br>"
                        f"来源: {event['source']}<br>"
                        f"国家: {event.get('country', 'N/A')}<br>"
                        f"价格: {price:.2f}"
                    )
                    hover_texts.append(hover_text)
                
                # 选择颜色
                if sentiment_type == 'bullish':
                    color = bullish_colors[star_level_int]
                    symbol = 'triangle-up'  # 向上箭头
                    name = f'利好 {star_level_int}星'
                elif sentiment_type == 'bearish':
                    color = bearish_colors[star_level_int]
                    symbol = 'triangle-down'  # 向下箭头
                    name = f'利空 {star_level_int}星'
                else:
                    color = neutral_colors[star_level_int]
                    symbol = 'circle'  # 圆点
                    name = f'中性 {star_level_int}星'
                
                # 添加散点图
                fig.add_trace(go.Scatter(
                    x=star_events['ts_local'],
                    y=event_prices,
                    mode='markers',
                    name=name,
                    marker=dict(
                        size=8,  # 比之前的星号小一点
                        color=color,
                        symbol=symbol,
                        line=dict(width=1, color='white')
                    ),
                    text=hover_texts,
                    hovertemplate='%{text}<extra></extra>',
                    showlegend=True,
                    legendgroup=sentiment_type,  # 按情感分组
                ))
    
    # 更新布局
    fig.update_layout(
        title=f"{ticker} K 线图 + 事件标注（鼠标悬停查看详情）",
        xaxis_title="时间",
        yaxis_title="价格",
        xaxis_rangeslider_visible=False,
        height=700,
        hovermode='closest',  # 悬停模式：最近的点
        dragmode='pan',  # 默认拖拽模式为平移（左键拖动）
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1
        )
    )
    
    # 配置交互工具
    config = {
        'scrollZoom': True,  # 启用鼠标滚轮缩放
        'displayModeBar': True,  # 显示工具栏
        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
        'modeBarButtonsToRemove': [],
        'displaylogo': False,  # 隐藏 Plotly logo
        'toImageButtonOptions': {
            'format': 'png',
            'filename': f'{ticker}_kline',
            'height': 700,
            'width': 1200,
            'scale': 2
        }
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
    
    # 确保星级是整数类型
    star_count = int(event['star']) if not pd.isna(event['star']) else 0
    
    with col1:
        st.markdown(f"**时间**: {event['ts_local']}")
        st.markdown(f"**来源**: {event['source']}")
        st.markdown(f"**星级**: {'★' * star_count}")
    
    with col2:
        st.markdown(f"**国家**: {event.get('country', 'N/A')}")
        st.markdown(f"**事件 ID**: {event['event_id']}")
        # 显示 affect 标签（如果有）
        if 'affect' in event and not pd.isna(event['affect']) and event['affect']:
            affect_label = event['affect']
            # 根据标签设置颜色
            if '利多' in affect_label or '利好' in affect_label:
                st.markdown(f"**标签**: :green[{affect_label}]")
            elif '利空' in affect_label:
                st.markdown(f"**标签**: :red[{affect_label}]")
            else:
                st.markdown(f"**标签**: {affect_label}")
    
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
                query_type="news_analysis",
                event_time=event['ts_local']  # 传递事件的实际时间
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
