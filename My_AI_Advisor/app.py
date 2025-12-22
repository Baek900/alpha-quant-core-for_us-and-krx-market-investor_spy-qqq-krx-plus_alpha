import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import datetime
from dateutil import parser
import pytz
from supabase import create_client, Client
import os

# ==============================================================================
# 0. Market Statistics (Historical Daily Avg Return for 5-Day Period)
# ==============================================================================
MARKET_STATS = {
    "S&P 500 (SPY)": {
        "bear": -0.005482, 
        "neut": 0.000151, 
        "bull": 0.004402
    },
    "NASDAQ (QQQ)": {
        "bear": -0.006119, 
        "neut": 0.000125, 
        "bull": 0.005435
    },
    "KOSPI (Korea)": {
        "bear": -0.002751, 
        "neut": -0.001029, 
        "bull": 0.002401
    }
}

# ==============================================================================
# 1. Configuration & Custom CSS
# ==============================================================================
st.set_page_config(page_title="TITAN FLOW - AI Advisor", layout="wide", page_icon="T")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #FFFFFF !important;
        }
        
        .stApp {
            background-color: #0B121F !important;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #FFFFFF !important;
            font-weight: 700 !important;
            margin-bottom: 0.5rem !important; /* 헤더 아래 간격 조정 */
        }

        div[data-testid="stCaptionContainer"], small, .stCaption {
            color: #CCCCCC !important;
            font-size: 0.9rem !important;
            opacity: 1 !important;
        }
        
        p {
            color: #E0E0E0 !important;
            font-size: 1rem !important;
        }

        header {visibility: hidden !important;}
        footer {visibility: hidden !important;}
        [data-testid="stSidebar"] {display: none !important;}
        section[data-testid="stSidebar"] {display: none !important;}

        div[data-testid="stMetric"] {
            background-color: #121926 !important;
            border: 1px solid #444444 !important;
            padding: 15px;
            border-radius: 8px;
        }
        div[data-testid="stMetricLabel"] {
            color: #DDDDDD !important;
            font-weight: 500 !important;
        }
        div[data-testid="stMetricValue"] {
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }

        div.stButton > button {
            width: 100%;
            background-color: #2563EB;
            color: white !important;
            border: 1px solid #60A5FA;
            border-radius: 6px;
            font-weight: 600;
        }

        div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
            background-color: #121926 !important;
            color: white !important;
            border: 1px solid #555555 !important;
        }
        ul[data-testid="stSelectboxVirtualDropdown"] {
            background-color: #121926 !important;
        }
        li[role="option"] {
            color: white !important;
        }

        hr {
            border-top: 1px solid #555555 !important;
            margin: 1.5rem 0;
        }
        
        .streamlit-expanderHeader {
            background-color: #121926 !important;
        }
        div[data-testid="stExpanderDetails"] {
            background-color: #0B121F !important;
            border: 1px solid #444444;
        }

        div[role="dialog"] {
            background-color: #121926 !important;
            border: 2px solid #6B7280 !important;
            color: #FFFFFF !important;
        }
        div[role="dialog"] h2, div[role="dialog"] p, div[role="dialog"] label {
            color: #FFFFFF !important;
        }
        div[role="dialog"] button[aria-label="Close"] {
            color: #FFFFFF !important;
        }
        div[data-testid="stMarkdownContainer"] p {
             color: #E0E0E0 !important;
        }
        
        /* [수정] 툴팁(Help) 스타일 개선: 너비 제한 및 줄바꿈 적용 */
        div[data-testid="stTooltipContent"] {
            background-color: #1E293B !important; /* 조금 더 밝은 배경 */
            border: 1px solid #475569 !important;
            color: #E2E8F0 !important;
            font-family: 'Inter', monospace !important; /* 고정폭 글꼴 느낌 */
            white-space: pre-wrap !important; /* 줄바꿈 유지 */
            max-width: 300px !important; /* 최대 너비 제한 */
            font-size: 0.85rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State
if "password_correct" not in st.session_state:
    st.session_state["password_correct"] = False

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Home"

# Connect to Supabase
try:
    SUPABASE_URL = st.secrets["supabase"]["url"]
    SUPABASE_KEY = st.secrets["supabase"]["key"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# Helper Functions
def convert_utc_to_kst(utc_str):
    try:
        utc_time = parser.parse(utc_str)
        kst_zone = pytz.timezone('Asia/Seoul')
        kst_time = utc_time.astimezone(kst_zone)
        return kst_time.strftime('%Y-%m-%d %H:%M')
    except:
        return utc_str

def load_latest_analysis(market_name):
    try:
        response = supabase.table("prediction_logs") \
            .select("*") \
            .eq("market_name", market_name) \
            .order("created_at", desc=True) \
            .limit(2) \
            .execute()
        
        if response.data:
            current_data = response.data[0]
            previous_data = response.data[1] if len(response.data) > 1 else None
            return current_data, previous_data
        else:
            return None, None
    except:
        return None, None

def logout():
    st.session_state["password_correct"] = False
    st.session_state["logged_in_user"] = None
    st.session_state["current_page"] = "Home"
    st.rerun()

# Strategy Text Generation
def get_strategy_text(market_name, prev_signal, current_signal):
    if "NASDAQ" in market_name:
        target_long = "TQQQ (or QLD)"
        target_short = "SQQQ (or QID)"
    elif "S&P 500" in market_name:
        target_long = "UPRO (or SSO)"
        target_short = "SPXU (or SDS)"
    else: # KOSPI
        target_long = "KODEX Leverage"
        target_short = "KODEX 200 Inverse"

    if current_signal == "BUY":
        return f"**Bullish Trend Detected.** The AI model probabilities indicate a structural uptrend. It is recommended to accumulate **{target_long}**. Maintain exposure while monitoring for trend exhaustion."
    elif current_signal == "SELL":
        return f"**Bearish Risk Detected.** Market probabilities have shifted downward. It is advised to liquidate long positions and hedge downside risk using **{target_short}** or hold Cash."
    else:
        return "**Neutral / Uncertain Market.** No clear directional signal is present. It is advised to **Hold Cash** and wait for a breakout confirmation before entering new positions."

# ==============================================================================
# Login Dialog
# ==============================================================================
@st.dialog("Member Login") 
def login_dialog():
    st.write("Please enter your credentials.")
    username = st.text_input("Username") 
    password = st.text_input("Password", type="password")

    if st.button("Access Dashboard"): 
        if username in st.secrets["users"] and password == st.secrets["users"][username]:
            st.session_state["password_correct"] = True
            st.session_state["logged_in_user"] = username
            st.session_state["current_page"] = "Dashboard"
            st.rerun()
        else:
            st.error("Invalid credentials.")

if st.session_state["password_correct"]:
    st.session_state["current_page"] = "Dashboard"

# ------------------------------------------------------------------------------
# PAGE: HOME
# ------------------------------------------------------------------------------
if st.session_state["current_page"] == "Home":
    col_header, col_login = st.columns([6, 1])
    with col_header:
        st.markdown("<h1 style='font-size: 3rem; margin-bottom: 0;'>TITAN FLOW</h1>", unsafe_allow_html=True)
    with col_login:
        if st.button("🔑 Log In", use_container_width=True):
            login_dialog()

    st.markdown("<h3 style='color: #FFFFFF; font-weight: 500; margin-top: 10px;'>Advanced Financial Forecasting System powered by TMFG-LSTM</h3>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color: #E0E0E0; font-size: 1.1rem;'>
    This platform leverages deep learning architectures to analyze global market trends, macroeconomics, and sector rotation, providing institutional-grade insights.
    </p>
    """, unsafe_allow_html=True)
    
    st.divider()

    # (Home page chart logic omitted for brevity - same as before)
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "backtest_result.csv")
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        initial_capital = df['Strategy'].iloc[0]
        ai_returns = (df['Strategy'] - initial_capital) / initial_capital
        market_returns = (df['Benchmark'] - initial_capital) / initial_capital
        final_ai_ret = ai_returns.iloc[-1] * 100
        final_bm_ret = market_returns.iloc[-1] * 100
        alpha = final_ai_ret - final_bm_ret

        fig_bench = go.Figure()
        fig_bench.add_trace(go.Scatter(x=df.index, y=ai_returns, mode='lines', name='Alpha Strategy', line=dict(color='#00E396', width=3))) 
        fig_bench.add_trace(go.Scatter(x=df.index, y=market_returns, mode='lines', name='S&P 500', line=dict(color='#AAAAAA', dash='dot', width=2))) 
        fig_bench.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=30, b=0), xaxis=dict(showgrid=True, gridcolor='#333', color='#FFFFFF'), yaxis=dict(showgrid=True, gridcolor='#333', color='#FFFFFF', tickformat='.0%'), legend=dict(orientation="h", y=1.1, font=dict(color="white", size=12)), height=350)
        
        c1, c2 = st.columns([3, 1])
        with c1: st.plotly_chart(fig_bench, use_container_width=True)
        with c2:
            st.metric(label="Total Return", value=f"{final_ai_ret:+.2f}%", delta=f"{alpha:+.2f}% Alpha")
            st.metric(label="Benchmark", value=f"{final_bm_ret:+.2f}%")
            st.caption("Data source: Verified Backtest")
    except:
        st.error("Backtest data not found.")

    st.divider()
    st.markdown("#### Model Reliability Metrics")
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1: st.metric("Accuracy", "52.6%", "vs Random (33%)")
    with col_m2: st.metric("Precision (Buy)", "64.0%", "High Confidence")
    with col_m3: st.metric("Recall (Uptrend)", "55.0%", "Opportunity Capture")

# ------------------------------------------------------------------------------
# PAGE: DASHBOARD
# ------------------------------------------------------------------------------
elif st.session_state["current_page"] == "Dashboard":
    
    # [상단 네비게이션]
    nav_col1, nav_col2 = st.columns([3, 1])
    with nav_col1:
        market_option = st.selectbox("Select Market", ["NASDAQ (QQQ)", "S&P 500 (SPY)", "KOSPI (Korea)"], label_visibility="collapsed")
    with nav_col2:
        if st.button("Sign Out"): logout()
    
    st.divider()

    # [시장 제목 및 상태]
    top_col1, top_col2 = st.columns([4, 1])
    with top_col1:
        st.markdown(f"## {market_option}") 
        st.caption("Live Market Analysis & Signal Generation")
    with top_col2:
         st.markdown(f"<div style='text-align: right; color: #FFFFFF; font-weight: bold;'>Status: <span style='color: #00E396;'>● Live</span></div>", unsafe_allow_html=True)

    # [설정 및 데이터 로드]
    if market_option == "NASDAQ (QQQ)": IDX, LEV_LONG, LEV_SHORT = "QQQ", "QLD/TQQQ", "QID/SQQQ"
    elif market_option == "S&P 500 (SPY)": IDX, LEV_LONG, LEV_SHORT = "SPY", "SSO/UPRO", "SDS/SPXU"
    else: IDX, LEV_LONG, LEV_SHORT = "^KS11", "KODEX Leverage", "KODEX 200 Inverse"
        
    latest_data, prev_data = load_latest_analysis(market_option)

    # [수정] 레이아웃 재구성: 상단 2단 컬럼 (왼쪽: 시그널, 오른쪽: 뉴스)
    col_signal, col_news = st.columns([1.2, 1]) # 비율 조정 (시그널 쪽을 조금 더 넓게)

    if latest_data:
        date_str = convert_utc_to_kst(latest_data['created_at'])
        
        # 데이터 추출
        f_up = latest_data.get('fin_prob_up', 0.0)
        f_down = latest_data.get('fin_prob_down', 0.0)
        f_neutral = latest_data.get('fin_prob_neutral', 0.0)
        t_up = latest_data.get('tech_prob_up', 0.0)
        t_down = latest_data.get('tech_prob_down', 0.0)
        t_neutral = latest_data.get('tech_prob_neutral', 0.0)
        
        # =========================================================
        # [상단 왼쪽] Final Ensemble Probabilities & Signal
        # =========================================================
        with col_signal:
            st.markdown(f"**Analysis Time:** {date_str}")
            st.markdown("##### 🎯 Final Ensemble Probabilities")
            st.caption("The delta values (Δ) indicate how **News Sentiment** adjusted the Technical Baseline.")
            
            m1, m2, m3 = st.columns(3)
            # (Tooltip 스타일이 CSS로 적용됨)
            m1.metric("Bullish", f"{f_up*100:.1f}%", delta=f"{f_up-t_up:.1%}",
                help=f"🤖 Tech Model: {t_up*100:.1f}%\n📰 News Impact: {f_up-t_up:+.1%}\n━━━━━━━━━━━━━━━\n🎯 Final: {f_up*100:.1f}%")
            m2.metric("Bearish", f"{f_down*100:.1f}%", delta=f"{f_down-t_down:.1%}", delta_color="inverse",
                help=f"🤖 Tech Model: {t_down*100:.1f}%\n📰 News Impact: {f_down-t_down:+.1%}\n━━━━━━━━━━━━━━━\n🎯 Final: {f_down*100:.1f}%")
            m3.metric("Neutral", f"{f_neutral*100:.1f}%", delta=f"{f_neutral-t_neutral:.1%}", delta_color="off",
                help=f"🤖 Tech Model: {t_neutral*100:.1f}%\n📰 News Impact: {f_neutral-t_neutral:+.1%}\n━━━━━━━━━━━━━━━\n🎯 Final: {f_neutral*100:.1f}%")
            
            with st.expander("📊 View Technical Model Baseline", expanded=False):
                st.caption("Raw probabilities from TMFG-LSTM model (Before News adjustment)")
                t1, t2, t3 = st.columns(3)
                t1.markdown(f"**Tech Bull:** `{t_up*100:.1f}%`")
                t2.markdown(f"**Tech Bear:** `{t_down*100:.1f}%`")
                t3.markdown(f"**Tech Neut:** `{t_neutral*100:.1f}%`")

            decision = latest_data.get('action', "HOLD")
            d_color = "#CCCCCC"
            if decision == "BUY": d_color = "#00E396"
            elif decision == "SELL": d_color = "#FF4560"
            
            st.markdown(f"""<div style='margin-top: 20px; padding: 20px; border: 3px solid {d_color}; border-radius: 8px; background-color: #121926; text-align: center;'><span style='color: #FFFFFF; font-size: 1.1rem; font-weight: bold;'>Primary Signal (Weighted)</span><br><span style='color: {d_color}; font-size: 2.5rem; font-weight: 900;'>{decision}</span></div>""", unsafe_allow_html=True)
            
            prev_signal = prev_data['action'] if prev_data else None
            strategy_text = get_strategy_text(market_option, prev_signal, decision)

            with st.expander("View Strategy Details", expanded=True):
                st.write("") 
                st.markdown(f"**Signal Change:** `{prev_signal}` ➜ **`{decision}`**")
                st.markdown(f"""<div style="margin-top: 10px; margin-bottom: 15px; padding: 15px; background-color: #000000; border: 1px solid #7C3AED; border-radius: 4px;"><p style="color: #FFFFFF; font-size: 1rem; line-height: 1.6; margin: 0; font-weight: 600;">{strategy_text}</p></div>""", unsafe_allow_html=True)
                st.markdown(f"""<div style='font-size: 0.9rem; color: #CCCCCC; margin-top: 10px; border-top: 2px solid #555; padding-top: 10px; font-weight: 500;'>* 📈 <b>Long Target:</b> {LEV_LONG}<br>* 📉 <b>Short Target:</b> {LEV_SHORT}</div>""", unsafe_allow_html=True)
                st.write("")
            
            if st.button("Refresh Analysis", use_container_width=True): st.rerun()

        # =========================================================
        # [상단 오른쪽] Global Sentiment & Macro Insights (위치 이동됨)
        # =========================================================
        with col_news:
            st.markdown("##### 📰 Global Sentiment & Macro Insights")
            st.caption("AI-driven analysis of latest market news and economic data.")

            sent_score = latest_data.get('news_sentiment', 0.0)
            sent_label = "Neutral"
            if sent_score > 0.3: sent_label = "Positive"
            elif sent_score < -0.3: sent_label = "Negative"
            
            rel_score = latest_data.get('news_reliability', 0.0)

            nc1, nc2 = st.columns(2)
            with nc1: st.metric("Sentiment Score", f"{sent_score * 100:.1f}%", sent_label)
            with nc2: st.metric("News Reliability", f"{rel_score * 100:.1f}%", help="Based on source credibility and consensus.")
            
            summary = latest_data.get('news_summary', "No summary available.")
            # [수정] 뉴스 요약창 높이 고정 및 스크롤 적용
            st.markdown(f"""
            <div style="background-color: #121926; border: 1px solid #444444; border-radius: 8px; padding: 15px; height: 300px; overflow-y: auto;">
                <p style="color: #E0E0E0; font-size: 0.95rem; line-height: 1.5;">{summary}</p>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.warning("Data syncing... Please wait.")

    # =========================================================
    # [하단] Price Action & Forecast (넓게 배치)
    # =========================================================
    st.divider()
    st.markdown(f"##### 📈 Price Action & Forecast ({IDX})")
    
    try:
        with st.spinner("Fetching market data..."):
            chart_df = yf.download(IDX, period="6mo", progress=False, auto_adjust=True)
        
        if not chart_df.empty:
            if isinstance(chart_df.columns, pd.MultiIndex):
                try: chart_df = chart_df.xs(IDX, axis=1, level=0)
                except: chart_df.columns = chart_df.columns.get_level_values(0)

            if 'Close' in chart_df.columns:
                chart_data = chart_df['Close'].dropna()
                current_price = chart_data.iloc[-1]
                
                if latest_data:
                    stats = MARKET_STATS.get(market_option, MARKET_STATS["S&P 500 (SPY)"])
                    f_up = latest_data.get('fin_prob_up', 0.0)
                    f_down = latest_data.get('fin_prob_down', 0.0)
                    f_neutral = latest_data.get('fin_prob_neutral', 0.0)
                    
                    daily_expected_return = (f_up * stats['bull']) + \
                                            (f_down * stats['bear']) + \
                                            (f_neutral * stats['neut'])
                    
                    future_price_5d = current_price * ((1 + daily_expected_return) ** 5)
                    total_return = (future_price_5d / current_price - 1) * 100
                else:
                    future_price_5d = current_price
                    total_return = 0

                daily_ret = 0
                if len(chart_data) >= 2:
                    daily_ret = (chart_data.iloc[-1] / chart_data.iloc[-2] - 1) * 100

                # [수정] 차트 위 메트릭을 4분할로 넓게 표시
                pc1, pc2, pc3, pc4 = st.columns(4)
                with pc1: st.metric("Current Price", f"{current_price:,.2f}")
                with pc2: st.metric("Daily Return", f"{daily_ret:+.2f}%")
                with pc3: st.metric("AI Target (5d)", f"{future_price_5d:,.2f}")
                with pc4: st.metric("Exp. Return (5d)", f"{total_return:+.2f}%")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data, mode='lines', name='Price', line=dict(color='#2563EB', width=3))) 

                if latest_data:
                    trend_color = '#FFFFFF' 
                    if total_return > 0: trend_color = '#00E396'
                    elif total_return < 0: trend_color = '#FF4560'

                    last_date = chart_data.index[-1]
                    future_dates = [last_date] + [last_date + datetime.timedelta(days=i) for i in range(1, 6)]
                    
                    future_prices = [current_price]
                    for i in range(1, 6):
                            future_prices.append(current_price * ((1 + daily_expected_return) ** i))

                    fig.add_trace(go.Scatter(x=future_dates, y=future_prices, mode='lines', name='Forecast', line=dict(color=trend_color, width=4, dash='dot'))) 

                # [수정] 차트 높이 증가
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=20, b=0), xaxis=dict(showgrid=True, gridcolor='#333', color='#FFFFFF'), yaxis=dict(showgrid=True, gridcolor='#333', color='#FFFFFF'), height=500, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    except Exception as e: st.error(f"Chart Error: {e}")
