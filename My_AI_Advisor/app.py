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
# 0. Configuration & Global Constants
# ==============================================================================
st.set_page_config(page_title="TITAN FLOW - AI Advisor", layout="wide", page_icon="T")

# 시장별 통계 데이터
# 시장별 통계 데이터
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

# 챌린지 시뮬레이션용 자산 정보
CHALLENGE_ASSETS = {
    "NASDAQ (QQQ)": {"ticker": "QQQ", "lev_mult": 3},   # 미국 전략: 3배
    "S&P 500 (SPY)": {"ticker": "SPY", "lev_mult": 3},   # 미국 전략: 3배
    "KOSPI (Korea)": {"ticker": "^KS11", "lev_mult": 2}  # 한국 전략: 2배
}

# [CSS 스타일링]
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');
        
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #FFFFFF !important; }
        .stApp { background-color: #0B121F !important; }
        h1, h2, h3, h4, h5, h6 { color: #FFFFFF !important; font-weight: 700 !important; }
        div[data-testid="stCaptionContainer"], small, .stCaption { color: #CCCCCC !important; font-size: 0.9rem !important; opacity: 1 !important; }
        p { color: #E0E0E0 !important; font-size: 1rem !important; }
        
        header {visibility: hidden !important;}
        footer {visibility: hidden !important;}
        [data-testid="stSidebar"] {display: none !important;}

        div[data-testid="stMetric"] { background-color: #121926 !important; border: 1px solid #444444 !important; padding: 15px; border-radius: 8px; }
        div[data-testid="stMetricLabel"] { color: #DDDDDD !important; font-weight: 500 !important; }
        div[data-testid="stMetricValue"] { color: #FFFFFF !important; font-weight: 700 !important; }

        div.stButton > button { width: 100%; background-color: #2563EB; color: white !important; border: 1px solid #60A5FA; border-radius: 6px; font-weight: 600; }

        div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background-color: #121926 !important; color: white !important; border: 1px solid #555555 !important; }
        ul[data-testid="stSelectboxVirtualDropdown"] { background-color: #121926 !important; }
        li[role="option"] { color: white !important; }
        
        div[data-baseweb="radio"] > div { color: white !important; }
        
        .streamlit-expanderHeader { background-color: #121926 !important; color: #FFFFFF !important; }
        div[data-testid="stExpanderDetails"] { background-color: #0B121F !important; border: 1px solid #444444; color: #E0E0E0 !important; }
        div[data-testid="stExpanderDetails"] p, div[data-testid="stExpanderDetails"] span { color: #E0E0E0 !important; }

        button[data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
        
        div[role="dialog"] { background-color: #121926 !important; border: 2px solid #6B7280 !important; color: #FFFFFF !important; }
        div[role="dialog"] h2, div[role="dialog"] p, div[role="dialog"] label { color: #FFFFFF !important; }
        div[role="dialog"] button[aria-label="Close"] { color: #FFFFFF !important; }
    </style>
""", unsafe_allow_html=True)

# Session Init
if "password_correct" not in st.session_state: st.session_state["password_correct"] = False
if "current_page" not in st.session_state: st.session_state["current_page"] = "Home"

# Supabase Init & Test Mode Check
try:
    SUPABASE_URL = st.secrets["supabase"]["url"]
    SUPABASE_KEY = st.secrets["supabase"]["key"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    IS_TEST_MODE = st.secrets.get("test_mode", False)
    TABLE_PREDS = "prediction_logs_test" if IS_TEST_MODE else "prediction_logs"
    TABLE_REFS = "news_reference_logs_test" if IS_TEST_MODE else "news_reference_logs"
    
    if IS_TEST_MODE:
        st.toast("⚠️ Test Mode Activated: Using `_test` tables.")

except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# ==============================================================================
# Helper Functions
# ==============================================================================
def convert_utc_to_kst(utc_str):
    try:
        utc_time = parser.parse(utc_str)
        kst_zone = pytz.timezone('Asia/Seoul')
        kst_time = utc_time.astimezone(kst_zone)
        return kst_time.strftime('%Y-%m-%d %H:%M')
    except:
        return utc_str

def load_latest_analysis(market_name):
    """라이브 분석용 최신 데이터 로드 (Signal + Reference)"""
    try:
        res_pred = supabase.table(TABLE_PREDS) \
            .select("*") \
            .eq("market_name", market_name) \
            .order("created_at", desc=True) \
            .limit(2) \
            .execute()
            
        res_ref = supabase.table(TABLE_REFS) \
            .select("*") \
            .eq("market_name", market_name) \
            .order("created_at", desc=True) \
            .limit(1) \
            .execute()
        
        current_data = res_pred.data[0] if res_pred.data else None
        previous_data = res_pred.data[1] if len(res_pred.data) > 1 else None
        ref_data = res_ref.data[0] if res_ref.data else None
        
        return current_data, previous_data, ref_data
    except:
        return None, None, None

def load_all_predictions(market_name):
    """챌린지용 전체 로그 로드"""
    try:
        response = supabase.table(TABLE_PREDS) \
            .select("*") \
            .eq("market_name", market_name) \
            .order("created_at", desc=True) \
            .execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            df['created_at'] = pd.to_datetime(df['created_at'])
            df['date_only'] = df['created_at'].dt.date
            df = df.sort_values('created_at').drop_duplicates('date_only', keep='last')
            return df
        return pd.DataFrame()
    except:
        return pd.DataFrame()

def logout():
    st.session_state["password_correct"] = False
    st.session_state["logged_in_user"] = None
    st.session_state["current_page"] = "Home"
    st.rerun()

def get_strategy_text(market_name, prev_signal, current_signal):
    if "NASDAQ" in market_name:
        target_long, target_short = "TQQQ (or QLD)", "SQQQ (or QID)"
    elif "S&P 500" in market_name:
        target_long, target_short = "UPRO (or SSO)", "SPXU (or SDS)"
    else: 
        target_long, target_short = "KODEX Leverage", "KODEX 200 Inverse"

    if current_signal == "BUY":
        return f"**Bullish Trend Detected.** The AI model probabilities indicate a structural uptrend. It is recommended to accumulate **{target_long}**."
    elif current_signal == "SELL":
        return f"**Bearish Risk Detected.** Market probabilities have shifted downward. It is advised to liquidate long positions and hedge downside risk using **{target_short}** or hold Cash."
    else:
        return "**Neutral / Uncertain Market.** No clear directional signal is present. It is advised to **Hold Cash** and wait for a breakout confirmation before entering new positions."

# [핵심 로직] 정수 단위 + 인버스 매매 시뮬레이션 엔진
def run_simulation(price_df, df_logs, init_cap, strategy_mode, lev_mult=1):
    """
    Args:
        lev_mult: 레버리지 배수 (1=1배/인버스1배, 3=3배/인버스3배)
    """
    
    dates = sorted(price_df.index)
    
    # 1. 등락률 안전하게 추출
    pct_changes_raw = price_df['pct_change']
    if isinstance(pct_changes_raw, pd.DataFrame):
        pct_changes = pct_changes_raw.iloc[:, 0].values
    else:
        pct_changes = pct_changes_raw.values
    
    # 2. 가상 자산 가격 생성 (Long & Short)
    # Long: 기초지수 * +lev_mult
    # Short: 기초지수 * -lev_mult (인버스)
    sim_long = [100.0]
    sim_short = [100.0]
    
    long_chg = pct_changes * lev_mult
    short_chg = pct_changes * (-lev_mult)
    
    for i in range(1, len(dates)):
        l_next = sim_long[-1] * (1 + long_chg[i])
        s_next = sim_short[-1] * (1 + short_chg[i])
        sim_long.append(max(0.01, l_next)) # 0이하 방지
        sim_short.append(max(0.01, s_next))
        
    sim_long_df = pd.DataFrame({'Close': sim_long}, index=dates)
    sim_short_df = pd.DataFrame({'Close': sim_short}, index=dates)
    
    # 3. 매매 시뮬레이션 Loop
    cash = float(init_cap)
    shares_long = 0
    shares_short = 0
    portfolio_value = []
    
    for i, date in enumerate(dates):
        p_long = sim_long_df['Close'].iloc[i]
        p_short = sim_short_df['Close'].iloc[i]
        
        # 신호 확인
        past_logs = df_logs[df_logs['date_only'] < date]
        signal = "HOLD"
        if not past_logs.empty:
            last_log = past_logs.iloc[-1]
            signal = last_log['action'] 
        
        # 매매 로직
        if signal == "BUY":
            # 1) Short 포지션 청산 (보유 시)
            if shares_short > 0:
                cash += shares_short * p_short
                shares_short = 0
            
            # 2) Long 포지션 진입
            if cash > p_long:
                amt = cash if strategy_mode == "Full Switching" else cash * 0.5
                n = int(amt // p_long)
                if n > 0:
                    cost = n * p_long
                    cash -= cost
                    shares_long += n
                    
        elif signal == "SELL":
            # 1) Long 포지션 청산 (보유 시)
            if shares_long > 0:
                cash += shares_long * p_long
                shares_long = 0
                
            # 2) Short(인버스) 포지션 진입
            if cash > p_short:
                amt = cash if strategy_mode == "Full Switching" else cash * 0.5
                n = int(amt // p_short)
                if n > 0:
                    cost = n * p_short
                    cash -= cost
                    shares_short += n
        
        # HOLD: 현 포지션 유지
        
        # 평가금 합산
        total_val = cash + (shares_long * p_long) + (shares_short * p_short)
        portfolio_value.append(total_val)
        
    return portfolio_value, dates


# ==============================================================================
# Auth & Dialogs
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

# ==============================================================================
# PAGE: HOME (복구됨)
# ==============================================================================
if st.session_state["current_page"] == "Home":
    col_header, col_login = st.columns([6, 1])
    with col_header:
        st.markdown("<h1 style='font-size: 3rem; margin-bottom: 0;'>TITAN FLOW</h1>", unsafe_allow_html=True)
    with col_login:
        if st.button("🔑 Log In", use_container_width=True):
            login_dialog()

    st.markdown("<h3 style='color: #FFFFFF; font-weight: 500; margin-top: 10px;'>Advanced Financial Forecasting System powered by TMFG-LSTM</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #E0E0E0; font-size: 1.1rem;'>This platform leverages deep learning architectures to analyze global market trends, macroeconomics, and sector rotation, providing institutional-grade insights.</p>", unsafe_allow_html=True)
    
    st.divider()

    st.markdown("#### Performance Benchmark (YTD)")
    st.caption("Strategy vs. S&P 500 (SPY) | Based on 12-month backtesting data")
    
    # [복구] 홈 화면 벤치마크 차트
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
    # [복구] 모델 신뢰도 지표
    st.markdown("#### Model Reliability Metrics")
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1: st.metric("Accuracy", "52.6%", "vs Random (33%)")
    with col_m2: st.metric("Precision (Buy)", "64.0%", "High Confidence")
    with col_m3: st.metric("Recall (Uptrend)", "55.0%", "Opportunity Capture")

# ==============================================================================
# PAGE: DASHBOARD (Main Logic)
# ==============================================================================
elif st.session_state["current_page"] == "Dashboard":
    
    nav_col1, nav_col2 = st.columns([3, 1])
    with nav_col1:
        market_option = st.selectbox("Select Market", ["NASDAQ (QQQ)", "S&P 500 (SPY)", "KOSPI (Korea)"], label_visibility="collapsed")
    with nav_col2:
        if st.button("Sign Out"): logout()
    
    st.divider()

    tab1, tab2 = st.tabs(["📊 Live Analysis", "🏆 Challenge"])

    # --------------------------------------------------------------------------
    # TAB 1: Live Analysis
    # --------------------------------------------------------------------------
    with tab1:
        top_col1, top_col2 = st.columns([4, 1])
        with top_col1:
            st.markdown(f"## {market_option}") 
            st.caption("Live Market Analysis & Signal Generation")
        with top_col2:
             st.markdown(f"<div style='text-align: right; color: #FFFFFF; font-weight: bold;'>Status: <span style='color: #00E396;'>● Live</span></div>", unsafe_allow_html=True)

        if market_option == "NASDAQ (QQQ)": IDX, LEV_LONG, LEV_SHORT = "QQQ", "QLD/TQQQ", "QID/SQQQ"
        elif market_option == "S&P 500 (SPY)": IDX, LEV_LONG, LEV_SHORT = "SPY", "SSO/UPRO", "SDS/SPXU"
        else: IDX, LEV_LONG, LEV_SHORT = "^KS11", "KODEX Leverage", "KODEX 200 Inverse"
            
        latest_data, prev_data, ref_data = load_latest_analysis(market_option)

        col1, col2 = st.columns([1, 1.5])

        # [LEFT] Signal & Reference News
        with col1:
            if latest_data:
                date_str = convert_utc_to_kst(latest_data['created_at'])
                
                f_up = latest_data.get('fin_prob_up', 0.0)
                f_down = latest_data.get('fin_prob_down', 0.0)
                f_neutral = latest_data.get('fin_prob_neutral', 0.0)
                
                t_up = latest_data.get('tech_prob_up', 0.0)
                t_down = latest_data.get('tech_prob_down', 0.0)
                t_neutral = latest_data.get('tech_prob_neutral', 0.0)

                st.markdown(f"**Analysis Time:** {date_str}")
                
                st.markdown("##### 🎯 Final Ensemble Probabilities")
                st.caption("The delta values (Δ) indicate how **News Sentiment** adjusted the Technical Baseline.")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Bullish", f"{f_up*100:.1f}%", delta=f"{f_up-t_up:.1%}", help=f"🤖 Tech Model: {t_up*100:.1f}%\n📰 News Impact: {f_up-t_up:+.1%}\n━━━━━━━━━━━━━━━\n🎯 Final: {f_up*100:.1f}%")
                m2.metric("Bearish", f"{f_down*100:.1f}%", delta=f"{f_down-t_down:.1%}", delta_color="inverse", help=f"🤖 Tech Model: {t_down*100:.1f}%\n📰 News Impact: {f_down-t_down:+.1%}\n━━━━━━━━━━━━━━━\n🎯 Final: {f_down*100:.1f}%")
                m3.metric("Neutral", f"{f_neutral*100:.1f}%", delta=f"{f_neutral-t_neutral:.1%}", delta_color="off", help=f"🤖 Tech Model: {t_neutral*100:.1f}%\n📰 News Impact: {f_neutral-t_neutral:+.1%}\n━━━━━━━━━━━━━━━\n🎯 Final: {f_neutral*100:.1f}%")
                
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
                
                if st.button("Refresh Analysis"): st.rerun()

                # [Reference News Section]
                st.write("")
                st.markdown("##### 📡 Market Monitoring (Reference)")
                if ref_data:
                    ref_time = convert_utc_to_kst(ref_data['created_at'])
                    with st.expander(f"Last Update: {ref_time}", expanded=True):
                        st.markdown(f"**Summary:** {ref_data.get('reference_summary', '-')}")
                        st.caption(f"**Risk Analysis:** {ref_data.get('detected_risks', '-')}")
                        st.markdown(f"**Risk Level:** `{ref_data.get('risk_level', 0.0)}`")
                else:
                    st.info("No monitoring data available.")

            else:
                st.warning("Data syncing...")

        # [RIGHT] Chart & Forecast & News
        with col2:
            st.write("")
            st.write("")
            st.write("") 
            st.write("") 
            st.write("") 
            st.write("") 

            # [복구] Price Forecast 차트
            st.markdown(f"**Price Action & Forecast ({IDX})**")
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

                        pc1, pc2 = st.columns(2)
                        with pc1: st.metric("Current Price", f"{current_price:,.2f}", f"{daily_ret:+.2f}% (Daily)")
                        with pc2: st.metric("AI Target (5 Days Later)", f"{future_price_5d:,.2f}", f"{total_return:+.2f}% (5d Exp.)")

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

                        fig.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)', 
                            plot_bgcolor='rgba(0,0,0,0)', 
                            margin=dict(l=0, r=0, t=10, b=0), 
                            xaxis=dict(showgrid=True, gridcolor='#333', color='#FFFFFF', fixedrange=True), 
                            yaxis=dict(showgrid=True, gridcolor='#333', color='#FFFFFF', fixedrange=True), 
                            height=350, 
                            showlegend=False,
                            dragmode=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False})

            except Exception as e: st.error(f"Chart Error: {e}")
                
            st.write("")
            st.markdown("##### Global Sentiment & Macro Insights")
            
            if latest_data:
                nc1, nc2 = st.columns([1, 2.5]) 
                with nc1:
                    sent_score = latest_data.get('news_sentiment', 0.0)
                    sent_label = "Neutral"
                    if sent_score > 0.3: sent_label = "Positive"
                    elif sent_score < -0.3: sent_label = "Negative"

                    st.metric("Sentiment Score", f"{sent_score * 100:.1f}%", sent_label)
                    
                    rel_score = latest_data.get('news_reliability', 0.0)
                    st.caption(f"News Reliability: {rel_score * 100:.1f}%")
                    
                with nc2:
                    summary = latest_data.get('news_summary', "No summary available.")
                    st.info(f"📰 **Market Summary (English):**\n\n{summary}")

    # --------------------------------------------------------------------------
    # TAB 2: Challenge
    # --------------------------------------------------------------------------
    with tab2:
        if "Korea" in market_option:
            init_cap = 5000000
            currency = "₩"
        else:
            init_cap = 5000
            currency = "$"
            
        st.subheader(f"🏆 {currency}{init_cap:,.0f} Challenge ({market_option})")
        st.caption(f"Real-time simulation with **Integer Shares** & **Cash Balance** tracking. (Start: **2025-12-22**)")

        st_col1, st_col2 = st.columns([1, 3])
        with st_col1:
            strategy_mode = st.radio(
                "Position Sizing Strategy",
                ["Full Switching", "Gradual Accumulation"],
                help="Full Switching: Buy 100% on Signal.\nGradual: Buy 50% of available cash on repeated Buy signals."
            )

        # 1. 로그 데이터 로드
        df_logs = load_all_predictions(market_option)
        target_start_date = datetime.date(2025, 12, 22)
        
        if not df_logs.empty:
            df_logs = df_logs[df_logs['date_only'] >= target_start_date]

        if df_logs.empty:
            st.warning("⚠️ No data available from 2025-12-22. Please check your DB.")
        else:
            # 2. 주가 데이터 로드 (Index 1x)
            try:
                asset_info = CHALLENGE_ASSETS[market_option]
                ticker = asset_info['ticker']
                lev_mult = asset_info['lev_mult']
                
                start_date = target_start_date
                end_date = datetime.date.today()
                
                with st.spinner(f"Simulating... (Strategy: {strategy_mode})"):
                    price_df = yf.download(ticker, start=start_date, end=end_date + datetime.timedelta(days=1), progress=False, auto_adjust=True)
                    
                    if price_df.empty:
                        st.warning(f"⚠️ Market data not available for {ticker}.")
                    else:
                        if isinstance(price_df.columns, pd.MultiIndex):
                             try: price_df = price_df.xs(ticker, axis=1, level=0)
                             except: pass 
                        
                        price_df = price_df[['Close']].copy()
                        price_df['pct_change'] = price_df['Close'].pct_change().fillna(0)
                        price_df.index = price_df.index.date
                        price_df = price_df[~price_df.index.duplicated(keep='last')] 

                        # [Fix: Ensure Series for benchmark calculation]
                        closes = price_df['Close']
                        if isinstance(closes, pd.DataFrame):
                            closes = closes.iloc[:, 0]
                        
                        # (A) Benchmark (Buy & Hold 1x) - Start 100% Allocation
                        bm_start_price = float(closes.iloc[0])
                        bm_shares = int(init_cap // bm_start_price)
                        bm_cash = init_cap - (bm_shares * bm_start_price)
                        # DataFrame/Series calculation to list -> Safe conversion
                        bench_curve = (closes * bm_shares + bm_cash).values.flatten().tolist()
                        
                        # (B) Strategies
                        df_tech = df_logs.copy()
                        def get_tech_action(row):
                            probs = {"SELL": row['tech_prob_down'], "HOLD": row['tech_prob_neutral'], "BUY": row['tech_prob_up']}
                            act = max(probs, key=probs.get)
                            return "HOLD" if probs[act] <= 0.45 else act
                        df_tech['action'] = df_tech.apply(get_tech_action, axis=1)

                        eq_ens_lev, plot_dates = run_simulation(price_df, df_logs, init_cap, strategy_mode, lev_mult=lev_mult)
                        eq_tech_lev, _ = run_simulation(price_df, df_tech, init_cap, strategy_mode, lev_mult=lev_mult)
                        
                        eq_ens_1x, _ = run_simulation(price_df, df_logs, init_cap, strategy_mode, lev_mult=1)
                        eq_tech_1x, _ = run_simulation(price_df, df_tech, init_cap, strategy_mode, lev_mult=1)

                        # --- CHART 1: Leverage Strategy ---
                        bm_name = "Index 1x (Buy&Hold)"
                        st.markdown(f"##### 1. Leverage Strategy ({lev_mult}x) vs {bm_name}")
                        
                        ret_el = (eq_ens_lev[-1] / init_cap - 1) * 100
                        ret_tl = (eq_tech_lev[-1] / init_cap - 1) * 100
                        ret_b1 = (bench_curve[-1] / init_cap - 1) * 100
                        
                        fig_lev = go.Figure()
                        fig_lev.add_trace(go.Scatter(x=plot_dates, y=eq_ens_lev, mode='lines', name=f'AI Ens ({lev_mult}x) {ret_el:+.1f}%', line=dict(color='#00E396', width=3)))
                        fig_lev.add_trace(go.Scatter(x=plot_dates, y=eq_tech_lev, mode='lines', name=f'Tech ({lev_mult}x) {ret_tl:+.1f}%', line=dict(color='#FEB019', width=2, dash='dash')))
                        fig_lev.add_trace(go.Scatter(x=plot_dates, y=bench_curve, mode='lines', name=f'{bm_name} {ret_b1:+.1f}%', line=dict(color='#FFFFFF', width=1, dash='dot')))
                        fig_lev.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(showgrid=True, gridcolor='#333'), yaxis=dict(showgrid=True, gridcolor='#333', tickprefix=currency), legend=dict(orientation="h", y=1.1), height=400)
                        st.plotly_chart(fig_lev, use_container_width=True)
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("AI Ensemble", f"{currency}{eq_ens_lev[-1]:,.0f}", f"{ret_el:+.2f}%")
                        m2.metric("Tech Only", f"{currency}{eq_tech_lev[-1]:,.0f}", f"{ret_tl:+.2f}%")
                        m3.metric("Buy & Hold", f"{currency}{bench_curve[-1]:,.0f}", f"{ret_b1:+.2f}%")

                        # --- CHART 2: No-Leverage Strategy ---
                        st.divider()
                        st.markdown(f"##### 2. Non-Leveraged Strategy (1x) vs {bm_name}")
                        
                        ret_e1 = (eq_ens_1x[-1] / init_cap - 1) * 100
                        ret_t1 = (eq_tech_1x[-1] / init_cap - 1) * 100
                        
                        fig_no = go.Figure()
                        fig_no.add_trace(go.Scatter(x=plot_dates, y=eq_ens_1x, mode='lines', name=f'AI Ens (1x) {ret_e1:+.1f}%', line=dict(color='#00E396', width=3)))
                        fig_no.add_trace(go.Scatter(x=plot_dates, y=eq_tech_1x, mode='lines', name=f'Tech (1x) {ret_t1:+.1f}%', line=dict(color='#FEB019', width=2, dash='dash')))
                        fig_no.add_trace(go.Scatter(x=plot_dates, y=bench_curve, mode='lines', name=f'{bm_name} {ret_b1:+.1f}%', line=dict(color='#FFFFFF', width=1, dash='dot')))
                        fig_no.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(showgrid=True, gridcolor='#333'), yaxis=dict(showgrid=True, gridcolor='#333', tickprefix=currency), legend=dict(orientation="h", y=1.1), height=400)
                        st.plotly_chart(fig_no, use_container_width=True)
                        
                        mm1, mm2, mm3 = st.columns(3)
                        mm1.metric("AI Ensemble (1x)", f"{currency}{eq_ens_1x[-1]:,.0f}", f"{ret_e1:+.2f}%")
                        mm2.metric("Tech Only (1x)", f"{currency}{eq_tech_1x[-1]:,.0f}", f"{ret_t1:+.2f}%")
                        mm3.metric("Buy & Hold", f"{currency}{bench_curve[-1]:,.0f}", f"{ret_b1:+.2f}%")

            except Exception as e:
                st.error(f"Simulation Failed: {e}")
