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
from strategy_logic import get_strategy_text

# ==============================================================================
# 0. Market Statistics 
# ==============================================================================
MARKET_STATS = {
    "S&P 500 (SPY)": {"bear": -0.005482, "neut": 0.000151, "bull": 0.004402},
    "NASDAQ (QQQ)": {"bear": -0.006119, "neut": 0.000125, "bull": 0.005435},
    "KOSPI (Korea)": {"bear": -0.002751, "neut": -0.001029, "bull": 0.002401}
}

# ==============================================================================
# 1. Configuration & Custom CSS
# ==============================================================================
st.set_page_config(page_title="TITAN FLOW - AI Advisor", layout="wide", page_icon="T")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #FFFFFF !important; }
        .stApp { background-color: #0B121F !important; }
        div[data-testid="stMetric"] { background-color: #121926 !important; border: 1px solid #444444 !important; border-radius: 8px; }
        div[data-testid="stMetricValue"] { color: #FFFFFF !important; font-weight: 700 !important; }
        div[data-testid="stMetricLabel"] { color: #DDDDDD !important; }
        div[data-testid="stMarkdownContainer"] p { color: #E0E0E0 !important; }
    </style>
""", unsafe_allow_html=True)

# Session State
if "password_correct" not in st.session_state: st.session_state["password_correct"] = False
if "current_page" not in st.session_state: st.session_state["current_page"] = "Home"

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
    except: return utc_str

def load_latest_analysis(market_name):
    try:
        response = supabase.table("prediction_logs") \
            .select("*").eq("market_name", market_name) \
            .order("created_at", desc=True).limit(2).execute()
        if response.data:
            return response.data[0], (response.data[1] if len(response.data) > 1 else None)
        else: return None, None
    except: return None, None

def logout():
    st.session_state["password_correct"] = False
    st.session_state["current_page"] = "Home"
    st.rerun()

# Login Dialog
@st.dialog("Member Login") 
def login_dialog():
    st.write("Please enter your credentials.")
    username = st.text_input("Username") 
    password = st.text_input("Password", type="password")
    if st.button("Access Dashboard"): 
        if username in st.secrets["users"] and password == st.secrets["users"][username]:
            st.session_state["password_correct"] = True
            st.session_state["current_page"] = "Dashboard"
            st.rerun()
        else: st.error("Invalid credentials.")

if st.session_state["password_correct"]: st.session_state["current_page"] = "Dashboard"

# ------------------------------------------------------------------------------
# PAGE: HOME
# ------------------------------------------------------------------------------
if st.session_state["current_page"] == "Home":
    col_header, col_login = st.columns([6, 1])
    with col_header: st.markdown("<h1 style='font-size: 3rem;'>TITAN FLOW</h1>", unsafe_allow_html=True)
    with col_login:
        if st.button("🔑 Log In", use_container_width=True): login_dialog()

    st.markdown("### Advanced Financial Forecasting System")
    st.divider()
    
    # (Home page chart code omitted for brevity - same as before)
    st.info("Please log in to view live signals.")

# ------------------------------------------------------------------------------
# PAGE: DASHBOARD
# ------------------------------------------------------------------------------
elif st.session_state["current_page"] == "Dashboard":
    nav_col1, nav_col2 = st.columns([3, 1])
    with nav_col1:
        market_option = st.selectbox("Select Market", ["NASDAQ (QQQ)", "S&P 500 (SPY)", "KOSPI (Korea)"], label_visibility="collapsed")
    with nav_col2:
        if st.button("Sign Out"): logout()
    
    st.divider()
    st.markdown(f"## {market_option}") 

    if market_option == "NASDAQ (QQQ)": IDX, L_LONG, L_SHORT = "QQQ", "QLD/TQQQ", "QID/SQQQ"
    elif market_option == "S&P 500 (SPY)": IDX, L_LONG, L_SHORT = "SPY", "SSO/UPRO", "SDS/SPXU"
    else: IDX, L_LONG, L_SHORT = "^KS11", "KODEX Leverage", "KODEX 200 Inverse"
        
    latest_data, prev_data = load_latest_analysis(market_option)

    col1, col2 = st.columns([1, 1.5])

    # [LEFT] Signal
    with col1:
        if latest_data:
            date_str = convert_utc_to_kst(latest_data['created_at'])
            
            # 앙상블 완료된 최종 확률들
            # DB 필드명: tech_prob -> Up, prob_down -> Down, prob_neutral -> Neutral
            p_up = latest_data.get('tech_prob', 0.0)
            p_down = latest_data.get('prob_down', 0.0)
            p_neutral = latest_data.get('prob_neutral', 0.0)
            
            # Action (DB에서 결정된 값)
            action = latest_data.get('action', 'HOLD')
            
            st.markdown(f"**Analysis Time:** {date_str}")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Bullish", f"{p_up*100:.1f}%")
            m2.metric("Bearish", f"{p_down*100:.1f}%") 
            m3.metric("Neutral", f"{p_neutral*100:.1f}%")
            
            d_color = "#CCCCCC"
            if action == "BUY": d_color = "#00E396"
            elif action == "SELL": d_color = "#FF4560"
            
            st.markdown(f"""
            <div style='margin-top: 20px; padding: 20px; border: 3px solid {d_color}; border-radius: 8px; background-color: #121926; text-align: center;'>
                <span style='color: #FFFFFF; font-size: 1.1rem; font-weight: bold;'>Primary Signal</span><br>
                <span style='color: {d_color}; font-size: 2.5rem; font-weight: 900;'>{action}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Strategy Details
            prev_signal = prev_data['action'] if prev_data else None
            strategy_text = get_strategy_text(prev_signal, action)
            
            with st.expander("View Strategy Details", expanded=True):
                st.markdown(f"**Signal Change:** `{prev_signal}` ➜ **`{action}`**")
                st.info(strategy_text)
                st.markdown(f"* 📈 **Long:** {L_LONG}\n* 📉 **Short:** {L_SHORT}")

            if st.button("Refresh"): st.rerun()

    # [RIGHT] Chart & Forecast
    with col2:
        try:
            chart_df = yf.download(IDX, period="6mo", progress=False, auto_adjust=True)
            if not chart_df.empty:
                if isinstance(chart_df.columns, pd.MultiIndex): chart_df = chart_df.xs(IDX, axis=1, level=0)
                chart_data = chart_df['Close'].dropna()
                current_price = chart_data.iloc[-1]
                
                # 강도 계산 (Action 기반 단순화)
                stats = MARKET_STATS.get(market_option, MARKET_STATS["S&P 500 (SPY)"])
                expected_return = 0
                if action == "BUY": expected_return = p_up * stats['bull'] * 2
                elif action == "SELL": expected_return = p_down * stats['bear'] * 2
                
                future_val = current_price * ((1 + expected_return) ** 5)
                ret_pct = (future_val / current_price - 1) * 100
                
                c_m1, c_m2 = st.columns(2)
                c_m1.metric("Current Price", f"{current_price:,.2f}")
                c_m2.metric("AI Target (5d)", f"{future_val:,.2f}", f"{ret_pct:+.2f}%")
                
                # Chart drawing (Simple Line)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data, line=dict(color='#2563EB', width=2)))
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e: st.error(f"Chart Error: {e}")

    # [BOTTOM] News & Sentiment
    st.markdown("---")
    st.markdown("**Global Sentiment & Macro Insights**")
    if latest_data:
        nc1, nc2 = st.columns([1, 3])
        with nc1:
            # [요청사항 반영] 소수점 4자리 % 표시
            sent_val = latest_data.get('news_sentiment', 0.0) # -1 ~ 1
            rel_val = latest_data.get('news_reliability', 0.0) # 0 ~ 1
            
            # 감정 점수 라벨링
            s_label = "Neutral"
            if sent_val > 0.3: s_label = "Positive"
            elif sent_val < -0.3: s_label = "Negative"
            
            # 감정을 %로 변환해서 보여줌 (Scale: -100% ~ 100%)
            st.metric("Sentiment Score", f"{sent_val * 100:.4f}%", s_label)
            st.caption(f"Reliability: {rel_val * 100:.4f}%")
            
        with nc2:
            summary = latest_data.get('news_summary', "No summary available.")
            st.info(f"📰 **Market Summary (English):**\n\n{summary}")
