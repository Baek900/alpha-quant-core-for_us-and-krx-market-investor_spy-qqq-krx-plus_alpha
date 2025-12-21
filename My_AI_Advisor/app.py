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
# 0. Market Statistics (Historical Daily Avg Return for 5-Day Period)
# ==============================================================================
# Define the probability data (actual daily average return) provided here.
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
# 1. Configuration & Custom CSS (The "Modern Fintech" Look)
# ==============================================================================
st.set_page_config(page_title="Global Asset Advisor", layout="wide", page_icon="G")

# [CSS Injection] Override default Streamlit styles for a professional look
st.markdown("""
    <style>
        /* 1. Import Google Fonts (Inter) for a clean financial look */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #E0E0E0;
        }

        /* 2. Metric Box Design: Styled as Cards */
        div[data-testid="stMetric"] {
            background-color: #1E1E2E; /* Dark Navy Background */
            border: 1px solid #2E2E3E;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        div[data-testid="stMetricLabel"] {
            font-size: 0.8rem !important;
            color: #9CA3AF !important; /* Muted Grey Text */
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
            font-weight: 700 !important;
            color: #FFFFFF !important;
        }

        /* 3. Button Styling */
        div.stButton > button {
            width: 100%;
            background-color: #2563EB; /* Professional Blue */
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            font-weight: 600;
        }
        div.stButton > button:hover {
            background-color: #1D4ED8;
        }

        /* 4. Hide default Streamlit header/footer */
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* 5. Divider Style */
        hr {
            margin-top: 2rem;
            margin-bottom: 2rem;
            border: 0;
            border-top: 1px solid #333;
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

# ==============================================================================
# 2. Helper Functions
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

# ==============================================================================
# 3. Login Dialog Logic
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

# ==============================================================================
# 4. Page Routing Logic
# ==============================================================================

if st.session_state["password_correct"]:
    st.session_state["current_page"] = "Dashboard"

# ------------------------------------------------------------------------------
# PAGE: HOME (Public Landing Page)
# ------------------------------------------------------------------------------
if st.session_state["current_page"] == "Home":
    
    # Header Layout
    col_header, col_login = st.columns([6, 1])
    
    with col_header:
        st.markdown("<h1 style='font-size: 3rem; font-weight: 800; letter-spacing: -1px;'>Global Asset Advisor</h1>", unsafe_allow_html=True)
    
    with col_login:
        if st.button("🔑 Log In", use_container_width=True):
            login_dialog()

    st.markdown("<h3 style='color: #888; font-weight: 400;'>Advanced Financial Forecasting System powered by TMFG-LSTM</h3>", unsafe_allow_html=True)
    st.write("This platform leverages deep learning architectures to analyze global market trends, macroeconomics, and sector rotation, providing institutional-grade insights.")
    
    st.divider()

    # Section 1: Benchmark Performance
    st.markdown("#### Performance Benchmark (YTD)")
    st.caption("Strategy vs. S&P 500 (SPY) | Based on 12-month backtesting data")
    
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

        # Chart (Clean Style)
        fig_bench = go.Figure()
        fig_bench.add_trace(go.Scatter(x=df.index, y=ai_returns, mode='lines', name='Alpha Strategy', line=dict(color='#00E396', width=2)))
        fig_bench.add_trace(go.Scatter(x=df.index, y=market_returns, mode='lines', name='S&P 500', line=dict(color='#4B5563', dash='dot')))
        
        fig_bench.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', # Transparent background
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(showgrid=False, color='#888'),
            yaxis=dict(showgrid=True, gridcolor='#333', color='#888', tickformat='.0%'),
            legend=dict(orientation="h", y=1.1),
            height=350
        )
        
        c1, c2 = st.columns([3, 1])
        with c1:
            st.plotly_chart(fig_bench, use_container_width=True)
        with c2:
            st.metric(label="Total Return", value=f"{final_ai_ret:+.2f}%", delta=f"{alpha:+.2f}% Alpha")
            st.metric(label="Benchmark", value=f"{final_bm_ret:+.2f}%")
            st.caption("Data source: Verified Backtest")

    except Exception as e:
        st.error(f"Data Error: {e}")

    st.divider()
    
    # Section 2: Model Reliability
    st.markdown("#### Model Reliability Metrics")
    st.caption("Validation on unseen test data (2025) | SPY Model")

    col_m1, col_m2, col_m3 = st.columns(3)
    
    with col_m1:
        st.metric(label="Accuracy", value="52.6%", delta="vs Random (33%)")
        st.caption("Consistent edge over random chance.")
        
    with col_m2:
        st.metric(label="Precision (Buy)", value="64.0%", delta="High Confidence")
        st.caption("Minimizes false positives in uptrends.")
        
    with col_m3:
        st.metric(label="Recall (Uptrend)", value="55.0%", delta="Opportunity Capture")
        st.caption("Captures the majority of market rallies.")

    st.divider()

    # Section 3: Architecture
    st.markdown("#### System Architecture")
    
    ac1, ac2 = st.columns(2)
    with ac1:
        st.markdown("**1. TMFG Network**")
        st.caption("Filters market noise to identify structural asset correlations.")
    with ac2:
        st.markdown("**2. LSTM + Attention**")
        st.caption("Captures temporal dependencies and sector rotation dynamics.")

# ------------------------------------------------------------------------------
# PAGE: DASHBOARD (Member Only)
# ------------------------------------------------------------------------------
elif st.session_state["current_page"] == "Dashboard":
    
    # Sidebar
    st.sidebar.markdown("### Member Menu") 
    st.sidebar.markdown(f"User: **{st.session_state.get('logged_in_user', 'Member')}**")
    
    market_option = st.sidebar.selectbox("Select Asset Class", ["NASDAQ (QQQ)", "S&P 500 (SPY)", "KOSPI (Korea)"]) 
    
    st.sidebar.markdown("---")
    if st.sidebar.button("Sign Out"):
        logout()

    # Dashboard Header
    top_col1, top_col2 = st.columns([4, 1])
    
    with top_col1:
        st.markdown(f"## {market_option}") 
        st.caption("Live Market Analysis & Signal Generation")
    
    with top_col2:
         st.markdown(f"<div style='text-align: right; color: #888;'>Status: <span style='color: #00E396;'>● Live</span></div>", unsafe_allow_html=True)

    st.markdown("---")

    # Configuration Map
    if market_option == "NASDAQ (QQQ)":
        IDX_TICKER, LEV_LONG, LEV_SHORT = "QQQ", "QLD (2x) / TQQQ (3x)", "QID (2x) / SQQQ (3x)"
    elif market_option == "S&P 500 (SPY)":
        IDX_TICKER, LEV_LONG, LEV_SHORT = "SPY", "SSO (2x) / UPRO (3x)", "SDS (2x) / SPXU (3x)"
    else: 
        IDX_TICKER, LEV_LONG, LEV_SHORT = "^KS11", "KODEX Leverage", "KODEX 200 Inverse 2X"
        
    # 1. Load Data
    latest_data, prev_data = load_latest_analysis(market_option)

    col1, col2 = st.columns([1, 1.5])

    # =========================================================
    # [Left Column] Signal and Strategy Display
    # =========================================================
    with col1:
        if latest_data:
            date_str = convert_utc_to_kst(latest_data['created_at'])
            
            up_prob = latest_data['final_prob']
            
            # Calculate remaining probabilities automatically if not in DB.
            down_prob = latest_data.get('prob_down', (1.0 - up_prob) * 0.5)
            hold_prob = latest_data.get('prob_neutral', (1.0 - up_prob) * 0.5)
            
            st.markdown(f"**Analysis Time:** {date_str}")
            
            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Bullish", f"{up_prob*100:.1f}%")
            m2.metric("Bearish", f"{down_prob*100:.1f}%") 
            m3.metric("Neutral", f"{hold_prob*100:.1f}%")
            
            # Primary Signal Decision
            decision = "HOLD"
            d_color = "#9CA3AF" # Grey
            if up_prob >= 0.45:
                decision = "BUY"
                d_color = "#00E396" # Green
            elif up_prob <= 0.2:
                decision = "SELL"
                d_color = "#FF4560" # Red
                
            st.markdown(f"""
            <div style='margin-top: 20px; padding: 20px; border: 1px solid {d_color}; border-radius: 8px; background-color: rgba(255,255,255,0.05); text-align: center;'>
                <span style='color: #888; font-size: 0.9rem;'>Primary Signal</span><br>
                <span style='color: {d_color}; font-size: 2rem; font-weight: bold;'>{decision}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Create strategy text
            prev_signal = prev_data['action'] if prev_data else None
            strategy_text = get_strategy_text(prev_signal, decision)

            with st.expander("View Strategy Details", expanded=True):
                st.write("") 
                
                # Signal change indication
                st.markdown(f"**Signal Change:** `{prev_signal if prev_signal else 'INIT'}` ➜ **`{decision}`**")
                
                # Strategy Description Box
                st.markdown(f"""
                <div style="
                    margin-top: 10px;
                    margin-bottom: 15px;
                    padding: 15px;
                    background-color: #2E2E3E; 
                    border-left: 4px solid #7C3AED; 
                    border-radius: 4px;
                ">
                    <p style="
                        color: #E0E0E0; 
                        font-size: 1rem; 
                        line-height: 1.6; 
                        margin: 0;
                        font-weight: 500;
                    ">
                        {strategy_text}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Leverage Information
                st.markdown(f"""
                <div style='font-size: 0.8rem; color: #999; margin-top: 10px; border-top: 1px solid #444; padding-top: 10px;'>
                * 📈 <b>Long Target:</b> {LEV_LONG}<br>
                * 📉 <b>Short Target:</b> {LEV_SHORT}
                </div>
                """, unsafe_allow_html=True)
                
                st.write("")
                
            if st.button("Refresh Analysis"):
                st.rerun()
        else:
            st.warning("Data syncing...")

    # =========================================================
    # [Right Column] Prices and Forecasts (New Weighted Logic)
    # =========================================================
    with col2:
        st.markdown(f"**Price Action & Forecast ({IDX_TICKER})**")
        try:
            with st.spinner("Fetching market data..."):
                chart_df = yf.download(IDX_TICKER, period="6mo", progress=False, auto_adjust=True)
            
            if not chart_df.empty:
                if isinstance(chart_df.columns, pd.MultiIndex):
                    try: chart_df = chart_df.xs(IDX_TICKER, axis=1, level=0)
                    except: chart_df.columns = chart_df.columns.get_level_values(0)

                if 'Close' in chart_df.columns:
                    chart_data = chart_df['Close'].replace(0, np.nan).dropna()
                    current_price = chart_data.iloc[-1]
                    
                    # ------------------------------------------------------------
                    # [NEW] Weighted Average Forecast Logic (Weighted Expected Return)
                    # ------------------------------------------------------------
                    if latest_data:
                        # 1. Get statistics (daily average return) for the currently selected market.
                        # Default is SPY.
                        stats = MARKET_STATS.get(market_option, MARKET_STATS["S&P 500 (SPY)"])
                        
                        # 2. Get model probabilities.
                        p_up = latest_data['final_prob']
                        # Calculate automatically if not in DB.
                        p_down = latest_data.get('prob_down', (1.0 - p_up) * 0.5)
                        p_neutral = latest_data.get('prob_neutral', (1.0 - p_up) * 0.5)

                        # 3. [Core] Calculate Weighted Average Daily Return (Daily Expected Return).
                        daily_expected_move = (p_down * stats['bear']) + \
                                              (p_neutral * stats['neut']) + \
                                              (p_up * stats['bull'])
                        
                        # 4. Calculate price after 5 days (applying compound interest).
                        future_price_5d = current_price * ((1 + daily_expected_move) ** 5)
                        
                        # 5. Total expected return for 5 days.
                        total_return = (future_price_5d / current_price - 1) * 100
                        
                    else:
                        daily_expected_move = 0
                        future_price_5d = current_price
                        total_return = 0

                    # ------------------------------------------------------------
                    # UI Display: Unify height and apply labels
                    # ------------------------------------------------------------
                    # Calculate daily return of Current Price (to balance box height).
                    daily_ret = 0
                    if len(chart_data) >= 2:
                        daily_ret = (chart_data.iloc[-1] / chart_data.iloc[-2] - 1) * 100

                    pc1, pc2 = st.columns(2)
                    with pc1:
                        # Current Price (Height adjustment: add delta).
                        st.metric(
                            label="Current Price", 
                            value=f"{current_price:,.2f}", 
                            delta=f"{daily_ret:+.2f}% (Daily)"
                        )
                    with pc2:
                        # AI Target (5 Days Later)
                        # Value: Forecasted price in 5 days, Delta: Total expected return for 5 days.
                        st.metric(
                            label="AI Target (5 Days Later)", 
                            value=f"{future_price_5d:,.2f}", 
                            delta=f"{total_return:+.2f}% (5d Exp.)"
                        )

                    # Draw chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data, mode='lines', name='Price', line=dict(color='#2563EB', width=2)))

                    if latest_data:
                        trend_color = '#9CA3AF'
                        if total_return > 0: trend_color = '#00E396'
                        elif total_return < 0: trend_color = '#FF4560'

                        last_date = chart_data.index[-1]
                        future_dates = [last_date] + [last_date + datetime.timedelta(days=i) for i in range(1, 6)]
                        
                        # Calculate chart plotting data based on daily_expected_move using compound interest.
                        future_prices = [current_price * ((1 + daily_expected_move) ** i) for i in range(0, 6)]
                        
                        fig.add_trace(go.Scatter(x=future_dates, y=future_prices, mode='lines', name='Forecast', line=dict(color=trend_color, width=3, dash='dot')))

                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=0, r=0, t=10, b=0),
                        xaxis=dict(showgrid=False, color='#888'),
                        yaxis=dict(showgrid=True, gridcolor='#333', color='#888'),
                        height=350,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Chart Error: {e}")
            
    # News Section
    st.markdown("---")
    st.markdown("**Global Sentiment & Macro Insights**")
    if latest_data:
        nc1, nc2 = st.columns([1, 3])
        with nc1:
            st.metric("Sentiment Score", f"{latest_data['news_score']}", delta="Neutral")
        with nc2:
            st.info("Live News Aggregation: System is processing global financial feeds...")
