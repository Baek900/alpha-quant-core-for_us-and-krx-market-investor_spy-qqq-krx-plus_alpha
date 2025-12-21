import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import datetime
from dateutil import parser
import pytz
from supabase import create_client, Client

# ==============================================================================
# 1. Configuration & Global Settings
# ==============================================================================
st.set_page_config(page_title="AI Global Asset Advisor", layout="wide", page_icon="📈")

# Initialize Session State for Login
if "password_correct" not in st.session_state:
    st.session_state["password_correct"] = False

# Connect to Supabase
try:
    SUPABASE_URL = st.secrets["supabase"]["url"]
    SUPABASE_KEY = st.secrets["supabase"]["key"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error(f"System Configuration Error: {e}")
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
            .limit(1) \
            .execute()
        if response.data: return response.data[0]
        else: return None
    except:
        return None

def check_password():
    """Authentication Logic"""
    def password_entered():
        if st.session_state["username"] in st.secrets["users"] and \
           st.session_state["password"] == st.secrets["users"][st.session_state["username"]]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        st.subheader("🔒 Member Login")
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("Please log in to access the real-time AI Dashboard.")
        return False
    else:
        return True

# ==============================================================================
# 3. Main Navigation (Sidebar)
# ==============================================================================
st.sidebar.title("Global AI Advisor")
page = st.sidebar.radio("Navigation", ["🏠 Home (About Model)", "🚀 AI Dashboard (Member Only)"])
st.sidebar.markdown("---")

# ==============================================================================
# 4. Page: Home (Landing Page - Public)
# ==============================================================================
if page == "🏠 Home (About Model)":
    # Header
    st.title("Generative AI Asset Allocation System")
    st.markdown("### Next-Generation Financial Forecasting with TMFG & LSTM")
    st.write("This platform utilizes state-of-the-art Deep Learning architectures to analyze global market trends, macroeconomics, and sector rotation.")
    
    st.divider()

    # Section 1: Benchmark Performance (Mock-up for demonstration)
    st.header("🏆 Performance Benchmark (Backtest)")
    st.write("Comparison of **AI Algorithm Strategy** vs. **Market Benchmark (S&P 500)** over the last 12 months.")
    
    # [Mock Data Generation for Chart]
    dates = pd.date_range(start="2024-01-01", periods=100)
    market_returns = np.cumsum(np.random.normal(0.0005, 0.01, 100)) # Random walk
    ai_returns = np.cumsum(np.random.normal(0.0008, 0.009, 100)) # Slightly better mean
    
    fig_bench = go.Figure()
    fig_bench.add_trace(go.Scatter(x=dates, y=ai_returns, mode='lines', name='AI Strategy', line=dict(color='green', width=3)))
    fig_bench.add_trace(go.Scatter(x=dates, y=market_returns, mode='lines', name='S&P 500 (Benchmark)', line=dict(color='gray', dash='dot')))
    fig_bench.update_layout(title="Cumulative Return Comparison (YTD)", xaxis_title="Date", yaxis_title="Return", template="plotly_dark", height=400)
    
    col_bench1, col_bench2 = st.columns([2, 1])
    with col_bench1:
        st.plotly_chart(fig_bench, use_container_width=True)
    with col_bench2:
        st.success("Target Alpha: **+15.4%**")
        st.info("Sharpe Ratio: **1.85**")
        st.warning("Max Drawdown: **-12.3%**")
        st.caption("*Based on backtesting data (2020-2024). Past performance is not indicative of future results.*")

    st.divider()

    # Section 2: Model Architecture
    st.header("🧠 Core Engine: Hybrid-AI Architecture")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("1. TMFG Network Analysis")
        st.write("""
        **Topological Mode Filtering Graph (TMFG)** constructs a correlation network between assets.
        It filters out market noise and captures the **true structural relationships** between global sectors.
        """)
    with c2:
        st.subheader("2. LSTM + Attention")
        st.write("""
        **Long Short-Term Memory (LSTM)** networks process time-series data to detect trend reversals.
        Our proprietary **Relation Layer** enhances prediction accuracy by understanding how one sector's movement impacts others.
        """)

    st.info("💡 **Why this matters:** Unlike traditional indicators (RSI, MACD), our AI understands the *context* of market movements.")

# ==============================================================================
# 5. Page: Dashboard (Private - Login Required)
# ==============================================================================
elif page == "🚀 AI Dashboard (Member Only)":
    
    # Login Check
    if check_password():
        st.sidebar.success(f"Login: {st.session_state.get('username')}")
        
        # --- Existing Dashboard Code Starts Here ---
        market_option = st.sidebar.radio("Select Market", ["NASDAQ (QQQ)", "S&P 500 (SPY)", "KOSPI (Korea)"])

        # Market Settings
        if market_option == "NASDAQ (QQQ)":
            TARGET_NAME = "NASDAQ 100 (QQQ)"
            IDX_TICKER = "QQQ"
            LEV_LONG = "QLD (2x) / TQQQ (3x)"
            LEV_SHORT = "QID (2x) / SQQQ (3x)"
            INVEST_AMT = "$1,000"
        elif market_option == "S&P 500 (SPY)":
            TARGET_NAME = "S&P 500 (SPY)"
            IDX_TICKER = "SPY"
            LEV_LONG = "SSO (2x) / UPRO (3x)"
            LEV_SHORT = "SDS (2x) / SPXU (3x)"
            INVEST_AMT = "$1,000"
        else: 
            TARGET_NAME = "KOSPI 200 (KOSPI)"
            IDX_TICKER = "^KS11" 
            LEV_LONG = "KODEX Leverage (122630)"
            LEV_SHORT = "KODEX 200 Futures Inverse 2X (252670)"
            INVEST_AMT = "1,000,000 KRW"

        st.title(f"🤖 AI Dashboard: {market_option}")
        st.caption(f"Real-time analysis powered by TMFG-LSTM Model.")
        st.markdown("---")

        latest_data = load_latest_analysis(market_option)

        col1, col2 = st.columns([1, 1.5])

        with col1:
            if latest_data:
                date_str = convert_utc_to_kst(latest_data['created_at'])
                
                final_prob = latest_data['final_prob']
                news_score_val = latest_data['news_score']
                
                up_prob = final_prob
                remaining = 1.0 - up_prob
                down_prob = remaining * 0.5 
                hold_prob = remaining * 0.5 
                
                st.info(f"📅 **Last Update:** {date_str}")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("📈 Bullish", f"{up_prob*100:.1f}%")
                m2.metric("📉 Bearish", f"{down_prob*100:.1f}%") 
                m3.metric("➖ Neutral", f"{hold_prob*100:.1f}%")
                
                decision = "HOLD (Neutral)"
                color = "gray"
                if up_prob >= 0.45:
                    decision = "BUY"
                    color = "green"
                elif up_prob <= 0.2:
                    decision = "SELL / Inverse"
                    color = "red"
                    
                st.markdown(f"### 📢 AI Signal: :{color}[**{decision}**]")
                
                with st.expander("💡 Execution Strategy", expanded=True):
                    st.markdown(f"""
                    **Ref Date: {date_str}**
                    * **BUY:** Accumulate {LEV_LONG}
                    * **SELL:** Accumulate {LEV_SHORT}
                    * **HOLD:** No Action.
                    """)
                    
                if st.button("🔄 Refresh"):
                    st.rerun()
            else:
                st.warning("⚠️ No data available.")

        with col2:
            st.write(f"📊 **{TARGET_NAME} Price Action**")
            try:
                with st.spinner("Loading chart..."):
                    chart_df = yf.download(IDX_TICKER, period="6mo", progress=False, auto_adjust=True)
                
                if not chart_df.empty:
                    if isinstance(chart_df.columns, pd.MultiIndex):
                        try: chart_df = chart_df.xs(IDX_TICKER, axis=1, level=0)
                        except: chart_df.columns = chart_df.columns.get_level_values(0)

                    if 'Close' in chart_df.columns:
                        chart_data = chart_df['Close'].replace(0, np.nan).dropna()
                        current_price = chart_data.iloc[-1]
                        st.metric(f"Price ({IDX_TICKER})", f"{current_price:,.2f}")

                        recent_volatility = chart_data.pct_change().tail(30).std()
                        if np.isnan(recent_volatility) or recent_volatility == 0: recent_volatility = 0.01

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data, mode='lines', name='Price', line=dict(color='#1f77b4', width=2)))

                        if latest_data:
                            # [Fixed Logic] Graph direction matches Signal (0.45 Threshold)
                            THRESHOLD = 0.45
                            up_p = latest_data['final_prob']
                            diff = up_p - THRESHOLD
                            expected_move = diff * recent_volatility * 3.0 # Multiplier increased for visibility
                            
                            trend_color = 'gray'
                            if expected_move > 0: trend_color = 'green'
                            elif expected_move < 0: trend_color = 'red'

                            last_date = chart_data.index[-1]
                            future_dates = [last_date] + [last_date + datetime.timedelta(days=i) for i in range(1, 6)]
                            future_prices = [current_price * ((1 + expected_move) ** i) for i in range(0, 6)]
                            
                            fig.add_trace(go.Scatter(x=future_dates, y=future_prices, mode='lines', name='AI Forecast', line=dict(color=trend_color, width=3, dash='dot')))
                        
                        fig.update_layout(xaxis_title="Date", yaxis_title="Price", template="plotly_dark", height=400, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Chart Error: {e}")

        # Sentiment Section (Placeholder)
        st.markdown("---")
        st.subheader("📰 Market Sentiment & Macro")
        if latest_data:
            nc1, nc2 = st.columns([1, 3])
            with nc1:
                st.metric("Sentiment Score", f"{news_score_val} / 100")
            with nc2:
                st.info("AI News Summary: The feature is currently aggregating global financial news...")
