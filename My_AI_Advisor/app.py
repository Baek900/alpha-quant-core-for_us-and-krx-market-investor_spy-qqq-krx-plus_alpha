import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import datetime
from dateutil import parser # Date conversion
import pytz # Timezone handling
from supabase import create_client, Client

# ==============================================================================
# 1. Configuration & DB Connection
# ==============================================================================
st.set_page_config(page_title="AI Global Asset Advisor", layout="wide", page_icon="📈")

# Connect to Supabase
try:
    # Use Streamlit Cloud Secrets
    SUPABASE_URL = st.secrets["supabase"]["url"]
    SUPABASE_KEY = st.secrets["supabase"]["key"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    # Fallback for local testing (if secrets are missing)
    st.error(f"Secret Configuration Error: {e}")
    st.stop()

def check_password():
    """Authentication Function"""
    def password_entered():
        if st.session_state["username"] in st.secrets["users"] and \
           st.session_state["password"] == st.secrets["users"][st.session_state["username"]]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.title("🔒 Global AI Advisor Login")
        st.write("Subscriber-only service. Please enter your credentials.")
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.title("🔒 Global AI Advisor Login")
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Incorrect username or password.")
        return False
    else:
        return True

if not check_password():
    st.stop()

# ==============================================================================
# 2. Data Loading Functions (DB & Chart)
# ==============================================================================

# Timezone Conversion Function (UTC -> Local/KST)
def convert_utc_to_kst(utc_str):
    try:
        utc_time = parser.parse(utc_str)
        # Using KST as base time, but displayed as standard time format
        kst_zone = pytz.timezone('Asia/Seoul')
        kst_time = utc_time.astimezone(kst_zone)
        return kst_time.strftime('%Y-%m-%d %H:%M') # e.g., 2024-05-20 08:00
    except:
        return utc_str

# [Core] Fetch latest analysis from DB
def load_latest_analysis(market_name):
    try:
        response = supabase.table("prediction_logs") \
            .select("*") \
            .eq("market_name", market_name) \
            .order("created_at", desc=True) \
            .limit(1) \
            .execute()
            
        if response.data:
            return response.data[0]
        else:
            return None
    except Exception as e:
        st.sidebar.error(f"DB Connection Failed: {e}")
        return None

# ==============================================================================
# 3. Main App Logic
# ==============================================================================
st.sidebar.title(f"Welcome, {st.session_state.get('username', 'Member')}! 👋")
st.sidebar.markdown("---")
market_option = st.sidebar.radio("Select Market", ["NASDAQ (QQQ)", "S&P 500 (SPY)", "KOSPI (Korea)"])

# Market Specific Settings
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
else: # KOSPI
    TARGET_NAME = "KOSPI 200 (KOSPI)"
    IDX_TICKER = "^KS11" 
    LEV_LONG = "KODEX Leverage (122630)"
    LEV_SHORT = "KODEX 200 Futures Inverse 2X (252670)"
    INVEST_AMT = "1,000,000 KRW"

st.title(f"🤖 Global AI Advisor: {market_option}")
st.write(f"**Daily Update**, AI automatically analyzes market trends and signals.")
st.markdown("---")

# Fetch Data from DB
latest_data = load_latest_analysis(market_option)

# Main Layout (2 Columns)
col1, col2 = st.columns([1, 1.5])

# Variables for news section (bottom)
news_score_val = 0
news_summary_text = "No Data"

with col1:
    if latest_data:
        # Convert time for display
        date_str = convert_utc_to_kst(latest_data['created_at'])
        
        final_prob = latest_data['final_prob'] # Bullish Probability
        news_score_val = latest_data['news_score'] # Stored for bottom section
        
        # Calculate Probabilities for Display
        # Assumption: Bull + Bear + Neutral = 100%
        up_prob = final_prob
        remaining = 1.0 - up_prob
        down_prob = remaining * 0.5 
        hold_prob = remaining * 0.5 # Neutral Probability
        
        # Display Info
        st.info(f"📅 **Last Update:** {date_str}")
        
        # Top Metrics (Added Neutral Prob, Removed News Score)
        m1, m2, m3 = st.columns(3)
        m1.metric("📈 Bullish Prob", f"{up_prob*100:.1f}%")
        m2.metric("📉 Bearish Prob", f"{down_prob*100:.1f}%") 
        m3.metric("➖ Neutral Prob", f"{hold_prob*100:.1f}%")
        
        # Decision Logic
        decision = "HOLD (Neutral)"
        color = "gray"
        
        if up_prob >= 0.45:
            decision = "BUY"
            color = "green"
        elif up_prob <= 0.2:
            decision = "SELL / Inverse"
            color = "red"
            
        st.markdown(f"### 📢 AI Final Decision: :{color}[**{decision}**]")
        
        with st.expander("💡 Investment Strategy Guide", expanded=True):
            st.markdown(f"""
            **Ref Date: {date_str}**
            * **BUY Signal:** Accumulate {LEV_LONG} **{INVEST_AMT}**
            * **SELL Signal:** Accumulate {LEV_SHORT} **{INVEST_AMT}**
            * **HOLD Signal:** Maintain current position. No new trades.
            """)
            
        # Manual Refresh Button
        if st.button("🔄 Refresh Data"):
            st.rerun()
            
    else:
        st.warning("⚠️ No analysis data available yet.")
        st.info("AI is currently collecting market data. Please check back later.")

with col2:
    st.write(f"📊 **{TARGET_NAME} Index Chart**")
    
    try:
        # Fetch real-time chart data using yfinance
        with st.spinner("Loading latest chart..."):
            chart_df = yf.download(IDX_TICKER, period="6mo", progress=False, auto_adjust=True)
        
        if chart_df.empty:
            st.warning("Failed to fetch chart data. Please try again later.")
        else:
            # MultiIndex Handling
            if isinstance(chart_df.columns, pd.MultiIndex):
                try: chart_df = chart_df.xs(IDX_TICKER, axis=1, level=0)
                except: chart_df.columns = chart_df.columns.get_level_values(0)

            if 'Close' in chart_df.columns:
                chart_data = chart_df['Close'].replace(0, np.nan).dropna()
                current_price = chart_data.iloc[-1]
                st.metric(label=f"Current Price ({IDX_TICKER})", value=f"{current_price:,.2f}")

                # Volatility Calculation
                recent_volatility = chart_data.pct_change().tail(30).std()
                if np.isnan(recent_volatility) or recent_volatility == 0: recent_volatility = 0.01

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data, mode='lines', name='History', line=dict(color='#1f77b4', width=2)))

                # Draw AI Forecast Line (Only if DB data exists)
                if latest_data:
                    # Expected move based on Bullish Probability
                    up_p = latest_data['final_prob']
                    expected_move = (up_p - 0.5) * recent_volatility * 1.5 
                    
                    trend_color = 'gray'
                    if expected_move > 0.0005: trend_color = 'green'
                    elif expected_move < -0.0005: trend_color = 'red'

                    last_date = chart_data.index[-1]
                    future_dates = [last_date] + [last_date + datetime.timedelta(days=i) for i in range(1, 6)]
                    future_prices = [current_price * ((1 + expected_move) ** i) for i in range(0, 6)]
                    
                    fig.add_trace(go.Scatter(x=future_dates, y=future_prices, mode='lines', name='AI Forecast', line=dict(color=trend_color, width=3, dash='dot')))
                    
                    total_return = (future_prices[-1] / current_price - 1) * 100
                    st.caption(f"💡 AI Prob-based 5-Day Forecast: **{total_return:+.2f}%**")

                # Layout Settings
                y_min = chart_data.min()
                y_max = chart_data.max()
                if 'future_prices' in locals():
                    y_min = min(y_min, min(future_prices))
                    y_max = max(y_max, max(future_prices))
                margin = (y_max - y_min) * 0.05
                fig.update_layout(
                    xaxis_title="Date", 
                    yaxis_title="Price", 
                    yaxis_range=[y_min - margin, y_max + margin], 
                    hovermode="x unified", 
                    margin=dict(l=20, r=20, t=40, b=20), 
                    height=400, 
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Invalid data format.")
    except Exception as e:
        st.error(f"Chart loading failed: {e}")

# ==============================================================================
# [New] Bottom News Section (UI Layout)
# ==============================================================================
st.markdown("---")
st.subheader("📰 AI Market Sentiment Analysis")

if latest_data:
    n_col1, n_col2 = st.columns([1, 3])
    
    with n_col1:
        st.metric("Sentiment Score", f"{news_score_val} / 100")
        st.caption("0 (Negative) ~ 100 (Positive)")
    
    with n_col2:
        # Placeholder for future summary feature
        st.info("💡 **Market News Summary**")
        st.markdown("""
        (News collection and summarization features are currently being integrated. 
        AI-generated summaries of key market issues will be displayed here.)
        """)
else:
    st.write("No data available.")
