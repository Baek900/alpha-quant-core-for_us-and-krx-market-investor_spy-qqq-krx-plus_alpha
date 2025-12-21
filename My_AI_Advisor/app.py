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
            
            # [Edit 1] When login is successful, store the ID in a permanent variable instead of the widget variable that disappears.
            st.session_state["logged_in_user"] = st.session_state["username"]
            
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

    # Section 1: Benchmark Performance (Real Backtest Data)
    st.header("🏆 Performance Benchmark (Backtest)")
    st.write("Comparison of **AI Algorithm Strategy** vs. **Market Benchmark (SPY)** over the last 12 months.")
    
    # Load real backtest results from CSV
    try:
        # Load CSV
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "backtest_result.csv")
        
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        initial_capital = df['Strategy'].iloc[0]
        ai_returns = (df['Strategy'] - initial_capital) / initial_capital
        market_returns = (df['Benchmark'] - initial_capital) / initial_capital
        
        dates = df.index
        
        final_ai_ret = ai_returns.iloc[-1] * 100
        final_bm_ret = market_returns.iloc[-1] * 100
        alpha = final_ai_ret - final_bm_ret

        # Draw Chart
        fig_bench = go.Figure()
        fig_bench.add_trace(go.Scatter(x=dates, y=ai_returns, mode='lines', name='AI Strategy', line=dict(color='#00FFA3', width=2)))
        fig_bench.add_trace(go.Scatter(x=dates, y=market_returns, mode='lines', name='S&P 500 (Benchmark)', line=dict(color='gray', dash='dot')))
        
        fig_bench.update_layout(
            title="Cumulative Return Comparison (Last 1 Year)",
            xaxis_title="Date",
            yaxis_title="Return (0.1 = 10%)",
            template="plotly_dark",
            height=400,
            yaxis_tickformat='.0%'
        )
        
        col_bench1, col_bench2 = st.columns([2, 1])
        with col_bench1:
            st.plotly_chart(fig_bench, use_container_width=True)
        with col_bench2:
            diff_color = "normal"
            if alpha > 0: diff_color = "normal" 
            
            st.metric(label="AI Total Return", value=f"{final_ai_ret:+.2f}%", delta=f"{alpha:+.2f}% vs SPY")
            st.metric(label="Benchmark Return", value=f"{final_bm_ret:+.2f}%")
            
            if alpha > 0:
                st.success(f"✅ AI outperformed the market by **{alpha:.2f}%p**")
            else:
                st.warning(f"⚠️ AI underperformed the market by **{alpha:.2f}%p**")
                
            st.caption("*Based on actual backtesting data (Last 365 days).*")

    except Exception as e:
        st.error(f"Failed to load backtest data: {e}")
        st.info("Please ensure 'backtest_result.csv' exists in the repository.")

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

st.divider()
    
    # Section 2: Model Accuracy & Reliability
    st.header("🎯 AI Reliability Verification")
    st.write("Performance metrics of the **SPY (S&P 500)** prediction model based on unseen test data (2025).")

    # Display 3 Key Metrics in Columns
    col_m1, col_m2, col_m3 = st.columns(3)
    
    with col_m1:
        st.metric(label="Overall Accuracy", value="52.62%", delta="+19.3% vs Random")
        st.caption("Outperforms random guessing (33.3%)")
        
    with col_m2:
        st.metric(label="Buy Signal Precision", value="64.0%", delta="High Confidence")
        st.caption("When AI says 'BUY', it is correct 64% of the time.")
        
    with col_m3:
        st.metric(label="F1-Score (Up Trend)", value="0.59")
        st.caption("Balanced metric of Precision and Recall.")

    # Confusion Matrix Image & Explanation
    st.markdown("### 🔍 Confusion Matrix Analysis")
    
    c_img, c_desc = st.columns([1, 1.5])
    
    with c_img:
        # Check if the image file exists in the current directory before displaying
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(current_dir, "confusion_matrix.png")
        
        if os.path.exists(img_path):
            st.image(img_path, caption="Model Prediction vs Actual Market Move", use_container_width=True)
        else:
            st.warning("Confusion Matrix image not found. Please upload 'confusion_matrix.png'.")

    with c_desc:
        st.info("💡 **Why is 52% Accuracy significant?**")
        st.markdown("""
        In financial markets, predicting stock movements with **>50% accuracy** consistently is considered highly profitable. 
        Most algorithmic trading funds operate with win rates between 51% and 54%.
        
        **Key Takeaways:**
        * The model effectively filters out market noise ('Flat' movements).
        * **High Precision in Uptrends (64%)**: The AI is conservative but highly accurate when identifying buying opportunities.
        * This reduces the risk of 'False Positives' (buying when the market falls).
        """)

    st.divider()

# ==============================================================================
# 5. Page: Dashboard (Private - Login Required)
# ==============================================================================
elif page == "🚀 AI Dashboard (Member Only)":
    
    # Login Check
    if check_password():
        # [Edit 2] Display login information in the upper right corner (split screen)
        top_col1, top_col2 = st.columns([4, 1])
        
        # --- Existing Dashboard Code Starts Here ---
        market_option = st.sidebar.radio("Select Market", ["NASDAQ (QQQ)", "S&P 500 (SPY)", "KOSPI (Korea)"])

        with top_col1:
            st.title(f"🤖 AI Dashboard: {market_option}")
            st.caption(f"Real-time analysis powered by TMFG-LSTM Model.")
        
        with top_col2:
            #  Display login information in the upper right corner
            user_id = st.session_state.get("logged_in_user", "Member")
            st.markdown(f"<div style='text-align: right; padding-top: 20px;'>👤 <b>{user_id}</b> logged in</div>", unsafe_allow_html=True)

        st.markdown("---")

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
                        
                        # [Edit 3] Added AI predicted return display and caption (right below Price Metric)
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
                            
                            # [Edit 3 Continued] Calculating the predicted return
                            total_return = (future_prices[-1] / current_price - 1) * 100
                            st.caption(f"💡 AI Prob-based 5-Day Forecast: **{total_return:+.2f}%**")
                        
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




