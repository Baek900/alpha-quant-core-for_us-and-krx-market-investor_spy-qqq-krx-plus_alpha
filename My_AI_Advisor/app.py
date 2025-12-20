import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import datetime
from dateutil import parser # 날짜 변환용
import pytz # 타임존 처리용
from supabase import create_client, Client

# ==============================================================================
# 1. 설정 및 DB 연결
# ==============================================================================
st.set_page_config(page_title="AI Global Asset Advisor", layout="wide", page_icon="📈")

# Supabase 연결
try:
    # 스트림릿 클라우드 Secret 사용 시
    SUPABASE_URL = st.secrets["supabase"]["url"]
    SUPABASE_KEY = st.secrets["supabase"]["key"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    # 로컬 테스트용 (secrets 파일이 없을 경우 예외처리)
    st.error(f"Secret 설정 오류: {e}")
    st.stop()

def check_password():
    """로그인 인증 함수"""
    def password_entered():
        if st.session_state["username"] in st.secrets["users"] and \
           st.session_state["password"] == st.secrets["users"][st.session_state["username"]]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.title("🔒 Global AI Advisor Login")
        st.write("구독 회원 전용 서비스입니다. 아이디와 비밀번호를 입력하세요.")
        st.text_input("아이디 (ID)", key="username")
        st.text_input("비밀번호 (Password)", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.title("🔒 Global AI Advisor Login")
        st.text_input("아이디 (ID)", key="username")
        st.text_input("비밀번호 (Password)", type="password", on_change=password_entered, key="password")
        st.error("😕 로그인 정보가 일치하지 않습니다.")
        return False
    else:
        return True

if not check_password():
    st.stop()

# ==============================================================================
# 2. 데이터 로드 함수 (DB & 차트용)
# ==============================================================================

# 시간대 변환 함수 (UTC -> KST)
def convert_utc_to_kst(utc_str):
    try:
        utc_time = parser.parse(utc_str)
        kst_zone = pytz.timezone('Asia/Seoul')
        kst_time = utc_time.astimezone(kst_zone)
        return kst_time.strftime('%Y-%m-%d %H:%M') # 예: 2024-05-20 08:00
    except:
        return utc_str

# [핵심] DB에서 최신 분석 결과 가져오기
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
        st.sidebar.error(f"DB 연결 실패: {e}")
        return None

# ==============================================================================
# 3. 메인 앱 로직
# ==============================================================================
st.sidebar.title(f"환영합니다, {st.session_state.get('username', 'Member')}님! 👋")
st.sidebar.markdown("---")
market_option = st.sidebar.radio("분석할 시장 선택", ["NASDAQ (QQQ)", "S&P 500 (SPY)", "KOSPI (Korea)"])

# 시장별 설정
if market_option == "NASDAQ (QQQ)":
    TARGET_NAME = "나스닥 100 (QQQ)"
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
    TARGET_NAME = "코스피 200 (KOSPI)"
    IDX_TICKER = "^KS11" 
    LEV_LONG = "KODEX 레버리지 (122630)"
    LEV_SHORT = "KODEX 200선물인버스2X (252670)"
    INVEST_AMT = "1,000,000원"

st.title(f"🤖 Global AI Advisor: {market_option}")
st.write(f"**매일 아침/저녁**, AI가 자동으로 시장을 분석하고 업데이트합니다.")
st.markdown("---")

# DB에서 데이터 가져오기
latest_data = load_latest_analysis(market_option)

col1, col2 = st.columns([1, 1.5])

with col1:
    if latest_data:
        # 시간 변환 (KST로 보여주기)
        date_str = convert_utc_to_kst(latest_data['created_at'])
        
        final_prob = latest_data['final_prob'] # 상승 확률
        news_score = latest_data['news_score']
        
        # 하락/보합 확률 추정 (화면 표시용)
        up_prob = final_prob
        remaining = 1.0 - up_prob
        down_prob = remaining * 0.5 
        
        # UI 표시
        st.info(f"📅 **최신 업데이트 (한국시간):** {date_str}")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("📈 상승 확률", f"{up_prob*100:.1f}%")
        m2.metric("📉 하락 확률", f"{down_prob*100:.1f}%") 
        m3.metric("📰 뉴스 점수", f"{news_score}점")
        
        # 판단 로직
        decision = "관망 (HOLD)"
        color = "gray"
        if up_prob >= 0.45:
            decision = "매수 (BUY)"
            color = "green"
        elif up_prob <= 0.2:
            decision = "매도/인버스 (SELL)"
            color = "red"
            
        st.markdown(f"### 📢 AI 최종 판단: :{color}[**{decision}**]")
        
        with st.expander("💡 상세 투자 가이드", expanded=True):
            st.markdown(f"""
            **기준일시: {date_str}**
            * **매수 신호:** {LEV_LONG} **{INVEST_AMT}** 분할 매수
            * **매도 신호:** {LEV_SHORT} **{INVEST_AMT}** 분할 매수
            * **관망:** 현금 보유 및 대기 (무리한 진입 금지)
            """)
            
        # 수동 업데이트 버튼
        if st.button("🔄 데이터 새로고침"):
            st.rerun()
            
    else:
        st.warning("⚠️ 아직 분석된 데이터가 없습니다.")
        st.info("AI 모델이 데이터를 수집 중입니다. 잠시 후 다시 확인해주세요.")

with col2:
    st.write(f"📊 **{TARGET_NAME} 지수 차트**")
    
    try:
        # 차트는 실시간 가격을 보여줘야 하므로 여기서 yfinance 호출
        with st.spinner("최신 차트 불러오는 중..."):
            chart_df = yf.download(IDX_TICKER, period="6mo", progress=False, auto_adjust=True)
        
        if chart_df.empty:
            st.warning("차트 데이터 수신 실패 (잠시 후 다시 시도하세요)")
        else:
            # MultiIndex 처리 (yfinance 버전에 따라 다를 수 있음)
            if isinstance(chart_df.columns, pd.MultiIndex):
                try: chart_df = chart_df.xs(IDX_TICKER, axis=1, level=0)
                except: chart_df.columns = chart_df.columns.get_level_values(0)

            if 'Close' in chart_df.columns:
                chart_data = chart_df['Close'].replace(0, np.nan).dropna()
                current_price = chart_data.iloc[-1]
                st.metric(label=f"현재 가격 ({IDX_TICKER})", value=f"{current_price:,.2f}")

                # 변동성 계산
                recent_volatility = chart_data.pct_change().tail(30).std()
                if np.isnan(recent_volatility) or recent_volatility == 0: recent_volatility = 0.01

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data, mode='lines', name='History', line=dict(color='#1f77b4', width=2)))

                # AI 예측선 그리기 (DB 데이터가 있을 때만)
                if latest_data:
                    # 상승 확률에 따른 예상 이동 경로
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
                    st.caption(f"💡 AI 확률 기반 5일 예상 추세: **{total_return:+.2f}%**")

                # 레이아웃
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
                st.warning("데이터 형식이 올바르지 않습니다.")
    except Exception as e:
        st.error(f"차트 로딩 실패: {e}")
