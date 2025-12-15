import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import yfinance as yf
from pykrx import stock
import datetime
import os
import plotly.graph_objects as go
from model_def import StockClassifierModel

# ==============================================================================
# 1. 페이지 설정 및 경로 설정
# ==============================================================================
st.set_page_config(page_title="AI Global Asset Advisor", layout="wide", page_icon="📈")

# [핵심] 현재 파일(app.py)의 절대 경로를 구해서 models 폴더 위치를 찾음
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
# 2. 데이터 처리 함수
# ==============================================================================

# [미국] yfinance 사용
def get_us_data(sectors):
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.datetime.now() - datetime.timedelta(days=150)).strftime("%Y-%m-%d")
    
    try:
        data = yf.download(sectors, start=start_date, end=end_date, group_by='ticker', progress=False, auto_adjust=True)
    except:
        return None, None

    processed_vectors = []
    
    for ticker in sectors:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                try:
                    df = data.xs(ticker, axis=1, level=0).copy()
                except KeyError: return None, None
            else:
                df = data.copy()

            df['Close'] = df['Close'].replace(0, np.nan).ffill()
            df['Volume'] = df['Volume'].replace(0, np.nan).fillna(0)

            df['Change'] = df['Close'].pct_change().fillna(0).clip(-0.3, 0.3)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-9)
            df['RSI_Scaled'] = (100 - (100 / (1 + rs))).fillna(50) / 100.0
            df['Vol_Ratio'] = (df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-9)).fillna(0).clip(0, 5)
            df['MA20_Disparity'] = ((df['Close'] / df['Close'].rolling(20).mean()) - 1).fillna(0)
            bb_std = df['Close'].rolling(20).std()
            bb_m = df['Close'].rolling(20).mean()
            df['BB_PctB'] = ((df['Close'] - (bb_m - 2*bb_std)) / (4*bb_std + 1e-9)).fillna(0.5).clip(-0.5, 1.5)
            
            feat = df[['Change', 'RSI_Scaled', 'Vol_Ratio', 'MA20_Disparity', 'BB_PctB']].dropna()
            
            if len(feat) < 14: return None, None
            processed_vectors.append(feat.iloc[-14:].values)
        except: return None, None

    if len(processed_vectors) != len(sectors): return None, None
    input_tensor = torch.FloatTensor(np.array(processed_vectors)).unsqueeze(0)
    return input_tensor, end_date

# [한국] pykrx 사용
def get_kr_data(lookback_days=14):
    target_date = datetime.datetime.now().strftime("%Y%m%d")
    
    for i in range(5):
        try:
            check_date = (datetime.datetime.now() - datetime.timedelta(days=i)).strftime("%Y%m%d")
            df_cap = stock.get_market_cap(check_date)
            if not df_cap.empty:
                top10_tickers = df_cap.sort_values(by='시가총액', ascending=False).head(10).index.tolist()
                target_date = check_date
                break
        except: continue
    
    start_date = (datetime.datetime.strptime(target_date, "%Y%m%d") - datetime.timedelta(days=150)).strftime("%Y%m%d")
    input_vectors = []

    for ticker in top10_tickers:
        try:
            df = stock.get_market_ohlcv(start_date, target_date, ticker)
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change'] 
            
            df['Close'] = df['Close'].replace(0, np.nan).ffill()
            df['Volume'] = df['Volume'].replace(0, np.nan).fillna(0)
            
            if 'Change' not in df.columns: df['Change'] = df['Close'].pct_change().fillna(0)
            df['Change'] = df['Change'].clip(-0.3, 0.3)

            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-9)
            df['RSI_Scaled'] = (100 - (100 / (1 + rs))).fillna(50) / 100.0

            df['Vol_Ratio'] = (df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-9)).fillna(0).clip(0, 5)
            df['MA20_Disparity'] = ((df['Close'] / df['Close'].rolling(20).mean()) - 1).fillna(0)
            
            bb_std = df['Close'].rolling(20).std()
            bb_m = df['Close'].rolling(20).mean()
            df['BB_PctB'] = ((df['Close'] - (bb_m - 2*bb_std)) / (4*bb_std + 1e-9)).fillna(0.5).clip(-0.5, 1.5)

            feat = df[['Change', 'RSI_Scaled', 'Vol_Ratio', 'MA20_Disparity', 'BB_PctB']].dropna()
            
            if len(feat) < lookback_days: return None, None
            input_vectors.append(feat.iloc[-lookback_days:].values)
        except: return None, None

    if len(input_vectors) != 10: return None, None
    input_tensor = torch.FloatTensor(np.array(input_vectors)).unsqueeze(0)
    return input_tensor, target_date

# ==============================================================================
# 3. 메인 앱 로직
# ==============================================================================
st.sidebar.title(f"환영합니다, {st.session_state.get('username', 'Member')}님! 👋")
st.sidebar.markdown("---")
market_option = st.sidebar.radio("분석할 시장 선택", ["NASDAQ (QQQ)", "S&P 500 (SPY)", "KOSPI (Korea)"])

# [안전장치] 시장을 바꿀 때마다 이전 분석 결과 초기화
if 'last_market' not in st.session_state:
    st.session_state.last_market = market_option

if st.session_state.last_market != market_option:
    st.session_state.analysis_result = None
    st.session_state.last_market = market_option

# 경로 및 투자금액 설정
if market_option == "NASDAQ (QQQ)":
    MODEL_FILE = os.path.join(BASE_DIR, "models", "us_sector_ai_model_qqq.pth")
    SECTORS = ['XLK', 'XLV', 'XLF', 'XLY', 'XLC', 'XLI', 'XLP', 'XLE', 'XLB', 'XLRE']
    TARGET_NAME = "나스닥 100 (QQQ)"
    IDX_TICKER = "QQQ"
    LEV_LONG = "QLD (2x) / TQQQ (3x)"
    LEV_SHORT = "QID (2x) / SQQQ (3x)"
    INVEST_AMT = "$1,000" # [설정] 미국 주식은 달러
elif market_option == "S&P 500 (SPY)":
    MODEL_FILE = os.path.join(BASE_DIR, "models", "us_spy_target_best_model.pth")
    SECTORS = ['XLK', 'XLV', 'XLF', 'XLY', 'XLC', 'XLI', 'XLP', 'XLE', 'XLB', 'XLRE']
    TARGET_NAME = "S&P 500 (SPY)"
    IDX_TICKER = "SPY"
    LEV_LONG = "SSO (2x) / UPRO (3x)"
    LEV_SHORT = "SDS (2x) / SPXU (3x)"
    INVEST_AMT = "$1,000" # [설정] 미국 주식은 달러
else: # KOSPI
    MODEL_FILE = os.path.join(BASE_DIR, "models", "kospi_model.pth")
    SECTORS = []
    TARGET_NAME = "코스피 200 (KOSPI)"
    IDX_TICKER = "^KS11" 
    LEV_LONG = "KODEX 레버리지 (122630)"
    LEV_SHORT = "KODEX 200선물인버스2X (252670)"
    INVEST_AMT = "1,000,000원" # [설정] 한국 주식은 원화

# 모델 로드
@st.cache_resource
def load_ai_model(path):
    device = torch.device('cpu')
    model = StockClassifierModel().to(device)
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model
    except FileNotFoundError:
        return "NOT_FOUND"
    except Exception as e:
        return None

# ==============================================================================
# 4. 화면 구성
# ==============================================================================
st.title(f"🤖 Global AI Advisor: {market_option}")
st.write(f"최근 데이터를 기반으로 **{TARGET_NAME}**의 향후 5일 추세를 분석합니다.")
st.markdown("---")

col1, col2 = st.columns([1, 1.5])

# 분석 결과 세션 상태 초기화
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

with col1:
    if st.button("🚀 AI 분석 실행", type="primary", use_container_width=True):
        model = load_ai_model(MODEL_FILE)
        
        if model == "NOT_FOUND":
            st.error(f"❌ 모델 파일을 찾을 수 없습니다.")
            st.warning("models 폴더가 app.py와 같은 위치에 있는지 확인해주세요.")
        elif model is None:
            st.error("❌ 모델 로드 중 오류가 발생했습니다.")
        else:
            with st.spinner("데이터 수집 및 AI 연산 중..."):
                if "KOSPI" in market_option:
                    input_tensor, date_str = get_kr_data()
                else:
                    input_tensor, date_str = get_us_data(SECTORS)
                
                if input_tensor is not None:
                    with torch.no_grad():
                        logits = model(input_tensor)
                        if isinstance(logits, tuple): logits = logits[1]
                        probs = F.softmax(logits, dim=1).squeeze().numpy()
                    
                    st.session_state.analysis_result = {
                        'probs': probs,
                        'date': date_str,
                        'market': market_option
                    }
                else:
                    st.error("데이터 수집 실패. 장 시작 전이거나 티커 오류일 수 있습니다.")

    # 결과 표시
    if st.session_state.analysis_result and st.session_state.analysis_result['market'] == market_option:
        res = st.session_state.analysis_result
        probs = res['probs']
        down, hold, up = probs[0], probs[1], probs[2]
        max_prob = np.max(probs)
        pred_idx = np.argmax(probs)

        st.success("분석 완료!")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("📈 상승 확률", f"{up*100:.1f}%")
        m2.metric("📉 하락 확률", f"{down*100:.1f}%")
        m3.metric("🧠 AI 확신도", f"{max_prob*100:.1f}%")
        
        decision = "관망 (HOLD)"
        color = "gray"
        
        if max_prob >= 0.45:
            if pred_idx == 2: 
                decision = "매수 (BUY)"
                color = "green"
            elif pred_idx == 0: 
                decision = "매도/인버스 (SELL)"
                color = "red"
        
        st.markdown(f"### 📢 AI 최종 판단: :{color}[**{decision}**]")
        
        # [수정] INVEST_AMT 변수 적용으로 시장별 통화 단위 자동 변경
        with st.expander("💡 상세 투자 가이드 (클릭)", expanded=True):
            st.markdown(f"""
            **기준일: {res['date']}**
            * **매수 신호:** {LEV_LONG} **{INVEST_AMT}** 적립
            * **매도 신호:** {LEV_SHORT} **{INVEST_AMT}** 적립 (또는 현금화)
            * **관망:** 현금 보유 및 대기
            """)

with col2:
    st.write(f"📊 **{TARGET_NAME} 지수 차트**")
    
    try:
        # 1. 기본 차트 데이터 가져오기 (6개월치)
        chart_df = yf.download(IDX_TICKER, period="6mo", progress=False, auto_adjust=True)
        
        if chart_df.empty:
            st.warning("차트 데이터를 불러올 수 없습니다.")
        else:
            if isinstance(chart_df.columns, pd.MultiIndex):
                try: chart_df = chart_df.xs(IDX_TICKER, axis=1, level=0)
                except: chart_df.columns = chart_df.columns.get_level_values(0)

            if 'Close' in chart_df.columns:
                chart_data = chart_df['Close'].replace(0, np.nan).dropna()
                current_price = chart_data.iloc[-1]
                st.metric(label=f"현재 가격 ({IDX_TICKER})", value=f"{current_price:,.2f}")

                # 최근 변동성 계산
                recent_volatility = chart_data.pct_change().tail(30).std()
                if np.isnan(recent_volatility) or recent_volatility == 0:
                    recent_volatility = 0.01

                # Plotly 차트
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=chart_data.index,
                    y=chart_data,
                    mode='lines',
                    name='History',
                    line=dict(color='#1f77b4', width=2)
                ))

                if st.session_state.analysis_result and st.session_state.analysis_result['market'] == market_option:
                    res = st.session_state.analysis_result
                    probs = res['probs']
                    prob_down, prob_flat, prob_up = probs[0], probs[1], probs[2]
                    
                    # [공식] 기대 변동률 (확률 가중치 반영)
                    expected_daily_move = (prob_up * recent_volatility) - (prob_down * recent_volatility)
                    
                    trend_color = 'gray'
                    if expected_daily_move > 0.0005: trend_color = 'green'
                    elif expected_daily_move < -0.0005: trend_color = 'red'

                    last_date = chart_data.index[-1]
                    future_dates = [last_date] + [last_date + datetime.timedelta(days=i) for i in range(1, 6)]
                    
                    future_prices = []
                    temp_price = current_price
                    future_prices.append(temp_price)

                    for _ in range(5):
                        temp_price = temp_price * (1 + expected_daily_move)
                        future_prices.append(temp_price)
                    
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=future_prices,
                        mode='lines',
                        name=f'AI Forecast',
                        line=dict(color=trend_color, width=3, dash='dot')
                    ))
                    
                    total_return = (future_prices[-1] / current_price - 1) * 100
                    st.caption(f"💡 AI 확률 기반 5일 예상 변동폭: **{total_return:+.2f}%** (확률 가중치 적용)")

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
