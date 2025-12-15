import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import yfinance as yf
from pykrx import stock
import datetime
from model_def import StockClassifierModel

# ==============================================================================
# 1. 페이지 및 로그인 설정
# ==============================================================================
st.set_page_config(page_title="AI Global Asset Advisor", layout="wide", page_icon="📈")

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
# 2. 데이터 처리 함수 (미국/한국 분리)
# ==============================================================================

# [미국] yfinance 사용
def get_us_data(sectors):
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.datetime.now() - datetime.timedelta(days=120)).strftime("%Y-%m-%d")
    
    try:
        data = yf.download(sectors, start=start_date, end=end_date, group_by='ticker', progress=False, auto_adjust=True)
    except:
        return None, None

    processed_vectors = []
    
    for ticker in sectors:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                df = data.xs(ticker, axis=1, level=0).copy()
            else:
                df = data.copy()

            # 지표 계산
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

# [한국] pykrx 사용 (시총 상위 10개 동적 추출)
def get_kr_data(lookback_days=14):
    target_date = datetime.datetime.now().strftime("%Y%m%d")
    
    # 1. 시가총액 상위 10개 추출 (가장 최근 영업일 기준)
    # pykrx는 휴장일이면 데이터가 없으므로 최근 5일 중 데이터 있는 날 찾기
    for i in range(5):
        try:
            check_date = (datetime.datetime.now() - datetime.timedelta(days=i)).strftime("%Y%m%d")
            df_cap = stock.get_market_cap(check_date)
            if not df_cap.empty:
                top10_tickers = df_cap.sort_values(by='시가총액', ascending=False).head(10).index.tolist()
                target_date = check_date
                break
        except: continue
    
    start_date = (datetime.datetime.strptime(target_date, "%Y%m%d") - datetime.timedelta(days=120)).strftime("%Y%m%d")
    input_vectors = []

    for ticker in top10_tickers:
        try:
            df = stock.get_market_ohlcv(start_date, target_date, ticker)
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change'] # 컬럼 영문 변환
            
            df['Close'] = df['Close'].replace(0, np.nan).ffill()
            df['Volume'] = df['Volume'].replace(0, np.nan).fillna(0)
            
            # 지표 계산
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

# 시장별 설정
if market_option == "NASDAQ (QQQ)":
    MODEL_FILE = "models/us_sector_ai_model_qqq.pth"
    SECTORS = ['XLK', 'XLV', 'XLF', 'XLY', 'XLC', 'XLI', 'XLP', 'XLE', 'XLB', 'XLRE']
    TARGET_NAME = "나스닥 100 (QQQ)"
    LEV_LONG = "QLD (2x) / TQQQ (3x)"
    LEV_SHORT = "QID (2x) / SQQQ (3x)"
elif market_option == "S&P 500 (SPY)":
    MODEL_FILE = "models/us_spy_target_best_model.pth"
    SECTORS = ['XLK', 'XLV', 'XLF', 'XLY', 'XLC', 'XLI', 'XLP', 'XLE', 'XLB', 'XLRE']
    TARGET_NAME = "S&P 500 (SPY)"
    LEV_LONG = "SSO (2x) / UPRO (3x)"
    LEV_SHORT = "SDS (2x) / SPXU (3x)"
else: # KOSPI
    MODEL_FILE = "models/kospi_model.pth"
    SECTORS = [] # 코스피는 동적 추출하므로 비워둠
    TARGET_NAME = "코스피 200 (KOSPI)"
    LEV_LONG = "KODEX 레버리지 (122630)"
    LEV_SHORT = "KODEX 200선물인버스2X (252670)"

# 모델 로드 (캐시 사용)
@st.cache_resource
def load_ai_model(path):
    device = torch.device('cpu')
    model = StockClassifierModel().to(device)
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model
    except Exception as e:
        return None

# 메인 화면
st.title(f"🤖 Global AI Advisor: {market_option}")
st.write(f"최근 시장 데이터를 기반으로 **{TARGET_NAME}**의 향후 5일 방향성을 예측합니다.")
st.markdown("---")

col1, col2 = st.columns([1, 1.5])

with col1:
    if st.button("🚀 AI 분석 실행", type="primary", use_container_width=True):
        model = load_ai_model(MODEL_FILE)
        
        if model is None:
            st.error(f"❌ 모델 파일을 찾을 수 없습니다: {MODEL_FILE}")
            st.warning("models 폴더에 .pth 파일이 있는지 확인해주세요.")
        else:
            with st.spinner("데이터 수집 및 AI 연산 중... (약 10~20초 소요)"):
                # 데이터 수집
                if "KOSPI" in market_option:
                    input_tensor, date_str = get_kr_data()
                else:
                    input_tensor, date_str = get_us_data(SECTORS)
                
                if input_tensor is not None:
                    # 예측
                    with torch.no_grad():
                        logits = model(input_tensor)
                        # 코스피 모델은 출력이 2개(stock_preds, market_logits)일 수 있음
                        # 제공해주신 코드를 보면 코스피도 return값이 2개인 경우가 있어 처리 필요
                        if isinstance(logits, tuple): 
                            logits = logits[1] # market_logits 사용
                        
                        probs = F.softmax(logits, dim=1).squeeze().numpy()
                    
                    down, hold, up = probs[0], probs[1], probs[2]
                    max_prob = np.max(probs)
                    pred_idx = np.argmax(probs)
                    
                    # 결과 출력
                    st.success("분석이 완료되었습니다!")
                    
                    # 1. 확률 메트릭
                    m1, m2, m3 = st.columns(3)
                    m1.metric("📈 상승 확률", f"{up*100:.1f}%")
                    m2.metric("📉 하락 확률", f"{down*100:.1f}%")
                    m3.metric("🧠 AI 확신도", f"{max_prob*100:.1f}%")
                    
                    # 2. 최종 판단
                    decision = "관망 (HOLD)"
                    color = "gray"
                    if max_prob >= 0.45: # Threshold
                        if pred_idx == 2: 
                            decision = "매수 (BUY)"
                            color = "green"
                        elif pred_idx == 0: 
                            decision = "매도/인버스 (SELL)"
                            color = "red"
                    
                    st.markdown(f"### 📢 AI 최종 판단: :{color}[**{decision}**]")
                    
                    # 3. 추천 액션
                    with st.expander("💡 상세 투자 가이드 (클릭)", expanded=True):
                        st.markdown(f"""
                        **기준일: {date_str}**
                        
                        * **매수 신호 시:** * {LEV_LONG}을 **$1,000 (또는 100만원)** 적립 매수하세요.
                            * 기존 숏 포지션은 모두 청산하세요.
                        * **매도 신호 시:**
                            * {LEV_SHORT}을 **$1,000 (또는 100만원)** 적립 매수하세요.
                            * 기존 롱 포지션은 모두 청산하세요.
                        * **관망 신호 시:**
                            * 현금을 보유하고 다음 신호를 기다리세요.
                        """)
                else:
                    st.error("데이터를 수집하지 못했습니다. (장 시작 전이거나 티커 오류)")

with col2:
    st.info("ℹ️ **서비스 안내**")
    st.markdown("""
    이 서비스는 **Transform-LSTM 기반 AI 모델**을 사용하여 시장의 흐름을 분석합니다.
    
    - **입력 데이터:** 주요 섹터 ETF 10종 또는 시총 상위 10개 종목
    - **분석 지표:** RSI, 거래량 비율, 이격도, 볼린저 밴드 등 기술적 지표
    - **업데이트:** 매월 1일 새로운 데이터로 모델이 재학습됩니다.
    """)
    
    # 간단한 참고용 차트
    st.write("📊 **참고용 지수 차트 (최근 3개월)**")
    try:
        idx_ticker = "^KS200" if "KOSPI" in market_option else ("QQQ" if "QQQ" in market_option else "SPY")
        chart_data = yf.download(idx_ticker, period="3mo", progress=False)['Close']
        st.line_chart(chart_data)
    except:
        st.write("차트 로딩 실패")