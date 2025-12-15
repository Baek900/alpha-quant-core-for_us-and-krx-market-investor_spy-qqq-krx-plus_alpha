import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import yfinance as yf
from pykrx import stock
import datetime
import os
from model_def import StockClassifierModel

# ==============================================================================
# 1. 페이지 설정 및 경로 설정 (에러 해결 핵심)
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
                df = data.xs(ticker, axis=1, level=0).copy()
            else:
                df = data.copy()

            # 데이터 정제 (0 제거)
            df['Close'] = df['Close'].replace(0, np.nan).ffill()
            df['Volume'] = df['Volume'].replace(0, np.nan).fillna(0)

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

# [한국] pykrx 사용
def get_kr_data(lookback_days=14):
    target_date = datetime.datetime.now().strftime("%Y%m%d")
    
    # 시가총액 상위 10개 추출
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

# [핵심] 경로 설정: os.path.join을 사용하여 절대 경로 생성
if market_option == "NASDAQ (QQQ)":
    MODEL_FILE = os.path.join(BASE_DIR, "models", "us_sector_ai_model_qqq.pth")
    SECTORS = ['XLK', 'XLV', 'XLF', 'XLY', 'XLC', 'XLI', 'XLP', 'XLE', 'XLB', 'XLRE']
    TARGET_NAME = "나스닥 100 (QQQ)"
    IDX_TICKER = "QQQ"
    LEV_LONG = "QLD (2x) / TQQQ (3x)"
    LEV_SHORT = "QID (2x) / SQQQ (3x)"
elif market_option == "S&P 500 (SPY)":
    MODEL_FILE = os.path.join(BASE_DIR, "models", "us_spy_target_best_model.pth")
    SECTORS = ['XLK', 'XLV', 'XLF', 'XLY', 'XLC', 'XLI', 'XLP', 'XLE', 'XLB', 'XLRE']
    TARGET_NAME = "S&P 500 (SPY)"
    IDX_TICKER = "SPY"
    LEV_LONG = "SSO (2x) / UPRO (3x)"
    LEV_SHORT = "SDS (2x) / SPXU (3x)"
else: # KOSPI
    MODEL_FILE = os.path.join(BASE_DIR, "models", "kospi_model.pth")
    SECTORS = []
    TARGET_NAME = "코스피 200 (KOSPI)"
    IDX_TICKER = "^KS200"
    LEV_LONG = "KODEX 레버리지 (122630)"
    LEV_SHORT = "KODEX 200선물인버스2X (252670)"

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

# 분석 결과 저장을 위한 세션 상태
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

with col1:
    if st.button("🚀 AI 분석 실행", type="primary", use_container_width=True):
        model = load_ai_model(MODEL_FILE)
        
        if model == "NOT_FOUND":
            st.error(f"❌ 모델 파일을 찾을 수 없습니다.")
            st.code(MODEL_FILE) # 디버깅용 경로 출력
            st.warning("models 폴더가 app.py와 같은 위치에 있는지 확인해주세요.")
        elif model is None:
            st.error("❌ 모델 로드 중 알 수 없는 오류가 발생했습니다.")
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
                    
                    # 결과 저장
                    st.session_state.analysis_result = {
                        'probs': probs,
                        'date': date_str,
                        'market': market_option
                    }
                else:
                    st.error("데이터 수집 실패. 장 시작 전이거나 티커 오류일 수 있습니다.")

    # 결과 표시 로직
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
        trend_direction = 0 # 0:Flat, 1:Up, -1:Down
        
        if max_prob >= 0.45:
            if pred_idx == 2: 
                decision = "매수 (BUY)"
                color = "green"
                trend_direction = 1
            elif pred_idx == 0: 
                decision = "매도/인버스 (SELL)"
                color = "red"
                trend_direction = -1
        
        st.markdown(f"### 📢 AI 최종 판단: :{color}[**{decision}**]")
        
        with st.expander("💡 상세 투자 가이드 (클릭)", expanded=True):
            st.markdown(f"""
            **기준일: {res['date']}**
            * **매수 신호:** {LEV_LONG} $1,000 적립
            * **매도 신호:** {LEV_SHORT} $1,000 적립 (또는 현금화)
            * **관망:** 현금 보유 및 대기
            """)

with col2:
    # ---------------------------------------------------------
    # [차트 개선] 현재 가격 표시 및 미래 예측 라인 추가
    # ---------------------------------------------------------
    st.write(f"📊 **{TARGET_NAME} 지수 차트 (AI 예측 포함)**")
    
    try:
        # 데이터 가져오기 (최근 90일)
        chart_df = yf.download(IDX_TICKER, period="3mo", progress=False, auto_adjust=True)
        
        # [수정] 0이나 NaN 제거 (그래프 뭉개짐 방지)
        if isinstance(chart_df.columns, pd.MultiIndex):
            chart_df = chart_df.xs(IDX_TICKER, axis=1, level=0)
        
        chart_data = chart_df['Close'].replace(0, np.nan).dropna()
        
        # 현재가 표시
        current_price = chart_data.iloc[-1]
        st.metric(label=f"현재 가격 ({IDX_TICKER})", value=f"{current_price:,.2f}")

        # 차트 데이터 구성
        # 1. 과거 데이터 (History)
        display_df = pd.DataFrame({'History': chart_data})
        
        # 2. 미래 예측 라인 (Prediction) 추가
        # 분석 결과가 있고, 현재 시장과 일치할 때만 그림
        if st.session_state.analysis_result and st.session_state.analysis_result['market'] == market_option:
            res = st.session_state.analysis_result
            probs = res['probs']
            pred_idx = np.argmax(probs)
            max_prob = np.max(probs)
            
            # 예측 방향 결정
            slope = 0
            if max_prob >= 0.45:
                if pred_idx == 2: slope = 0.015 # 상승 예측 시 +1.5% 기울기 (시각적용)
                elif pred_idx == 0: slope = -0.015 # 하락 예측 시 -1.5%
            
            # 미래 5일 날짜 생성
            last_date = chart_data.index[-1]
            future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 6)]
            
            # 미래 가격 생성 (단순 선형 추세)
            future_prices = [current_price * (1 + slope * (i/5)) for i in range(1, 6)]
            
            # DataFrame에 미래 데이터 추가 (History 컬럼은 NaN, Prediction 컬럼에 값)
            future_df = pd.DataFrame({
                'History': [np.nan]*5,
                'AI Prediction': future_prices
            }, index=future_dates)
            
            # 과거 데이터와 합치기 (과거 데이터의 Prediction 컬럼은 NaN)
            display_df['AI Prediction'] = np.nan
            
            # 연결 부위 매끄럽게 하기 위해 오늘 날짜에 점 하나 찍기
            display_df.loc[last_date, 'AI Prediction'] = current_price
            
            final_chart = pd.concat([display_df, future_df])
            
            # 선 차트 그리기 (색상 구분됨)
            st.line_chart(final_chart)
            
        else:
            # 분석 전에는 그냥 과거 차트만
            st.line_chart(chart_data)
            
    except Exception as e:
        st.error(f"차트 로딩 실패: {e}")
