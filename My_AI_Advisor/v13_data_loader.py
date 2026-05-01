import numpy as np
import pandas as pd
import yfinance as yf
import datetime

# ==============================================================================
# 1. 환경 설정 및 상수
# ==============================================================================
V13_SECTORS = ['XLV', 'XLF', 'XLY', 'XLI', 'XLP', 'XLE', 'XLB', 'XLU', 'IGV', 'SOXX']
MACRO_TICKERS = ['SPY', '^VIX', '^TNX', 'GLD', 'HYG', 'UUP']

def get_market_regime_and_features(target_date):
    """
    [V13.8 Data Loader] 
    오늘 날짜를 기준으로 250일치 데이터를 수집하여 13차원 피처를 생성합니다.
    - 반환값 1: is_bull_market (SPY 200일선 돌파 여부, Boolean)
    - 반환값 2: features_dict (각 섹터별 5개 시그널과 8개 매크로 지표가 담긴 딕셔너리)
    """
    print(f"📡 [Data Loader] 야후 파이낸스 실시간 데이터 수집 중... (기준일: {target_date})")
    
    # 충분한 이동평균선(200일) 계산을 위해 약 300일치 넉넉하게 다운로드
    start_date = target_date - datetime.timedelta(days=350)
    
    all_tickers = V13_SECTORS + MACRO_TICKERS
    # progress=False로 두어 GitHub Actions 로그가 지저분해지는 것을 방지
    raw_data = yf.download(all_tickers, start=start_date, end=target_date + datetime.timedelta(days=1), progress=False)
    
    df_close = raw_data['Close'].ffill()
    
    # ==============================================================================
    # 2. 글로벌 매크로 지표 (8 Features) 계산
    # ==============================================================================
    # 1) VIX 종가 (정규화: 20으로 나눔)
    vix_close = df_close['^VIX'].iloc[-1] / 20.0
    # 2) VIX 5일 변화율
    vix_5d_roc = (df_close['^VIX'].iloc[-1] / df_close['^VIX'].iloc[-6] - 1.0)
    # 3) SPY 20일 변화율 (단기 시장 모멘텀)
    spy_20d_roc = (df_close['SPY'].iloc[-1] / df_close['SPY'].iloc[-21] - 1.0)
    # 4) TNX(미 10년물 국채 금리) 종가 (정규화: 4로 나눔)
    tnx_close = df_close['^TNX'].iloc[-1] / 4.0
    # 5) TNX 5일 변화율 (금리 발작 감지)
    tnx_5d_roc = (df_close['^TNX'].iloc[-1] / df_close['^TNX'].iloc[-6] - 1.0)
    # 6) GLD(금) 20일 변화율 (안전자산 선호도)
    gld_20d_roc = (df_close['GLD'].iloc[-1] / df_close['GLD'].iloc[-21] - 1.0)
    # 7) HYG(하이일드 채권) 20일 변화율 (기업 신용 리스크)
    hyg_20d_roc = (df_close['HYG'].iloc[-1] / df_close['HYG'].iloc[-21] - 1.0)
    # 8) UUP(달러 인덱스) 20일 변화율 (달러 강세/약세)
    uup_20d_roc = (df_close['UUP'].iloc[-1] / df_close['UUP'].iloc[-21] - 1.0)
    
    macro_features = np.array([
        vix_close, vix_5d_roc, spy_20d_roc, tnx_close, tnx_5d_roc, 
        gld_20d_roc, hyg_20d_roc, uup_20d_roc
    ])
    
    # 🐂 강세장 판별 (SPY가 200일선 위에 있는가?)
    spy_sma200 = df_close['SPY'].tail(200).mean()
    is_bull_market = bool(df_close['SPY'].iloc[-1] > spy_sma200)

    # ==============================================================================
    # 3. 개별 섹터 기술적 시그널 (5 Features) 계산
    # ==============================================================================
    features_dict = {}
    
    for sec in V13_SECTORS:
        prices = df_close[sec]
        current_price = prices.iloc[-1]
        
        # 1) Price vs 20 SMA
        sma20 = prices.tail(20).mean()
        sig_sma20 = 1.0 if current_price > sma20 else -1.0
        
        # 2) Price vs 50 SMA
        sma50 = prices.tail(50).mean()
        sig_sma50 = 1.0 if current_price > sma50 else -1.0
        
        # 3) MACD Histogram (12, 26, 9)
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd.iloc[-1] - macd_signal.iloc[-1]
        sig_macd = 1.0 if macd_hist > 0 else -1.0
        
        # 4) RSI (14)
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        sig_rsi = 1.0 if rsi > 50 else -1.0
        
        # 5) Bollinger Bands %B (20, 2)
        std20 = prices.tail(20).std()
        upper_band = sma20 + (std20 * 2)
        lower_band = sma20 - (std20 * 2)
        # 0.5 이상이면 중심선 위, 아니면 아래
        bb_pb = (current_price - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0.5
        sig_bb = 1.0 if bb_pb > 0.5 else -1.0
        
        sector_sigs = np.array([sig_sma20, sig_sma50, sig_macd, sig_rsi, sig_bb])
        
        # 딕셔너리에 저장 (배치 봇이 가져가기 편한 구조)
        features_dict[sec] = {
            'sigs': sector_sigs,
            'macro': macro_features
        }
        
    return is_bull_market, features_dict

# (테스트용 코드)
if __name__ == "__main__":
    test_date = datetime.datetime.now().date()
    is_bull, feat_dict = get_market_regime_and_features(test_date)
    print(f"\n📊 Is Bull Market? {is_bull}")
    print(f"🔎 SOXX Signals (5-Dim): {feat_dict['SOXX']['sigs']}")
    print(f"🌍 Macro Features (8-Dim): {feat_dict['SOXX']['macro']}")
    print("✅ V13.8 데이터 로더 작동 이상 없음.")
