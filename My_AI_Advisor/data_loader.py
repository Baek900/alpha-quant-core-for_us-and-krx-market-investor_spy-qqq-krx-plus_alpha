# data_loader.py
import torch
import numpy as np
import pandas as pd
import yfinance as yf
import datetime

# [v2.0] 노트북의 지표 계산 로직을 엄격하게 통합 (13개 피처)
def calculate_v2_indicators(df):
    """
    노트북의 calculate_indicators_final 및 calculate_technical_indicators 로직을 
    오차 없이 통합한 함수 (13개 피처 반환)
    """
    df = df.copy()
    
    # 1. 수익률 및 가격 변동 (4개)
    df['Change'] = df['Close'].pct_change()
    df['Open_chg'] = (df['Open'] - df['Close'].shift(1)) / (df['Close'].shift(1) + 1e-9)
    df['High_chg'] = (df['High'] - df['Close'].shift(1)) / (df['Close'].shift(1) + 1e-9)
    df['Low_chg'] = (df['Low'] - df['Close'].shift(1)) / (df['Close'].shift(1) + 1e-9)
    
    # 2. 이동평균 이격도 (3개: 5, 20, 60)
    for window in [5, 20, 60]:
        ma = df['Close'].rolling(window).mean()
        df[f'MA{window}_Disparity'] = (df['Close'] / (ma + 1e-9)) - 1
        
    # 3. RSI (1개)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = (100 - (100 / (1 + gain / (loss + 1e-9)))) / 100.0
    
    # 4. MACD Oscillator (1개)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD_Oscillator'] = (macd - signal) / (df['Close'] + 1e-9)
    
    # 5. Bollinger Bands (2개: %B, Width)
    bb_m = df['Close'].rolling(20).mean()
    bb_s = df['Close'].rolling(20).std()
    df['BB_PctB'] = (df['Close'] - (bb_m - 2*bb_s)) / (4*bb_s + 1e-9)
    df['BB_Width'] = (4*bb_s) / (bb_m + 1e-9)
    
    # 6. Volume Ratio (1개)
    df['Vol_Ratio'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-9)
    
    # 7. MFI (1개: Money Flow Index)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    mf = tp * df['Volume']
    pf = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
    nf = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
    df['MFI'] = (100 - (100 / (1 + pf / (nf + 1e-9)))) / 100.0
    
    # 결측치 처리 및 클리핑 (노트북 전처리 로직)
    df = df.ffill().fillna(0)
    df['Vol_Ratio'] = df['Vol_Ratio'].clip(0, 10)
    df['Change'] = df['Change'].clip(-0.3, 0.3)
    
    return df

# [v2.0] 미국 모델용 데이터 로더 (QQQ/SPY 공용)
def get_us_v2_data(sectors, seq_len=14):
    """
    미국 섹터 ETF 10개 데이터를 로드하여 14일치 시퀀스 벡터 반환
    """
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.datetime.now() - datetime.timedelta(days=150)).strftime("%Y-%m-%d")
    
    # 데이터 다운로드
    data = yf.download(sectors, start=start_date, end=end_date, group_by='ticker', progress=False, auto_adjust=True)
    
    feature_cols = [
        'Change', 'Open_chg', 'High_chg', 'Low_chg', 'MA5_Disparity', 'MA20_Disparity', 'MA60_Disparity',
        'RSI', 'MACD_Oscillator', 'BB_PctB', 'BB_Width', 'Vol_Ratio', 'MFI'
    ]
    
    vectors = []
    for ticker in sectors:
        df = data[ticker] if len(sectors) > 1 else data
        df_processed = calculate_v2_indicators(df)
        vectors.append(df_processed[feature_cols].tail(seq_len).values)
        
    # Return shape: [1, 10, 14, 13]
    return torch.FloatTensor(np.array(vectors)).unsqueeze(0)

# [v2.0] 한국 모델용 데이터 로더 (KOSPI Ensemble)
def get_kr_v2_data(kr_sectors, macro_indicators, seq_len=20):
    """
    한국 섹터 10개 + 매크로(환율, 미국지수) 2개 = 총 12개 자산 로드 (20일치 시퀀스)
    """
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.datetime.now() - datetime.timedelta(days=150)).strftime("%Y-%m-%d")
    
    # 한국 섹터와 매크로 지표 통합 다운로드
    all_tickers = kr_sectors + macro_indicators
    data = yf.download(all_tickers, start=start_date, end=end_date, group_by='ticker', progress=False, auto_adjust=True)
    
    feature_cols = [
        'Change', 'Open_chg', 'High_chg', 'Low_chg', 'MA5_Disparity', 'MA20_Disparity', 'MA60_Disparity',
        'RSI', 'MACD_Oscillator', 'BB_PctB', 'BB_Width', 'Vol_Ratio', 'MFI'
    ]
    
    vectors = []
    # 노트북의 자산 순서 엄격 유지 (Sectors -> Macro)
    for ticker in all_tickers:
        df = data[ticker]
        df_processed = calculate_v2_indicators(df)
        vectors.append(df_processed[feature_cols].tail(seq_len).values)
        
    # Return shape: [1, 12, 20, 13]
    return torch.FloatTensor(np.array(vectors)).unsqueeze(0)