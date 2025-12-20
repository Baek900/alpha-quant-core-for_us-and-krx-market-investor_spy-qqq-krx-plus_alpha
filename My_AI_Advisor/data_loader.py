import torch
import numpy as np
import pandas as pd
import yfinance as yf
from pykrx import stock
import datetime

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
                try: df = data.xs(ticker, axis=1, level=0).copy()
                except KeyError: return None, None
            else: df = data.copy()

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
