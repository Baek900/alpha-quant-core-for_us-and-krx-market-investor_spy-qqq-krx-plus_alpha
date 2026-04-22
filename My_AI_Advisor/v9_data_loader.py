# v9_data_loader.py
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore') # 쓸데없는 yfinance 경고 숨김

START_DATE = '2000-01-01'

def fetch_v9_inference_data():
    end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    series_dict = {}
    
    # 1. FRED Macro Data
    fred_tickers = {'CPIAUCSL':'CPI', 'PPIACO':'PPI', 'UNRATE':'Unemployment', 
                    'M2REAL':'Real_M2', 'BAMLH0A0HYM2':'High_Yield', 'T10Y2Y':'Yield_Curve'}
    for t, n in fred_tickers.items():
        try:
            s = web.DataReader(t, 'fred', START_DATE, end_date)[t]
            s.name = n
            series_dict[n] = s
        except Exception as e: print(f"⚠️ FRED {n} 에러: {e}")

    # 2. YFinance Macro Data
    yf_tickers = {'GC=F':'Gold', 'HG=F':'Copper', 'CL=F':'WTI_Crude', '^VIX':'VIX'}
    for t, n in yf_tickers.items():
        try:
            s = yf.download(t, start=START_DATE, end=end_date, progress=False)['Close']
            if s.index.tz is not None: s.index = s.index.tz_localize(None)
            s.name = n
            series_dict[n] = s
        except Exception as e: print(f"⚠️ YF {n} 에러: {e}")

    final_df = pd.concat(series_dict.values(), axis=1).sort_index()
    
    # [수정됨] 최신 Pandas 문법 적용 (경고창 제거)
    final_df.ffill(inplace=True)
    final_df.bfill(inplace=True)
    
    macro = pd.DataFrame(index=final_df.index)
    cols = final_df.columns
    for c in ['CPI', 'PPI', 'Real_M2']:
        if c in cols: macro[c+'_YoY'] = final_df[c].pct_change(252)
    for c in ['Gold', 'Copper', 'WTI_Crude']:
        if c in cols: macro[c+'_Mom20'] = final_df[c].pct_change(20)
    for c in ['Unemployment', 'High_Yield', 'Yield_Curve', 'VIX']:
        if c in cols:
            macro[c] = final_df[c]
            macro[c+'_Diff'] = final_df[c].diff(20)
            
    macro.dropna(inplace=True)
    
    # PCA 차원 축소 (실시간 6차원 변환)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(macro.values)
    pca = PCA(n_components=6)
    latent = pca.fit_transform(scaled)
    
    # [수정됨] 텐서 차원 붕괴 해결 (unsqueeze(0) 제거 -> 완벽한 [1, 6] 차원)
    macro_tensor = torch.FloatTensor(latent[-1:])
    
    # 3. 10 Sectors Data Fetch
    sectors = ['XLK', 'XLV', 'XLF', 'XLY', 'XLC', 'XLI', 'XLP', 'XLE', 'XLB', 'XLRE']
    sec_start = (datetime.datetime.today() - datetime.timedelta(days=150)).strftime('%Y-%m-%d')
    sec_data = yf.download(sectors, start=sec_start, end=end_date, group_by='ticker', progress=False, auto_adjust=True)
    
    d_list, w_list, m_list = [], [], []
    for sec in sectors:
        # [수정됨] SettingWithCopyWarning 제거를 위한 .copy() 사용
        df = sec_data[sec].copy()
        df['Change'] = df['Close'].pct_change()
        df = df.fillna(0)
        feats = df[['Close', 'Volume', 'Change']]
        
        d_list.append(feats.tail(20).values)
        # [수정됨] 최신 pandas 리샘플링 문법 (BM -> BME) 및 ffill 적용
        w_list.append(feats.resample('W-FRI').last().ffill().tail(20).values)
        m_list.append(feats.resample('BME').last().ffill().tail(20).values)
        
    d_tensor = torch.FloatTensor(np.array(d_list)).unsqueeze(0) 
    w_tensor = torch.FloatTensor(np.array(w_list)).unsqueeze(0)
    m_tensor = torch.FloatTensor(np.array(m_list)).unsqueeze(0)
    
    return d_tensor, w_tensor, m_tensor, macro_tensor
