import os
import datetime
import pytz
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from supabase import create_client, Client

# -----------------------------------------------------------------------------
# [Next Step 타겟] 다음 단계에서 업데이트할 V13 전용 모듈들
# -----------------------------------------------------------------------------
from v13_data_loader import get_market_regime_and_features
from v13_model import MetaLabelingNet

print("="*80)
print("🔥 [Titan Flow V13.8 Batch Bot] 실전 하이브리드 레버리지 + 시계열 로그")
print("="*80)

# ==============================================================================
# 1. 환경 설정 및 상수 정의
# ==============================================================================
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
V13_SECTORS = ['XLV', 'XLF', 'XLY', 'XLI', 'XLP', 'XLE', 'XLB', 'XLU', 'IGV', 'SOXX']
TRADING_MAP = {
    'SOXX': 'SOXL', 'IGV': 'TECL', 'XLF': 'FAS', 'XLV': 'CURE',
    'XLY': 'XLY', 'XLI': 'XLI', 'XLP': 'XLP', 'XLE': 'XLE', 'XLB': 'XLB', 'XLU': 'XLU'
}

# ⚙️ V13.8 핵심 파라미터
AI_APPROVAL_THRESHOLD = 0.30  # 허들 완화 (타협형)
HARD_STOP_LIMIT = 0.88       # -12% 하드스탑
TRAILING_STOP_LIMIT = 0.85   # -15% 트레일링스탑
COOLDOWN_PERIOD = 5          # 휩소 철벽 방어 (5일)

# ==============================================================================
# 2. 모델 로드 (V13 가중치)
# ==============================================================================
print("📥 V13.8 AI 문지기(Meta-Labeler) 로드 중...")
model = MetaLabelingNet(input_dim=13).to(DEVICE)
# 향후 GitHub Actions 환경에 맞게 모델 가중치 경로 수정
model.load_state_dict(torch.load('models/v13_metalabel_weights.pth', map_location=DEVICE))
model.eval()

# ==============================================================================
# 3. 메인 배치 실행 함수
# ==============================================================================
def run_daily_batch():
    # 기준일 설정 (미국 시간 기준)
    us_tz = pytz.timezone('US/Eastern')
    today_date = datetime.datetime.now(us_tz).date()
    today_str = today_date.strftime('%Y-%m-%d')
    print(f"📅 기준일 (US Time): {today_str}")

    # 1️⃣ DB에서 어제(최근)의 봇 상태 10개 섹터 읽어오기
    print("🔍 Supabase에서 이전 포트폴리오 상태를 조회합니다...")
    response = supabase.table('v13_daily_log').select('*').order('target_date', desc=True).limit(10).execute()
    past_states = {row['sector']: row for row in response.data}
    
    if not past_states:
        print("🚨 [경고] DB에 초기 상태값이 없습니다. 마이그레이션 SQL을 확인하세요.")
        return

    # 2️⃣ 실전 호가 데이터(오늘의 종가) 수집
    print("📊 야후 파이낸스에서 실전 티커 종가 수집 중...")
    trading_tickers = list(TRADING_MAP.values())
    raw_trade = yf.download(trading_tickers + ['SPY'], period='5d', progress=False)
    today_closes = raw_trade['Close'].iloc[-1]

    # 3️⃣ 데이터 로더를 통한 피처 추출 (다음 스텝에서 구현할 부분)
    # is_bull_market (SPY 200일선 기준), sec_sigs(기술적 시그널), mac_feats(매크로 지표)
    is_bull_market, features_dict = get_market_regime_and_features(today_date)

    # 4️⃣ V13.8 상태 업데이트 및 스탑로스 계산
    new_states = {}
    sectors_to_remove = []
    
    with torch.no_grad():
        for sec in V13_SECTORS:
            prev = past_states[sec]
            trade_tic = TRADING_MAP[sec]
            cur_price = today_closes[trade_tic]
            
            sec_sigs = features_dict[sec]['sigs']
            mac_feats = features_dict[sec]['macro']
            
            # 현재 상태 복사 및 쿨다운 차감
            state = prev.copy()
            if state['cooldown_timer'] > 0:
                state['cooldown_timer'] -= 1

            if state['is_holding']:
                state['days_held'] += 1
                state['high_p'] = max(state['high_p'], float(cur_price))
                
                entry_p = state['entry_p']
                high_p = state['high_p']
                
                # 🚨 스탑로스 체크
                if cur_price <= entry_p * HARD_STOP_LIMIT or cur_price <= high_p * TRAILING_STOP_LIMIT:
                    sectors_to_remove.append(sec)
                    state['cooldown_timer'] = COOLDOWN_PERIOD
                # 정상 매도 시그널
                elif -1 in sec_sigs:
                    sectors_to_remove.append(sec)
                # ⏱️ 20일 타임아웃 롤오버 심사
                elif state['days_held'] >= 20:
                    if 1 in sec_sigs and state['cooldown_timer'] == 0:
                        x_tensor = torch.FloatTensor(np.concatenate([sec_sigs, mac_feats])).unsqueeze(0).to(DEVICE)
                        if model(x_tensor).item() >= AI_APPROVAL_THRESHOLD:
                            # 롤오버 성공
                            state['days_held'] = 0
                            state['entry_p'] = float(cur_price)
                            state['high_p'] = float(cur_price)
                        else:
                            sectors_to_remove.append(sec)
                    else:
                        sectors_to_remove.append(sec)
            
            new_states[sec] = state

        # 매도 확정 종목 비우기
        for sec in sectors_to_remove:
            new_states[sec]['is_holding'] = False
            new_states[sec]['target_weight'] = 0.0

        # 🎯 신규 진입 심사
        for sec in V13_SECTORS:
            state = new_states[sec]
            if not state['is_holding'] and sec not in sectors_to_remove and state['cooldown_timer'] == 0:
                sec_sigs = features_dict[sec]['sigs']
                if 1 in sec_sigs:
                    mac_feats = features_dict[sec]['macro']
                    x_tensor = torch.FloatTensor(np.concatenate([sec_sigs, mac_feats])).unsqueeze(0).to(DEVICE)
                    if model(x_tensor).item() >= AI_APPROVAL_THRESHOLD:
                        # 매수 승인 (실제 진입가는 내일 시가지만, DB 기록용으로 오늘 종가 세팅)
                        cur_price = float(today_closes[TRADING_MAP[sec]])
                        state['is_holding'] = True
                        state['days_held'] = 0
                        state['entry_p'] = cur_price
                        state['high_p'] = cur_price

        # ⚖️ 타겟 비중(Weight) 리밸런싱
        active_sectors = [sec for sec in V13_SECTORS if new_states[sec]['is_holding']]
        num_holdings = len(active_sectors)
        MAX_WEIGHT = 0.50 if is_bull_market else 0.33
        
        # 비보유 종목 비중 0 초기화
        for sec in V13_SECTORS:
            new_states[sec]['target_weight'] = 0.0

        if num_holdings > 0:
            target_w = min(1.0 / num_holdings, MAX_WEIGHT)
            for sec in active_sectors:
                # 잦은 리밸런싱 방지 (5% 이상 차이날 때만 비중 조절)
                prev_w = past_states[sec]['target_weight']
                if prev_w == 0.0 or abs(prev_w - target_w) > 0.05:
                    new_states[sec]['target_weight'] = float(target_w)
                else:
                    new_states[sec]['target_weight'] = float(prev_w)

    # 5️⃣ Supabase DB에 새로운 시계열 로그 Insert
    print("💾 연산 완료! Supabase에 내일자 타겟 비중을 기록합니다...")
    insert_data = []
    for sec in V13_SECTORS:
        st = new_states[sec]
        insert_data.append({
            'target_date': today_str,
            'sector': sec,
            'trade_ticker': TRADING_MAP[sec],
            'target_weight': st['target_weight'],
            'is_holding': st['is_holding'],
            'days_held': st['days_held'],
            'entry_p': st['entry_p'],
            'high_p': st['high_p'],
            'cooldown_timer': st['cooldown_timer']
        })
    
    # DB Insert 실행
    supabase.table('v13_daily_log').insert(insert_data).execute()
    print("✅ [성공] V13.8 배치 작업이 완벽하게 종료되었습니다.")

if __name__ == "__main__":
    run_daily_batch()
