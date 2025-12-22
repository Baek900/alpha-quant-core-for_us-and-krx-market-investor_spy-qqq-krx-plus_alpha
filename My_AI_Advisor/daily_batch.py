import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from supabase import create_client, Client

# 모델 정의 불러오기
from model_def import StockClassifierModel
from data_loader import get_us_data, get_kr_data

# 환경변수 로드
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL:
    print("⚠️ Supabase 설정 없음. 로컬 테스트 모드")
    supabase = None
else:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def save_to_supabase(market, probs_list, news_score, final_prob, w_tech, w_news):
    """
    probs_list: [prob_down, prob_neutral, prob_up]
    """
    if not supabase: return False
    try:
        # Action 기준은 최종 상승 확률(final_prob)로 판단
        action = "BUY" if final_prob >= 0.45 else ("SELL" if final_prob <= 0.2 else "HOLD")
        
        data = {
            "market_name": market,
            "prob_down": round(float(probs_list[0]), 4),    # [NEW] 하락
            "prob_neutral": round(float(probs_list[1]), 4), # [NEW] 횡보
            "tech_prob": round(float(probs_list[2]), 4),    # 상승 (기존 tech_prob 유지)
            "news_score": int(news_score),
            "final_prob": round(float(final_prob), 4),
            "w_tech": float(w_tech),
            "w_news": float(w_news),
            "action": action
        }
        
        supabase.table("prediction_logs").insert(data).execute()
        print(f"✅ [{market}] 저장 완료 (Down:{data['prob_down']}, Neut:{data['prob_neutral']}, Up:{data['tech_prob']})")
        return True
    except Exception as e:
        print(f"❌ [{market}] 저장 실패: {e}")
        return False

def run_analysis_batch(market_option):
    print(f"🚀 분석 시작: {market_option}")
    
    # 1. 모델 경로 및 섹터 설정
    if market_option == "NASDAQ (QQQ)":
        MODEL_FILE = os.path.join(BASE_DIR, "models", "us_sector_ai_model_qqq.pth")
        SECTORS = ['XLK', 'XLV', 'XLF', 'XLY', 'XLC', 'XLI', 'XLP', 'XLE', 'XLB', 'XLRE']
    elif market_option == "S&P 500 (SPY)":
        MODEL_FILE = os.path.join(BASE_DIR, "models", "us_spy_target_best_model.pth")
        SECTORS = ['XLK', 'XLV', 'XLF', 'XLY', 'XLC', 'XLI', 'XLP', 'XLE', 'XLB', 'XLRE']
    else: # KOSPI
        MODEL_FILE = os.path.join(BASE_DIR, "models", "kospi_model.pth")
        SECTORS = []

    # 2. 모델 로드
    device = torch.device('cpu')
    model = StockClassifierModel().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
        model.eval()
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    # 3. 데이터 수집
    if "KOSPI" in market_option:
        input_tensor, _ = get_kr_data()
    else:
        input_tensor, _ = get_us_data(SECTORS)

    if input_tensor is None:
        print("❌ 데이터 수집 실패")
        return

    # 4. 예측 수행 (3가지 클래스 확률 추출)
    with torch.no_grad():
        logits = model(input_tensor)
        if isinstance(logits, tuple): logits = logits[1]
        probs = F.softmax(logits, dim=1).squeeze().numpy()
    
    # [중요] 모델의 출력 순서는 학습 시 라벨 순서인 [0:하락, 1:횡보, 2:상승] 입니다.
    tech_down = probs[0]
    tech_neutral = probs[1]
    tech_up = probs[2]

    # 5. 뉴스 분석 (일단 기본값, 추후 News Agent가 업데이트)
    news_score = 50 
    news_prob = 0.5 

    # 6. 앙상블 (상승 확률만 보정하여 Signal 생성용으로 사용)
    W_TECH = 0.7
    W_NEWS = 0.3
    final_up = (tech_up * W_TECH) + (news_prob * W_NEWS)

    # 7. DB 저장 (3가지 확률 모두 전달)
    save_to_supabase(market_option, [tech_down, tech_neutral, tech_up], news_score, final_up, W_TECH, W_NEWS)

if __name__ == "__main__":
    mode = "all"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

    if mode == "kr":
        markets = ["KOSPI (Korea)"]
    elif mode == "us":
        markets = ["NASDAQ (QQQ)", "S&P 500 (SPY)"]
    else:
        markets = ["NASDAQ (QQQ)", "S&P 500 (SPY)", "KOSPI (Korea)"]

    print(f"🔄 Mode: {mode} / Targets: {markets}")

    for m in markets:
        run_analysis_batch(m)
