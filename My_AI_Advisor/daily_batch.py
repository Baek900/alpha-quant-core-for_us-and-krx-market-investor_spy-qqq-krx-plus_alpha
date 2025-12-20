import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from supabase import create_client, Client

# 만든 모듈들 가져오기
from model_def import StockClassifierModel
from news_agent import analyze_market_sentiment
from data_loader import get_us_data, get_kr_data # 방금 만든 파일

# 환경변수에서 키 가져오기 (GitHub Actions용)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# 로컬 테스트용 (키가 없을 때 에러 방지)
if not SUPABASE_URL:
    print("⚠️ Supabase URL이 설정되지 않았습니다. DB 저장을 건너뜁니다.")
    supabase = None
else:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def save_to_supabase(market, tech_prob, news_score, final_prob, w_tech, w_news):
    if not supabase: return False
    try:
        action = "BUY" if final_prob >= 0.45 else ("SELL" if final_prob <= 0.2 else "HOLD")
        
        data = {
            "market_name": market,
            "tech_prob": round(float(tech_prob), 4),
            "news_score": int(news_score),
            "final_prob": round(float(final_prob), 4),
            "w_tech": float(w_tech),
            "w_news": float(w_news),
            "action": action
        }
        # DB에 저장
        supabase.table("prediction_logs").insert(data).execute()
        print(f"✅ [{market}] Supabase 저장 완료")
        return True
    except Exception as e:
        print(f"❌ [{market}] 저장 실패: {e}")
        return False

def run_analysis_batch(market_option):
    print(f"🚀 분석 시작: {market_option}")
    
    # 1. 모델 경로
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

    # 3. 데이터 수집 (data_loader 사용)
    if "KOSPI" in market_option:
        input_tensor, _ = get_kr_data()
    else:
        input_tensor, _ = get_us_data(SECTORS)

    if input_tensor is None:
        print("❌ 데이터 수집 실패")
        return

    # 4. 예측
    with torch.no_grad():
        logits = model(input_tensor)
        if isinstance(logits, tuple): logits = logits[1]
        probs = F.softmax(logits, dim=1).squeeze().numpy()
    
    tech_up = probs[2]

    # 5. 뉴스 분석 (Gemini)
    try:
        news_score, _ = analyze_market_sentiment(market_option)
    except:
        news_score = 50 # 에러 시 중립
        print("⚠️ 뉴스 분석 에러 (기본값 50 적용)")
        
    news_prob = news_score / 100.0

    # 6. 앙상블 (가중치 0.7:0.3 고정)
    W_TECH = 0.7
    W_NEWS = 0.3
    final_up = (tech_up * W_TECH) + (news_prob * W_NEWS)

    # 7. DB 저장
    save_to_supabase(market_option, tech_up, news_score, final_up, W_TECH, W_NEWS)

if __name__ == "__main__":
    markets = ["NASDAQ (QQQ)", "S&P 500 (SPY)", "KOSPI (Korea)"]
    for m in markets:
        run_analysis_batch(m)
