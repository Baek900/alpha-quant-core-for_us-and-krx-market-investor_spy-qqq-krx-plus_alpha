import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import time
from supabase import create_client, Client

# 모듈 임포트
from model_def import StockClassifierModel
from data_loader import get_us_data, get_kr_data
from news_agent import get_news_analysis 

# 환경변수
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL:
    print("⚠️ Supabase 설정 없음. 로컬 테스트 모드")
    supabase = None
else:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# [설정] 모델별 검증된 정확도 (Baseline Accuracy)
MODEL_ACCURACY = {
    "S&P 500 (SPY)": 0.53,
    "NASDAQ (QQQ)": 0.58,
    "KOSPI (Korea)": 0.42
}

def save_to_supabase(market, probs_list, news_data, final_prob, w_tech, w_news):
    if not supabase: return False
    try:
        # Action 결정 (임계값 0.45 / 0.2)
        action = "BUY" if final_prob >= 0.45 else ("SELL" if final_prob <= 0.2 else "HOLD")
        
        # News Score (Display용 0~100)
        news_score_display = int((news_data['sentiment'] + 1) * 50)
        
        data = {
            "market_name": market,
            "prob_down": float(probs_list[0]),
            "prob_neutral": float(probs_list[1]),
            "tech_prob": float(probs_list[2]), # 상승 확률
            
            "news_sentiment": float(news_data['sentiment']),
            "news_reliability": float(news_data['reliability']),
            "news_summary": news_data['summary'],
            "news_score": news_score_display,
            
            "final_prob": round(float(final_prob), 4),
            "w_tech": round(float(w_tech), 2),
            "w_news": round(float(w_news), 2),
            "action": action
        }
        
        supabase.table("prediction_logs").insert(data).execute()
        print(f"✅ [{market}] 저장 완료 | Tech: {data['tech_prob']}(w={data['w_tech']}) + News: {data['news_sentiment']}(w={data['w_news']}) -> Final: {data['final_prob']}")
        return True
    except Exception as e:
        print(f"❌ DB 저장 실패: {e}")
        return False

def run_analysis_batch(market_option):
    print(f"🚀 배치 시작: {market_option}")
    
    # 1. 설정 로드
    if market_option == "NASDAQ (QQQ)":
        MODEL_FILE = os.path.join(BASE_DIR, "models", "us_sector_ai_model_qqq.pth")
        SECTORS = ['XLK', 'XLV', 'XLF', 'XLY', 'XLC', 'XLI', 'XLP', 'XLE', 'XLB', 'XLRE']
        search_query = "latest market sentiment news for NASDAQ 100 QQQ ETF today macro economics"
    elif market_option == "S&P 500 (SPY)":
        MODEL_FILE = os.path.join(BASE_DIR, "models", "us_spy_target_best_model.pth")
        SECTORS = ['XLK', 'XLV', 'XLF', 'XLY', 'XLC', 'XLI', 'XLP', 'XLE', 'XLB', 'XLRE']
        search_query = "latest market sentiment news for S&P 500 SPY ETF today macro economics"
    else: # KOSPI
        MODEL_FILE = os.path.join(BASE_DIR, "models", "kospi_model.pth")
        SECTORS = []
        search_query = "latest south korea kospi stock market news today macro economics"

    # 2. 기술적 모델 예측
    device = torch.device('cpu')
    model = StockClassifierModel().to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
        model.eval()
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    if "KOSPI" in market_option:
        input_tensor, _ = get_kr_data()
    else:
        input_tensor, _ = get_us_data(SECTORS)

    if input_tensor is None:
        print("❌ 데이터 수집 실패")
        return

    with torch.no_grad():
        logits = model(input_tensor)
        if isinstance(logits, tuple): logits = logits[1]
        probs = F.softmax(logits, dim=1).squeeze().numpy()
    
    # [하락, 횡보, 상승]
    tech_down, tech_neutral, tech_up = probs[0], probs[1], probs[2]

    # 3. 뉴스 분석 수행
    news_data = get_news_analysis(market_option, search_query)
    
    # 4. [핵심] 앙상블 가중치 계산 (Dynamic Weighting)
    # Baseline Accuracy
    acc_model = MODEL_ACCURACY.get(market_option, 0.5) 
    # News Reliability
    rel_news = news_data['reliability']
    
    # 가중치 비율 계산 (정확도 vs 신뢰도)
    total_weight = acc_model + rel_news
    if total_weight == 0: total_weight = 1 # 방어 코드
    
    w_tech = acc_model / total_weight
    w_news = rel_news / total_weight
    
    # 뉴스 감정점수(-1~1)를 확률(0~1)로 변환
    news_prob = 0.5 + (news_data['sentiment'] / 2.0)
    
    # 최종 확률 계산
    final_up = (tech_up * w_tech) + (news_prob * w_news)

    # 5. 저장
    save_to_supabase(market_option, [tech_down, tech_neutral, tech_up], news_data, final_up, w_tech, w_news)

if __name__ == "__main__":
    mode = "all"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

    if mode == "kr": markets = ["KOSPI (Korea)"]
    elif mode == "us": markets = ["NASDAQ (QQQ)", "S&P 500 (SPY)"]
    else: markets = ["NASDAQ (QQQ)", "S&P 500 (SPY)", "KOSPI (Korea)"]

    print(f"🔄 Mode: {mode} / Targets: {markets}")
    
    for i, m in enumerate(markets):
        run_analysis_batch(m)
        
        if i < len(markets) - 1:
            print("⏳ API 요청 제한 방지를 위해 20초 대기 중...")
            time.sleep(20)
