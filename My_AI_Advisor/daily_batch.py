import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
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

def save_to_supabase(market, tech_probs, fin_probs, news_data, w_tech, w_news, action):
    if not supabase: return False
    try:
        # News Score (Display용 0~100)
        news_score_display = int((news_data['sentiment'] + 1) * 50)
        
        data = {
            "market_name": market,
            
            # [신규] 기술적 모델 확률 (Raw)
            "tech_prob_down": round(float(tech_probs[0]), 4),
            "tech_prob_neutral": round(float(tech_probs[1]), 4),
            "tech_prob_up": round(float(tech_probs[2]), 4),
            
            # [신규] 앙상블 최종 확률 (Final)
            "fin_prob_down": round(float(fin_probs[0]), 4),
            "fin_prob_neutral": round(float(fin_probs[1]), 4),
            "fin_prob_up": round(float(fin_probs[2]), 4),
            
            # 뉴스 데이터
            "news_sentiment": round(float(news_data['sentiment']), 4),
            "news_reliability": round(float(news_data['reliability']), 4),
            "news_summary": news_data['summary'],
            "news_score": news_score_display,
            
            # 가중치
            "w_tech": round(float(w_tech), 4),
            "w_news": round(float(w_news), 4),
            
            "action": action
        }
        
        supabase.table("prediction_logs").insert(data).execute()
        print(f"✅ [{market}] 저장 완료 | Action: {action} (Fin Up: {data['fin_prob_up']})")
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
    
    # [기술적 확률]
    t_down, t_neutral, t_up = probs[0], probs[1], probs[2]

    # 3. 뉴스 분석 수행
    news_data = get_news_analysis(market_option, search_query)
    sentiment = news_data['sentiment']   # -1 ~ 1
    reliability = news_data['reliability'] # 0 ~ 1
    
    # 4. [핵심] 앙상블 가중치 및 확률 계산
    
    # (1) 가중치 결정
    acc_model = MODEL_ACCURACY.get(market_option, 0.5)
    # [수정] 뉴스 신뢰도에 0.5(또는 0.7)를 곱해 반영 비율을 강제로 낮춤 (Conservative Weighting)
    # 아무리 신뢰도가 높아도 기술적 모델(acc_model)을 완전히 압도하지 못하게 함
    damped_reliability = news_data['reliability'] * 0.6  # 60%만 인정
    
    total_weight = acc_model + damped_reliability
    if total_weight == 0: total_weight = 1
    
    w_tech = acc_model / total_weight
    w_news = damped_reliability / total_weight
    
    # (2) 뉴스 감정을 확률 벡터로 변환
    n_up = max(0.0, sentiment)
    n_down = max(0.0, -sentiment)
    n_neutral = 1.0 - abs(sentiment)
    
    # (3) 최종 확률 앙상블 (Weighted Sum)
    final_down = (t_down * w_tech) + (n_down * w_news)
    final_neutral = (t_neutral * w_tech) + (n_neutral * w_news)
    final_up = (t_up * w_tech) + (n_up * w_news)
    
    # 합이 1이 되도록 정규화
    total_prob = final_down + final_neutral + final_up
    final_down /= total_prob
    final_neutral /= total_prob
    final_up /= total_prob
    
    # 5. 의사 결정 (Threshold 0.45 Rule)
    prob_map = {
        "SELL": final_down,
        "HOLD": final_neutral,
        "BUY": final_up
    }
    
    best_action = max(prob_map, key=prob_map.get)
    max_prob = prob_map[best_action]
    
    if max_prob < 0.45:
        final_action = "HOLD"
        print(f"⚖️ 판단 보류: 최대 확률({max_prob:.4f})이 임계값(0.45) 미달 -> HOLD 강제")
    else:
        final_action = best_action
        
    # 6. 저장 (변경된 스키마에 맞춰 인자 전달)
    save_to_supabase(
        market_option, 
        [t_down, t_neutral, t_up],        # 기술적 확률
        [final_down, final_neutral, final_up], # 최종 확률
        news_data, 
        w_tech, 
        w_news,
        final_action
    )

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
