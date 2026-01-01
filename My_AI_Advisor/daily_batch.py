import os
import sys
from dotenv import load_dotenv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

import torch
import torch.nn.functional as F
from supabase import create_client, Client
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

# 모듈 임포트
from model_def import StockClassifierModel
from data_loader import get_us_data, get_kr_data
from news_agent import get_news_analysis

# 환경변수
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_ACCURACY = {
    "S&P 500 (SPY)": 0.53, "NASDAQ (QQQ)": 0.58, "KOSPI (Korea)": 0.42
}
IS_TEST_MODE = os.environ.get("TEST_MODE", "false").lower() == "true"
PREDICTION_TABLE = "prediction_logs_test" if IS_TEST_MODE else "prediction_logs"

if IS_TEST_MODE:
    print(f"⚠️ [TEST MODE] 데이터가 '{PREDICTION_TABLE}'에 저장됩니다.")
    
# [NEW] 2026년 휴장일 데이터 (YYYY-MM-DD)
HOLIDAYS_2026 = {
    "us": {
        "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03",
        "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
        "2026-11-26", "2026-12-25"
    },
    "kr": {
        "2026-01-01", "2026-02-16", "2026-02-17", "2026-02-18",
        "2026-03-02", "2026-05-01", "2026-05-05", "2026-05-25",
        "2026-06-06", "2026-08-17", "2026-09-24", "2026-09-25",
        "2026-09-26", "2026-10-05", "2026-10-09", "2026-12-25",
        "2026-12-31"
    }
}

def is_market_open(market_type, current_date):
    """
    오늘이 휴장일인지 확인
    """
    date_str = current_date.strftime("%Y-%m-%d")
    
    # 1. 주말 체크 (5=토, 6=일)
    if current_date.weekday() >= 5:
        print(f"🛑 [Market Closed] {date_str} is Weekend.")
        return False
        
    # 2. 휴장일 체크
    if date_str in HOLIDAYS_2026.get(market_type, set()):
        print(f"🛑 [Market Closed] {date_str} is a Holiday.")
        return False
        
    return True

def get_signal_cutoff(market_option):
    now_utc = datetime.now(ZoneInfo("UTC"))
    
    if "Korea" in market_option:
        tz = ZoneInfo("Asia/Seoul")
        market_type = "kr"
        local_now = now_utc.astimezone(tz)
        # 한국: 아침 7~8시에 돌리므로, '오늘 아침 6시'까지의 뉴스 반영
        cutoff = local_now.replace(hour=6, minute=0, second=0, microsecond=0)
    else:
        tz = ZoneInfo("US/Eastern")
        market_type = "us"
        local_now = now_utc.astimezone(tz)
        # 미국: 장 시작 전(오전)에 실행하므로 '현재 시각'을 그대로 사용
        cutoff = local_now 

    return tz, cutoff, market_type

# [수정] tech_action 인자 추가 및 데이터 저장 로직 반영
def save_prediction(market, tech_probs, fin_probs, news_data, w_tech, w_news, action, tech_action):
    if not supabase: return
    try:
        data = {
            "market_name": market,
            "tech_prob_down": round(float(tech_probs[0]), 4),
            "tech_prob_neutral": round(float(tech_probs[1]), 4),
            "tech_prob_up": round(float(tech_probs[2]), 4),
            "fin_prob_down": round(float(fin_probs[0]), 4),
            "fin_prob_neutral": round(float(fin_probs[1]), 4),
            "fin_prob_up": round(float(fin_probs[2]), 4),
            "w_tech": round(float(w_tech), 4),
            "w_news": round(float(w_news), 4),
            "news_sentiment": news_data.sentiment,
            "news_reliability": news_data.reliability,
            "news_summary": news_data.final_summary,
            "news_score": int((news_data.sentiment + 1) * 50),
            "news_risk_score": news_data.risk_score,
            "news_metadata": news_data.dict(),
            "action": action,
            "tech_action": tech_action  # [NEW] 기술적 모델 단독 신호 저장
        }
        supabase.table(PREDICTION_TABLE).insert(data).execute()
        print(f"✅ [Prediction] 저장 완료 ({PREDICTION_TABLE}): {market}")
    except Exception as e:
        print(f"❌ [Prediction] 저장 실패: {e}")

def run_prediction_batch(market_option):
    print(f"🚀 [Prediction Job] 시작: {market_option}")
    
    # 1. 시간 및 휴장일 확인
    tz, cutoff_time, market_type = get_signal_cutoff(market_option)
    cutoff_str = cutoff_time.strftime("%Y-%m-%d %H:%M")
    
    # 휴장일 체크
    if not is_market_open(market_type, cutoff_time.date()):
        print("💤 휴장일이므로 예측 프로세스를 건너뜁니다.")
        return

    print(f"🔒 News Cutoff Time: {cutoff_str} ({tz})")

    # 2. 기술적 모델 (PyTorch)
    if market_option == "NASDAQ (QQQ)":
        MODEL_FILE = os.path.join(BASE_DIR, "models", "us_sector_ai_model_qqq.pth")
        SECTORS = ['XLK', 'XLV', 'XLF', 'XLY', 'XLC', 'XLI', 'XLP', 'XLE', 'XLB', 'XLRE']
        query = "latest market sentiment news for NASDAQ 100 QQQ ETF today macro economics"
    elif market_option == "S&P 500 (SPY)":
        MODEL_FILE = os.path.join(BASE_DIR, "models", "us_spy_target_best_model.pth")
        SECTORS = ['XLK', 'XLV', 'XLF', 'XLY', 'XLC', 'XLI', 'XLP', 'XLE', 'XLB', 'XLRE']
        query = "latest market sentiment news for S&P 500 SPY ETF today macro economics"
    else:
        MODEL_FILE = os.path.join(BASE_DIR, "models", "kospi_model.pth")
        SECTORS = []
        query = "latest south korea kospi stock market news today macro economics"

    # 모델 로드 및 추론
    device = torch.device('cpu')
    model = StockClassifierModel().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
        model.eval()
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    if market_type == "kr": input_tensor, _ = get_kr_data()
    else: input_tensor, _ = get_us_data(SECTORS)

    if input_tensor is None: return

    with torch.no_grad():
        logits = model(input_tensor)
        if isinstance(logits, tuple): logits = logits[1]
        probs = F.softmax(logits, dim=1).squeeze().numpy()
    
    t_down, t_neutral, t_up = probs[0], probs[1], probs[2]

    # 3. 뉴스 에이전트
    news_obj = get_news_analysis(market_option, query, cutoff_str, str(tz))
    if not news_obj: return

    # 4. 앙상블 (Ensemble: Balanced Risk Adjustment)
    acc_model = MODEL_ACCURACY.get(market_option, 0.5)
    
    # 4-1. 가중치 계산 (신뢰도 반영)
    w_tech = acc_model / (acc_model + (news_obj.reliability**2 * 0.8))
    w_news = 1 - w_tech
    
    # 4-2. [핵심 수정] 리스크를 반영한 심리 점수 보정 (Gravity Model)
    # Risk Score(0~1)가 높을수록 Sentiment를 아래로(Bearish 쪽으로) 끌어당김
    # 민감도 계수 0.5 적용: 리스크가 MAX(1.0)여도 Sentiment를 -0.5만큼만 낮춤 (상승 모멘텀이 +0.8이면 여전히 +0.3 매수 유지)
    
    risk_gravity = news_obj.risk_score * 0.5
    adjusted_sentiment = news_obj.sentiment - risk_gravity
    
    # 값 범위 클리핑 (-1.0 ~ 1.0)
    adjusted_sentiment = max(-1.0, min(1.0, adjusted_sentiment))
    
    # 보정된 Sentiment를 확률로 변환
    news_prob_up = max(0, adjusted_sentiment)
    news_prob_down = max(0, -adjusted_sentiment)
    
    # 나머지는 중립 확률로 (합이 1이 안 될 경우를 대비)
    # 예: adj_sentiment가 0.5면 -> Up 0.5, Down 0 -> Neutral 0.5
    # 예: adj_sentiment가 -0.3이면 -> Up 0, Down 0.3 -> Neutral 0.7
    news_prob_neutral = 1.0 - (news_prob_up + news_prob_down)
    
    # 4-3. 최종 앙상블 계산
    final_down = (t_down * w_tech) + (news_prob_down * w_news)
    final_neutral = (t_neutral * w_tech) + (news_prob_neutral * w_news)
    final_up = (t_up * w_tech) + (news_prob_up * w_news)
    
    # Normalize (부동소수점 오차 보정)
    total = final_down + final_neutral + final_up
    final_down, final_neutral, final_up = final_down/total, final_neutral/total, final_up/total
    
    # 5. 최종 Action 결정 (Ensemble)
    prob_map = {"SELL": final_down, "HOLD": final_neutral, "BUY": final_up}
    best_action = max(prob_map, key=prob_map.get)
    
    # 확신 부족 시 HOLD 처리 (Threshold: 45%)
    if prob_map[best_action] < 0.45: 
        best_action = "HOLD"

    # [NEW] 5-1. 기술적 모델 단독 Action 결정 (Tech Only)
    # 0.45를 넘은 확률 중 가장 높은 값을 신호로 보고, 없으면 HOLD
    tech_map = {"SELL": t_down, "HOLD": t_neutral, "BUY": t_up}
    best_tech_action = max(tech_map, key=tech_map.get)
    
    if tech_map[best_tech_action] < 0.45:
        best_tech_action = "HOLD"

    # 6. 저장 (tech_action 추가 전달)
    save_prediction(market_option, [t_down, t_neutral, t_up], 
                    [final_down, final_neutral, final_up], 
                    news_obj, w_tech, w_news, best_action, best_tech_action)

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "us"
    markets = ["NASDAQ (QQQ)", "S&P 500 (SPY)"] if target == "us" else ["KOSPI (Korea)"]
    for m in markets: run_prediction_batch(m)
