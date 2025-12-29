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

# [수정된 함수]
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
        
        # 미국: 장 시작 전(07:00~09:00 EST)에 돌림.
        # 따라서 '오늘 날짜'인지가 중요함. 과거로 돌리지 않고 '현재 시각'을 기준으로 함.
        # 뉴스는 "실행 시점(현재)"까지 나온 모든 속보를 반영하는 것이 유리함.
        cutoff = local_now 

    return tz, cutoff, market_type

def save_prediction(market, tech_probs, fin_probs, news_data, w_tech, w_news, action):
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
            "action": action
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
    
    # [수정] cutoff_time(현재 시각)의 날짜(Date)가 휴장일인지 체크
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

    # 4. 앙상블 (Ensemble)
    acc_model = MODEL_ACCURACY.get(market_option, 0.5)
    w_tech = acc_model / (acc_model + (news_obj
