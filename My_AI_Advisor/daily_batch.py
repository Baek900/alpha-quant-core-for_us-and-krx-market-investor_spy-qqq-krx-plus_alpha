import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import lightgbm as lgb
from datetime import datetime
from zoneinfo import ZoneInfo
from supabase import create_client, Client
from dotenv import load_dotenv

# 모듈 임포트
from model_def import StockClassifierModel
from data_loader import get_us_v2_data, get_kr_v2_data
from news_agent import get_news_analysis

# 1. 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 모델 파일들이 위치한 절대 경로 (My_AI_Advisor/models)
MODEL_DIR = os.path.join(BASE_DIR, "models")

load_dotenv(os.path.join(BASE_DIR, ".env"))

# 2. Supabase 및 테스트 모드 설정
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None

IS_TEST_MODE = os.environ.get("TEST_MODE", "false").lower() == "true"
PREDICTION_TABLE = "prediction_logs_test" if IS_TEST_MODE else "prediction_logs"

# 3. 시장별 설정 (모델 파일명 직접 지정)
MARKET_CONFIG = {
    "NASDAQ (QQQ)": {
        "is_kr": False, "seq_len": 14, "num_assets": 10, "hidden": 128, "threshold": 0.3,
        "assets": ['XLK','XLV','XLF','XLY','XLC','XLI','XLP','XLE','XLB','XLRE'],
        "model_file": "us_sector_ai_model_qqq_kmeans_5class_v2.pth"
    },
    "S&P 500 (SPY)": {
        "is_kr": False, "seq_len": 14, "num_assets": 10, "hidden": 128, "threshold": 0.3,
        "assets": ['XLK','XLV','XLF','XLY','XLC','XLI','XLP','XLE','XLB','XLRE'],
        "model_file": "us_sector_ai_model_spy_kmeans_5class_v1.pth"
    },
    "KOSPI (Korea)": {
        "is_kr": True, "seq_len": 20, "num_assets": 12, "hidden": 256, "threshold": 0.3,
        "lstm_w": 0.3, "lgbm_w": 0.7,
        "sectors": ['091160.KS','305720.KS','091170.KS','091180.KS','117460.KS','102120.KS','117700.KS','139230.KS','140700.KS','266370.KS'],
        "macro": ['USDKRW=X', 'SPY'],
        "model_file": "kospi_macro_ai_model_5class_v2.pth",
        "lgbm_file": "lgbm_model_v2_721.txt"
    }
}

# 2026년 휴장일 검증 (생략하지 않고 유지)
HOLIDAYS_2026 = {
    "us": {"2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25"},
    "kr": {"2026-01-01", "2026-02-16", "2026-02-17", "2026-02-18", "2026-03-02", "2026-05-01", "2026-05-05", "2026-05-25", "2026-06-06", "2026-08-17", "2026-09-24", "2026-09-25", "2026-09-26", "2026-10-05", "2026-10-09", "2026-12-25", "2026-12-31"}
}

def is_market_open(market_type, current_date):
    date_str = current_date.strftime("%Y-%m-%d")
    if current_date.weekday() >= 5 or date_str in HOLIDAYS_2026.get(market_type, set()):
        return False
    return True

def get_market_info(market_option):
    now_utc = datetime.now(ZoneInfo("UTC"))
    if "Korea" in market_option:
        tz, m_type = ZoneInfo("Asia/Seoul"), "kr"
        local_now = now_utc.astimezone(tz)
        cutoff = local_now.replace(hour=6, minute=0, second=0, microsecond=0)
    else:
        tz, m_type = ZoneInfo("US/Eastern"), "us"
        local_now = now_utc.astimezone(tz)
        if local_now.hour < 12:
            cutoff = (local_now - timedelta(days=1)).replace(hour=20, minute=0, second=0)
        else:
            cutoff = local_now.replace(hour=20, minute=0, second=0)
    return tz, cutoff, m_type

def save_prediction(market, tech_probs, final_probs, news_data, best_action, tech_action, weights):
    if not supabase: return
    try:
        data = {
            "market_name": market,
            "tech_prob_down": round(float(tech_probs[0] + tech_probs[1]), 4),
            "tech_prob_neutral": round(float(tech_probs[2]), 4),
            "tech_prob_up": round(float(tech_probs[3] + tech_probs[4]), 4),
            "fin_prob_down": round(float(final_probs[0]), 4),
            "fin_prob_neutral": round(float(final_probs[1]), 4),
            "fin_prob_up": round(float(final_probs[2]), 4),
            "w_tech": round(float(weights[0]), 4),
            "w_news": round(float(weights[1]), 4),
            "news_sentiment": news_data.sentiment if news_data else 0.0,
            "news_reliability": news_data.reliability if news_data else 0.5,
            "news_summary": news_data.final_summary if news_data else "No news",
            "news_score": int((news_data.sentiment + 1) * 50) if news_data else 50,
            "action": best_action,
            "tech_action": tech_action,
            "news_risk_score": news_data.risk_score if news_data else 0.0,
            "news_metadata": news_data.dict() if news_data else {}
        }
        supabase.table(PREDICTION_TABLE).insert(data).execute()
        print(f"✅ [DB] {market} saved successfully.")
    except Exception as e:
        print(f"❌ [DB Error]: {e}")

def run_prediction_v2(market_option):
    cfg = MARKET_CONFIG[market_option]
    tz, cutoff_time, m_type = get_market_info(market_option)
    
    if not is_market_open(m_type, cutoff_time.date()):
        print(f"💤 {market_option}: 휴장일 스킵")
        return

    print(f"🚀 [Batch] {market_option} 시작")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 데이터 로드
    if cfg["is_kr"]:
        input_tensor = get_kr_v2_data(cfg["sectors"], cfg["macro"], cfg["seq_len"])
    else:
        input_tensor = get_us_v2_data(cfg["assets"], cfg["seq_len"])
    
    if input_tensor is None: return
    input_tensor = input_tensor.to(device)

    # 2. LSTM 추론 (경로를 직접 조인)
    model = StockClassifierModel(num_sectors=cfg["num_assets"], hidden_dim=cfg["hidden"], is_kr=cfg["is_kr"]).to(device)
    model_full_path = os.path.join(MODEL_DIR, cfg["model_file"])
    model.load_state_dict(torch.load(model_full_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        logits = model(input_tensor)
        lstm_probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    # 3. KOSPI LGBM 앙상블
    if cfg["is_kr"]:
        lgbm_full_path = os.path.join(MODEL_DIR, cfg["lgbm_file"])
        lgbm_model = lgb.Booster(model_file=lgbm_full_path)
        x_tabular = input_tensor.cpu().numpy()[0, :, -1, :].reshape(1, -1)
        lgbm_probs = lgbm_model.predict(x_tabular)[0]
        tech_probs = (lstm_probs * cfg["lstm_w"]) + (lgbm_probs * cfg["lgbm_w"])
    else:
        tech_probs = lstm_probs

    # 4. 기술 지표 기반 판단 (3진 변환)
    t_down = tech_probs[0] + tech_probs[1]
    t_neutral = tech_probs[2]
    t_up = tech_probs[3] + tech_probs[4]
    
    tech_map = {"BUY": t_up, "SELL": t_down, "HOLD": t_neutral}
    tech_action = max(tech_map, key=tech_map.get)
    if np.max(tech_probs) < cfg["threshold"]: tech_action = "HOLD"

    # 5. 뉴스 에이전트 분석
    news_res = get_news_analysis(market_option, f"latest {market_option} news", str(cutoff_time), str(tz))
    
    # 6. [핵심] 리스크 기반 앙상블 로직
    acc_model = 0.7  # 기술 신뢰도 고정
    news_rel = news_res.reliability if news_res else 0.5
    
    # 6-1. 가중치 계산
    w_tech = acc_model / (acc_model + (news_rel**2 * 0.8))
    w_news = 1 - w_tech

    # 6-2. 리스크 민감도 적용 Sentiment 보정 (Gravity Model)
    risk_score = news_res.risk_score if news_res else 0.0
    sentiment = news_res.sentiment if news_res else 0.0
    sensitivity = 0.3 if risk_score < 0.6 else 0.5
    
    adj_sentiment = max(-1.0, min(1.0, sentiment - (risk_score * sensitivity)))

    # 6-3. 보정된 Sentiment를 확률로 변환
    news_prob_up = max(0, adj_sentiment)
    news_prob_down = max(0, -adj_sentiment)
    news_prob_neutral = 1.0 - (news_prob_up + news_prob_down)

    # 6-4. 최종 확률 결합
    f_up = (t_up * w_tech) + (news_prob_up * w_news)
    f_down = (t_down * w_tech) + (news_prob_down * w_news)
    f_neutral = (t_neutral * w_tech) + (news_prob_neutral * w_news)
    
    prob_map = {"BUY": f_up, "SELL": f_down, "HOLD": f_neutral}
    best_action = max(prob_map, key=prob_map.get)

    if np.max(tech_probs) < cfg["threshold"]:
        best_action = "HOLD"

    # 7. 저장
    save_prediction(market_option, tech_probs, [f_down, f_neutral, f_up], news_res, best_action, tech_action, (w_tech, w_news))

def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "us"
    markets = ["NASDAQ (QQQ)", "S&P 500 (SPY)"] if target == "us" else ["KOSPI (Korea)"]
    for m in markets:
        try:
            run_prediction_v2(m)
        except Exception as e:
            print(f"❌ {m} 오류: {e}")

if __name__ == "__main__":
    main()
