import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import lightgbm as lgb
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from supabase import create_client, Client
from dotenv import load_dotenv

# 모듈 임포트
from model_def import StockClassifierModel
from data_loader import get_us_v2_data, get_kr_v2_data
from news_agent import get_news_analysis

# 1. 환경 설정 및 테스트 모드 구분
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None

# [기능 유지] 테스트 모드에 따른 테이블 분기 로직
IS_TEST_MODE = os.environ.get("TEST_MODE", "false").lower() == "true"
PREDICTION_TABLE = "prediction_logs_test" if IS_TEST_MODE else "prediction_logs"

if IS_TEST_MODE:
    print(f"⚠️ [TEST MODE] 데이터가 '{PREDICTION_TABLE}' 테이블에 저장됩니다.")

# [기능 유지] 2026년 휴장일 데이터
HOLIDAYS_2026 = {
    "us": {
        "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
        "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25"
    },
    "kr": {
        "2026-01-01", "2026-02-16", "2026-02-17", "2026-02-18", "2026-03-02",
        "2026-05-01", "2026-05-05", "2026-05-25", "2026-06-06", "2026-08-17",
        "2026-09-24", "2026-09-25", "2026-09-26", "2026-10-05", "2026-10-09",
        "2026-12-25", "2026-12-31"
    }
}

# [v2.0 엄격 설정] 노트북 백테스팅 로직 및 임계값 반영
MARKET_CONFIG = {
    "NASDAQ (QQQ)": {
        "is_kr": False, "seq_len": 14, "num_assets": 10, "hidden": 128, "threshold": 0.33,
        "assets": ['XLK','XLV','XLF','XLY','XLC','XLI','XLP','XLE','XLB','XLRE'],
        "model_env": "us_sector_ai_model_qqq_kmeans_5class_v2.pth"
    },
    "S&P 500 (SPY)": {
        "is_kr": False, "seq_len": 14, "num_assets": 10, "hidden": 128, "threshold": 0.33,
        "assets": ['XLK','XLV','XLF','XLY','XLC','XLI','XLP','XLE','XLB','XLRE'],
        "model_env": "us_sector_ai_model_spy_kmeans_5class_v1.pth"
    },
    "KOSPI (Korea)": {
        "is_kr": True, "seq_len": 20, "num_assets": 12, "hidden": 256, "threshold": 0.3,
        "lstm_w": 0.3, "lgbm_w": 0.7, # 노트북 코드 가중치 우선 적용
        "sectors": ['091160.KS','305720.KS','091170.KS','091180.KS','117460.KS','102120.KS','117700.KS','139230.KS','140700.KS','266370.KS'],
        "macro": ['USDKRW=X', 'SPY'],
        "model_env": "kospi_macro_ai_model_5class_v2.pth",
        "lgbm_env": "lgbm_model_v2_721.txt"
    }
}

# [기능 유지] 휴장일 및 주말 검증 로직
def is_market_open(market_type, current_date):
    date_str = current_date.strftime("%Y-%m-%d")
    if current_date.weekday() >= 5: # 토(5), 일(6)
        print(f"🛑 [Market Closed] {date_str} is Weekend.")
        return False
    if date_str in HOLIDAYS_2026.get(market_type, set()):
        print(f"🛑 [Market Closed] {date_str} is a Holiday.")
        return False
    return True

# [기능 유지] 시장별 타임존 및 Cutoff 계산
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
    """
    DB 스키마(v2026)에 맞춘 저장 로직
    weights: (w_tech, w_news) 튜플
    """
    if not supabase: return
    try:
        # 기술적 확률 (5진 -> 3진 통합)
        t_down = round(float(tech_probs[0] + tech_probs[1]), 4)
        t_neutral = round(float(tech_probs[2]), 4)
        t_up = round(float(tech_probs[3] + tech_probs[4]), 4)

        data = {
            "market_name": market,
            "tech_prob_down": t_down,
            "tech_prob_neutral": t_neutral,
            "tech_prob_up": t_up,
            "fin_prob_down": round(float(final_probs[0]), 4),
            "fin_prob_neutral": round(float(final_probs[1]), 4),
            "fin_prob_up": round(float(final_probs[2]), 4),
            "w_tech": weights[0],
            "w_news": weights[1],
            "news_sentiment": news_data.sentiment if news_data else 0.0,
            "news_reliability": news_data.reliability if news_data else 0.5, # 필드 추가
            "news_summary": news_data.final_summary if news_data else "No news analysis",
            "news_score": int((news_data.sentiment + 1) * 50) if news_data else 50, # integer 변환
            "action": best_action,
            "tech_action": tech_action, # 기술 지표만의 판단 결과 저장
            "news_risk_score": news_data.risk_score if news_data else 0.0,
            "news_metadata": news_data.dict() if news_data else {}
        }

        # [기능 유지] 테스트 모드 구분에 따른 테이블 선택
        target_table = PREDICTION_TABLE 
        supabase.table(target_table).insert(data).execute()
        print(f"✅ [DB Insert] {market} -> {best_action} (Table: {target_table})")
        
    except Exception as e:
        print(f"❌ [DB Error] 데이터 저장 실패: {e}")

def run_prediction_v2(market_option):
    cfg = MARKET_CONFIG[market_option]
    tz, cutoff_time, m_type = get_market_info(market_option)
    
    # [기능 유지] 휴장일 검증
    if not is_market_open(m_type, cutoff_time.date()):
        print(f"💤 {market_option}: 휴장일이므로 스킵합니다.")
        return

    print(f"🚀 [Batch] Processing {market_option} (Cutoff: {cutoff_time.strftime('%Y-%m-%d %H:%M')})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 데이터 로드 (v2.0 매크로 포함)
    if cfg["is_kr"]:
        input_tensor = get_kr_v2_data(cfg["sectors"], cfg["macro"], cfg["seq_len"])
    else:
        input_tensor = get_us_v2_data(cfg["assets"], cfg["seq_len"])
    
    if input_tensor is None: return
    input_tensor = input_tensor.to(device)

    # 2. LSTM 모델 추론
    model = StockClassifierModel(num_sectors=cfg["num_assets"], hidden_dim=cfg["hidden"], is_kr=cfg["is_kr"]).to(device)
    model_path = os.environ.get(cfg["model_env"])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        logits = model(input_tensor)
        lstm_probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    # 3. KOSPI 전용 LGBM 앙상블 (노트북 로직)
    if cfg["is_kr"]:
        lgbm_model = lgb.Booster(model_file=os.environ.get(cfg["lgbm_env"]))
        x_tabular = input_tensor.cpu().numpy()[0, :, -1, :].reshape(1, -1)
        lgbm_probs = lgbm_model.predict(x_tabular)[0]
        tech_probs = (lstm_probs * cfg["lstm_w"]) + (lgbm_probs * cfg["lgbm_w"])
    else:
        tech_probs = lstm_probs

    # 4. 5진 -> 3진 변환 (노트북: 0,1:Down / 2:Neutral / 3,4:Up)
    t_down = tech_probs[0] + tech_probs[1]
    t_neutral = tech_probs[2]
    t_up = tech_probs[3] + tech_probs[4]
    
    # [추가] 기술적 판단만 따로 계산 (tech_action 저장용)
    tech_map = {"BUY": t_up, "SELL": t_down, "HOLD": t_neutral}
    tech_action = max(tech_map, key=tech_map.get)
    # 기술 모델도 확신(threshold)이 없으면 HOLD
    if np.max(tech_probs) < cfg["threshold"]:
        tech_action = "HOLD"


    # 5. 뉴스 에이전트 분석
    news_res = get_news_analysis(market_option, f"latest {market_option} market news", str(cutoff_time), str(tz))
    news_obj = news_res if news_res else None

    # 6. [핵심] 개선된 앙상블 로직 (Balanced Risk Adjustment)
    # 6-1. 가중치 계산 (기술 신뢰도 70% 고정)
    acc_model = 0.7  # 사용자 요청에 의한 고정
    news_rel = news_obj.reliability if news_obj else 0.5
    
    w_tech = acc_model / (acc_model + (news_rel**2 * 0.8))
    w_news = 1 - w_tech

    # 6-2. 리스크 반영 Sentiment 보정 (Gravity Model)
    # 리스크 점수가 0.6 미만이면 민감도를 0.3으로 낮춰 비관론 억제
    risk_score = news_obj.risk_score if news_obj else 0.0
    sensitivity = 0.3 if risk_score < 0.6 else 0.5
    
    sentiment = news_obj.sentiment if news_obj else 0.0
    adjusted_sentiment = max(-1.0, min(1.0, sentiment - (risk_score * sensitivity)))

    # 6-3. 보정된 Sentiment를 뉴스 확률로 변환
    news_prob_up = max(0, adjusted_sentiment)
    news_prob_down = max(0, -adjusted_sentiment)
    news_prob_neutral = 1.0 - (news_prob_up + news_prob_down)

    # 6-4. 최종 결합 (기술 + 뉴스)
    f_up = (t_up * w_tech) + (news_prob_up * w_news)
    f_down = (t_down * w_tech) + (news_prob_down * w_news)
    f_neutral = (t_neutral * w_tech) + (news_prob_neutral * w_news)

    prob_map = {"BUY": f_up, "SELL": f_down, "HOLD": f_neutral}
    best_action = max(prob_map, key=prob_map.get)

    # [엄격] 기술적 확신 부족 시 최종 결론도 HOLD
    if np.max(tech_probs) < cfg["threshold"]:
        best_action = "HOLD"

    # 7. 저장 (이제 tech_action과 가중치를 함께 보냅니다)
    save_prediction(
        market_option, 
        tech_probs, 
        [f_down, f_neutral, f_up], 
        news_res, 
        best_action, 
        tech_action, 
        (w_tech, w_news)
    )

def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "us"
    markets = ["NASDAQ (QQQ)", "S&P 500 (SPY)"] if target == "us" else ["KOSPI (Korea)"]
    for m in markets:
        try:
            run_prediction_v2(m)
        except Exception as e:
            print(f"❌ {m} 처리 중 오류 발생: {e}")

if __name__ == "__main__":
    main()