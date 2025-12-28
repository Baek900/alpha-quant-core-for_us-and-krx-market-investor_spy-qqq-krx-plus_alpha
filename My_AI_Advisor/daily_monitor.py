import os
import sys
from dotenv import load_dotenv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))
from supabase import create_client, Client
from datetime import datetime
from zoneinfo import ZoneInfo
from news_agent import get_news_analysis

# 환경변수
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None
IS_TEST_MODE = os.environ.get("TEST_MODE", "false").lower() == "true"
REFERENCE_TABLE = "news_reference_logs_test" if IS_TEST_MODE else "news_reference_logs"

if IS_TEST_MODE:
    print(f"⚠️ [TEST MODE] 데이터가 '{REFERENCE_TABLE}'에 저장됩니다.")

def save_reference_log(market, news_data):
    if not supabase: return
    try:
        # 리스크 요약 텍스트
        risk_text = (f"Infl: {news_data.inflation_jobs_summary[:50]}.. | "
                     f"Pol: {news_data.monetary_policy_summary[:50]}.. | "
                     f"Geo: {news_data.geopolitics_summary[:50]}..")
        
        data = {
            "market_name": market,
            "reference_summary": news_data.final_summary,
            "detected_risks": risk_text,
            "risk_level": news_data.risk_score,
            "news_source": "TitanFlow_Monitor"
        }
        supabase.table(REFERENCE_TABLE).insert(data).execute()
        print(f"✅ [Monitor] 저장 완료 ({REFERENCE_TABLE}): {market}")
    except Exception as e:
        print(f"❌ [Monitor] 저장 실패: {e}")

def run_monitor_batch(market_option):
    print(f"📡 [Monitor Job] 리스크 감시 시작: {market_option}")
    
    # 1. 시간 설정 (현재 시간 기준)
    now_utc = datetime.now(ZoneInfo("UTC"))
    if "Korea" in market_option:
        tz = ZoneInfo("Asia/Seoul")
        query = "latest south korea kospi stock market news today macro economics"
    else:
        tz = ZoneInfo("US/Eastern")
        query = "latest market sentiment news for US stock market NASDAQ SPY today"
        
    local_now = now_utc.astimezone(tz)
    cutoff_str = local_now.strftime("%Y-%m-%d %H:%M") # 현재 시간까지 모두 포함
    
    print(f"🕒 Scanning news up to: {cutoff_str} ({tz})")

    # 2. 뉴스 분석 (모델 없이 뉴스만 분석)
    news_obj = get_news_analysis(market_option, query, cutoff_str, str(tz))
    
    if news_obj:
        save_reference_log(market_option, news_obj)
    else:
        print("⚠️ 뉴스 데이터 없음")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "us"
    markets = ["NASDAQ (QQQ)", "S&P 500 (SPY)"] if target == "us" else ["KOSPI (Korea)"]
    for m in markets: run_monitor_batch(m)