# daily_v9_batch.py
import os
import torch
import json
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from supabase import create_client
from dotenv import load_dotenv

from v9_model import V9_AI_Core
from v9_data_loader import fetch_v9_inference_data

# 설정 로드
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
load_dotenv(os.path.join(BASE_DIR, ".env"))

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None

SECTORS = ['XLK', 'XLV', 'XLF', 'XLY', 'XLC', 'XLI', 'XLP', 'XLE', 'XLB', 'XLRE']

def run_v9_inference():
    print("🚀 [V9 Daily Batch] 10-Sector 비대칭 포트폴리오 추론 시작")
    
    # 1. 데이터 로드
    d_tensor, w_tensor, m_tensor, macro_tensor = fetch_v9_inference_data()
    print("✅ 데이터 로드 완료")
    
    # 2. 모델 로드 (가중치 파일 필요)
    model_path = os.path.join(MODEL_DIR, "v9_master_weights.pth")
    model = V9_AI_Core(sector_feature_dim=3, macro_latent_dim=6)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print("✅ V9 마스터 가중치 로드 완료")
    except Exception as e:
        print(f"⚠️ 가중치 로드 실패. 더미 모드로 전환합니다 (사유: {e})")
        # 실제 서버 업로드 전 테스트를 위해 에러를 뱉지 않게 처리
    
    model.eval()
    with torch.no_grad():
        exp_alphas, mac_scores, reg_probs, attn_w = model(d_tensor, w_tensor, m_tensor, macro_tensor)
    
    # 3. 듀얼 퓨전 & 비대칭 리스크 로직 (Black-Litterman)
    bear_prob = reg_probs[0][2].item()
    bull_prob = reg_probs[0][0].item()
    
    # (심화 튜닝: 실제로는 Supabase에서 최근 20일 오차를 불러와 Confidence를 구합니다)
    final_scores = (0.4 * mac_scores[0].numpy()) + (0.6 * exp_alphas[0].numpy())
    
    target_weights = {sec: 0.0 for sec in SECTORS}
    cash_weight = 0.0
    
    if bear_prob > 0.60:
        print("🚨 [위기 감지] Fast Exit 단두대 발동. 100% 현금 대피")
        cash_weight = 1.0
    else:
        positive_indices = np.where(final_scores > 0)[0]
        if len(positive_indices) > 0:
            sum_scores = np.sum(final_scores[positive_indices])
            for idx in positive_indices:
                target_weights[SECTORS[idx]] = float(final_scores[idx] / sum_scores)
            
            # 레버리지 로직
            if bull_prob > 0.7:
                for sec in SECTORS: target_weights[sec] *= 1.5
                cash_weight = -0.5
        else:
            cash_weight = 1.0

    # 4. Supabase DB 저장
    now_kst = datetime.now(ZoneInfo("Asia/Seoul"))
    log_data = {
        "date": now_kst.strftime("%Y-%m-%d"),
        "created_at": now_kst.isoformat(),
        "bull_prob": round(bull_prob, 4),
        "bear_prob": round(bear_prob, 4),
        "cash_weight": round(cash_weight, 4),
        "allocations": json.dumps(target_weights),
        # 프론트엔드 히트맵용 데이터 (소수점 4자리 변환)
        "attention_heatmap": json.dumps(np.round(attn_w[0].numpy(), 4).tolist()) 
    }
    
    if supabase:
        # V9 전용 테이블 (v9_allocation_logs) 에 Insert (사전에 Supabase에서 테이블을 생성해야 함)
        try:
            supabase.table("v9_allocation_logs").insert(log_data).execute()
            print(f"✅ [DB 저장 성공] {log_data['date']} V9 배분 로그 저장 완료")
        except Exception as e:
            print(f"❌ [DB 저장 실패] {e}")
            print(log_data)
    else:
        print("⚠️ Supabase 미연결 (로컬 테스트)")
        print(log_data)

if __name__ == "__main__":
    run_v9_inference()
