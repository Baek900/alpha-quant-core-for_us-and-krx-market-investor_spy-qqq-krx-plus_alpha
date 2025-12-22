# ... (상단 import 및 CSS 설정은 기존 코드 유지) ...

    # [Left Column] Signal 수정 부분
    with col1:
        if latest_data:
            date_str = convert_utc_to_kst(latest_data['created_at'])
            
            # DB에서 가져온 Raw 확률 (없으면 0.0 처리)
            p_up = latest_data.get('tech_prob', 0.0)
            p_down = latest_data.get('prob_down', 0.0)
            p_neutral = latest_data.get('prob_neutral', 0.0)
            
            # 최종 확률 (앙상블 결과)
            final_prob = latest_data['final_prob']
            
            # 가중치 정보 (툴팁이나 설명용)
            w_tech = latest_data.get('w_tech', 0.7)
            w_news = latest_data.get('w_news', 0.3)

            st.markdown(f"**Analysis Time:** {date_str}")
            
            # 3가지 확률 표시
            m1, m2, m3 = st.columns(3)
            m1.metric("Bullish", f"{p_up*100:.1f}%")
            m2.metric("Bearish", f"{p_down*100:.1f}%") 
            m3.metric("Neutral", f"{p_neutral*100:.1f}%")
            
            # Primary Signal (Final Prob 기준)
            decision = "HOLD"
            d_color = "#CCCCCC"
            if final_prob >= 0.45:
                decision = "BUY"
                d_color = "#00E396"
            elif final_prob <= 0.2:
                decision = "SELL"
                d_color = "#FF4560"
            
            st.markdown(f"""
            <div style='margin-top: 20px; padding: 20px; border: 3px solid {d_color}; border-radius: 8px; background-color: #121926; text-align: center;'>
                <span style='color: #FFFFFF; font-size: 1.1rem; font-weight: bold;'>Primary Signal (Weighted)</span><br>
                <span style='color: {d_color}; font-size: 2.5rem; font-weight: 900;'>{decision}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # ... (Strategy Details 생략) ...

    # [Right Column] Prediction 수정 부분
    with col2:
        # ... (차트 데이터 로드 생략) ...
        
        if latest_data:
             stats = MARKET_STATS.get(market_option, MARKET_STATS["S&P 500 (SPY)"])
             
             # [중요] 5일 뒤 예측 계산 시:
             # 기술적 분석 확률(p_up/down/neutral)을 사용할지, 
             # 뉴스까지 반영된 final_prob를 역산해서 사용할지 결정해야 합니다.
             # 여기서는 '앙상블된 최종 방향성'을 반영하기 위해 아래와 같이 조정합니다.
             
             # final_prob(상승확률)가 높아지면 -> Bullish 힘 증가
             # final_prob가 낮아지면 -> Bearish 힘 증가
             
             # 단순화를 위해 앙상블된 Net Score 계산 (0.5 기준)
             # 0.5 이상이면 상승분, 0.5 미만이면 하락분
             ensemble_score = (final_prob - 0.5) * 2  # -1 ~ 1
             
             # 통계적 기대 수익률 (가중 평균)
             # 상승/하락/횡보 확률을 앙상블된 점수로 재분배하여 기대값 산출
             # (복잡한 수식 대신, direction에 따라 stats['bull'] 또는 stats['bear']를 강도만큼 적용)
             
             if ensemble_score > 0:
                 expected_return = ensemble_score * stats['bull']
             else:
                 expected_return = abs(ensemble_score) * stats['bear']
                 
             # 5일 복리 적용
             future_price_5d = current_price * ((1 + expected_return) ** 5)
             total_return = (future_price_5d / current_price - 1) * 100

        # ... (UI 및 차트 그리기 코드는 기존 유지) ...

    # [Bottom] News Section 수정
    st.markdown("---")
    st.markdown("**Global Sentiment & Macro Insights**")
    if latest_data:
        nc1, nc2 = st.columns([1, 3])
        with nc1:
            # 감정 점수 (-1 ~ 1) 표시
            sent_score = latest_data.get('news_sentiment', 0.0)
            sent_label = "Neutral"
            if sent_score > 0.3: sent_label = "Positive"
            elif sent_score < -0.3: sent_label = "Negative"
            
            st.metric("Sentiment", f"{sent_score:.2f}", sent_label)
            
            # 신뢰도 표시 (Reliability)
            rel_score = latest_data.get('news_reliability', 0.0)
            st.caption(f"News Reliability: {rel_score*100:.0f}%")
            
        with nc2:
            # 뉴스 요약 표시
            summary = latest_data.get('news_summary', "No summary available.")
            st.info(f"📰 **Market Summary:**\n\n{summary}")
