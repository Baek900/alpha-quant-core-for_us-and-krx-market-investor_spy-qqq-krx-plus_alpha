import os
import json
from datetime import datetime
from langchain_community.tools import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# [Pydantic] 출력 데이터 구조 정의
class MarketRiskAnalysis(BaseModel):
    # 1. 핵심 지표 (모델 반영용)
    sentiment: float = Field(description="Market sentiment score from -1.0 (Negative) to 1.0 (Positive)")
    reliability: float = Field(description="Reliability of the news sources from 0.0 to 1.0")
    
    # 2. 리스크 분석 (입체적 판단)
    risk_score: float = Field(description="Overall market risk score from 0.0 (Safe) to 1.0 (Extreme Risk)")
    inflation_jobs_summary: str = Field(description="Summary of Inflation (CPI/PPI) and Jobs data")
    monetary_policy_summary: str = Field(description="Summary of Fed/Central Bank policy and rates")
    geopolitics_summary: str = Field(description="Summary of Geopolitical risks and wars")
    
    # 3. 종합 요약
    final_summary: str = Field(description="Executive summary of the market situation (Max 3 sentences)")

    # 4. 시점 구분 (Signal vs Reference)
    is_signal_valid: bool = Field(description="True if enough news exists BEFORE the cutoff time to form a signal")

def get_news_analysis(market_name, query, cutoff_datetime_str, time_zone_str):
    """
    Args:
        market_name: 대상 시장 (예: NASDAQ)
        query: 검색 쿼리
        cutoff_datetime_str: 뉴스 반영 마감 시간 (YYYY-MM-DD HH:MM format)
        time_zone_str: 해당 시장의 타임존 (예: US/Eastern)
    """
    
    # 1. API 키 확인
    tavily_key = os.environ.get("TAVILY_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")
    
    if not tavily_key or not google_key:
        print("⚠️ API Key 누락. 뉴스 분석 건너뜀")
        return None

    # 2. 도구 설정
    tavily_tool = TavilySearchResults(
        tavily_api_key=tavily_key,
        max_results=7, # 더 많은 문맥 확보를 위해 증가
        include_raw_content=False 
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-light", 
        google_api_key=google_key,
        temperature=0.1
    )

    parser = PydanticOutputParser(pydantic_object=MarketRiskAnalysis)

    # 3. 프롬프트 고도화 (Macro Specialist)
    # 시간 통제(Look-ahead Bias) 및 카테고리별 분석 지시
    news_prompt = PromptTemplate(
        template="""
        You are a conservative Chief Risk Officer (CRO) for a quantitative hedge fund.
        Analyze the provided news snippets for '{market_name}'.

        [CRITICAL: Time Control & Look-ahead Bias]
        - Current Market Timezone: {time_zone}
        - SIGNAL CUTOFF TIME: {cutoff_time}
        - The news snippets may contain relative times (e.g., '2 hours ago').
        - IGNORE any events that occurred AFTER the 'SIGNAL CUTOFF TIME' for the 'sentiment' calculation.
        - Events AFTER the cutoff should ONLY influence the 'risk_score' if they pose an immediate crash threat, otherwise treat them as 'Reference'.

        [Analysis Categories]
        1. Inflation & Jobs: CPI, PPI, Payrolls.
        2. Monetary Policy: Fed/Bank rates, FOMC minutes.
        3. Geopolitics: Wars, Oil prices, Political instability.
        4. Market Sentiment: General fear/greed.

        [Scoring Rules]
        - Sentiment: -1.0 (Crash/Bear) to 1.0 (Rally/Bull). Be conservative.
        - Reliability: 0.5 start. +0.2 for Bloomberg/Reuters. -0.2 for rumors. Max 0.9.
        - Risk Score: 0.0 (Calm) to 1.0 (Panic). High inflation/War = High Risk.

        [News Context]
        {news_context}

        [Output Format]
        {format_instructions}
        """,
        input_variables=["market_name", "time_zone", "cutoff_time", "news_context"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    try:
        print(f"🔍 [News Agent] '{market_name}' 검색 (Cutoff: {cutoff_datetime_str})...")
        
        search_results = tavily_tool.invoke({"query": query})
        
        if not search_results:
            return None

        # 검색 결과 텍스트 변환
        news_context = "\n".join([f"- {r['content']}" for r in search_results])
        
        # Chain 실행
        chain = news_prompt | llm | parser
        analysis_result = chain.invoke({
            "market_name": market_name, 
            "time_zone": time_zone_str,
            "cutoff_time": cutoff_datetime_str,
            "news_context": news_context
        })
        
        # 수치 보정 (소수점 4자리)
        analysis_result.sentiment = round(analysis_result.sentiment, 4)
        analysis_result.reliability = round(analysis_result.reliability, 4)
        analysis_result.risk_score = round(analysis_result.risk_score, 4)

        print(f"✅ 분석 완료: Sent {analysis_result.sentiment} / Risk {analysis_result.risk_score}")
        return analysis_result

    except Exception as e:
        print(f"❌ 뉴스 분석 실패: {e}")
        # 실패 시 기본 안전 객체 반환
        return MarketRiskAnalysis(
            sentiment=0.0, reliability=0.0, risk_score=0.0,
            inflation_jobs_summary="Error", monetary_policy_summary="Error",
            geopolitics_summary="Error", final_summary=f"Analysis Failed: {str(e)}",
            is_signal_valid=False
        )
