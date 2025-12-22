import os
import json
import warnings
import google.generativeai as genai

# 경고 메시지 제어
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.tools import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get_news_analysis(market_name, query):
    # 1. API 키 확인
    tavily_key = os.environ.get("TAVILY_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")
    
    if not tavily_key or not google_key:
        print("⚠️ API Key 누락. 뉴스 분석 건너뜀 (기본값 반환)")
        return {"sentiment": 0.0, "reliability": 0.0, "summary": "API Key Missing"}

    # 2. 도구 설정
    tavily_tool = TavilySearchResults(
        tavily_api_key=tavily_key,
        max_results=5,
        include_raw_content=True
    )

    # 사용자 지정 모델 (성공 확인된 버전)
    target_model = "gemini-2.5-flash-lite"

    llm = ChatGoogleGenerativeAI(
        model=target_model,
        google_api_key=google_key,
        temperature=0.1
    )

    # 3. 프롬프트 (영어 요약 요청)
    news_prompt = PromptTemplate.from_template("""
    You are a senior quant analyst. Analyze the provided news for '{market_name}'.

    [Output Requirements]
    1. sentiment: -1.0 (Very Negative) to 1.0 (Very Positive).
    2. reliability: 0.0 to 1.0 based on source credibility and consensus.
    3. summary: Summarize key points in English (max 3 sentences).

    [News Data]
    {news_context}

    [Format]
    JSON only:
    {{
        "sentiment": float,
        "reliability": float,
        "summary": "string"
    }}
    """)

    try:
        print(f"🔍 [News Agent] '{market_name}' 검색 및 분석 중... (Model: {target_model})")
        
        search_results = tavily_tool.invoke({"query": query})
        
        if not search_results:
            print("⚠️ 검색 결과 없음.")
            return {"sentiment": 0.0, "reliability": 0.0, "summary": "No news found"}

        news_context = "\n".join([f"- {r['content'][:300]}" for r in search_results])
        
        chain = news_prompt | llm | StrOutputParser()
        result_raw = chain.invoke({"market_name": market_name, "news_context": news_context})
        
        clean_json = result_raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        
        print(f"✅ 뉴스 분석 완료: 감정 {data['sentiment']} / 신뢰도 {data['reliability']}")
        return data

    except Exception as e:
        print(f"❌ 뉴스 분석 실패: {e}")
        return {"sentiment": 0.0, "reliability": 0.0, "summary": f"Analysis Failed: {str(e)}"}
