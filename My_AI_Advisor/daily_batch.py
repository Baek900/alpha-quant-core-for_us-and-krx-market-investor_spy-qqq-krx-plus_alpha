import os
import json
# [최신 문법 1] TavilySearchResults는 이제 langchain_tavily 패키지에서 가져옵니다.
from langchain_tavily import TavilySearchResults
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
        search_depth="advanced",
        include_raw_content=True
    )

    llm = ChatGoogleGenerativeAI(
        # [최신 문법 2] 존재하는 최신 모델명으로 변경 (gemini-3 -> gemini-1.5)
        model="gemini-1.5-flash", 
        google_api_key=google_key,
        temperature=0.1
    )

    # 3. 프롬프트
    news_prompt = PromptTemplate.from_template("""
    You are a senior quant analyst. Analyze the provided news for '{market_name}'.

    [Output Requirements]
    1. sentiment: -1.0 (Very Negative) to 1.0 (Very Positive).
    2. reliability: 0.0 to 1.0 based on source credibility and consensus.
    3. summary: Summarize key points in Korean (max 3 sentences).

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
        print(f"🔍 [News Agent] '{market_name}' 검색 및 분석 중...")
        # 검색
        search_results = tavily_tool.invoke({"query": query})
        news_context = "\n".join([f"- {r['content'][:300]}" for r in search_results])
        
        # 분석
        chain = news_prompt | llm | StrOutputParser()
        result_raw = chain.invoke({"market_name": market_name, "news_context": news_context})
        
        # JSON 파싱
        clean_json = result_raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        
        print(f"✅ 뉴스 분석 완료: 감정 {data['sentiment']} / 신뢰도 {data['reliability']}")
        return data

    except Exception as e:
        print(f"❌ 뉴스 분석 실패: {e}")
        return {"sentiment": 0.0, "reliability": 0.0, "summary": f"Analysis Failed: {str(e)}"}
