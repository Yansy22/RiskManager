import json
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from analysis import perform_attribution_analysis

def run_feedback():
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY가 설정되지 않았습니다.")
        return

    portfolio_file = "portfolio.json"
    if not os.path.exists(portfolio_file):
        print(f"Error: {portfolio_file} 파일이 없습니다.")
        return
        
    with open(portfolio_file, "r") as f:
        user_data = json.load(f)
        
    holdings = user_data.get("holdings", {})
    watchlist = user_data.get("watchlist", {})
    all_tickers = list(holdings.keys()) + list(watchlist.keys())
    
    if not all_tickers:
        print("포트폴리오에 종목이 없습니다.")
        return
        
    print("[1/3] 최근 주가 데이터를 불러오는 중...")
    data = yf.download(all_tickers, period="5d", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        current_prices = data['Close'].iloc[-1]
    else:
        if len(all_tickers) == 1:
            current_prices = pd.Series({all_tickers[0]: data['Close'].iloc[-1]})
        else:
            current_prices = data['Close'].iloc[-1]
            
    print("[2/3] 과거 예측 기록과 비교하여 기여도 채점 중...")
    try:
        df_attr = perform_attribution_analysis(current_prices)
    except Exception as e:
        print(f"채점표 생성 실패: {e}")
        return
        
    if df_attr.empty:
        print("이전 포트폴리오 예측 기록이 없어 복기를 생략합니다.")
        return
        
    print("[3/3] AI 수석 리스크 관리자의 사후 복기 브리핑 생성 중...\n")
    
    prompt = f"""
    <System_Role> 당신은 수석 리스크 관리자입니다.
    지난주 예측과 실제 결과를 비교한 다음의 채점표(기여도 분석)를 바탕으로, 무엇이 성공적이었고 무엇이 실패(오차)의 원인이었는지 사후 복기 브리핑을 작성하세요.
    특히 'error_responsibility_pct'가 높은 종목을 지적하며 원인을 분석하세요.
    
    [핵심] 파라미터 제안:
    브리핑의 마지막에는 반드시 사용자가 `portfolio.json`을 수정할 수 있도록 구체적인 파라미터 변경을 제안해야 합니다.
    예시: "VST의 Confidence를 4점으로 낮추고, 클러스터 임계값을 0.9로 수정할 것을 권장합니다."
    
    [채점표 데이터]
    {df_attr.to_string()}
    """
    
    messages = [
        SystemMessage(content="You are a strict Chief Risk Officer."),
        HumanMessage(content=prompt)
    ]
    
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.2)
    response = llm.invoke(messages)
    
    # response.content가 list 형태일 수도 있으므로 처리
    report_content = response.content
    if isinstance(report_content, list):
        report_text = "".join([block.get("text", "") if isinstance(block, dict) else str(block) for block in report_content])
    else:
        report_text = str(report_content)
        
    print("="*60)
    print("📊 AI Guardian 사후 복기 브리핑")
    print("="*60)
    print(report_text)
    print("\n[안내] 위 제안을 참고하여 주말 동안 portfolio.json의 점수와 확신도를 수정하신 후, main.py를 다시 실행하세요.")

if __name__ == "__main__":
    run_feedback()
