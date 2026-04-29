import os
from dotenv import load_dotenv
from workflow import build_guardian_graph

def main():
    # 환경 변수 로드 (API 키 등)
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY가 환경 변수(.env)에 설정되지 않았습니다.")
        print(".env 파일을 만들고 GOOGLE_API_KEY=당신의_키 를 입력해주세요.")
        return

    # 워크플로우 그래프 생성
    app = build_guardian_graph()
    
    import json
    import yfinance as yf
    import pandas as pd

    # 1. 포트폴리오 파일 읽기
    portfolio_file = "portfolio.json"
    if not os.path.exists(portfolio_file):
        print(f"Error: {portfolio_file} 파일이 없습니다. 파일을 생성해주세요.")
        return
        
    with open(portfolio_file, "r") as f:
        user_data = json.load(f)
        
    holdings = user_data.get("holdings", {})
    cash = user_data.get("cash", 0.0)
    watchlist_targets = user_data.get("watchlist", {})
    
    if not holdings:
        print("Error: 보유 주식(holdings) 정보가 없습니다.")
        return

    # 2. 실시간 주가를 기반으로 현재 비중(Weight) 자동 계산
    print("[1/3] 실시간 주가를 바탕으로 현재 자산 가치를 계산 중입니다...")
    tickers = list(holdings.keys())
    data = yf.download(tickers, period="5d", progress=False)
    
    # yfinance 결과 구조에 따른 가격 추출 (MultiIndex 대응)
    if isinstance(data.columns, pd.MultiIndex):
        current_prices = data['Close'].iloc[-1]
    else:
        # 단일 종목이거나 MultiIndex가 아닌 경우
        if len(tickers) == 1:
            price_val = data['Close'].iloc[-1]
            current_prices = pd.Series({tickers[0]: price_val})
        else:
            current_prices = data['Close'].iloc[-1]

    total_stock_value = sum([holdings[t] * current_prices[t] for t in tickers])
    initial_value = float(total_stock_value + cash)
    
    # 현재 포트폴리오 비중 (현금 비중을 제외한 종목별 비중)
    current_portfolio = {t: (holdings[t] * current_prices[t]) / initial_value for t in tickers}
    
    # 3. 워크플로우 초기 상태 설정 (Raw Data 전달)
    initial_state = {
        "holdings": holdings,
        "cash": cash,
        "watchlist": watchlist_targets,
        "current_portfolio": current_portfolio,
        "initial_value": initial_value,
        "start_date": "2023-01-01",
        "end_date": pd.Timestamp.today().strftime("%Y-%m-%d")
    }
    
    print("============================================================")
    print("AI Portfolio Guardian 시스템을 시작합니다")
    print("============================================================\n")
    
    # LangGraph 워크플로우 실행
    try:
        result = app.invoke(initial_state)
    except Exception as e:
        print(f"\n워크플로우 실행 중 오류가 발생했습니다: {e}")
        return
        
    print("\n" + "="*60)
    print("AI 가디언 최종 리스크 진단 리포트")
    print("="*60 + "\n")
    print(result["final_report"])

if __name__ == "__main__":
    main()
