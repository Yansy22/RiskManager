import os
import json
import traceback
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from workflow import build_guardian_graph
from data_provider import get_market_caps, get_risk_free_rate

def main():
    # 1. 환경 변수 로드
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY가 설정되지 않았습니다.")
        return

    # 2. 포트폴리오 데이터 로드
    portfolio_file = "portfolio.json"
    if not os.path.exists(portfolio_file):
        print(f"Error: {portfolio_file} 파일이 없습니다.")
        return
        
    with open(portfolio_file, "r") as f:
        user_data = json.load(f)
        
    holdings = user_data.get("holdings", {})
    watchlist = user_data.get("watchlist", {})
    cash = user_data.get("cash", 0.0)
    settings = user_data.get("settings", {"risk_free_rate": 0.035, "sector_limit": 0.3})
    
    if not holdings and not watchlist:
        print("Error: 보유 주식(holdings) 및 관심 종목(watchlist) 정보가 모두 없습니다.")
        return

    if not holdings and cash <= 0:
        print("Error: 현재 보유한 주식이 없으며, 투자 가능한 현금(cash)도 없습니다.")
        return

    print("\n[1/3] 기초 시장 데이터 수집 및 전처리 중...")
    lookback_years = settings.get("lookback_years", 3)
    start_date = (pd.Timestamp.today() - pd.DateOffset(years=lookback_years)).strftime("%Y-%m-%d")
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    
    all_tickers = list(holdings.keys()) + list(watchlist.keys())
    market_caps = get_market_caps(all_tickers)
    rf_rate = get_risk_free_rate()
    settings["risk_free_rate"] = rf_rate # 실시간 금리 반영
    
    # 실시간 주가 다운로드 (비중 계산용)
    data = yf.download(all_tickers, period="5d", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        current_prices = data['Close'].iloc[-1]
    else:
        # 단일 종목 처리
        if len(all_tickers) == 1:
            current_prices = pd.Series({all_tickers[0]: data['Close'].iloc[-1]})
        else:
            current_prices = data['Close'].iloc[-1]

    # 4. 현재 자산 가치 및 비중 계산
    stock_value = sum([holdings[t]["quantity"] * current_prices[t] for t in holdings])
    initial_value = float(stock_value + cash)
    
    current_weights = {t: (holdings[t]["quantity"] * current_prices[t]) / initial_value for t in holdings}
    current_weights["CASH"] = cash / initial_value
    
    # watchlist 종목은 현재 비중 0%
    for t in watchlist:
        if t not in current_weights:
            current_weights[t] = 0.0

    # 5. 워크플로우 초기 상태 설정
    initial_state = {
        "user_holdings": holdings,
        "user_watchlist": watchlist,
        "cash": cash,
        "settings": settings,
        "all_tickers": all_tickers,
        "market_caps": market_caps,
        "initial_value": initial_value,
        "current_weights": current_weights,
        "start_date": start_date,
        "end_date": end_date
    }
    
    print("============================================================")
    print("🛡️ AI Portfolio Guardian: 고도화 엔진 가동")
    print(f"   - 총 자산: ${initial_value:,.2f}")
    print(f"   - 분석 대상: {len(all_tickers)}개 종목")
    print("============================================================\n")
    
    # 6. 워크플로우 실행
    app = build_guardian_graph()
    try:
        # 튜플 형태가 아닌 딕셔너리 형태로 전달
        result = app.invoke(initial_state)
    except Exception as e:
        print(f"\n시스템 실행 중 오류가 발생했습니다: {e}")
        traceback.print_exc() # 상세 에러 위치 출력
        return
        
    print("\n" + "="*60)
    print("📜 AI 가디언 최종 전략 분석 리포트")
    print("="*60 + "\n")
    
    report = result["final_report"]
    if isinstance(report, list):
        # 리스트 형태인 경우 텍스트 블록만 추출하여 합침
        report_text = "".join([block.get("text", "") if isinstance(block, dict) else str(block) for block in report])
    else:
        report_text = str(report)
        
    print(report_text)
    
    if result.get("pdf_path") and result["pdf_path"] != "Error":
        print("\n" + "="*60)
        print(f"📁 PDF 리포트가 생성되었습니다: {os.path.abspath(result['pdf_path'])}")
        print("="*60 + "\n")

if __name__ == "__main__":
    main()
