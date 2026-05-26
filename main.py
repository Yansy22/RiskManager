import os
import json
import traceback
import pandas as pd
from workflow import build_guardian_graph

def main():
    # 1. 포트폴리오 데이터 로드
    portfolio_file = "portfolio.json"
    if not os.path.exists(portfolio_file):
        print(f"Error: {portfolio_file} 파일이 없습니다.")
        return
        
    with open(portfolio_file, "r") as f:
        user_data = json.load(f)
        
    holdings = user_data.get("holdings", {})
    watchlist = user_data.get("watchlist", {})
    settings = user_data.get("settings", {"sector_limit": 0.3, "lookback_years": 3, "clustering_threshold": 1.0})
    
    if not holdings and not watchlist:
        print("Error: 보유 주식(holdings) 및 관심 종목(watchlist) 정보가 모두 없습니다.")
        return
        
    # 중복 제거한 전체 티커 리스트 추출
    all_tickers = list(set(holdings.keys()) | set(watchlist.keys()))
    
    # 2. 분석 기간 설정 (최근 N년 과거 데이터 수집용)
    lookback_years = settings.get("lookback_years", 3)
    start_date = (pd.Timestamp.today() - pd.DateOffset(years=lookback_years)).strftime("%Y-%m-%d")
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    
    # 3. 워크플로우 초기 상태 설정
    initial_state = {
        "all_tickers": all_tickers,
        "settings": settings,
        "start_date": start_date,
        "end_date": end_date
    }
    
    print("============================================================")
    print("AI Portfolio Guardian: 리스크 설계 진단 가동")
    print(f"   - 분석 대상: {len(all_tickers)}개 종목 ({all_tickers})")
    print(f"   - 임계점(t): {settings.get('clustering_threshold', 1.0)}")
    print("============================================================\n")
    
    # 4. 워크플로우 실행
    app = build_guardian_graph()
    try:
        result = app.invoke(initial_state)
    except Exception as e:
        print(f"\n시스템 실행 중 오류가 발생했습니다: {e}")
        traceback.print_exc()
        return
        
    print("\n" + "="*60)
    print("AI 가디언 리스크 설계 보고서")
    print("="*60 + "\n")
    
    print(result["final_report"])
    
    if result.get("pdf_path") and result["pdf_path"] != "Error":
        print("\n" + "="*60)
        print(f"PDF 전략 리포트가 생성되었습니다: {os.path.abspath(result['pdf_path'])}")
        print("="*60 + "\n")

if __name__ == "__main__":
    main()
