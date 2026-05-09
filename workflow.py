import os
import numpy as np
import pandas as pd
from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from data_provider import get_monte_carlo_params, get_market_caps, get_risk_free_rate
from simulation import run_portfolio_simulation, get_simulation_stats
from engine import calculate_risk_clusters, black_litterman_posterior, optimize_portfolio, calculate_rebalancing_plan
from pdf_generator import generate_pdf_report
from analysis import perform_attribution_analysis

# 1. 상태(State) 정의
class PortfolioState(TypedDict):
    # 입력 데이터
    user_holdings: Dict[str, Any]       # 종목별 수량, 점수, 확신도
    user_watchlist: Dict[str, Any]      # 관심 종목, 점수, 확신도
    cash: float                         # 보유 현금
    settings: Dict[str, Any]            # 금리, 섹터제한 등
    
    # 분석 데이터
    all_tickers: List[str]
    market_caps: Dict[str, float]
    initial_value: float
    
    # 엔진 결과
    clusters: Dict[int, List[str]]
    posterior_returns: pd.Series
    cov_matrix: pd.DataFrame
    
    # 비중 데이터
    current_weights: Dict[str, float]
    optimal_weights: Dict[str, float]
    rebalance_plan: Dict[str, Any]      # 정수 단위 매매 계획
    
    start_date: str
    end_date: str
    
    # 통계 결과
    current_stats: Dict[str, Any]
    optimal_stats: Dict[str, Any]
    
    # 최종 결과물
    final_report: str
    pdf_path: str                       # 생성된 PDF 파일 경로

# 2. 노드(Nodes) 정의

def analysis_node(state: PortfolioState) -> PortfolioState:
    """Phase 2, 3, 4: 클러스터링, 블랙-리터만, 최적화를 수행하는 노드"""
    print("[Node 1] 전략 수립 엔진 가동 (Clustering & Black-Litterman)...")
    
    tickers = state["all_tickers"]
    user_views = {}
    for t in state["user_holdings"]:
        user_views[t] = {
            "score": state["user_holdings"][t].get("score", 5),
            "confidence": state["user_holdings"][t].get("confidence", 5)
        }
    for t in state["user_watchlist"]:
        user_views[t] = {
            "score": state["user_watchlist"][t].get("score", 5),
            "confidence": state["user_watchlist"][t].get("confidence", 5)
        }

    prices, _, _, cov = get_monte_carlo_params(tickers, state["start_date"], state["end_date"])
    returns = np.log(prices / prices.shift(1)).dropna()
    
    threshold = state["settings"].get("clustering_threshold", 0.4)
    clusters = calculate_risk_clusters(returns, threshold=threshold)
    
    rf_rate = state["settings"].get("risk_free_rate", 0.035)
    mu_bl, _ = black_litterman_posterior(state["market_caps"], cov, user_views, rf_rate)
    
    mu_bl_with_cash = mu_bl.copy()
    mu_bl_with_cash["CASH"] = rf_rate
    
    cov_with_cash = cov.copy()
    cov_with_cash["CASH"] = 0.0
    cash_row = pd.Series(0.0, index=cov_with_cash.columns, name="CASH")
    cov_with_cash = pd.concat([cov_with_cash, cash_row.to_frame().T])
    
    sector_limit = state["settings"].get("sector_limit", 0.3)
    optimal_w = optimize_portfolio(mu_bl_with_cash, cov_with_cash, clusters, sector_limit, rf_rate)
    
    return {
        "clusters": clusters,
        "posterior_returns": mu_bl_with_cash,
        "cov_matrix": cov_with_cash,
        "optimal_weights": optimal_w
    }

def simulation_node(state: PortfolioState) -> PortfolioState:
    """Phase 5: 몬테카를로 스트레스 테스트 노드"""
    print("[Node 2] 최적 전략 스트레스 테스트 중 (10,000회 시뮬레이션)...")
    
    init_val = state["initial_value"]
    cov = state["cov_matrix"]
    mu_bl = state["posterior_returns"]
    
    tickers = state["all_tickers"]
    prices, _, _, _ = get_monte_carlo_params(tickers, state["start_date"], state["end_date"])
    current_prices = prices.iloc[-1]

    current_prices_with_cash = current_prices.copy()
    current_prices_with_cash["CASH"] = 1.0
    
    cov_stable = cov.copy()
    for t in cov_stable.index:
        cov_stable.loc[t, t] += 1e-9

    curr_paths, _ = run_portfolio_simulation(current_prices_with_cash, mu_bl, cov_stable, state["current_weights"], init_val)
    curr_stats = get_simulation_stats(curr_paths, init_val)
    
    opt_paths, _ = run_portfolio_simulation(current_prices_with_cash, mu_bl, cov_stable, state["optimal_weights"], init_val)
    opt_stats = get_simulation_stats(opt_paths, init_val)
    
    rebalance_plan = calculate_rebalancing_plan(
        state["current_weights"],
        state["optimal_weights"],
        current_prices_with_cash,
        init_val,
        state["user_holdings"]
    )
    
    from db_manager import RiskManagerDB
    db = RiskManagerDB()
    
    # 포트폴리오 로그 저장
    opt_expected_ret = opt_stats["expected_return"]
    opt_mdd = opt_stats["expected_mdd"]
    cash_weight = state["optimal_weights"].get("CASH", 0.0)
    
    current_date = pd.Timestamp.today()
    week_id = f"{current_date.year}-W{current_date.weekofyear:02d}"
    
    db.upsert_portfolio_log(
        week_id=week_id,
        total_aum=init_val,
        cash_weight=cash_weight,
        portfolio_expected_ret=opt_expected_ret,
        monte_carlo_mdd=opt_mdd
    )
    
    db.delete_asset_logs(week_id)
    
    # 종목별 상세(클러스터 ID 맵핑)
    ticker_to_cluster = {}
    for cid, members in state["clusters"].items():
        for m in members:
            ticker_to_cluster[m] = cid
            
    # 사용자 견해 맵핑
    user_views = {}
    for t in state["user_holdings"]:
        user_views[t] = state["user_holdings"][t]
    for t in state["user_watchlist"]:
        user_views[t] = state["user_watchlist"][t]
        
    # 거래 정보 맵핑
    trades_map = {t["ticker"]: t for t in rebalance_plan["trades"]}
    
    # 자산 로그 저장
    for ticker, target_w in state["optimal_weights"].items():
        if ticker == "CASH": continue
        
        expected_ret = state["posterior_returns"].get(ticker, 0.0)
        u_view = user_views.get(ticker, {}).get("score", 5)
        u_conf = user_views.get(ticker, {}).get("confidence", 5)
        c_id = ticker_to_cluster.get(ticker, -1)
        
        trade_info = trades_map.get(ticker, {})
        t_action = trade_info.get("action", "HOLD")
        t_qty = trade_info.get("shares", 0.0)
        t_price = current_prices_with_cash.get(ticker, 0.0)
        
        db.insert_asset_log(
            week_id=week_id,
            ticker=ticker,
            target_weight=target_w,
            expected_return=expected_ret,
            implied_equilibrium_ret=0.0, # Pi is not saved in state currently
            user_view=u_view,
            view_confidence=u_conf,
            cluster_id=c_id,
            trade_action=t_action,
            trade_qty=t_qty,
            execution_price=t_price
        )
    
    return {
        "current_stats": curr_stats, 
        "optimal_stats": opt_stats,
        "rebalance_plan": rebalance_plan
    }

def strategy_node(state: PortfolioState) -> PortfolioState:
    """Phase 6: AI 가디언 리포트 생성 노드"""
    print("[Node 3] AI 가디언 전략 리포트 생성 중...")
    
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.2)
    
    current = state["current_stats"]
    optimal = state["optimal_stats"]
    
    # 오늘 날짜 명시 (hallucination 방지)
    today_str = pd.Timestamp.today().strftime("%Y년 %m월 %d일")
    
    prompt = f"""
    [필수 준수 사항]
    1. 오늘 날짜는 **{today_str}**입니다. 리포트의 모든 날짜 관련 내용은 이 날짜를 기준으로 작성하세요.
    2. **리포트 최상단에 반드시 아래와 같이 작성 일자를 명시하십시오.**
       - **작성 일자: {today_str}**

    [역할 설정]
    당신은 글로벌 헤지펀드의 수석 퀀트 전략가이자 리스크 관리 총괄 책임자(CRO)입니다.
    제공된 데이터를 바탕으로 [AI 가디언 전략 리포트]를 작성하세요.

    [1. 분석 환경 및 기초 데이터]
    - 보고서 작성일: {today_str}
    - 분석 기간: {state['start_date']} ~ {state['end_date']} (최근 {state['settings'].get('lookback_years')}년 데이터 활용)
    - 초기 투자금: ${state['initial_value']:,.2f}
    - 무위험 수익률(Benchmark): {state['settings'].get('risk_free_rate', 0.035)*100:.2f}% (현재 국채 금리 기준)
    - 리스크 관리 제약: 특정 클러스터 비중 합계 {state['settings'].get('sector_limit', 0.3)*100}% 이내 제한
    - 클러스터링 임계값: {state['settings'].get('clustering_threshold', 0.4)}

    [2. 포트폴리오 비교 요약]
    A. 현재 포트폴리오 (Current):
    - 기대 수익률(연율화): {current['expected_return']*100:.2f}%
    - 기대 변동성: {current['expected_volatility']*100:.2f}%
    - 최대 낙폭(MDD): {current['expected_mdd']*100:.2f}%
    - 원금 보존 승률(1년 뒤): {current['win_rate']*100:.2f}%
    - 샤프 지수: {current['sharpe_ratio']:.2f}

    B. 최적 포트폴리오 (Target - BL Optimization):
    - 기대 수익률(연율화): {optimal['expected_return']*100:.2f}%
    - 기대 변동성: {optimal['expected_volatility']*100:.2f}%
    - 최대 낙폭(MDD): {optimal['expected_mdd']*100:.2f}%
    - 원금 보존 승률(1년 뒤): {optimal['win_rate']*100:.2f}%
    - 샤프 지수: {optimal['sharpe_ratio']:.2f}

    [3. 비중 데이터]
    - 현재 비중: {state['current_weights']}
    - 최적 비중: {state['optimal_weights']}
    
    [4. 정수 단위 리밸런싱 실행 계획 (Prioritized)]
    - 실행 순서 및 수량: {state['rebalance_plan']['trades']}
    - 예상 최종 현금 잔고: ${state['rebalance_plan']['estimated_final_cash']:,.2f}
    - 총 매도 대금: ${state['rebalance_plan']['total_sell_amount']:,.2f}
    - 총 매수 대금: ${state['rebalance_plan']['total_buy_amount']:,.2f}

    [5. 리스크 클러스터링 구조]
    - 식별된 그룹: {state['clusters']}

    ------------------------------------------------------------
    [보고서 작성 가이드라인]
    1. **요약(Executive Summary)**: 차이점 정의 및 개선 방향 명시.
    2. **블랙-리터만 전략 분석 (The Alpha)**: 의견 반영 및 비중 변화 배경 분석.
    3. **리스크 클러스터링 및 상관관계 (The Defense)**: 분산 진단 및 방어막 분석.
    4. **스트레스 테스트 및 몬테카를로 결과 (The Resilience)**: MDD와 승률 중심 해석.
    5. **최종 리밸런싱 가이드 (The Execution)**: 
       - 상기 제공된 '정수 단위 실행 계획'을 바탕으로 **매도(SELL) -> 매수(BUY)** 순서로 정렬된 테이블을 작성하십시오.
       - 테이블 컬럼명은 반드시 다음과 같이 짧게 구성하십시오: 
         [순서, 종목, 작업, 수량, 가격, 금액, 비중 변화]
       - 각 종목별로 현재 보유량 대비 몇 주를 더 사거나 팔아야 하는지 명확히 제시하십시오.
       - 마지막에 예상되는 최종 현금 잔고(${state['rebalance_plan']['estimated_final_cash']:,.2f})를 언급하며 마무리하십시오.
    6. **CRO의 결론 및 투자 조언**.

    작성 언어: 한국어 (전문적이고 신뢰감 있는 어조)
    출력 형식: GitHub Flavored Markdown
    """
    
    messages = [
        SystemMessage(content="You are an expert quantitative portfolio risk analyst."),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    
    # 리포트 내용을 문자열로 강제 변환 (리스트 형태 방지)
    report_content = response.content
    if isinstance(report_content, list):
        report_text = "".join([block.get("text", "") if isinstance(block, dict) else str(block) for block in report_content])
    else:
        report_text = str(report_content)
        
    return {"final_report": report_text}

def pdf_node(state: PortfolioState) -> PortfolioState:
    """최종 마크다운 리포트를 PDF 파일로 변환하는 노드"""
    print("[Node 4] PDF 전략 보고서 생성 중...")
    
    timestamp = pd.Timestamp.today().strftime("%Y%m%d_%H%M")
    filename = f"RiskManager_Report_{timestamp}.pdf"
    
    try:
        pdf_path = generate_pdf_report(state["final_report"], filename)
        return {"pdf_path": pdf_path}
    except Exception as e:
        print(f"      [Error] PDF 생성 중 오류 발생: {e}")
        return {"pdf_path": "Error"}

# 3. 그래프(Graph) 생성
def build_guardian_graph():
    workflow = StateGraph(PortfolioState)
    
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("simulate", simulation_node)
    workflow.add_node("report", strategy_node)
    workflow.add_node("pdf", pdf_node) # PDF 노드 추가
    
    workflow.set_entry_point("analysis")
    workflow.add_edge("analysis", "simulate")
    workflow.add_edge("simulate", "report")
    workflow.add_edge("report", "pdf") # report -> pdf 연결
    workflow.add_edge("pdf", END)
    
    return workflow.compile()
