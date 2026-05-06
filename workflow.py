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
from engine import calculate_risk_clusters, black_litterman_posterior, optimize_portfolio

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
    
    start_date: str
    end_date: str
    
    # 통계 결과
    current_stats: Dict[str, Any]
    optimal_stats: Dict[str, Any]
    
    # 최종 결과물
    final_report: str

# 2. 노드(Nodes) 정의

def analysis_node(state: PortfolioState) -> PortfolioState:
    """Phase 2, 3, 4: 클러스터링, 블랙-리터만, 최적화를 수행하는 노드"""
    print("[Node 1] 전략 수립 엔진 가동 (Clustering & Black-Litterman)...")
    
    tickers = state["all_tickers"]
    user_views = {}
    # holdings와 watchlist에서 score/confidence 추출
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

    # 데이터 로드
    prices, _, _, cov = get_monte_carlo_params(tickers, state["start_date"], state["end_date"])
    returns = np.log(prices / prices.shift(1)).dropna()
    
    # Phase 2: 리스크 클러스터링
    clusters = calculate_risk_clusters(returns)
    
    # Phase 3: 블랙-리터만 기대수익률 산출
    rf_rate = state["settings"].get("risk_free_rate", 0.035)
    mu_bl, _ = black_litterman_posterior(state["market_caps"], cov, user_views, rf_rate)
    
    # Phase 4: 제약 조건 기반 최적화
    sector_limit = state["settings"].get("sector_limit", 0.3)
    optimal_w = optimize_portfolio(mu_bl, cov, clusters, sector_limit, rf_rate)
    
    return {
        "clusters": clusters,
        "posterior_returns": mu_bl,
        "cov_matrix": cov,
        "optimal_weights": optimal_w
    }

def simulation_node(state: PortfolioState) -> PortfolioState:
    """Phase 5: 몬테카를로 스트레스 테스트 노드"""
    print("[Node 2] 최적 전략 스트레스 테스트 중 (10,000회 시뮬레이션)...")
    
    init_val = state["initial_value"]
    cov = state["cov_matrix"]
    mu_bl = state["posterior_returns"]
    
    # 현재 가격 가져오기 (마지막 행)
    tickers = state["all_tickers"]
    prices, _, _, _ = get_monte_carlo_params(tickers, state["start_date"], state["end_date"])
    current_prices = prices.iloc[-1]

    # A. 현재 포트폴리오 시뮬레이션
    curr_paths, _ = run_portfolio_simulation(current_prices, mu_bl, cov, state["current_weights"], init_val)
    curr_stats = get_simulation_stats(curr_paths, init_val)
    
    # B. 최적 포트폴리오 시뮬레이션
    opt_paths, _ = run_portfolio_simulation(current_prices, mu_bl, cov, state["optimal_weights"], init_val)
    opt_stats = get_simulation_stats(opt_paths, init_val)
    
    return {"current_stats": curr_stats, "optimal_stats": opt_stats}

def strategy_node(state: PortfolioState) -> PortfolioState:
    """Phase 6: AI 가디언 리포트 생성 노드"""
    print("[Node 3] AI 가디언 전략 리포트 생성 중...")
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    
    current = state["current_stats"]
    optimal = state["optimal_stats"]
    
    prompt = f"""
    당신은 글로벌 헤지펀드의 수석 퀀트 전략가이자 리스크 관리 총괄 책임자(CRO)입니다.
    사용자의 '현재 포트폴리오'와 수학적 최적화 엔진(Black-Litterman & QP)이 산출한 '최적 포트폴리오'를 비교하여, 전문가 수준의 [AI 가디언 전략 리포트]를 작성하세요.

    [1. 분석 환경 및 기초 데이터]
    - 분석 기간: {state['start_date']} ~ {state['end_date']} (최근 {state['settings'].get('lookback_years')}년 데이터 활용)
    - 초기 투자금: ${state['initial_value']:,.2f}
    - 무위험 수익률(Benchmark): {state['settings'].get('risk_free_rate', 0.035)*100:.2f}% (현재 국채 금리 기준)
    - 리스크 관리 제약: 특정 클러스터 비중 합계 {state['settings'].get('sector_limit', 0.3)*100}% 이내 제한

    [2. 포트폴리오 비교 데이터]
    A. 현재 포트폴리오 (Current):
    - 구성 및 비중: {state['current_weights']}
    - 기대 수익률(연율화): {current['expected_return']*100:.2f}%
    - 최대 낙폭(MDD): {current['expected_mdd']*100:.2f}%
    - 원금 보존 승률(1년 뒤): {current['win_rate']*100:.2f}%

    B. 최적 포트폴리오 (Target - BL Optimization):
    - 권장 비중: {state['optimal_weights']}
    - 기대 수익률(연율화): {optimal['expected_return']*100:.2f}%
    - 최대 낙폭(MDD): {optimal['expected_mdd']*100:.2f}%
    - 원금 보존 승률(1년 뒤): {optimal['win_rate']*100:.2f}%

    [3. 리스크 클러스터링 진단 (통계적 상관관계 그룹)]
    - 식별된 클러스터 구조: {state['clusters']}

    ------------------------------------------------------------
    [보고서 작성 지침 - 다음 섹션들을 반드시 포함할 것]

    1. **Executive Summary**: 현재 포트폴리오의 치명적인 약점과 최적화 후 얻게 될 전략적 이점(수익률 제고, 리스크 방어 등)을 요약하십시오.
    
    2. **Black-Litterman Insight**: 사용자의 주관적 견해(Score/Confidence)가 시장 데이터와 결합되어 어떻게 '수정 기대수익률'을 형성했는지, 그리고 이것이 비중 변화에 어떤 영향을 주었는지 설명하십시오.
    
    3. **Risk Clustering Analysis**: 현재 클러스터링 임계값 설정(0.4)을 통해 아주 세밀하게 리스크를 분산했습니다. 특정 그룹에 자금이 쏠리지 않도록 엔진이 어떻게 방어막을 쳤는지 분석하십시오.
    
    4. **Stress Test Results**: 1만 번의 몬테카를로 시뮬레이션 결과를 바탕으로, 향후 1년간 겪을 수 있는 최악의 시나리오와 이를 극복할 가능성(승률)을 언급하십시오.
    
    5. **Final Rebalancing Guide (Trade List)**: 
       - 종목별 현재 비중 대비 매수/매도 필요량을 명확히 표기하십시오.
       - 총 투자금 ${state['initial_value']:,.2f}를 기준으로 실제 집행해야 할 달러($) 금액을 계산하여 테이블 형식으로 제시하십시오.
    
    6. **Closing Advice**: 투자자가 심리적으로 흔들리지 않고 이 전략을 완수하기 위한 CRO로서의 마지막 조언을 남기십시오.

    작성 언어: 한국어 (전문적이고 신뢰감 있는 어조)
    출력 형식: GitHub Flavored Markdown
    """
    
    messages = [
        SystemMessage(content="You are an expert quantitative portfolio risk analyst."),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    return {"final_report": response.content}

# 3. 그래프(Graph) 생성
def build_guardian_graph():
    workflow = StateGraph(PortfolioState)
    
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("simulate", simulation_node)
    workflow.add_node("report", strategy_node)
    
    workflow.set_entry_point("analysis")
    workflow.add_edge("analysis", "simulate")
    workflow.add_edge("simulate", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()
