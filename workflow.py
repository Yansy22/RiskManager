import os
import numpy as np
import pandas as pd
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from data_provider import get_monte_carlo_params
from simulation import run_portfolio_simulation, get_simulation_stats

# 1. 상태(State) 정의
class PortfolioState(TypedDict):
    # 입력 데이터
    holdings: Dict[str, float]           # 티커와 수량
    cash: float                         # 보유 현금
    watchlist: Dict[str, float]         # 관심 종목과 목표 비중 (0.0 ~ 1.0)
    
    # 계산된 비중 (워크플로우 내부 사용)
    current_portfolio: Dict[str, float]  # 티커와 비중 (현재)
    new_portfolio: Dict[str, float]      # 티커와 비중 (편입 후)
    initial_value: float                 # 총 자본
    start_date: str                      # 과거 데이터 시작일
    end_date: str                        # 과거 데이터 종료일
    
    # 노드 간 전달될 중간 데이터
    current_stats: Dict[str, Any]
    new_stats: Dict[str, Any]
    
    # 최종 결과물
    final_report: str

# 2. 노드(Nodes) 정의
def rebalance_node(state: PortfolioState) -> PortfolioState:
    """관심 종목을 추가하기 위해 기존 자산을 재배분하는 노드"""
    print("[Node 1] 포트폴리오 리밸런싱 전략 수립 중...")
    
    # 모든 관련 티커 합집합
    current_tickers = list(state["holdings"].keys())
    watchlist_tickers = list(state["watchlist"].keys())
    all_tickers = list(set(current_tickers + watchlist_tickers))
    
    # 상관관계 데이터 가져오기
    _, _, _, cov = get_monte_carlo_params(all_tickers, state["start_date"], state["end_date"])
    
    # 상관행렬 계산
    std_dev = np.sqrt(np.diag(cov))
    corr_matrix = cov / np.outer(std_dev, std_dev)
    
    # 1. 현재 비중 계산 (main.py에서 이미 계산해서 넘겨줬을 수 있지만 여기서 재검증 가능)
    # 현재는 main.py에서 넘어온 current_portfolio 사용
    new_portfolio = state["current_portfolio"].copy()
    current_cash_weight = state["cash"] / state["initial_value"]
    
    target_new_weight = sum(state["watchlist"].values())
    
    print(f"   -> 목표 관심종목 비중: {target_new_weight*100:.1f}%")
    print(f"   -> 가용 현금 비중: {current_cash_weight*100:.1f}%")

    # 관심 종목 추가 로직
    # A. 현금을 우선적으로 사용
    cash_to_use = min(current_cash_weight, target_new_weight)
    remaining_to_fund = target_new_weight - cash_to_use
    
    # B. 부족한 비중은 기존 주식에서 차감 (상관계수 기반)
    if remaining_to_fund > 0:
        print(f"   -> 현금 부족분({remaining_to_fund*100:.1f}%)을 기존 주식에서 충당합니다.")
        # 관심 종목들과의 평균 상관계수가 가장 높은 기존 종목 순으로 정렬
        correlations = {}
        for ct in current_tickers:
            avg_corr = np.mean([corr_matrix.loc[ct, wt] for wt in watchlist_tickers])
            correlations[ct] = avg_corr
        
        # 상관계수 내림차순 정렬
        sorted_tickers = sorted(correlations, key=correlations.get, reverse=True)
        
        total_current_stock_weight = sum(state["current_portfolio"].values())
        
        for ct in sorted_tickers:
            if remaining_to_fund <= 0: break
            
            current_w = new_portfolio[ct]
            reduction = min(current_w, remaining_to_fund)
            new_portfolio[ct] -= reduction
            remaining_to_fund -= reduction
            print(f"      [조정] {ct} 비중 {reduction*100:.1f}% 차감 (관심종목 유사도 높음)")

    # 관심 종목 비중 설정
    for wt, weight in state["watchlist"].items():
        new_portfolio[wt] = weight
        
    return {"new_portfolio": new_portfolio}


def simulation_node(state: PortfolioState) -> PortfolioState:
    """몬테카를로 시뮬레이션을 실행하여 통계 데이터를 추출하는 노드"""
    print("[Node 2] 몬테카를로 시뮬레이션 엔진 가동 중...")
    start_date = state["start_date"]
    end_date = state["end_date"]
    init_val = state["initial_value"]
    
    def run_sim_for_portfolio(portfolio):
        tickers = list(portfolio.keys())
        prices, mean_ret, vol, cov = get_monte_carlo_params(tickers, start_date, end_date)
        
        if prices is None or prices.empty:
            raise ValueError(f"데이터를 불러오지 못했습니다: {tickers}")
            
        current_prices = prices.iloc[-1]
        paths, _ = run_portfolio_simulation(current_prices, mean_ret, cov, portfolio, init_val)
        stats = get_simulation_stats(paths, init_val)
        return stats

    print("   -> 현재 포트폴리오(Current) 10,000회 시뮬레이션 중...")
    current_stats = run_sim_for_portfolio(state["current_portfolio"])
    
    print("   -> 관심 종목 편입 포트폴리오(What-If) 10,000회 시뮬레이션 중...")
    new_stats = run_sim_for_portfolio(state["new_portfolio"])
    
    return {"current_stats": current_stats, "new_stats": new_stats}


def strategy_node(state: PortfolioState) -> PortfolioState:
    """LLM을 이용해 시뮬레이션 결과를 해석하고 전략을 세우는 노드"""
    print(f"[Node 3] AI 가디언 리포트 생성 중... (LLM 추론 시작, 약 5~10초 소요될 수 있습니다)")
    
    # LLM 초기화 (gemini-2.5-flash 모델 사용)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    
    current = state["current_stats"]
    new = state["new_stats"]
    
    prompt = f"""
    당신은 최고 수준의 금융 리스크 관리 AI 가디언입니다.
    사용자의 '현재 포트폴리오'와 관심 종목을 편입한 '새로운 포트폴리오'의 몬테카를로 시뮬레이션(10,000회, 1년 후) 결과가 주어집니다.
    수치들을 객관적으로 비교 분석하여 리스크와 수익률 측면에서 사용자에게 전문적이고 직관적인 조언을 제공하세요.
    
    [시뮬레이션 조건]
    - 초기 투자금: ${state['initial_value']:,.2f}
    
    [현재 포트폴리오 비중]
    {state['current_portfolio']}
    
    [현재 포트폴리오 결과]
    - 예상 기대 수익률: {current['expected_return']*100:.2f}%
    - 승률 (원금 보존 확률): {current['win_rate']*100:.2f}%
    - Value at Risk (95% 신뢰구간 최악 5% 손실률): {current['var_95']*100:.2f}%
    - Expected MDD (평균 최대 낙폭): {current['expected_mdd']*100:.2f}%
    
    [관심 종목 편입 후 포트폴리오 비중]
    {state['new_portfolio']}
    
    [편입 후 포트폴리오 결과]
    - 예상 기대 수익률: {new['expected_return']*100:.2f}%
    - 승률 (원금 보존 확률): {new['win_rate']*100:.2f}%
    - Value at Risk (95% 신뢰구간 최악 5% 손실률): {new['var_95']*100:.2f}%
    - Expected MDD (평균 최대 낙폭): {new['expected_mdd']*100:.2f}%
    
    보고서 작성 지침:
    1. 편입 전/후의 수익률과 리스크(MDD, VaR) 변화를 수치와 함께 명확히 비교하세요.
    2. 편입하려는 종목이 전체 포트폴리오 리스크에 미치는 영향을 평가하세요 (예: 하방 리스크 증가, 분산 효과 여부 등).
    3. 리스크 대비 수익 효율성을 판단하여 최종 권고안(비중 유지, 축소, 확대, 대안 제시 등)을 내려주세요.
    4. 친절하고 신뢰감 있는 마크다운 포맷으로 가독성 좋게 작성하세요.
    """
    
    messages = [
        SystemMessage(content="You are an expert quantitative portfolio risk analyst."),
        HumanMessage(content=prompt)
    ]
    
    print("   -> 프롬프트 작성 완료. Google Gemini API에 요청을 보냅니다...")
    response = llm.invoke(messages)
    print("   -> 리포트 생성이 완료되었습니다!")
    
    return {"final_report": response.content}


# 3. 그래프(Graph) 생성
def build_guardian_graph():
    workflow = StateGraph(PortfolioState)
    
    # 노드 추가
    workflow.add_node("rebalance", rebalance_node)
    workflow.add_node("simulate", simulation_node)
    workflow.add_node("analyze", strategy_node)
    
    # 엣지 연결 (흐름 제어)
    workflow.set_entry_point("rebalance")
    workflow.add_edge("rebalance", "simulate")
    workflow.add_edge("simulate", "analyze")
    workflow.add_edge("analyze", END)
    
    return workflow.compile()
