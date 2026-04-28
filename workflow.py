import os
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from data_provider import get_monte_carlo_params
from simulation import run_portfolio_simulation, get_simulation_stats

# 1. 상태(State) 정의
class PortfolioState(TypedDict):
    current_portfolio: Dict[str, float]  # 티커와 비중 (현재)
    new_portfolio: Dict[str, float]      # 티커와 비중 (관심 종목 편입 후)
    initial_value: float                 # 초기 자본
    start_date: str                      # 과거 데이터 시작일
    end_date: str                        # 과거 데이터 종료일
    
    # 노드 간 전달될 중간 데이터
    current_stats: Dict[str, Any]
    new_stats: Dict[str, Any]
    
    # 최종 결과물
    final_report: str

# 2. 노드(Nodes) 정의
def simulation_node(state: PortfolioState) -> PortfolioState:
    """몬테카를로 시뮬레이션을 실행하여 통계 데이터를 추출하는 노드"""
    print("📊 [Node 1] 몬테카를로 시뮬레이션 엔진 가동 중...")
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
    print(f"🧠 [Node 2] AI 가디언 리포트 생성 중... (LLM 추론 시작, 약 5~10초 소요될 수 있습니다)")
    
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
    workflow.add_node("simulate", simulation_node)
    workflow.add_node("analyze", strategy_node)
    
    # 엣지 연결 (흐름 제어)
    workflow.set_entry_point("simulate")
    workflow.add_edge("simulate", "analyze")
    workflow.add_edge("analyze", END)
    
    return workflow.compile()
