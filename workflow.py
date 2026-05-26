import os
import numpy as np
import pandas as pd
from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, END

from data_provider import get_monte_carlo_params
from engine import calculate_risk_clusters, generate_dendrogram_plot
from pdf_generator import generate_pdf_report

# 1. 상태(State) 정의
class PortfolioState(TypedDict):
    # 입력 데이터
    all_tickers: List[str]
    settings: Dict[str, Any]
    start_date: str
    end_date: str
    
    # 분석 데이터
    clusters: Dict[int, List[str]]
    cluster_limits: Dict[int, float]
    dendrogram_path: str
    volatilities: Dict[str, float]
    
    # 최종 결과물
    final_report: str
    pdf_path: str

# 2. 노드(Nodes) 정의

def analysis_node(state: PortfolioState) -> PortfolioState:
    """Phase 1: 상관관계 트리 생성, 종목 그룹화 및 그룹별 리스크 한계치 산출"""
    print("[Node 1] 포트폴리오 리스크 구조 분석 가동 (Clustering & Volatility Limits)...")
    
    tickers = state["all_tickers"]
    
    # 1. 2개년 과거 가격 데이터 수집 및 연환산 변동성 계산
    prices, _, volatilities, cov = get_monte_carlo_params(tickers, state["start_date"], state["end_date"])
    if prices is None or prices.empty:
        raise ValueError("가격 데이터를 가져오는 데 실패했습니다.")
        
    returns = np.log(prices / prices.shift(1)).dropna()
    corr = returns.corr()
    
    # 2. 상관계수 기반 덴드로그램 계층 트리 생성 및 이미지 저장
    dendrogram_path = "dendrogram.png"
    threshold = state["settings"].get("clustering_threshold", 1.0)
    generate_dendrogram_plot(corr, threshold=threshold, output_path=dendrogram_path)
    
    # 3. 임계점에 따른 계층적 자산 그룹화
    clusters = calculate_risk_clusters(returns, threshold=threshold)
    
    # 4. 각 그룹별 변동성에 따른 투자 한계 비중(Dynamic Limit) 결정
    market_avg_vol = volatilities.mean()
    base_limit = state["settings"].get("sector_limit", 0.3)
    
    cluster_limits = {}
    for cid, members in clusters.items():
        cluster_vols = volatilities[members]
        cluster_avg_vol = cluster_vols.mean()
        
        # 전체 시장 평균 대비 상대적 크기로 한계치 조절 (10% ~ 50% 제한)
        if cluster_avg_vol > 0:
            dynamic_limit = base_limit * (market_avg_vol / cluster_avg_vol)
        else:
            dynamic_limit = base_limit
        dynamic_limit = np.clip(dynamic_limit, 0.10, 0.50)
        cluster_limits[cid] = float(dynamic_limit)
        
    return {
        "clusters": clusters,
        "cluster_limits": cluster_limits,
        "dendrogram_path": dendrogram_path,
        "volatilities": volatilities.to_dict()
    }

def report_node(state: PortfolioState) -> PortfolioState:
    """Phase 2: 리스크 배분 및 구조화 종합 보고서 마크다운 포맷팅"""
    print("[Node 2] 리스크 구조화 최종 분석 보고서 생성 중...")
    
    clusters = state["clusters"]
    limits = state["cluster_limits"]
    dendrogram_path = state["dendrogram_path"]
    volatilities = state["volatilities"]
    
    threshold = state["settings"].get("clustering_threshold", 1.0)
    base_limit = state["settings"].get("sector_limit", 0.3)
    today_str = pd.Timestamp.today().strftime("%Y년 %m월 %d일")
    
    markdown = []
    markdown.append(f"# AI Portfolio Guardian: 리스크 배분 및 구조화 보고서")
    markdown.append(f"**작성 일자: {today_str}**")
    
    markdown.append(f"\n## Executive Summary")
    markdown.append(f"본 보고서는 포트폴리오 후보 자산군 간의 상관관계와 변동성을 분석하여 거시적 리스크 구조를 설계한 결과입니다.")
    markdown.append(f"기계적인 개별 종목 최적화(3단계)를 배제하고, 자산 그룹화 및 그룹별 리스크 허용 한도(Dynamic Limit)를 도출하여 투자자가 자율적으로 세부 비중을 조절할 수 있도록 안전펜스를 설계하였습니다.")
    markdown.append(f"\n* **그룹 설정 기준 임계점 (Threshold):** {threshold:.2f}")
    markdown.append(f"* **그룹별 기준 리스크 한계 비중 (Base Limit):** {base_limit*100:.1f}%")
    
    markdown.append(f"\n## 1. 상관계수 기반 자산 계층 트리 (Dendrogram)")
    markdown.append(f"아래 차트는 종목 간 일별 로그 수익률의 상관관계를 거리(Distance = sqrt(2 * (1 - rho)))로 환산하여 도출한 계층 구조 트리입니다.")
    markdown.append(f"세로축의 리스크 거리가 임계선(Threshold={threshold}) 이하인 종목들이 하나의 리스크 그룹으로 묶이게 됩니다.")
    markdown.append(f"\n![Dendrogram]({dendrogram_path})")
    
    markdown.append(f"\n## 2. 임계점에 따른 주식 그룹화 결과 (Risk Clusters)")
    markdown.append(f"상관관계 거리가 가까운 자산들을 그룹화한 리스크 클러스터 진단 결과입니다.")
    markdown.append(f"\n| 리스크 그룹 ID | 소속 종목 리스트 |")
    markdown.append(f"| :---: | :--- |")
    for cid, members in clusters.items():
        members_str = ", ".join(members)
        markdown.append(f"| **Cluster {cid}** | {members_str} |")
        
    markdown.append(f"\n## 3. 그룹별 변동성에 따른 권고 투자 한계 비중 (Dynamic Limits)")
    markdown.append(f"각 그룹에 속한 자산들의 변동성을 연환산 기준으로 분석하여, **변동성이 낮은 안전 그룹에는 높은 투자 한도를 부여하고 고변동성 위험 그룹에는 타이트한 한도를 강제**한 결과입니다.")
    markdown.append(f"각 그룹에 투자하실 때 이 합산 비중 제한선(Dynamic Limit)을 넘지 않도록 자율 투자 포트폴리오를 관리하시기 바랍니다.")
    markdown.append(f"\n| 리스크 그룹 ID | 소속 종목 | 그룹 평균 변동성 | 권고 투자 한계 비중 (Dynamic Limit) |")
    markdown.append(f"| :---: | :--- | :---: | :---: |")
    for cid, members in clusters.items():
        members_str = ", ".join(members)
        cluster_vols = [volatilities[m] for m in members]
        cluster_avg_vol = np.mean(cluster_vols)
        limit_pct = limits[cid] * 100
        markdown.append(f"| **Cluster {cid}** | {members_str} | {cluster_avg_vol*100:.2f}% | **{limit_pct:.2f}%** |")
        
    markdown.append(f"\n\n**[안내]** 본 리포트의 그룹별 투자 한계 비중은 각 그룹의 연환산 표준편차를 기준으로 시장 평균 대비 가중치를 산출한 수치입니다. 자산 배분 가이드라인으로 활용해 주시기 바랍니다.")
    
    report_text = "\n".join(markdown)
    return {"final_report": report_text}

def pdf_node(state: PortfolioState) -> PortfolioState:
    """Phase 3: 최종 마크다운 리포트를 PDF 파일로 변환하여 reports/ 폴더에 빌드"""
    print("[Node 3] PDF 보고서 생성 및 파일 빌드 중...")
    
    timestamp = pd.Timestamp.today().strftime("%Y%m%d_%H%M")
    filename = f"RiskManager_Report_{timestamp}.pdf"
    
    os.makedirs("reports", exist_ok=True)
    output_path = os.path.join("reports", filename)
    
    try:
        pdf_path = generate_pdf_report(state["final_report"], output_path)
        return {"pdf_path": pdf_path}
    except Exception as e:
        print(f"      [Error] PDF 생성 중 오류 발생: {e}")
        return {"pdf_path": "Error"}

# 3. 그래프(Graph) 생성 및 컴파일
def build_guardian_graph():
    workflow = StateGraph(PortfolioState)
    
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("report", report_node)
    workflow.add_node("pdf", pdf_node)
    
    workflow.set_entry_point("analysis")
    workflow.add_edge("analysis", "report")
    workflow.add_edge("report", "pdf")
    workflow.add_edge("pdf", END)
    
    return workflow.compile()
