import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from typing import Dict, List, Tuple

def calculate_risk_clusters(returns: pd.DataFrame, threshold: float = 0.4) -> Dict[int, List[str]]:
    """
    수익률 데이터를 기반으로 종목들을 리스크 그룹(클러스터)으로 묶습니다.
    
    Args:
        returns: 종목별 일별 수익률 DataFrame
        threshold: 클러스터를 나눌 거리 임계값 (작을수록 더 촘촘하게 묶임)
        
    Returns:
        clusters: {클러스터ID: [종목리스트]} 형태의 딕셔너리
    """
    # 1. 상관계수 행렬 계산
    corr = returns.corr()
    
    # 2. 상관계수를 거리(Distance)로 변환: d = sqrt(2 * (1 - rho))
    # 상관계수가 1이면 거리는 0, -1이면 거리는 2
    dist_matrix = np.sqrt(2 * (1 - corr))
    
    # 3. 계층적 군집화 (Ward's Method)
    # condensed distance matrix 형태로 변환 후 linkage 수행
    condensed_dist = squareform(dist_matrix, checks=False)
    linkage_matrix = linkage(condensed_dist, method='ward')
    
    # 4. 클러스터 할당
    # 임계값 기준으로 클러스터를 자름
    cluster_labels = fcluster(linkage_matrix, threshold, criterion='distance')
    
    # 5. 결과 정리
    clusters = {}
    for ticker, label in zip(corr.index, cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(ticker)
        
    print(f"      [Engine] 총 {len(clusters)}개의 리스크 클러스터가 식별되었습니다.")
    for cid, members in clusters.items():
        print(f"        - Cluster {cid}: {members}")
        
    return clusters

def check_cluster_exposure(weights: Dict[str, float], clusters: Dict[int, List[str]], limit: float = 0.3) -> Dict[int, float]:
    """
    각 클러스터별 현재 비중 합계를 계산하고 제한선 초과 여부를 확인합니다.
    """
    exposure = {}
    for cid, members in clusters.items():
        total_w = sum(weights.get(m, 0.0) for m in members)
        exposure[cid] = total_w
        if total_w > limit:
            print(f"      [Warning] Cluster {cid} 비중({total_w*100:.1f}%)이 제한선({limit*100:.1f}%)을 초과했습니다.")
            
    return exposure

# --- Phase 3: Black-Litterman Engine ---

def calculate_implied_risk_aversion(market_returns: pd.Series, risk_free_rate: float = 0.035) -> float:
    """시장 데이터를 바탕으로 위험 회피 계수(Delta)를 계산합니다."""
    excess_return = market_returns.mean() * 252 - risk_free_rate
    variance = (market_returns.std() * np.sqrt(252)) ** 2
    delta = excess_return / variance
    return max(delta, 1e-6) # 최소값 보장

def black_litterman_posterior(
    market_caps: Dict[str, float],
    cov_matrix: pd.DataFrame,
    user_views: Dict[str, Dict[str, float]],
    risk_free_rate: float = 0.035,
    tau: float = 0.05
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    시장 균형 수익률과 사용자의 견해를 결합하여 수정된 기대수익률을 산출합니다.
    """
    tickers = list(cov_matrix.index)
    num_assets = len(tickers)
    
    # 1. 시장 시가총액 비중 (Market Weights)
    total_cap = sum(market_caps.values())
    w_mkt = np.array([market_caps.get(t, 0.0) / total_cap for t in tickers])
    
    # 2. 시장 균형 수익률 (Implied Equilibrium Returns - Pi)
    # Delta (Risk Aversion)는 일반적으로 2.5 ~ 3.0 수준 사용
    delta = 3.0 
    Pi = delta * (cov_matrix @ w_mkt)
    
    # 3. 사용자 견해 처리 (P, Q, Omega)
    # 여기서는 각 종목에 대한 절대적 견해(Absolute Views)로 단순화
    P = np.eye(num_assets)
    Q = []
    Omega_diag = []
    
    for t in tickers:
        view = user_views.get(t, {"score": 5, "confidence": 5})
        # Score(1~10)를 기대 수익률로 매핑 (5점=Pi, 10점=Pi+10%, 1점=Pi-10% 등)
        # Pi[t]가 시장의 기대치라면, 사용자의 점수는 그에 대한 상대적 확신
        excess_view = (view['score'] - 5) * 0.02 # 1점당 2%p 가중
        Q.append(Pi.loc[t] + excess_view)
        
        # Confidence(1~10)를 오차 분산으로 매핑 (10점=매우 낮음, 1점=매우 높음)
        # 분산이 클수록 시장 수익률(Pi)에 더 의존하게 됨
        variance_factor = (11 - view['confidence']) * 0.01
        Omega_diag.append(max(variance_factor, 1e-6))
        
    Q = np.array(Q)
    Omega = np.diag(Omega_diag)
    
    # 4. 베이지안 업데이트 수식 (Black-Litterman Formula)
    # mu_bl = [ (tau*Sigma)^-1 + P^T*Omega^-1*P ]^-1 * [ (tau*Sigma)^-1*Pi + P^T*Omega^-1*Q ]
    tau_sigma_inv = np.linalg.inv(tau * cov_matrix)
    omega_inv = np.linalg.inv(Omega)
    
    # 공통 부분 계산
    term1 = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
    term2 = tau_sigma_inv @ Pi + P.T @ omega_inv @ Q
    
    mu_bl = term1 @ term2
    
    print(f"      [Engine] 블랙-리터만 수정 수익률 산출 완료 (평균: {np.mean(mu_bl)*100:.2f}%)")
    
    return pd.Series(mu_bl, index=tickers), cov_matrix

# --- Phase 4: Constrained Optimization ---

from scipy.optimize import minimize

def optimize_portfolio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    clusters: Dict[int, List[str]],
    cluster_limit: float = 0.3,
    risk_free_rate: float = 0.035
) -> Dict[str, float]:
    """
    샤프 지수를 극대화하는 최적 비중을 산출합니다 (동적 클러스터 한계치 및 내적 역변동성 제약 조건 포함).
    """
    tickers = list(expected_returns.index)
    num_assets = len(tickers)
    
    # 1. 개별 연환산 변동성 및 시장 평균 변동성 계산
    asset_vols = pd.Series(np.sqrt(np.diag(cov_matrix.values)), index=tickers)
    market_avg_vol = asset_vols.mean()
    
    # 2. 목적 함수: -Sharpe Ratio (minimize를 위해 부호 반전)
    def objective(weights):
        p_return = np.sum(expected_returns.values * weights)
        p_vol = np.sqrt(weights.T @ cov_matrix.values @ weights)
        if p_vol == 0: return 0
        return -(p_return - risk_free_rate) / p_vol

    # 3. 제약 조건 (Constraints)
    constraints = []
    
    # A. 비중 합계 = 1.0
    constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    
    # B. 클러스터별 동적 비중 제한 및 클러스터 내부 역변동성 등식 제약 조건
    for cid, members in clusters.items():
        # 각 멤버의 인덱스 찾기
        indices = [tickers.index(m) for m in members if m in tickers]
        if not indices:
            continue
            
        # (1) 동적 한계치 계산
        cluster_avg_vol = asset_vols[members].mean()
        # 평균 변동성에 반비례하게 한계치 결정 (평균 변동성 대비 동적 조절)
        if cluster_avg_vol > 0:
            dynamic_limit = cluster_limit * (market_avg_vol / cluster_avg_vol)
        else:
            dynamic_limit = cluster_limit
        dynamic_limit = float(np.clip(dynamic_limit, 0.10, 0.50))
        
        print(f"      [Engine] Risk Cluster {cid} (종목: {members}, 평균 변동성: {cluster_avg_vol*100:.1f}%) -> 동적 한계치: {dynamic_limit*100:.1f}%")
        
        # 해당 클러스터 멤버들의 비중 합이 dynamic_limit보다 작아야 함 (dynamic_limit - sum >= 0)
        constraints.append({
            'type': 'ineq', 
            'fun': lambda w, idx=indices, limit=dynamic_limit: limit - np.sum(w[idx])
        })
        
        # (2-2) 클러스터 내부 종목 간의 역변동성 등식 제약 조건 추가 (w_i * vol_i = w_0 * vol_0)
        if len(indices) > 1:
            first_idx = indices[0]
            first_vol = asset_vols.iloc[first_idx]
            
            # 클러스터 내의 임의의 두 종목 i, j에 대해 w_i * vol_i = w_j * vol_j 여야 하므로,
            # 기준 종목(indices[0])과 나머지 종목 간의 차이를 0으로 고정시킵니다.
            for idx in indices[1:]:
                vol = asset_vols.iloc[idx]
                constraints.append({
                    'type': 'eq',
                    'fun': lambda w, idx=idx, f_idx=first_idx, v=vol, f_v=first_vol: w[idx] * v - w[f_idx] * f_v
                })

    # 4. 각 종목별 비중 범위 (0.0 ~ 1.0, 공매도 금지)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    
    # 5. 초기값 (균등 배분)
    init_guess = np.array([1.0 / num_assets] * num_assets)
    
    # 6. 최적화 실행
    print(f"      [Engine] 최적화 알고리즘 실행 중 (목표: Max Sharpe Ratio, 동적 한계치 & 역변동성 제약 포함)...")
    result = minimize(
        objective, 
        init_guess, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    
    if not result.success:
        print(f"      [Warning] 최적화가 완벽히 수렴하지 않았습니다: {result.message}")
        
    optimal_weights = dict(zip(tickers, result.x))
    
    # 결과 요약 출력
    print(f"      [Engine] 최적화 완료.")
    for t, w in optimal_weights.items():
        if w > 0.001: # 0.1% 이상 비중만 출력
            print(f"        - {t}: {w*100:.2f}% (개별 변동성: {asset_vols[t]*100:.1f}%)")
            
    return optimal_weights

def calculate_rebalancing_plan(
    current_weights: Dict[str, float],
    optimal_weights: Dict[str, float],
    current_prices: pd.Series,
    total_value: float,
    current_holdings: Dict[str, Any],
    tolerance: float = 0.02 # 2% 미만 변화는 무시
) -> Dict[str, Any]:
    """
    정수 단위 매매 수량과 실행 우선순위를 포함한 리밸런싱 계획을 수립합니다.
    """
    trades = []
    
    # 1. 모든 종목 리스트 (CASH 제외)
    all_tickers = list(set(current_weights.keys()) | set(optimal_weights.keys()))
    if "CASH" in all_tickers: all_tickers.remove("CASH")
    
    for ticker in all_tickers:
        price = current_prices.get(ticker)
        if not price or price <= 0: continue
        
        target_w = optimal_weights.get(ticker, 0.0)
        curr_w = current_weights.get(ticker, 0.0)
        
        # Tolerance 체크: 변화폭이 너무 작으면 유지
        if abs(target_w - curr_w) < tolerance:
            continue
            
        target_shares = int((total_value * target_w) // price)
        curr_shares = current_holdings.get(ticker, {}).get("quantity", 0)
        
        diff_shares = target_shares - curr_shares
        
        if diff_shares != 0:
            trades.append({
                "ticker": ticker,
                "action": "BUY" if diff_shares > 0 else "SELL",
                "shares": abs(diff_shares),
                "price": float(price),
                "amount": abs(diff_shares * price),
                "weight_diff": target_w - curr_w
            })
            
    # 2. 우선순위 정렬
    # SELL: 비중 감소폭이 큰 순서대로 (현금 확보 우선)
    sells = sorted([t for t in trades if t["action"] == "SELL"], key=lambda x: x["weight_diff"])
    # BUY: 비중 증가폭이 큰 순서대로
    buys = sorted([t for t in trades if t["action"] == "BUY"], key=lambda x: x["weight_diff"], reverse=True)
    
    # 3. 예상 최종 현금 계산
    total_sell_amount = sum(t["amount"] for t in sells)
    total_buy_amount = sum(t["amount"] for t in buys)
    # 초기 현금에서 매도액을 더하고 매수액을 뺀 값
    initial_cash = current_weights.get("CASH", 0.0) * total_value
    estimated_final_cash = initial_cash + total_sell_amount - total_buy_amount
    
    return {
        "trades": sells + buys, # 매도 후 매수 순서로 합침
        "estimated_final_cash": estimated_final_cash,
        "total_sell_amount": total_sell_amount,
        "total_buy_amount": total_buy_amount
    }

def generate_dendrogram_plot(corr_matrix: pd.DataFrame, threshold: float = 1.0, output_path: str = "dendrogram.png"):
    """
    상관관계 거리를 바탕으로 계층 트리(덴드로그램)를 시각화하여 이미지 파일로 저장합니다.
    """
    import matplotlib
    matplotlib.use('Agg') # 비대화형 백엔드 사용 (GUI 창 팝업 방지)
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    
    # 1. 상관계수를 거리(Distance)로 변환: d = sqrt(2 * (1 - rho))
    dist_matrix = np.sqrt(2 * (1 - corr_matrix))
    condensed_dist = squareform(dist_matrix, checks=False)
    
    # 2. 계층적 군집화 (Ward's Method)
    Z = linkage(condensed_dist, method='ward')
    
    # 3. 덴드로그램 그리기
    plt.figure(figsize=(10, 6))
    
    # 한국어 폰트 설정 (PDF와 일치시키기 위해 맑은 고딕 사용 권장)
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass
        
    dendrogram(
        Z, 
        labels=corr_matrix.index, 
        leaf_rotation=45, 
        leaf_font_size=10
    )
    
    plt.title("Portfolio Correlation Hierarchical Tree (Dendrogram)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Stocks (Tickers)", fontsize=11, labelpad=10)
    plt.ylabel("Risk Distance", fontsize=11, labelpad=10)
    
    # 임계점 가로선 추가 (빨간색 점선)
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold (t={threshold})')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"      [Engine] 덴드로그램 시각화 이미지 저장 완료: {output_path}")

