import numpy as np
import pandas as pd
from typing import Dict, Tuple

def run_portfolio_simulation(
    current_prices: pd.Series,
    annual_mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    weights: Dict[str, float],
    initial_portfolio_value: float = 10000.0,
    num_simulations: int = 10000,
    num_days: int = 252
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    기하 브라운 운동(GBM)과 촐레스키 분해를 이용한 포트폴리오 몬테카를로 시뮬레이션
    
    Args:
        current_prices: 현재 주가 (Series)
        annual_mean_returns: 연환산 기대 수익률 (Series)
        cov_matrix: 연환산 공분산 행렬 (DataFrame)
        weights: 종목별 투자 비중 (Dict, 합이 1이어야 함)
        initial_portfolio_value: 초기 투자 원금
        num_simulations: 시뮬레이션 반복 횟수 (기본 10,000회)
        num_days: 예측할 미래 거래일 수 (기본 252일, 1년)
        
    Returns:
        portfolio_value (np.ndarray): (시뮬레이션 횟수, 거래일 수) 크기의 포트폴리오 가치 궤적 행렬
        asset_paths (Dict): 각 종목별 주가 궤적 행렬 (분석용)
    """
    tickers = list(weights.keys())
    
    # 1. 입력 데이터 정렬 (티커 순서 일치)
    weights_arr = np.array([weights[t] for t in tickers])
    S0 = current_prices[tickers].values
    mu_annual = annual_mean_returns[tickers].values
    cov_annual = cov_matrix.loc[tickers, tickers].values
    
    # 2. 일일 파라미터로 변환
    mu_daily = mu_annual / 252
    cov_daily = cov_annual / 252
    vol_daily = np.sqrt(np.diag(cov_daily))
    
    # 3. 촐레스키 분해 (Cholesky Decomposition)
    # L * L^T = cov_daily 를 만족하는 하삼각행렬 L 생성
    L = np.linalg.cholesky(cov_daily)
    
    num_assets = len(tickers)
    
    # 4. 난수 생성 및 기하 브라운 운동 (Vectorized Calculation)
    # 표준 정규 난수 Z 생성 (시뮬레이션 횟수 x 날짜 x 종목 수)
    Z = np.random.normal(0, 1, size=(num_simulations, num_days, num_assets))
    
    # 상관관계가 반영된 난수: Z와 L의 전치행렬 곱
    corr_Z = Z @ L.T
    
    # 매일의 로그 수익률: Drift + Random Shock
    drift = mu_daily - 0.5 * (vol_daily ** 2)
    daily_log_returns = drift + corr_Z
    
    # 누적 수익률 계산 (누적합 후 지수 변환)
    cumulative_returns = np.exp(np.cumsum(daily_log_returns, axis=1))
    
    # 시뮬레이션된 가격 궤적 (초기 가격 * 누적 수익률)
    simulated_prices = S0 * cumulative_returns
    
    # 5. 포트폴리오 가치 집계 (Portfolio Aggregation)
    # 초기 자본을 가중치대로 분배하여 구매 가능한 주식 수(shares) 계산
    initial_capital_per_asset = initial_portfolio_value * weights_arr
    shares = initial_capital_per_asset / S0
    
    # 매일의 포트폴리오 가치 = sum(시뮬레이션 가격 * 주식 수)
    portfolio_value = np.sum(simulated_prices * shares, axis=2)
    
    # 종목별 시각화를 위해 사전 형태로 궤적 반환
    asset_paths = {t: simulated_prices[:, :, i] for i, t in enumerate(tickers)}
    
    return portfolio_value, asset_paths


def get_simulation_stats(portfolio_paths: np.ndarray, initial_value: float) -> dict:
    """
    시뮬레이션 결과 궤적에서 주요 리스크/수익 통계 지표를 추출합니다.
    """
    # 1년 뒤(마지막 날)의 포트폴리오 가치들
    final_values = portfolio_paths[:, -1]
    
    # 수익률 분포
    returns = (final_values - initial_value) / initial_value
    
    # 핵심 지표 산출
    expected_return = np.mean(returns) # 평균 기대 수익률
    median_return = np.median(returns) # 중간값 수익률
    
    # VaR (Value at Risk) - 95% 신뢰수준에서의 최대 손실
    var_95 = np.percentile(returns, 5)
    
    # CVaR (Conditional VaR) - 하위 5% 최악의 시나리오들의 평균 손실
    cvar_95 = np.mean(returns[returns <= var_95])
    
    # 승률 (원금을 잃지 않고 수익을 낼 확률)
    win_rate = np.mean(final_values > initial_value)
    
    # MDD (Maximum Drawdown, 최대 낙폭) 추정
    # 각 시나리오 궤적별로 고점 대비 최대 하락폭을 구한 뒤 그 평균을 냄
    running_max = np.maximum.accumulate(portfolio_paths, axis=1)
    # 고점이 현재 값과 같다면 0으로 처리, 방어 로직 추가
    running_max = np.where(running_max == 0, 1e-9, running_max) 
    drawdowns = (portfolio_paths - running_max) / running_max
    max_drawdowns = np.min(drawdowns, axis=1) # 궤적별 최악의 MDD (음수)
    expected_mdd = np.mean(max_drawdowns)
    
    return {
        "expected_return": expected_return,
        "median_return": median_return,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "win_rate": win_rate,
        "expected_mdd": expected_mdd,
        "final_values": {
            "mean": np.mean(final_values),
            "std": np.std(final_values),
            "worst_case": np.min(final_values),
            "best_case": np.max(final_values)
        }
    }
