import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
from data_provider import get_monte_carlo_params

# 벤치마크용 10개 대표 종목 (빅테크, 금융, 헬스케어, 에너지, 필수소비재, 금, 채권)
BENCHMARK_TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "JNJ", "XOM", "PG", "TSLA", "GLD", "TLT"]

def run_benchmark_lab():
    print(f"\n[실험 1] 표준 10개 종목 벤치마크 분석 중...")
    lookback_years = 3
    start_date = (pd.Timestamp.today() - pd.DateOffset(years=lookback_years)).strftime("%Y-%m-%d")
    
    # 데이터 가져오기
    prices, _, _, _ = get_monte_carlo_params(BENCHMARK_TICKERS, start_date, pd.Timestamp.today().strftime("%Y-%m-%d"))
    returns = np.log(prices / prices.shift(1)).dropna()
    
    # 1. 상관계수 행렬 계산 및 시각화
    corr = returns.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
    plt.title(f"Standard Market Correlation Matrix (Last {lookback_years} Years)")
    plt.savefig("benchmark_correlation_heatmap.png")
    print("      [V] 상관계수 히트맵 저장 완료 (benchmark_correlation_heatmap.png)")
    
    # 2. 덴드로그램 시각화
    dist_matrix = np.sqrt(2 * (1 - corr))
    condensed_dist = squareform(dist_matrix, checks=False)
    Z = linkage(condensed_dist, method='ward')
    
    plt.figure(figsize=(12, 8))
    dendrogram(Z, labels=corr.index, leaf_rotation=90)
    plt.title("Standard Market Hierarchical Clustering Dendrogram")
    plt.axhline(y=1.0, color='r', linestyle='--', label='Ref Threshold (1.0)')
    plt.legend()
    plt.savefig("benchmark_dendrogram.png")
    print("      [V] 벤치마크 덴드로그램 저장 완료 (benchmark_dendrogram.png)")
    
    print("\n[벤치마크 인사이트]")
    print("- 기술주(AAPL, MSFT, NVDA) 간의 상관계수는 보통 0.7 이상으로 매우 높게 나타납니다.")
    print("- 반면, 주식과 금(GLD) 또는 채권(TLT)은 상관계수가 낮거나 음수로 나타나 분산 효과가 큼을 알 수 있습니다.")

def run_user_portfolio_lab():
    # 기존 portfolio.json 분석 로직 (중략 및 유지)
    with open("portfolio.json", "r") as f:
        user_data = json.load(f)
    
    holdings = user_data.get("holdings", {})
    watchlist = user_data.get("watchlist", {})
    settings = user_data.get("settings", {})
    sector_limit = settings.get("sector_limit", 0.3)
    lookback_years = settings.get("lookback_years", 3)
    
    tickers = list(holdings.keys()) + list(watchlist.keys())
    if len(tickers) < 3:
        print("\n[알림] 사용자 포트폴리오 종목이 부족하여 벤치마크 실험만 진행합니다.")
        return

    print(f"\n[실험 2] 사용자 포트폴리오({tickers}) 정밀 분석 중...")
    start_date = (pd.Timestamp.today() - pd.DateOffset(years=lookback_years)).strftime("%Y-%m-%d")
    prices, _, _, _ = get_monte_carlo_params(tickers, start_date, pd.Timestamp.today().strftime("%Y-%m-%d"))
    returns = np.log(prices / prices.shift(1)).dropna()
    
    corr = returns.corr()
    dist_matrix = np.sqrt(2 * (1 - corr))
    condensed_dist = squareform(dist_matrix, checks=False)
    Z = linkage(condensed_dist, method='ward')
    
    # 최적 임계점 스캔 (기존 로직)
    min_required_k = int(np.ceil(1.0 / sector_limit))
    best_t = 1.0
    max_sil = -1.0
    
    print(f"\n{'Threshold (t)':<15} | {'Clusters (K)':<12} | {'Silhouette':<12}")
    print("-" * 45)
    for t in np.arange(0.1, 1.6, 0.1):
        labels = fcluster(Z, t, criterion='distance')
        num_clusters = len(np.unique(labels))
        sil = silhouette_score(dist_matrix, labels, metric='precomputed') if 1 < num_clusters < len(tickers) else 0.0
        print(f"{t:<15.1f} | {num_clusters:<12} | {sil:<12.3f}")
        if num_clusters >= min_required_k and sil > max_sil:
            max_sil = sil
            best_t = t
    print("-" * 45)
    print(f"👉 사용자 포트폴리오 추천 임계점(t): {best_t:.1f}")

if __name__ == "__main__":
    # seaborn 스타일 설정
    sns.set_theme(style="whitegrid")
    run_benchmark_lab()
    run_user_portfolio_lab()
