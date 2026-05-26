import numpy as np
import pandas as pd
from data_provider import get_monte_carlo_params
from engine import calculate_risk_clusters, optimize_portfolio

def main():
    # Use 5 stocks across different sectors to create distinct clusters:
    # Tech: AAPL, MSFT, NVDA
    # Financials / Healthcare: JPM, JNJ
    tickers = ["AAPL", "MSFT", "NVDA", "JPM", "JNJ"]
    
    # 2 years historical window
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    start_date = (pd.Timestamp.today() - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    
    print(f"[Test] Fetching historical parameters for tickers: {tickers}")
    prices, mean_returns, volatilities, cov_matrix = get_monte_carlo_params(tickers, start_date, end_date)
    
    if prices is None:
        print("[Error] Failed to fetch data.")
        return
        
    returns = np.log(prices / prices.shift(1)).dropna()
    
    print("\n[Test] Running Risk Clustering (threshold = 1.0)...")
    # Clustering will group AAPL, MSFT, NVDA together, and JPM, JNJ together
    clusters = calculate_risk_clusters(returns, threshold=1.0)
    
    print("\n[Test] Running Volatility-Integrated Optimization...")
    optimal_weights = optimize_portfolio(
        expected_returns=mean_returns,
        cov_matrix=cov_matrix,
        clusters=clusters,
        cluster_limit=0.3, # Base Limit
        risk_free_rate=0.035
    )
    
    # --- VERIFICATION ---
    print("\n" + "="*50)
    print("VERIFICATION OF MATHEMATICAL CONSTRAINTS")
    print("="*50)
    
    # 1. Budget Constraint: Sum of weights == 1.0
    sum_weights = sum(optimal_weights.values())
    print(f"1. Sum of Portfolio Weights: {sum_weights:.6f} (Target: 1.000000)")
    assert abs(sum_weights - 1.0) < 1e-4, f"Budget constraint failed! Sum of weights: {sum_weights}"
    print("   [V] Budget Constraint Met!")
    
    # 2. Dynamic Cluster Limits & Inverse Volatility constraints
    print("\n2. Checking Cluster-Specific Constraints:")
    for cid, members in clusters.items():
        print(f"\n   --- Cluster {cid} (Members: {members}) ---")
        
        # Calculate dynamic limit for verification
        cluster_vols = volatilities[members]
        cluster_avg_vol = cluster_vols.mean()
        market_avg_vol = volatilities.mean()
        dynamic_limit = 0.3 * (market_avg_vol / cluster_avg_vol)
        dynamic_limit = np.clip(dynamic_limit, 0.10, 0.50)
        
        # Actual weights sum in this cluster
        actual_sum = sum(optimal_weights[m] for m in members)
        print(f"      Target Dynamic Limit: {dynamic_limit*100:.2f}%")
        print(f"      Actual Cluster Sum:   {actual_sum*100:.2f}%")
        assert actual_sum <= dynamic_limit + 1e-4, f"Cluster {cid} limit breached! Limit: {dynamic_limit}, Actual: {actual_sum}"
        print("      [V] Cluster Limit Constraint Met!")
        
        # Inverse Volatility Constraint check: w_i * vol_i == w_j * vol_j (equal risk contribution)
        if len(members) > 1:
            print("      Checking Intra-cluster Inverse-Volatility ratios:")
            risk_contributions = []
            for m in members:
                w = optimal_weights[m]
                vol = volatilities[m]
                risk_contribution = w * vol
                risk_contributions.append((m, w, vol, risk_contribution))
                print(f"         - {m}: Weight = {w*100:.3f}%, Volatility = {vol*100:.1f}%, Risk Product (w * vol) = {risk_contribution:.6f}")
            
            # Check deviation between the first active element and others
            ref_rc = risk_contributions[0][3]
            for m, w, vol, rc in risk_contributions[1:]:
                # If weights are extremely close to zero, it means the whole cluster weight is zero or near-zero, which is also fine.
                rc_diff = abs(rc - ref_rc)
                print(f"         - Difference ({m} vs {risk_contributions[0][0]}): {rc_diff:.8f}")
                assert rc_diff < 1e-4, f"Inverse Volatility ratio breached for {m}! Diff: {rc_diff}"
            print("      [V] Inverse-Volatility Equal Risk Constraint Met!")
            
    print("\n" + "="*50)
    print("ALL TESTS PASSED SUCCESSFULLY! MATHEMATICAL INTEGRATION IS 100% CORRECT!")
    print("="*50)

if __name__ == "__main__":
    main()
