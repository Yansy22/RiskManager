import pandas as pd
from data_provider import get_monte_carlo_params
from simulation import run_portfolio_simulation, get_simulation_stats

def main():
    tickers = ["AAPL", "NVDA", "MSFT"]
    start_date = "2023-01-01"
    end_date = "2024-04-28"
    
    print(f"Fetching data for {tickers}...")
    prices, mean_ret, vol, cov = get_monte_carlo_params(tickers, start_date, end_date)
    
    if prices is None or prices.empty:
        print("Failed to get data.")
        return

    # Extract current prices (last row of prices DataFrame)
    current_prices = prices.iloc[-1]
    
    # Portfolio Weights (Example: 40% AAPL, 40% MSFT, 20% NVDA)
    weights = {"AAPL": 0.4, "MSFT": 0.4, "NVDA": 0.2}
    
    print("\n--- Running Monte Carlo Simulation ---")
    portfolio_paths, asset_paths = run_portfolio_simulation(
        current_prices=current_prices,
        annual_mean_returns=mean_ret,
        cov_matrix=cov,
        weights=weights,
        initial_portfolio_value=10000.0,
        num_simulations=10000,
        num_days=252
    )
    
    print("Simulation completed. Extracting statistics...")
    stats = get_simulation_stats(portfolio_paths, initial_value=10000.0)
    
    print("\n[Simulation Results (1 Year Horizon)]")
    print(f"Expected Return: {stats['expected_return'] * 100:.2f}%")
    print(f"Median Return:   {stats['median_return'] * 100:.2f}%")
    print(f"Win Rate:        {stats['win_rate'] * 100:.2f}%")
    print(f"Value at Risk (95%): {stats['var_95'] * 100:.2f}% (최악 5%의 기준선 손실률)")
    print(f"Conditional VaR (95%): {stats['cvar_95'] * 100:.2f}% (최악 5%의 평균 손실률)")
    print(f"Expected MDD:    {stats['expected_mdd'] * 100:.2f}%")
    
    print("\n[Final Value ($10,000 Initial)]")
    print(f"Mean:  ${stats['final_values']['mean']:,.2f}")
    print(f"Worst: ${stats['final_values']['worst_case']:,.2f}")
    print(f"Best:  ${stats['final_values']['best_case']:,.2f}")

if __name__ == "__main__":
    main()
