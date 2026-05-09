import sqlite3
import pandas as pd
import numpy as np

def get_latest_portfolio_log(db_path="riskmanager.db"):
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM portfolio_logs ORDER BY timestamp DESC LIMIT 1", conn)
    return df.iloc[0] if not df.empty else None

def get_asset_logs(week_id, db_path="riskmanager.db"):
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f"SELECT * FROM asset_logs WHERE week_id = '{week_id}'", conn)
    return df

def perform_attribution_analysis(current_prices: pd.Series, db_path="riskmanager.db") -> pd.DataFrame:
    """
    과거(지난주) 예측치와 현재 실제 수익률을 비교하여 각 종목별 오차 기여도를 계산합니다.
    """
    portfolio_log = get_latest_portfolio_log(db_path)
    if portfolio_log is None:
        print("이전 포트폴리오 예측 기록이 없습니다.")
        return pd.DataFrame()
        
    week_id = portfolio_log['week_id']
    asset_logs = get_asset_logs(week_id, db_path)
    
    results = []
    
    for _, row in asset_logs.iterrows():
        ticker = row['ticker']
        target_weight = row['target_weight']
        expected_ret = row['expected_return'] 
        
        if ticker == "CASH":
            continue
            
        past_price = row['execution_price']
        curr_price = current_prices.get(ticker)
        
        if past_price and past_price > 0 and curr_price:
            actual_ret = (curr_price - past_price) / past_price
        else:
            actual_ret = 0.0
            
        contribution = target_weight * (actual_ret - expected_ret)
        
        results.append({
            'ticker': ticker,
            'target_weight': target_weight,
            'past_price': past_price,
            'current_price': curr_price,
            'expected_return': expected_ret,
            'actual_return': actual_ret,
            'error_contribution': contribution,
            'cluster_id': row['cluster_id']
        })
        
    df_result = pd.DataFrame(results)
    
    if not df_result.empty:
        total_abs_error = df_result['error_contribution'].abs().sum()
        if total_abs_error > 0:
            df_result['error_responsibility_pct'] = (df_result['error_contribution'].abs() / total_abs_error) * 100
        else:
            df_result['error_responsibility_pct'] = 0.0
            
    return df_result

