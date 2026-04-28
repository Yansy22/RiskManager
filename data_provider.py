import os
import numpy as np
import pandas as pd
import yfinance as yf
from stockstats import wrap
from typing import Annotated, Dict, Optional, List, Tuple
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Internal utility from stockstats_utils.py (Removed because it's not in this project)
# from .stockstats_utils import StockstatsUtils


# 특정 기간 동안의 야후 파이낸스(Yahoo Finance) 주식 데이터를 온라인에서 가져오는 함수
def get_YFin_data_online(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
):
    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")

        # Create ticker object
        ticker = yf.Ticker(symbol.upper())

        # Fetch historical data for the specified date range
        data = ticker.history(start=start_date, end=end_date)

        # Check if data is empty
        if data.empty:
            return (
                f"No data found for symbol '{symbol}' between {start_date} and {end_date}"
            )

        # Remove timezone info from index for cleaner output
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        # Round numerical values to 2 decimal places for cleaner display
        numeric_columns = ["Open", "High", "Low", "Close", "Adj Close"]
        for col in numeric_columns:
            if col in data.columns:
                data[col] = data[col].round(2)

        # Convert DataFrame to CSV string
        csv_string = data.to_csv()

        # Add header information
        header = f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
        header += f"# Total records: {len(data)}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        return header + csv_string
    except Exception as e:
        return f"Error fetching stock data: {str(e)}"

# 조회 기간 동안의 특정 기술적 지표 값을 가져와서 보고서 형식으로 반환하는 함수
def get_stock_stats_indicators_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:

    best_ind_params = {
        "close_50_sma": "50 SMA: A medium-term trend indicator.",
        "close_200_sma": "200 SMA: A long-term trend benchmark.",
        "close_10_ema": "10 EMA: A responsive short-term average.",
        "macd": "MACD: Momentum via differences of EMAs.",
        "macds": "MACD Signal: EMA smoothing of the MACD line.",
        "macdh": "MACD Histogram: Gap between MACD and signal.",
        "rsi": "RSI: Momentum overbought/oversold indicator.",
        "boll": "Bollinger Middle: 20 SMA.",
        "boll_ub": "Bollinger Upper Band.",
        "boll_lb": "Bollinger Lower Band.",
        "atr": "ATR: Measures volatility.",
        "vwma": "VWMA: Volume weighted average.",
        "mfi": "MFI: Money Flow Index.",
    }

    if indicator not in best_ind_params:
        return f"Indicator {indicator} is not supported. Supported: {list(best_ind_params.keys())}"

    end_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_date_dt = end_date_dt - relativedelta(days=look_back_days)

    try:
        indicator_data = _get_stock_stats_bulk(symbol, indicator, curr_date)
        
        current_dt = end_date_dt
        date_values = []
        
        while current_dt >= start_date_dt:
            date_str = current_dt.strftime('%Y-%m-%d')
            if date_str in indicator_data:
                indicator_value = indicator_data[date_str]
            else:
                indicator_value = "N/A"
            
            date_values.append((date_str, indicator_value))
            current_dt = current_dt - relativedelta(days=1)
        
        ind_string = "\n".join([f"{d}: {v}" for d, v in date_values])
        
    except Exception as e:
        print(f"Error getting bulk stockstats data: {e}")
        return f"Error calculating indicators: {str(e)}"

    result_str = (
        f"## {indicator} values from {start_date_dt.strftime('%Y-%m-%d')} to {curr_date}:\n\n"
        + ind_string
        + "\n\n"
        + best_ind_params.get(indicator, "")
    )

    return result_str


def _get_stock_stats_bulk(
    symbol: str,
    indicator: str,
    curr_date: str
) -> dict:
    # Local defaults
    data_cache_dir = os.getenv("DATA_CACHE_DIR", "data_cache")
    
    today_date = pd.Timestamp.today()
    end_date_str = today_date.strftime("%Y-%m-%d")
    start_date = today_date - pd.DateOffset(years=15)
    start_date_str = start_date.strftime("%Y-%m-%d")
    
    os.makedirs(data_cache_dir, exist_ok=True)
    
    data_file = os.path.join(
        data_cache_dir,
        f"{symbol}-YFin-bulk-{start_date_str}-{end_date_str}.csv",
    )
    
    if os.path.exists(data_file):
        data = pd.read_csv(data_file)
        data["Date"] = pd.to_datetime(data["Date"])
    else:
        data = yf.download(
            symbol,
            start=start_date_str,
            end=end_date_str,
            multi_level_index=False,
            progress=False,
            auto_adjust=True,
        )
        data = data.reset_index()
        data.to_csv(data_file, index=False)
    
    df = wrap(data)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    
    # Calculate the indicator
    _ = df[indicator]
    
    result_dict = {}
    for _, row in df.iterrows():
        date_str = row["Date"]
        indicator_value = row[indicator]
        result_dict[date_str] = str(indicator_value) if not pd.isna(indicator_value) else "N/A"
    
    return result_dict


def get_fundamentals(ticker: str, curr_date: str = None):
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        info = ticker_obj.info
        if not info: return f"No fundamentals found for {ticker}"
        
        fields = ["longName", "sector", "industry", "marketCap", "trailingPE", "forwardPE", "dividendYield"]
        result = [f"{f}: {info.get(f)}" for f in fields if info.get(f) is not None]
        return f"# Fundamentals for {ticker.upper()}\n" + "\n".join(result)
    except Exception as e:
        return f"Error: {str(e)}"

def get_balance_sheet(ticker: str, freq: str = "quarterly"):
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        data = ticker_obj.quarterly_balance_sheet if freq == "quarterly" else ticker_obj.balance_sheet
        return data.to_csv() if not data.empty else "No data"
    except Exception as e: return f"Error: {str(e)}"

def get_cashflow(ticker: str, freq: str = "quarterly"):
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        data = ticker_obj.quarterly_cashflow if freq == "quarterly" else ticker_obj.cashflow
        return data.to_csv() if not data.empty else "No data"
    except Exception as e: return f"Error: {str(e)}"

def get_income_statement(ticker: str, freq: str = "quarterly"):
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        data = ticker_obj.quarterly_income_stmt if freq == "quarterly" else ticker_obj.income_stmt
        return data.to_csv() if not data.empty else "No data"
    except Exception as e: return f"Error: {str(e)}"

def get_insider_transactions(ticker: str):
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        data = ticker_obj.insider_transactions
        return data.to_csv() if data is not None and not data.empty else "No data"
    except Exception as e: return f"Error: {str(e)}"

# --- New News & Performance Tools ---

def get_stock_news(ticker: str) -> str:
    """Fetch recent news for a specific stock."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        news = ticker_obj.news
        if not news:
            return f"No news found for {ticker}"
        
        formatted_news = []
        for item in news[:10]: # Limit to 10 latest
            content = item.get("content", {})
            title = content.get("title")
            provider = content.get("provider", {}).get("displayName")
            link = content.get("canonicalUrl", {}).get("url")
            
            if title:
                formatted_news.append(f"- Title: {title}\n  Publisher: {provider}\n  Link: {link}\n")
            
        if not formatted_news:
            return f"No valid news content found for {ticker}"

        return f"# Recent News for {ticker.upper()}\n" + "\n".join(formatted_news)
    except Exception as e:
        return f"Error fetching news: {str(e)}"

def get_macro_news() -> str:
    """Fetch recent market-wide (macro) news using S&P 500 index."""
    try:
        # Using S&P 500 news as a proxy for macro news
        ticker_obj = yf.Ticker("^GSPC")
        news = ticker_obj.news
        if not news:
            return "No macro news found."
        
        formatted_news = []
        for item in news[:10]:
            content = item.get("content", {})
            title = content.get("title")
            provider = content.get("provider", {}).get("displayName")
            link = content.get("canonicalUrl", {}).get("url")
            
            if title:
                formatted_news.append(f"- Title: {title}\n  Publisher: {provider}\n  Link: {link}\n")
            
        if not formatted_news:
            return "No valid macro news content found."

        return "# Market-wide (Macro) News\n" + "\n".join(formatted_news)
    except Exception as e:
        return f"Error fetching macro news: {str(e)}"

def get_price_performance(ticker: str, curr_date: str) -> str:
    """Compare current price with 1 week and 1 month ago."""
    try:
        end_date = datetime.strptime(curr_date, "%Y-%m-%d")
        start_date = end_date - relativedelta(months=2) # Fetch enough data
        
        ticker_obj = yf.Ticker(ticker.upper())
        df = ticker_obj.history(start=start_date.strftime("%Y-%m-%d"), end=(end_date + relativedelta(days=1)).strftime("%Y-%m-%d"))
        
        if df.empty:
            return f"No price data found for {ticker}"
        
        # Get target price (at curr_date or last available before it)
        target_df = df[:curr_date]
        if target_df.empty:
             return f"No price data available up to {curr_date}"
        
        curr_price = target_df.iloc[-1]["Close"]
        curr_date_actual = target_df.index[-1].strftime("%Y-%m-%d")
        
        # 1 week ago
        one_week_ago = end_date - relativedelta(weeks=1)
        df_1w = df[:one_week_ago.strftime("%Y-%m-%d")]
        price_1w = df_1w.iloc[-1]["Close"] if not df_1w.empty else None
        
        # 1 month ago
        one_month_ago = end_date - relativedelta(months=1)
        df_1m = df[:one_month_ago.strftime("%Y-%m-%d")]
        price_1m = df_1m.iloc[-1]["Close"] if not df_1m.empty else None
        
        report = f"# Price Performance for {ticker.upper()} (as of {curr_date_actual})\n"
        report += f"- Current Price: ${curr_price:.2f}\n"
        
        if price_1w:
            diff_1w = ((curr_price - price_1w) / price_1w) * 100
            report += f"- 1 Week Ago: ${price_1w:.2f} ({diff_1w:+.2f}%)\n"
        else:
            report += "- 1 Week Ago: Data not available\n"
            
        if price_1m:
            diff_1m = ((curr_price - price_1m) / price_1m) * 100
            report += f"- 1 Month Ago: ${price_1m:.2f} ({diff_1m:+.2f}%)\n"
        else:
            report += "- 1 Month Ago: Data not available\n"
            
        return report
    except Exception as e:
        return f"Error calculating price performance: {str(e)}"

# --- New Monte Carlo Simulation Tools ---

def get_monte_carlo_params(
    tickers: List[str],
    start_date: str,
    end_date: str
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series], Optional[pd.DataFrame]]:
    """
    몬테카를로 시뮬레이션에 필요한 통계 파라미터를 계산합니다.
    
    Returns:
    - prices (DataFrame): 정제된 수정 종가 데이터
    - annual_mean_returns (Series): 연환산 기대 수익률 (Drift)
    - annual_volatility (Series): 연환산 표준편차 (Volatility)
    - cov_matrix (DataFrame): 연환산 공분산 행렬 (Covariance Matrix)
    """
    try:
        # 1. 과거 가격 데이터 다운로드
        print(f"      [Data] yfinance에서 {tickers} 데이터 다운로드 중... (기간: {start_date} ~ {end_date})")
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            raise ValueError(f"No data found for tickers {tickers}")

        # yfinance 반환 객체 구조 처리 (다중 티커 vs 단일 티커)
        if len(tickers) > 1 and isinstance(data.columns, pd.MultiIndex):
            prices = data['Close']
        else:
            prices = pd.DataFrame({tickers[0]: data['Close']})
            
        # 2. 결측치(NaN) 처리
        # 상장일이 다르거나 휴장일이 겹치는 경우를 위해 앞선 가격으로 채우고 남은 결측치 제거
        prices = prices.ffill().dropna()
        
        if prices.empty:
            raise ValueError("Data became empty after handling missing values.")

        # 3. 일별 로그 수익률 계산
        returns = np.log(prices / prices.shift(1)).dropna()
        
        # 4. 연환산 통계량 계산 (1년 = 252 거래일)
        annual_mean_returns = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        cov_matrix = returns.cov() * 252
        
        return prices, annual_mean_returns, annual_volatility, cov_matrix

    except Exception as e:
        print(f"Error calculating Monte Carlo parameters: {e}")
        return None, None, None, None