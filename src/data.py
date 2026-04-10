import yfinance as yf
import pandas as pd

def get_stock_data(ticker, start_date, end_date):
    # ticker = Stock Symbol (e.g. APPL)
    # start_date = Start date (YYYY-MM-DD)
    # end_date = End date (YYYY-MM-DD)
    try:
        stock = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if stock.empty:
            raise ValueError(f"No data found for {ticker}")

        if isinstance(stock.columns, pd.MultiIndex):
            stock.columns = stock.columns.get_level_values(0)

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

    return stock

def validate_ticker(ticker):
    try:
        data = yf.Ticker(ticker)
        info = data.info
        return info.get("symbol") is not None
    except Exception:
        return False
