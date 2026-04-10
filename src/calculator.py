import pandas as pd
import math

def calc_simple_return(initial_investment, start_price, current_price):
    shares = (initial_investment / start_price).item()
    current_value = shares * current_price
    total_return = current_value - initial_investment
    return_percentage = (total_return / initial_investment) * 100

    return {
        'shares' : shares,
        'current_value' : current_value,
        'total_return' : total_return,
        'return_percentage' : return_percentage
    }

def calc_annualised_return(start_value, end_value, years):
    if years <= 0:
        return 0
    
    # CAGR = Compound Annual Growth Rate
    cagr = ((end_value / start_value) ** (1 / years) - 1) * 100

    return cagr

def calc_with_monthly_contributions(stock_data, initial_investment, monthly_contribution):
   
    # Get monthly resampled data
    monthly_prices = stock_data['Close'].squeeze().resample('ME').last().astype(float)
    
    total_shares = initial_investment / monthly_prices.iloc[0]
    total_invested = initial_investment

    # Skip first month (already invested)
    for price in monthly_prices.iloc[1:]:
        shares_bought = monthly_contribution / float(price)
        total_shares += shares_bought
        total_invested += monthly_contribution

    current_value = total_shares * monthly_prices.iloc[-1]
    total_return = current_value - total_invested
    return_percent = (total_return / total_invested) * 100

    return {
        'total_shares' : total_shares,
        'total_invested' : total_invested,
        'current_value' : current_value,
        'total_return' : total_return,
        'return_percent' : return_percent
    }

def calc_volatility(stock_data):
    daily_returns = stock_data['Close'].pct_change()
    ann_volatility = daily_returns.std() * math.sqrt(252) * 100
    
    return ann_volatility # Volatility measures how much the stock price fluctuates


def calc_max_drawdown(stock_data):
    cumul_returns = (1 + stock_data['Close'].pct_change()).cumprod()
    running_max = cumul_returns.cummax()
    drawdown = (cumul_returns - running_max) / running_max
    min_drawdown = drawdown.min() * 100

    return min_drawdown # Drawdown shoes the largest drop from a peak value