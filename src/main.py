from datetime import datetime
from data import validate_ticker

from data import get_stock_data
from calculator import calc_simple_return, calc_with_monthly_contributions, calc_volatility, calc_max_drawdown
from plot import plot_investment_growth

def get_valid_ticker():
    while True:
        ticker = input("Enter stock ticker: ").upper()

        if validate_ticker(ticker):
            return ticker
        else:
            print("Invalid Ticker")

def get_valid_date(prompt):
    while True:
        date_str = input(prompt)

        try:
            datetime.strptime(date_str, '%Y-%m-%d')

            return date_str
            
        except ValueError:
            print("Invalid format. Use YYYY-MM-DD")

def get_valid_amount(prompt, allow_zero=False):
    while True:
        try:
            amount = float(input(prompt))
            if (allow_zero and amount >= 0) or (not allow_zero and amount > 0):
                return amount
            else:
                print("Amount must be positive. Please try again.")
        except ValueError:
            print("Not valid, please try again.")


def main():
    # Get inputs from user
    print("=" * 50 + "\nWelcome to XYZ Investment Return Calculator!\n" + "=" * 50)

    ticker = get_valid_ticker()

    # Get and validate date order
    while True:
        start_date = get_valid_date("Enter Start Date (YYYY-MM-DD): ")
        end_date = get_valid_date("Enter End Date (YYYY-MM-DD): ")
        
        if datetime.strptime(start_date, '%Y-%m-%d') < datetime.strptime(end_date, '%Y-%m-%d'):
            break
        else:
            print("Start date must be before end date. Please try again.\n")

    # Get investment values
    initial_investment = get_valid_amount("Enter Initial Investment Amount (£): ")
    monthly_contribution = get_valid_amount("Enter Monthly Contribution (£): ", allow_zero=True)

    while True:
        # Fetch stock data
        print(f"Fetching data for {ticker} ...")
        stock_data = get_stock_data(ticker, start_date, end_date)

        if stock_data is None:
            print(f"Could not fetch data for {ticker}. Please try another ticker.")
            ticker = get_valid_ticker()
        else:
            break



    start_price = stock_data['Close'].iloc[0]
    current_price = stock_data['Close'].iloc[-1]

    # Calculate returns
    if monthly_contribution > 0:
        results = calc_with_monthly_contributions(stock_data, initial_investment, monthly_contribution)
    else:
        results = calc_simple_return(initial_investment, start_price, current_price)

    # Display reults
    # Display results
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)

    if monthly_contribution > 0:
        print(f"Total Invested:       £{results['total_invested']:.2f}")
        print(f"Current Value:        £{results['current_value']:.2f}")
        print(f"Total Return:         £{results['total_return']:.2f}")
        print(f"Return Percentage:    {results['return_percent']:.2f}%")
    else:
        print(f"Shares Bought:        {results['shares']:.4f}")
        print(f"Current Value:        £{results['current_value']:.2f}")
        print(f"Total Return:         £{results['total_return']:.2f}")
        print(f"Return Percentage:    {results['return_percentage']:.2f}%")

    volatility = calc_volatility(stock_data)
    max_drawdown = calc_max_drawdown(stock_data)
    print(f"Annualised Volatility: {volatility:.2f}%")
    print(f"Max Drawdown:          {max_drawdown:.2f}%")
    print("="*50 + "\n")


    # Generate graphs
    chart_choice = input("Generate chart? (y/n): ")
    if chart_choice.lower() == 'y':
        plot_investment_growth(stock_data, initial_investment, save_path=None)

if __name__ == '__main__':
    main()