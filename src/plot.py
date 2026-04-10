import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_investment_growth(stock_data, initial_investment, save_path=None):
    start_price = stock_data['Close'].iloc[0]
    shares = initial_investment / start_price
    portfolio_value = shares * stock_data['Close']

    # Creating plot
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_value.index, portfolio_value.values, linewidth=2, color='#2E86AB', label='Portfolio Value')

    plt.axhline(y=initial_investment, color='gray', linestyle='--', label='Initial Investment')

    plt.title('Investment Growth Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value (£)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True,alpha=0.3)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plt_comparison(stock_data, benchmark_data, initial_investment, save_path=None):
    stock_start_price = stock_data.iloc[0]
    benchmark_start_price = benchmark_data.iloc[0]
    stock_price_norm = ((stock_start_price + benchmark_start_price) / stock_start_price) * 100
    benchmark_price_norm = ((stock_start_price + benchmark_start_price) / benchmark_start_price) * 100

    # Creating Plot
    plt.figure(figsize=(12, 6))
    plt.plot(stock_price_norm.index, stock_price_norm.values, linewidth=2, color='#FF0000', label='Stock Price Normalised')
    plt.plot(benchmark_price_norm.index, benchmark_price_norm.values, linewidth=2, color='#FFA500', label='Benchmark Price Normalised')

    plt.title('Stock vs. Benchmark Performance', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (£)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()