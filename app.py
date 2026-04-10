from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg') # Important (Non-GUI backend)
import matplotlib.pyplot as plt
import io
import base64
from src.data import get_stock_data, validate_ticker
from src.calculator import calc_simple_return, calc_with_monthly_contributions, calc_volatility, calc_max_drawdown


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/calculator')
def calculator():
    return render_template('calculator.html')


@app.route('/calculate', methods=['POST'])
def calculate():
    ticker = request.form.get('ticker').upper()
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    initial_investment = float(request.form.get('initial_investment'))
    monthly_contribution = float(request.form.get('monthly_contribution'))

    if not validate_ticker(ticker):
        return render_template('calculator.html', error="Invalid ticker symbol.")

    stock_data = get_stock_data(ticker, start_date, end_date)

    if stock_data is None:
        return render_template('calculator.html', error="Could not fetch stock data.")

    start_price = stock_data['Close'].iloc[0]
    current_price = stock_data['Close'].iloc[-1]

    if monthly_contribution > 0:
        results = calc_with_monthly_contributions(stock_data, initial_investment, monthly_contribution)
    else:
        results = calc_simple_return(initial_investment, start_price, current_price)

    # Convert to base64 for embedding in HTML
    chart = generate_chart(stock_data, initial_investment)

    volatility = calc_volatility(stock_data)
    max_drawdown = calc_max_drawdown(stock_data)

    return render_template('results.html', results=results, ticker=ticker, chart=chart, volatility=volatility, max_drawdown=max_drawdown)


def generate_chart(stock_data, initial_investment):
    start_price = stock_data['Close'].iloc[0]
    shares = initial_investment / start_price
    portfolio_value = shares * stock_data['Close']

    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_value.index, portfolio_value.values, linewidth=2, color='#C9A84C')
    plt.axhline(y=initial_investment, color='gray', linestyle='--', alpha=0.7)

    plt.title('Investment Growth over Time', fontsize=14, fontweight='bold', color='#0A1628')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (£)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Convert to base64 for embedding in HTML
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return f"data:image/png;base64,{image_base64}"


if __name__ == '__main__':
    app.run(debug=True)
