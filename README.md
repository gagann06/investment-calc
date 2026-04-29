# Investment Return Calculator

A full-stack web application for analysing historical stock performance and calculating investment returns. Built with Python (Flask) on the backend and a finance-professional HTML/CSS frontend.

---

## Features

- **Real market data** fetches historical stock prices via [yfinance](https://github.com/ranaroussi/yfinance)
- **Two calculation modes**: simple return (lump sum) or monthly contribution (DCA) strategy
- **Risk metrics**: annualised volatility and maximum drawdown
- **Interactive chart**: portfolio growth visualisation generated with matplotlib
- **Ticker autocomplete**: dropdown search for stock symbols as you type
- **Input validation**: ticker verification, date ordering checks, and amount validation
- **Loading spinner**: visual feedback while fetching and processing data
- **3-page frontend**: home page, calculator form, and results page

---

## Project Structure

```
investment-calculator/
├── app.py                  # Flask application and routes
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not committed)
├── src/
│   ├── data.py             # Stock data fetching and ticker validation
│   ├── calculator.py       # Return, volatility, and drawdown calculations
│   ├── plot.py             # Matplotlib chart generation (CLI version)
│   ├── main.py             # CLI entry point
│   └── tickers.py          # Ticker autocomplete list
└── templates/
    ├── index.html          # Home page
    ├── calculator.html     # Input form
    └── results.html        # Results display
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation (for Windows)

1. **Clone the repository**

```bash
git clone https://github.com/your-username/investment-calculator.git
cd investment-calculator
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the project root:

```
SECRET_KEY=your-random-secret-key-here
```

Generate a secure key with:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

5. **Run the application**

```bash
python app.py
```

6. **If using VS Code, install the Live Server extension and press "Go Live" on the bottom right of the window**


7. **Open in your browser**

Visit [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Usage

### Web Application

1. Navigate to the home page and click **LAUNCH CALCULATOR**
2. Enter a stock ticker (e.g. `AAPL`, `MSFT`, `TSLA`) and use the autocomplete dropdown to search
3. Select a start and end date (start must be before end)
4. Enter your initial investment amount
5. Optionally enter a monthly contribution amount (enter `0` for none)
6. Click **CALCULATE RETURNS** and wait for results

### CLI Version

You can also run the calculator from the command line:

```bash
python src/main.py
```

---

## Calculations

### Simple Return (no monthly contributions)

Calculates the return on a one-time lump sum investment:

- **Shares bought** = Initial investment / Start price
- **Current value** = Shares x End price
- **Total return** = Current value - Initial investment
- **Return %** = (Total return / Initial investment) x 100

### Monthly Contributions (DCA)

Models a dollar-cost averaging strategy:

- Initial lump sum invested at the start price
- Additional shares purchased each month at that month's closing price
- Final value = Total shares x Final price

### Risk Metrics

| Metric | Description |
|--------|-------------|
| **Annualised Volatility** | Standard deviation of daily returns x sqrt(252) x 100 |
| **Max Drawdown** | Largest peak-to-trough percentage decline over the period |
| **CAGR** | Compound Annual Growth Rate: `((end / start) ^ (1 / years) - 1) x 100` |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `flask` | Web framework |
| `yfinance` | Stock data fetching |
| `pandas` | Data processing |
| `matplotlib` | Chart generation |
| `python-dotenv` | Environment variable loading |

Install all dependencies:

```bash
pip install flask yfinance pandas matplotlib python-dotenv
```

---

## Disclaimer

This tool is for **educational purposes only**. It does not constitute financial advice. Past performance is not indicative of future results. Do not use to inform future investments without prior consultation with a financial professional.
