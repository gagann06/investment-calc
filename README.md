# Portfolio Simulator

A historical backtester for multi-asset portfolios. You give it holdings and
weights, a contribution schedule and a rebalancing rule; it tells you what that
would have been worth, how much of the result was skill rather than market
exposure, and how bad the worst stretch was.

The core question it exists to answer honestly is one most retail calculators
get wrong: **paying money into a portfolio is not the same as the portfolio
performing well.** A calculator that reports "your investment grew 300%" when
most of that came from your own deposits is measuring your bank transfers.

```
157 tests
multi-asset, weighted, with calendar rebalancing
time-weighted and money-weighted returns reported side by side
benchmarked against identical cashflows into an index
```

---

## Quick start

Requires Python 3.10 or newer, and `git`. On macOS use `python3` and `pip3` if
`python` points at the system Python 2.

Clone the repository and move into it:

```bash
git clone https://github.com/gagann06/investment-calc.git
cd investment-calc
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Run the tests:

```bash
pytest
```

Start the app at <http://localhost:5000>:

```bash
portfolio-simulator
```

`pip install -e .` puts that command on your path. `--host`, `--port` and
`--no-debug` are available, and `python -m portfolio.api` does the same thing if
you would rather not install the package.

The whole suite runs offline in about two seconds. Only `portfolio/data.py`
touches the network, and the tests replace it.

---

## What it reports

Enter any number of tickers with weights, a date range, an opening investment
and an optional monthly contribution. Choose whether to rebalance, and what to
benchmark against.

**Returns.** Time-weighted (the investment's performance), money-weighted
(yours, given when you actually committed money), and both as annualised rates.

**Risk.** Annualised volatility, Sharpe, Sortino, maximum drawdown with the
dates it began, bottomed and recovered, and Calmar.

**Against a benchmark.** The same cashflows, on the same days, invested in an
index instead — plus beta and Jensen's alpha, so you can see how much of the
result was market exposure rather than selection.

**Composition.** Target versus final weights, per-holding drift, and a
correlation matrix of daily returns.

A real run — £10,000 opening, £250 a month, 40/35/25 across AAPL, MSFT and JNJ,
rebalanced quarterly, 2019 to 2024:

| | |
|---|---:|
| Paid in | £27,750 |
| Final value | £74,223 |
| Time-weighted return | 317.05% |
| Annualised | 26.93% |
| **Money-weighted** | **24.61%** |
| Volatility | 23.30% |
| Sharpe / Sortino | 0.97 / 0.96 |
| Max drawdown | −28.38% |
| Beta / alpha vs S&P 500 | 1.01 / +11.40% |
| Same cashflows into the S&P 500 | £50,754 |

The drawdown is dated 10 February to 23 March 2020, recovered 3 June 2020 —
which is the COVID crash, to the day.

Note the gap between the annualised 26.93% and the money-weighted 24.61%. The
investment compounded at 26.93%; the investor earned 24.61%, because each
monthly contribution had less time to work than the money before it. Only one of
those numbers is what actually happened to this person.

---

## API

| method | path | |
|---|---|---|
| `GET` | `/` | the page |
| `POST` | `/api/simulate` | run a simulation, returns the full bundle as JSON |
| `GET` | `/health` | liveness |

`/api/simulate` takes a JSON body:

```json
{
  "tickers": ["AAPL", "MSFT", "JNJ"],
  "weights": [40, 35, 25],
  "start_date": "2019-01-01",
  "end_date": "2024-12-31",
  "initial_investment": 10000,
  "monthly_contribution": 250,
  "rebalance": "quarterly",
  "benchmark": "^GSPC",
  "risk_free_rate": 4
}
```

Weights may be given in any scale — percentages, fractions or raw ratios — and
are normalised. `rebalance` is one of `none`, `monthly`, `quarterly`, `annual`.
Omit `weights` entirely to split evenly.

The response carries `summary`, `benchmark`, `allocation`, `series`,
`correlation` and `annual_returns`. Anything the request layer rejects comes
back as a `400` with a readable `error`, rather than a 500 from inside pandas.

---

## Design notes

The decisions worth explaining, and why they were made that way.

### Prices are adjusted, always

`yf.download` returns raw closes unless told otherwise, and raw closes silently
drop every dividend. Over a decade that is not a rounding error — it is most of
the gap between an index's price return and its total return. `auto_adjust=True`
is passed explicitly rather than relied on as a default, because that default
has changed between yfinance versions.

### A portfolio is share counts, not money

Value is derived as `shares × prices` and never tracked alongside. A running
total kept in parallel is a second source of truth, and on the day it disagrees
with the holdings there is no way to tell which is wrong. Deriving it means the
two cannot drift; a test asserts the identity across contributions and
rebalances.

It also makes contributions and rebalancing trivial. Both are operations on a
share vector, and everything downstream follows without special cases.

### The simulation loops over events, not days

Holdings only change on two kinds of day: a contribution, or a rebalance. Every
other day inherits yesterday's. So the loop runs over the event dates only,
writing a share vector into an otherwise-empty frame, and a forward-fill turns
that into a daily history in one operation.

For a six-year monthly-rebalanced portfolio that is 71 iterations rather than
1,509, and the cost scales with rebalance frequency instead of with the length
of history. The frame is initialised to `NaN` rather than zero deliberately: a
row that never gets written then fails loudly, instead of producing a portfolio
silently worth nothing.

### Rebalancing cannot create money

A rebalance values the entire holding at today's prices and rebuys it at target
weights. The same total goes back in, so the portfolio is worth the same
immediately before and after — it is a reallocation, not a deposit. A test
asserts the value series does not jump on any rebalance date, which is the
property a naive implementation breaks.

### Time-weighted return is the headline

The daily return of a portfolio's value is wrong on any day money arrived,
because the value jumped for a reason that was not performance. Each day's
cashflow is removed before comparing against the previous day, so the numerator
is "what yesterday's holdings are worth today".

It is subtracted from the numerator rather than added to the denominator because
in this model the contribution buys at that day's price — it is in the portfolio
but has not earned anything yet. The formula follows from the simulation rather
than being a convention adopted from elsewhere.

### Money-weighted return is reported next to it

Time-weighted return deliberately ignores when money arrived, which makes it the
right way to judge a fund and the wrong way to tell an investor what they
earned. So both are shown.

There is no closed form, so the rate is found by bisection on the net present
value of the dated cashflows. Newton–Raphson converges faster but can diverge,
and the input is whatever a stranger typed into a form. Halving a bracketed
interval cannot fail. The loop is bounded rather than running until convergence,
because a loop that can only exit on convergence can hang, and this runs inside
a web request — sixty passes exhausts double precision over the bracket used,
and it runs two hundred.

### The benchmark runs the same cashflows

Comparing a funded portfolio against an index's headline return compares two
different things. Instead the benchmark is a full simulation of the *same*
deposits on the *same* days into the index, so the comparison answers "what if
this money had bought the index instead".

Both are summarised by the same function, called twice. Written out separately,
a fix to one would quietly leave the comparison uneven.

### Forward-filling prices is right; forward-filling returns is not

Prices are forward-filled to bridge mismatched exchange holidays — a shut market
means the price did not change, which is true. Rows before every holding has
listed survive that fill and are trimmed, because a portfolio cannot hold an
asset that does not exist yet.

Return series are aligned by an inner join instead. Forward-filling a return
would invent a move that never happened, inflating the covariance and skewing
beta. Different meaning, different treatment.

### Correlation is between holdings, not within the portfolio

It is computed from each asset's own returns, before weighting blends them.
That answers the question actually worth asking: are these genuinely different
assets? Two holdings correlating at 0.95 offer roughly one asset's worth of
diversification while appearing on the page as two.

### JSON safety is pinned at the boundary

`jsonify` refuses numpy scalars and emits invalid JSON for `NaN`, and the
failure mode is a blank page with no error anywhere. So every scalar leaving the
analysis layer is a plain `float`, every date a string, and one helper coerces
anything that slips through. It is the last point at which the data is still
under our control.

### The engine never touches the network

`portfolio/data.py` is the only module that makes a request. Everything else
takes price frames and returns objects, which is why 157 tests run offline in
two seconds against hand-written prices with known answers — a doubling asset, a
flat one, a 50% crash with a dated recovery.

---

## Tests

```
tests/test_simulation.py   44   contributions, rebalancing, share accounting
tests/test_metrics.py      32   annualisation, Sharpe, Sortino, drawdown dates
tests/test_analysis.py     31   summaries, beta and alpha, benchmark bundling
tests/test_data.py         23   provider layouts, alignment, failure paths
tests/test_xirr.py         14   bisection, degenerate cashflows
tests/test_returns.py      13   the cashflow adjustment
```

The tests that matter most are the ones asserting a property rather than a
number. `test_contributions_into_flat_prices_still_produce_no_return` fails for
any implementation that treats deposits as gains.
`test_rebalancing_moves_money_without_creating_any` fails for any rebalance that
does not conserve value. `test_a_hand_computed_case` asserts the naive answer
(+300%) alongside the correct one (+100%), so the difference is documented in
the suite rather than only in prose.

The browser layer has **no automated tests**. `templates/index.html` was
verified by driving the live page.

---

## Layout

```
portfolio/
  data.py         adjusted price retrieval - the only module that does I/O
  simulation.py   share accounting, contributions, rebalancing, time-weighting
  metrics.py      volatility, Sharpe, Sortino, drawdown, Calmar, XIRR
  analysis.py     assembly into a reportable bundle, beta and alpha
  api.py          Flask app factory, request validation, serialisation
  templates/
    index.html    the single-page front end
tests/
```

---

## Limitations

- **No transaction costs.** Rebalancing is free and fills are at the close.
  A monthly-rebalanced portfolio would cost real money to run.
- **No tax.** Dividends are reinvested gross.
- **Fractional shares** are assumed throughout.
- **Contributions land on the first trading day of each month**, not on a date
  you choose.
- **Survivorship bias** is inherent to picking tickers that exist today.
- Yahoo Finance is not an authoritative source, and it has no SLA.

---

## Coming soon

- Transaction costs and a spread assumption, so rebalancing has a price
- Threshold rebalancing — trade when a weight drifts past a band, not on a date
- Monte Carlo: bootstrap the return series to stress a portfolio across paths
  it did not happen to take
- Efficient-frontier weights rather than user-supplied ones
- A CLI, for running a portfolio without the browser

---

## License

MIT. See [LICENSE](LICENSE).
