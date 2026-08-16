"""Return and risk metrics.

Every function takes a *time-weighted* daily return series rather than a
portfolio value, so deposits are never mistaken for performance.

The trap throughout is annualisation, because the two quantities scale to a year
differently. Volatility grows with the square root of time - variance adds over
independent periods, so its square root grows with sqrt(t) - while returns
compound, so they are raised to a power. Mixing the two produces a number that
is wrong by a factor of sixteen and looks entirely plausible.

Every scalar returned here is a plain float. Numpy scalars cannot be serialised
to JSON, and pinning the type at the source is cheaper than discovering it as a
blank page in the browser.
"""

import numpy as np

TRADING_DAYS = 252


def annualised_volatility(returns):
    """Standard deviation of daily returns, scaled to a year, in percent.

    A measure of uncertainty, not of loss: an asset that reliably rises 2% a day
    has high volatility and no risk of the kind an investor minds.
    """
    if len(returns) < 2:
        return 0.0

    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100)


def annualised_return(returns):
    """The constant yearly rate that would have produced the same total growth.

    Compounds the series into total growth, then takes the root of however many
    years it spans - so 21% over two years annualises to 10%, not 10.5%.
    """
    total_growth = (1 + returns).prod()
    years = len(returns) / TRADING_DAYS

    if  years <= 0:
        return 0.0

    if total_growth <= 0:
        return -100.0
    
    return float((total_growth ** (1 / years) - 1) * 100)


def sharpe_ratio(returns, risk_free_rate=0.0):
    """Excess return per unit of total volatility, annualised.

    `risk_free_rate` is an annual decimal (0.04 for 4%) and is compounded down
    to a daily rate, not divided by 252. It is subtracted because a portfolio
    returning 5% while cash pays 4% has earned almost nothing for the risk taken
    - what is being measured is the reward for bearing risk at all.
    """
    if len(returns) < 2:
        return 0.0

    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    excess = returns - daily_rf
    deviation = excess.std(ddof=1)

    # Compared against a tolerance rather than zero: the deviation of a series
    # of identical floats comes out around 2e-19, not 0.0, and dividing by that
    # gives a Sharpe ratio of 10^15 instead of the intended zero.
    if deviation < 1e-12:
        return 0.0

    return float(excess.mean() / deviation * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns, risk_free_rate=0.0):
    """Excess return per unit of *downside* volatility, annualised.

    Sharpe punishes a portfolio that occasionally leaps upwards, which nobody
    actually minds. Only the negative excess returns go into the denominator
    here; the numerator stays the mean over every day.
    """
    if len(returns) < 2:
        return 0.0

    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    excess = returns - daily_rf
    downside = excess[excess < 0]

    # Never fell: there is no downside risk to divide by.
    if len(downside) < 2:
        return 0.0

    deviation = np.sqrt((downside**2).mean())

    if deviation < 1e-12:
        return 0.0

    return float(excess.mean() / deviation * np.sqrt(TRADING_DAYS))


def drawdown_series(growth):
    """Percent below the running peak at every point - the "underwater" curve.

    Measured against `cummax`, the high-water mark, rather than against the
    starting value: a portfolio up 50% and then down 25% from its high is in a
    25% drawdown, however far above its start it still sits.
    """
    peak = growth.cummax()
    return (growth / peak - 1) * 100


def max_drawdown(growth):
    """Worst peak-to-trough fall, as (depth, peak, trough, recovery).

    `recovery` is None if the prior peak was never regained. Arguably the most
    honest risk number available: volatility is symmetric and abstract, while
    this is the fall an investor actually had to sit through, and the date they
    got back to level.

    The trough is found first and the peak looked for behind it - done the other
    way round, the highest point could be paired with a trough that preceded it.
    """
    if len(growth) < 2:
        return (0, None, None, None)

    drawdown = drawdown_series(growth)
    trough = drawdown.idxmin()
    depth = float(drawdown.loc[trough])

    if depth >= 0:
        return (0, None, None, None) # never fell

    peak = growth.loc[:trough].idxmax()
    peak_value = growth.loc[peak]

    after = growth.loc[trough:]
    recovered = after[after >= peak_value]

    if len(recovered) > 0:
        recovery = recovered.index[0]
    else:
        recovery = None
    
    return (depth, peak, trough, recovery)                    


def calmar_ratio(annual_return_pct, max_drawdown_pct):
    """Annualised return divided by the depth of the worst fall.

    Absolute value, because drawdown is reported negative and the ratio should
    come out positive for a profitable portfolio.
    """
    if max_drawdown_pct == 0:
        return 0.0
    return float(annual_return_pct / abs(max_drawdown_pct))


def xirr(cashflows, final_value, final_date):
    """Money-weighted annual return, in percent.

    Where a time-weighted return measures the investment, this measures the
    *investor*: it accounts for when money was actually committed, so a good
    fund bought mostly near the top still reports a poor personal return.

    It is the rate at which the discounted value of every cashflow sums to zero.
    There is no closed form, so it is solved numerically. `cashflows` maps date
    to amount paid in, as positive numbers.
    """
    flows = [(date, -amount) for date, amount in cashflows.items()] + [(final_date, +final_value)]

    if len(flows) < 2:
        return 0.0

    t0 = min(date for date, amount in flows)
    amounts = [amount for date, amount in flows]
    years = [(date - t0).days / 365.0 for date, amount in flows]

    def npv(r):
        return sum (a / (1 + r) ** t for (a, t) in zip(amounts, years))

    # Bisection rather than Newton-Raphson. Newton converges faster but can
    # diverge, and the cashflows come from whatever a stranger typed into a
    # form. Halving a bracketed interval cannot fail to converge.
    low = -0.9999    # not -1: (1 + r) ** t divides by zero there
    high = 10.0

    # Bisection needs the root bracketed. Same sign at both ends means there is
    # nothing between them to find, so bail rather than loop pointlessly.
    if np.sign(npv(low)) == np.sign(npv(high)):
        return 0.0

    # Bounded, not `while abs(value) > tol`: a loop that can only exit on
    # convergence can hang forever on degenerate input, and this runs inside a
    # web request. 60 passes exhausts double precision over this bracket; 200 is
    # a margin that costs microseconds.
    for i in range(0, 200):
        mid = (low + high) / 2
        value = npv(mid)

        if abs(value) < 1e-9:
            break

        if npv(low) * value < 0:
            high  = mid
        else:
            low = mid
    
    return float(mid * 100)