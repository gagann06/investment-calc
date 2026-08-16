"""Assembly of a finished simulation into a reportable bundle.

This module computes almost nothing of its own - `metrics` owns the arithmetic
and `simulation` owns the simulation. What lives here is the decision about
*what a user should be shown*, which is a product question rather than a
mathematical one. The payoff is that changing the page changes this file and
nothing behind it.

It is also the last point at which the data is under our control, so it is where
everything is made safe to serialise: plain floats, no NaN, dates as strings.
"""

import pandas as pd
import numpy as np
from portfolio.metrics import TRADING_DAYS, annualised_return, annualised_volatility, sharpe_ratio, sortino_ratio, drawdown_series, max_drawdown, calmar_ratio, xirr

def beta_alpha(returns, benchmark_returns, risk_free_rate=0):
    """Sensitivity to a benchmark, and the return left over once it is paid for.

    Beta is how hard the portfolio moves when the benchmark moves: 1.0 is
    lockstep, 1.5 amplifies it by half again. Alpha is what was earned beyond
    what that exposure alone would have produced - beta is leverage on the
    market, alpha is everything else.

    Returns (beta, alpha_pct). Alpha is a percentage; the annualised returns are
    converted to decimals first so they are in the same units as the risk-free
    rate, and converted back at the end.
    """
    # Inner join, because comparing days the two series do not share is
    # meaningless - a US index and a London listing disagree on several trading
    # days a year. Deliberately not forward-filled: filling a *price* means the
    # market was shut, but filling a *return* invents a move that never happened.
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    port  = aligned.iloc[:, 0]
    bench = aligned.iloc[:, 1]

    if len(aligned) < 3:
        return (0.0, 0.0)

    port_annual  = annualised_return(port) / 100
    bench_annual = annualised_return(bench) / 100

    if bench.var(ddof=1) < 1e-12:
        return (0.0, 0.0)
    
    beta = port.cov(bench) / bench.var(ddof=1)

    # The bracket is what the exposure alone should have earned: the risk-free
    # rate, plus beta's share of whatever the market paid above it.
    expected = risk_free_rate + beta * (bench_annual - risk_free_rate)
    alpha = (port_annual - expected) * 100

    return (float(beta), float(alpha))

def tracking_error(returns, benchmark_returns):
    """How far the portfolio typically drifts from the benchmark, per year.

    It is a volatility - of the *difference* between the two rather than of
    either one - so it annualises the same way. Zero means exact tracking.
    """
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    port  = aligned.iloc[:, 0]
    bench = aligned.iloc[:, 1]

    diff = port - bench
    te = float(diff.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100)

    return te

def format_date(value):
    """Timestamp to "YYYY-MM-DD", passing None through untouched."""
    return value.strftime("%Y-%m-%d") if value is not None else None

def summarise(result, risk_free_rate):
    """Every headline figure for one simulation.

    Called once for the portfolio and once for the benchmark, so that the two
    are measured by identical code. Written out twice, a fix applied to one
    would quietly leave the comparison uneven.

    `beta` and `alpha_pct` are seeded as None so the keys always exist and
    callers can read them without checking - absence is a value, not a gap.
    """
    final_value = float(result.values.iloc[-1])
    total_invested = float(result.invested.iloc[-1])
    first, last = result.prices.index[0], result.prices.index[-1]
    years = (last - first).days / 365.25
    cagr = annualised_return(result.returns)
    depth, peak, trough, recovery = max_drawdown(result.growth)

    return {
        "final_value":        final_value,
        "total_invested":     total_invested,
        "profit":             final_value - total_invested,
        "years":              float(years),
        "twr_pct":            float((result.growth.iloc[-1] - 1) * 100),
        "cagr_pct":           cagr,
        "volatility_pct":     annualised_volatility(result.returns),
        "sharpe":             sharpe_ratio(result.returns, risk_free_rate),
        "sortino":            sortino_ratio(result.returns, risk_free_rate),
        "max_drawdown_pct":   float(depth),
        "drawdown_peak":      format_date(peak),
        "drawdown_trough":    format_date(trough),
        "drawdown_recovery":  format_date(recovery),
        "calmar":             calmar_ratio(cagr, depth),
        "money_weighted_pct": xirr(result.cashflows, final_value, last),
        "beta":               None,
        "alpha_pct":          None,
    }

def analyse(result, benchmark=None, risk_free_rate=0):
    """Bundle a simulation, and optionally a benchmark, into a reportable dict.

    Keys: `summary`, `benchmark_summary`, `correlation`, `annual_returns` and
    `drawdown`. The benchmark is expected to be a simulation run on *identical*
    cashflows, so the comparison answers "what if the same money, paid in on the
    same days, had bought the index instead".
    """
    summary = summarise(result, risk_free_rate)

    benchmark_summary = None
    if benchmark is not None:
        benchmark_summary  = summarise(benchmark, risk_free_rate)
        beta, alpha = beta_alpha(result.returns, benchmark.returns, risk_free_rate)
        summary["beta"] = beta
        summary["alpha_pct"] = alpha

    # Correlation between the *holdings*, not within the blended portfolio: two
    # assets correlating at 0.95 offer roughly one asset's worth of
    # diversification while appearing on the page as two.
    correlation = result.prices.pct_change().dropna().corr()

    # Compounded within each year, never summed. Returns chain multiplicatively,
    # so +10% twice is +21%.
    by_year = (1 + result.returns).groupby(result.returns.index.year).prod() - 1

    bench_by_year = None
    if benchmark is not None:
        bench_by_year = (1 + benchmark.returns).groupby(benchmark.returns.index.year).prod() - 1

    annual = []
    for (year, value) in by_year.items():
        # The two can cover different year ranges, so a year missing from the
        # benchmark reports None rather than raising.
        bench_value = None
        if bench_by_year is not None and year in bench_by_year.index:
            bench_value = float(bench_by_year.loc[year] * 100)
        annual.append({
            "year":      int(year),
            "portfolio": float(value * 100),
            "benchmark": bench_value,
        })

    return {
        "summary": summary,
        "benchmark_summary": benchmark_summary,
        "correlation": correlation,
        "annual_returns": annual,
        "drawdown": drawdown_series(result.growth),
    }