"""The portfolio simulation.

A portfolio is modelled as a vector of *share counts*, not a pot of money. That
is the decision everything else follows from: share counts only change on the
days you actually trade, so value at any moment is just shares times price, and
the two can never drift out of agreement.

Holdings change on two kinds of day - a contribution, when new money buys more
of everything at target weights, and a rebalance, when the existing holding is
redistributed back to target. On every other day the shares are yesterday's.
The loop therefore runs over those event days only, not over every trading day,
so cost scales with rebalance frequency rather than with the length of history.

This module never touches the network. It takes a price frame and returns
objects, which is what makes it testable against hand-written prices.
"""

import pandas as pd
import numpy as np

class SimulationError(Exception):
    """Raised when the inputs cannot describe a valid simulation."""

class SimulationResult:
    """Everything a completed simulation produced.

    prices     what was simulated against
    weights    the *normalised* target allocation - [2, 1] becomes 0.667/0.333
    shares     units held of each asset, per day
    values     portfolio value, per day, derived from shares x prices
    invested   cumulative cash paid in, per day (dense - for charting and profit)
    cashflows  date -> amount, only days money moved (sparse - for XIRR and TWR)
    returns    daily time-weighted return, with contributions removed
    growth     those returns compounded, starting at 1.0
    """

    def __init__(self, prices, weights, shares, values, invested, cashflows, returns, growth):
        self.prices = prices
        self.weights = weights
        self.shares = shares
        self.values = values
        self.invested = invested
        self.cashflows = cashflows
        self.returns = returns
        self.growth = growth

REBALANCE_FREQUENCIES = {"none": None, "monthly": "M", "quarterly": "Q", "annual": "Y"}

def _period_starts(index, freq):
    """First trading day of each period after the first.

    Stamping every date with the period it belongs to and keeping the first of
    each gives real trading days, so an event never lands on a weekend or a
    holiday. The opening period is dropped because the initial investment
    already covers it: you do not contribute again on the day you started.
    """
    periods = index.to_period(freq)
    return index[~periods.duplicated()][1:]


def simulate(prices, weights, initial_investment, monthly_contribution=0.0, rebalance="none"):
    """Run a portfolio through a price history and return a SimulationResult.

    `weights` may be given in any scale - percentages, fractions or raw ratios -
    and are normalised to sum to one. `rebalance` is one of the keys of
    REBALANCE_FREQUENCIES.

    Contributions land on the first trading day of each month; rebalances on the
    first trading day of each chosen period. Where a date is both, the money is
    paid in *first* so that it is included in the reallocation.
    """
    w = np.asarray(weights, float)

    if len(prices) < 2:
        raise SimulationError("Cannot have a single row of prices")
    
    if len(weights) != len(prices.columns):
        raise SimulationError("Number of weights must match the number of assets.")
    
    if initial_investment <= 0:
        raise SimulationError("Initial investment must be greater than 0.")

    if monthly_contribution < 0:
        raise SimulationError("Cannot have negative monthly contributions.")

    if rebalance not in REBALANCE_FREQUENCIES:
        raise SimulationError("Unrecognised rebalance frequency.")
    
    if (w < 0).any():
        raise SimulationError("Cannot have any negative weights.")

    if w.sum() <= 0:
        raise SimulationError("Sum of weights cannot equal 0.")

    w = w / w.sum()

    weights = pd.Series(w, index=prices.columns)
    contributions = set()
    rebalances = set()

    if monthly_contribution > 0:
        contributions = set(_period_starts(prices.index, "M"))

    code = REBALANCE_FREQUENCIES[rebalance]

    if code is not None:
        rebalances = set(_period_starts(prices.index, code))

    # Both sides are indexed by ticker, so pandas aligns them by name. Using
    # numpy arrays here would align by position and silently buy the wrong asset
    # whenever the provider returned the columns in a different order.
    opening = initial_investment * weights / prices.iloc[0]

    # NaN means "nothing happened on this day"; the ffill below turns that into
    # "so you still hold what you held before". NaN rather than zero because a
    # missed row then shows up as an obvious failure instead of a silent
    # portfolio worth nothing.
    shares = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    shares.iloc[0] = opening
    cashflows = {prices.index[0]: float(initial_investment)}

    held = opening # running share vector
    for date in sorted(contributions | rebalances):
        price_row = prices.loc[date]

        if date in contributions:
            held = held  + (monthly_contribution * weights / price_row)
            cashflows[date] = cashflows.get(date, 0) + monthly_contribution

        if date in rebalances:
            # Value the whole holding, then rebuy it from scratch at target.
            # The same total goes back in, so a rebalance can never create or
            # destroy money - it only moves it between assets.
            total = (held * price_row).sum()
            held = total * weights / price_row

        shares.loc[date] = held  # snapshot

    shares = shares.ffill()

    # Derived, never tracked alongside. A running total kept in parallel is a
    # second source of truth, and the day it disagrees with the holdings there
    # is no way to tell which one is wrong.
    values = (shares * prices).sum(axis=1)

    flows = pd.Series(cashflows).reindex(prices.index).fillna(0.0)
    invested = flows.cumsum()

    # Time-weighted return: paying money in raises the value without being
    # performance, so the day's cashflow is removed before comparing against
    # yesterday. The numerator becomes "what yesterday's holdings are worth
    # today". Subtracted rather than added to the denominator because the
    # contribution buys at today's price and so has not earned anything yet.
    previous = values.shift(1)
    returns = (values - flows) / previous - 1
    returns.iloc[0] = 0.0    # no previous day to compare the first one against
    growth = (1 + returns).cumprod()
    
    return SimulationResult(prices, weights, shares, values, invested, cashflows, returns, growth)