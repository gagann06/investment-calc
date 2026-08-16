"""Tests for the portfolio simulation.

The properties asserted here are the ones that would be silently wrong in a
naive implementation: that rebalancing moves money between assets without
creating any, that contributions land on real trading days, and that a
portfolio which never gains anything ends up worth exactly what was paid in.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import price_frame
from portfolio.simulation import SimulationError, simulate


def period_starts(index, freq):
    """First trading day of each period after the first - the event schedule.

    The opening period is excluded because the initial investment already
    covers it: you do not contribute again on the day you started.
    """
    periods = index.to_period(freq)
    return index[~periods.duplicated()][1:]


def flat(days=253, price=100.0, tickers=("FLAT",)):
    """A frame where nothing moves, so any change in value came from cashflow."""
    return price_frame({t: [price] * days for t in tickers})


def weight_split(result, date):
    """The fraction of portfolio value held in each asset on a given date."""
    holdings = result.shares.loc[date] * result.prices.loc[date]
    return holdings / holdings.sum()


# --------------------------- validation ---------------------------

def test_weights_must_match_the_number_of_assets(two_assets):
    with pytest.raises(SimulationError):
        simulate(two_assets, [1.0], 1000)


def test_negative_weights_are_rejected(two_assets):
    with pytest.raises(SimulationError):
        simulate(two_assets, [1.5, -0.5], 1000)


def test_weights_summing_to_zero_are_rejected(two_assets):
    with pytest.raises(SimulationError):
        simulate(two_assets, [0.0, 0.0], 1000)


@pytest.mark.parametrize("amount", [0, -1, -1000])
def test_initial_investment_must_be_positive(flat_prices, amount):
    with pytest.raises(SimulationError):
        simulate(flat_prices, [1.0], amount)


def test_negative_contributions_are_rejected(flat_prices):
    with pytest.raises(SimulationError):
        simulate(flat_prices, [1.0], 1000, monthly_contribution=-50)


def test_unknown_rebalance_frequency_is_rejected(flat_prices):
    with pytest.raises(SimulationError):
        simulate(flat_prices, [1.0], 1000, rebalance="fortnightly")


def test_a_single_row_of_prices_is_rejected():
    with pytest.raises(SimulationError):
        simulate(flat(days=1), [1.0], 1000)


@pytest.mark.parametrize(
    "given, expected",
    [
        ([1, 1], [0.5, 0.5]),
        ([2, 1], [2 / 3, 1 / 3]),
        ([50, 50], [0.5, 0.5]),
        ([0.5, 0.5], [0.5, 0.5]),
        ([0, 1], [0.0, 1.0]),
    ],
)
def test_weights_are_normalised_to_sum_to_one(two_assets, given, expected):
    """Percentages, fractions and raw ratios should all mean the same thing."""
    result = simulate(two_assets, given, 1000)

    assert list(result.weights) == pytest.approx(expected)


# --------------------------- lump sum ---------------------------

def test_day_one_value_is_the_initial_investment(two_assets):
    result = simulate(two_assets, [0.5, 0.5], 1000)

    assert result.values.iloc[0] == pytest.approx(1000)


def test_flat_prices_never_change_the_value(flat_prices):
    result = simulate(flat_prices, [1.0], 1000)

    assert result.values.min() == pytest.approx(1000)
    assert result.values.max() == pytest.approx(1000)


def test_a_doubling_asset_doubles_the_money(doubling_prices):
    result = simulate(doubling_prices, [1.0], 1000)

    assert result.values.iloc[-1] == pytest.approx(2000)


def test_shares_never_change_without_contributions_or_rebalancing(two_assets):
    result = simulate(two_assets, [0.5, 0.5], 1000)

    assert result.shares.nunique().tolist() == [1, 1]


def test_initial_money_is_split_by_weight(two_assets):
    result = simulate(two_assets, [0.75, 0.25], 1000)
    opening = weight_split(result, two_assets.index[0])

    assert opening.tolist() == pytest.approx([0.75, 0.25])


def test_a_zero_weight_asset_is_never_held(two_assets):
    result = simulate(two_assets, [1.0, 0.0], 1000)

    assert (result.shares.iloc[:, 1] == 0).all()


def test_invested_stays_flat_with_no_contributions(two_assets):
    result = simulate(two_assets, [0.5, 0.5], 1000)

    assert result.invested.unique().tolist() == pytest.approx([1000])


# --------------------------- contributions ---------------------------

def test_contributions_land_on_the_first_trading_day_of_each_month():
    prices = flat()
    result = simulate(prices, [1.0], 1000, monthly_contribution=100)

    expected = set(period_starts(prices.index, "M"))
    expected.add(prices.index[0])          # the opening investment

    assert set(result.cashflows) == expected


def test_total_invested_is_initial_plus_every_contribution():
    prices = flat()
    months = len(period_starts(prices.index, "M"))

    result = simulate(prices, [1.0], 1000, monthly_contribution=100)

    assert result.invested.iloc[-1] == pytest.approx(1000 + 100 * months)


def test_with_flat_prices_you_end_up_with_exactly_what_you_paid_in():
    """No gain and no loss: every pound in is a pound out. This is the test
    that catches contributions being double-counted or dropped."""
    prices = flat()

    result = simulate(prices, [1.0], 1000, monthly_contribution=100)

    assert result.values.iloc[-1] == pytest.approx(result.invested.iloc[-1])


def test_contributions_buy_shares_at_that_days_price():
    prices = price_frame([100.0] * 20 + [200.0] * 233, tickers=["STEP"])
    first_of_month = period_starts(prices.index, "M")

    result = simulate(prices, [1.0], 1000, monthly_contribution=1000)

    for date in first_of_month:
        before = result.shares.shift(1).loc[date, "STEP"]
        bought = result.shares.loc[date, "STEP"] - before
        assert bought == pytest.approx(1000 / prices.loc[date, "STEP"])


def test_shares_only_grow_when_contributing():
    prices = flat()
    result = simulate(prices, [1.0], 1000, monthly_contribution=100)

    assert result.shares["FLAT"].is_monotonic_increasing


def test_zero_contribution_means_one_cashflow_only(flat_prices):
    result = simulate(flat_prices, [1.0], 1000, monthly_contribution=0)

    assert list(result.cashflows.values()) == pytest.approx([1000])


def test_contributions_are_split_by_weight(two_assets):
    """New money follows the target allocation, not the drifted one."""
    result = simulate(two_assets, [0.75, 0.25], 1000, monthly_contribution=400)
    date = period_starts(two_assets.index, "M")[0]

    before = result.shares.shift(1).loc[date]
    bought = (result.shares.loc[date] - before) * two_assets.loc[date]

    assert (bought / bought.sum()).tolist() == pytest.approx([0.75, 0.25])


# --------------------------- rebalancing ---------------------------

def test_without_rebalancing_the_weights_drift(two_assets):
    result = simulate(two_assets, [0.5, 0.5], 1000, rebalance="none")
    final = weight_split(result, two_assets.index[-1])

    assert final["RISE"] > 0.6, "the riser should have taken over the portfolio"


def test_rebalancing_restores_the_target_weights(two_assets):
    result = simulate(two_assets, [0.5, 0.5], 1000, rebalance="monthly")

    for date in period_starts(two_assets.index, "M"):
        assert weight_split(result, date).tolist() == pytest.approx([0.5, 0.5])


def test_rebalancing_moves_money_without_creating_any(two_assets):
    """A rebalance is a reallocation. The portfolio is worth the same the
    instant before and the instant after, so the value series must not jump."""
    result = simulate(two_assets, [0.5, 0.5], 1000, rebalance="monthly")

    for date in period_starts(two_assets.index, "M"):
        previous = result.shares.shift(1).loc[date]
        untouched = (previous * two_assets.loc[date]).sum()
        assert result.values.loc[date] == pytest.approx(untouched)


def test_rebalancing_adds_no_cashflow(two_assets):
    result = simulate(two_assets, [0.5, 0.5], 1000, rebalance="monthly")

    assert list(result.cashflows.values()) == pytest.approx([1000])


def test_rebalancing_a_single_asset_does_nothing(doubling_prices):
    held = simulate(doubling_prices, [1.0], 1000, rebalance="none")
    traded = simulate(doubling_prices, [1.0], 1000, rebalance="monthly")

    pd.testing.assert_series_equal(held.values, traded.values)


def test_less_frequent_rebalancing_trades_less_often(two_assets):
    def days_traded(result):
        moved = result.shares.diff().abs().sum(axis=1)
        return int((moved > 1e-9).sum())

    monthly = simulate(two_assets, [0.5, 0.5], 1000, rebalance="monthly")
    quarterly = simulate(two_assets, [0.5, 0.5], 1000, rebalance="quarterly")

    assert days_traded(monthly) > days_traded(quarterly) > 0


@pytest.mark.parametrize("frequency", ["none", "monthly", "quarterly", "annual"])
def test_every_frequency_is_accepted(two_assets, frequency):
    result = simulate(two_assets, [0.5, 0.5], 1000, rebalance=frequency)

    assert len(result.values) == len(two_assets)


def test_rebalancing_flat_prices_changes_nothing():
    prices = flat(tickers=("A", "B"))

    result = simulate(prices, [0.5, 0.5], 1000, rebalance="monthly")

    assert result.values.min() == pytest.approx(1000)
    assert result.values.max() == pytest.approx(1000)


# --------------------------- invariants ---------------------------

def test_every_series_is_indexed_by_the_price_dates(two_assets):
    result = simulate(two_assets, [0.5, 0.5], 1000, monthly_contribution=50)

    pd.testing.assert_index_equal(result.values.index, two_assets.index)
    pd.testing.assert_index_equal(result.shares.index, two_assets.index)
    pd.testing.assert_index_equal(result.invested.index, two_assets.index)


def test_value_is_always_shares_times_price(two_assets):
    """The value series must be derived from the holdings, not tracked separately."""
    result = simulate(two_assets, [0.6, 0.4], 1000, monthly_contribution=50,
                      rebalance="quarterly")

    expected = (result.shares * two_assets).sum(axis=1)

    pd.testing.assert_series_equal(result.values, expected, check_names=False)


def test_nothing_comes_back_as_nan(two_assets):
    result = simulate(two_assets, [0.5, 0.5], 1000, monthly_contribution=50,
                      rebalance="monthly")

    assert not result.values.isna().any()
    assert not result.shares.isna().any().any()
    assert not result.invested.isna().any()


def test_invested_never_decreases(two_assets):
    result = simulate(two_assets, [0.5, 0.5], 1000, monthly_contribution=50)

    assert result.invested.is_monotonic_increasing


def test_the_original_prices_are_not_modified(two_assets):
    before = two_assets.copy()

    simulate(two_assets, [0.5, 0.5], 1000, monthly_contribution=50,
             rebalance="monthly")

    pd.testing.assert_frame_equal(two_assets, before)
