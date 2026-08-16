"""Tests for time-weighted returns.

A portfolio's value going up is not the same as the portfolio performing well:
paying money in also raises the value. Time-weighted return strips the deposits
out, which is why it is the measure a fund is judged on and the one that lets
two portfolios funded differently be compared at all.

Nearly every test here is the same assertion in a different disguise: the return
series must not react to cashflows, only to prices.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import price_frame
from portfolio.simulation import simulate


def flat(days=253, price=100.0, ticker="FLAT"):
    return price_frame({ticker: [price] * days})


# --------------------------- the defining property ---------------------------

def test_flat_prices_produce_no_return(flat_prices):
    result = simulate(flat_prices, [1.0], 1000)

    assert result.returns.abs().max() == pytest.approx(0.0)


def test_contributions_into_flat_prices_still_produce_no_return():
    """Money paid in is not performance. This is the whole point of the measure."""
    result = simulate(flat(), [1.0], 1000, monthly_contribution=500)

    assert result.returns.abs().max() == pytest.approx(0.0)
    assert result.growth.iloc[-1] == pytest.approx(1.0)


def test_the_size_of_contributions_does_not_change_the_return(doubling_prices):
    """Two investors in the same asset earned the same return on their money,
    however differently they funded it."""
    small = simulate(doubling_prices, [1.0], 1000, monthly_contribution=10)
    large = simulate(doubling_prices, [1.0], 1000, monthly_contribution=10_000)

    pd.testing.assert_series_equal(small.returns, large.returns)


def test_contributions_do_not_change_the_growth_of_a_doubling_asset(doubling_prices):
    held = simulate(doubling_prices, [1.0], 1000)
    funded = simulate(doubling_prices, [1.0], 1000, monthly_contribution=500)

    assert held.growth.iloc[-1] == pytest.approx(2.0, rel=1e-6)
    assert funded.growth.iloc[-1] == pytest.approx(2.0, rel=1e-6)


# --------------------------- worked example ---------------------------

def test_a_hand_computed_case():
    """Five days across a month boundary, with one contribution.

        prices   100  100  100  100  200
        day 3 is 1 February, so 1000 is paid in at 100
        day 4 the price doubles

    Holdings: 10 shares, then 20 after the contribution.
    Values:   1000 1000 1000 2000 4000

    The value went from 1000 to 4000, but 1000 of that was deposited. The
    investment doubled once, on the last day. TWR must say +100%, not +300%.
    """
    index = pd.bdate_range("2024-01-29", periods=5)
    prices = price_frame({"X": [100.0, 100.0, 100.0, 100.0, 200.0]}, index=index)

    result = simulate(prices, [1.0], 1000, monthly_contribution=1000)

    assert result.values.tolist() == pytest.approx([1000, 1000, 1000, 2000, 4000])
    assert result.returns.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 1.0])
    assert result.growth.tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0, 2.0])

    naive = result.values.iloc[-1] / result.values.iloc[0] - 1
    assert naive == pytest.approx(3.0), "the measure TWR exists to replace"


def test_a_steady_riser_compounds():
    """10% a day for two days is 21% in total, not 20%."""
    prices = price_frame({"X": [100.0, 110.0, 121.0]})

    result = simulate(prices, [1.0], 1000)

    assert result.returns.tolist() == pytest.approx([0.0, 0.1, 0.1])
    assert result.growth.iloc[-1] == pytest.approx(1.21)


def test_a_fall_is_negative():
    prices = price_frame({"X": [100.0, 50.0]})

    result = simulate(prices, [1.0], 1000)

    assert result.returns.iloc[-1] == pytest.approx(-0.5)
    assert result.growth.iloc[-1] == pytest.approx(0.5)


# --------------------------- shape and consistency ---------------------------

def test_the_first_day_has_no_return(two_assets):
    """There is no previous day to compare against, so the series opens at zero."""
    result = simulate(two_assets, [0.5, 0.5], 1000)

    assert result.returns.iloc[0] == pytest.approx(0.0)
    assert result.growth.iloc[0] == pytest.approx(1.0)


def test_growth_is_the_compounded_return_series(two_assets):
    result = simulate(two_assets, [0.5, 0.5], 1000, monthly_contribution=100)

    expected = (1 + result.returns).cumprod()

    pd.testing.assert_series_equal(result.growth, expected, check_names=False)


def test_returns_and_growth_are_indexed_by_the_price_dates(two_assets):
    result = simulate(two_assets, [0.5, 0.5], 1000, monthly_contribution=100)

    pd.testing.assert_index_equal(result.returns.index, two_assets.index)
    pd.testing.assert_index_equal(result.growth.index, two_assets.index)


def test_no_nan_in_either_series(two_assets):
    result = simulate(two_assets, [0.5, 0.5], 1000, monthly_contribution=100,
                      rebalance="monthly")

    assert not result.returns.isna().any()
    assert not result.growth.isna().any()


def test_rebalancing_is_not_a_gain():
    """Selling one asset to buy another moves money sideways. With prices flat
    there is nothing to earn, so a rebalanced portfolio must still show zero."""
    prices = price_frame({"A": [100.0] * 253, "B": [100.0] * 253})

    result = simulate(prices, [0.7, 0.3], 1000, rebalance="monthly")

    assert result.returns.abs().max() == pytest.approx(0.0)
    assert result.growth.iloc[-1] == pytest.approx(1.0)


def test_the_first_rebalance_cannot_change_anything(two_assets):
    """Up to the first rebalance date the two portfolios are identical holdings,
    so their returns on that date must agree exactly."""
    plain = simulate(two_assets, [0.5, 0.5], 1000, rebalance="none")
    traded = simulate(two_assets, [0.5, 0.5], 1000, rebalance="monthly")

    periods = two_assets.index.to_period("M")
    first_rebalance = two_assets.index[~periods.duplicated()][1]

    upto = two_assets.index <= first_rebalance
    pd.testing.assert_series_equal(traded.returns[upto], plain.returns[upto])
