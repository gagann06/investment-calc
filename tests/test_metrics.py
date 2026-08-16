"""Tests for the risk and return metrics.

These functions take a return series rather than a portfolio, so they can be
tested against hand-built inputs with known answers. Most of the traps are in
annualisation: getting a formula right daily and then scaling it wrongly to a
year is the classic way these come out plausible but incorrect.
"""

import numpy as np
import pandas as pd
import pytest

from portfolio.metrics import (
    TRADING_DAYS,
    annualised_return,
    annualised_volatility,
    calmar_ratio,
    drawdown_series,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)


def series(values, start="2024-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)),
                     dtype="float64")


def constant_returns(daily, days=TRADING_DAYS):
    return series([daily] * days)


def growth_from(returns):
    return (1 + returns).cumprod()


# --------------------------- volatility ---------------------------

def test_a_constant_return_has_no_volatility():
    """Volatility measures variation. A series that never varies has none,
    however large the return itself is."""
    assert annualised_volatility(constant_returns(0.01)) == pytest.approx(0.0)


def test_volatility_is_the_daily_deviation_scaled_to_a_year():
    returns = series([0.01, -0.02, 0.015, -0.005, 0.02] * 20)

    expected = returns.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100

    assert annualised_volatility(returns) == pytest.approx(expected)


def test_doubling_every_move_doubles_the_volatility():
    returns = series([0.01, -0.02, 0.015, -0.005] * 25)

    assert annualised_volatility(returns * 2) == pytest.approx(
        annualised_volatility(returns) * 2
    )


def test_volatility_of_too_short_a_series_is_zero():
    assert annualised_volatility(series([0.01])) == pytest.approx(0.0)


# --------------------------- annualised return ---------------------------

def test_a_year_of_daily_growth_compounds_to_the_annual_figure():
    """0.1% a day for 252 days is not 25.2% - it compounds."""
    daily = 0.001
    expected = ((1 + daily) ** TRADING_DAYS - 1) * 100

    assert annualised_return(constant_returns(daily)) == pytest.approx(expected)


def test_doubling_over_one_trading_year_is_one_hundred_percent():
    daily = 2 ** (1 / TRADING_DAYS) - 1

    assert annualised_return(constant_returns(daily)) == pytest.approx(100.0)


def test_doubling_over_two_trading_years_is_not_one_hundred_percent():
    """Half the growth rate per year, so roughly 41%, not 50%."""
    daily = 2 ** (1 / (TRADING_DAYS * 2)) - 1

    result = annualised_return(constant_returns(daily, days=TRADING_DAYS * 2))

    assert result == pytest.approx((2 ** 0.5 - 1) * 100)


def test_no_movement_is_no_return():
    assert annualised_return(constant_returns(0.0)) == pytest.approx(0.0)


def test_losses_annualise_negative():
    assert annualised_return(constant_returns(-0.001)) < 0


# --------------------------- drawdown ---------------------------

def test_a_series_that_only_rises_never_draws_down():
    growth = growth_from(constant_returns(0.001))

    assert drawdown_series(growth).min() == pytest.approx(0.0)
    assert max_drawdown(growth)[0] == pytest.approx(0.0)


def test_a_halving_is_a_fifty_percent_drawdown():
    growth = series([1.0, 1.2, 0.6, 0.9, 1.3])

    depth, _, _, _ = max_drawdown(growth)

    assert depth == pytest.approx(-50.0)


def test_drawdown_is_measured_from_the_peak_not_the_start():
    """Starting at 1.0, peaking at 2.0 and falling to 1.5 is -25%, not +50%."""
    growth = series([1.0, 2.0, 1.5])

    assert max_drawdown(growth)[0] == pytest.approx(-25.0)


def test_the_peak_and_trough_dates_bracket_the_fall():
    growth = series([1.0, 1.2, 0.6, 0.9, 1.3])

    _, peak, trough, _ = max_drawdown(growth)

    assert peak == growth.index[1]
    assert trough == growth.index[2]


def test_the_recovery_date_is_when_the_old_peak_is_regained():
    growth = series([1.0, 1.2, 0.6, 0.9, 1.3])

    *_, recovery = max_drawdown(growth)

    assert recovery == growth.index[4], "1.3 is the first value back above 1.2"


def test_a_portfolio_still_under_water_has_no_recovery_date():
    growth = series([1.0, 1.2, 0.6, 0.9, 1.1])

    *_, recovery = max_drawdown(growth)

    assert recovery is None


def test_the_underwater_curve_is_zero_at_every_new_high():
    growth = series([1.0, 1.1, 1.05, 1.2, 1.3])
    underwater = drawdown_series(growth)

    assert underwater.iloc[0] == pytest.approx(0.0)
    assert underwater.iloc[1] == pytest.approx(0.0)
    assert underwater.iloc[2] < 0
    assert underwater.iloc[3] == pytest.approx(0.0)


def test_the_underwater_curve_is_never_positive():
    growth = growth_from(series([0.01, -0.03, 0.02, -0.01, 0.005] * 20))

    assert drawdown_series(growth).max() <= 0


# --------------------------- sharpe ---------------------------

def test_sharpe_is_excess_return_over_volatility_annualised():
    returns = series([0.01, -0.005, 0.008, -0.002] * 30)

    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(TRADING_DAYS)

    assert sharpe_ratio(returns) == pytest.approx(expected)


def test_a_riskless_return_has_no_sharpe_ratio():
    """Zero volatility means dividing by zero; report 0 rather than infinity."""
    assert sharpe_ratio(constant_returns(0.001)) == pytest.approx(0.0)


def test_raising_the_risk_free_rate_lowers_sharpe():
    returns = series([0.01, -0.005, 0.008, -0.002] * 30)

    assert sharpe_ratio(returns, 0.05) < sharpe_ratio(returns, 0.0)


def test_a_portfolio_that_only_loses_has_a_negative_sharpe():
    returns = series([-0.01, -0.005, -0.008, -0.002] * 30)

    assert sharpe_ratio(returns) < 0


# --------------------------- sortino ---------------------------

def test_sortino_ignores_upside_volatility():
    """Two series with identical downside but different upside. Sharpe punishes
    the wilder one; Sortino should not."""
    steady = series([0.01, -0.01] * 60)
    spiky = series([0.05, -0.01] * 60)

    assert sortino_ratio(spiky) > sortino_ratio(steady)


def test_sortino_exceeds_sharpe_when_gains_are_larger_than_losses():
    returns = series([0.04, -0.01] * 60)

    assert sortino_ratio(returns) > sharpe_ratio(returns)


def test_a_series_that_never_falls_has_no_sortino():
    assert sortino_ratio(constant_returns(0.001)) == pytest.approx(0.0)


# --------------------------- calmar ---------------------------

def test_calmar_is_return_divided_by_the_worst_fall():
    assert calmar_ratio(20.0, -10.0) == pytest.approx(2.0)


def test_calmar_uses_the_size_of_the_drawdown_not_its_sign():
    assert calmar_ratio(15.0, -30.0) == pytest.approx(0.5)


def test_calmar_without_a_drawdown_is_zero():
    assert calmar_ratio(20.0, 0.0) == pytest.approx(0.0)


# --------------------------- shape ---------------------------

def test_drawdown_series_keeps_the_index():
    growth = growth_from(constant_returns(0.001, days=50))

    pd.testing.assert_index_equal(drawdown_series(growth).index, growth.index)


@pytest.mark.parametrize(
    "function", [annualised_volatility, annualised_return, sharpe_ratio, sortino_ratio]
)
def test_metrics_return_plain_floats(function):
    """A numpy scalar serialises to JSON differently; pin the type at the source."""
    assert type(function(series([0.01, -0.01] * 30))) is float
