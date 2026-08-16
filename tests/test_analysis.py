"""Tests for the analysis layer.

`analyse` is the assembly step: it takes a finished simulation, optionally a
benchmark simulation run on identical cashflows, and produces the bundle the web
layer serialises. It should compute almost nothing itself - the arithmetic lives
in `metrics` - but it decides what gets reported and how the benchmark is
compared, which is where the judgement is.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import price_frame
from portfolio.analysis import analyse, beta_alpha, tracking_error
from portfolio.simulation import simulate


def run(prices, weights=None, initial=10_000, monthly=0.0, rebalance="none"):
    if weights is None:
        weights = [1.0] * len(prices.columns)
    return simulate(prices, weights, initial,
                    monthly_contribution=monthly, rebalance=rebalance)


def rising(rate=0.0005, days=253, ticker="X"):
    return price_frame({ticker: [100.0 * (1 + rate) ** i for i in range(days)]})


# --------------------------- headline numbers ---------------------------

def test_the_summary_reports_the_closing_value(doubling_prices):
    result = run(doubling_prices)

    summary = analyse(result)["summary"]

    assert summary["final_value"] == pytest.approx(result.values.iloc[-1])


def test_total_invested_is_every_pound_paid_in():
    result = run(rising(), monthly=200)

    summary = analyse(result)["summary"]

    assert summary["total_invested"] == pytest.approx(result.invested.iloc[-1])


def test_profit_is_the_gap_between_them():
    result = run(rising(), monthly=200)

    summary = analyse(result)["summary"]

    assert summary["profit"] == pytest.approx(
        summary["final_value"] - summary["total_invested"]
    )


def test_a_doubling_portfolio_reports_a_hundred_percent_time_weighted(doubling_prices):
    summary = analyse(run(doubling_prices))["summary"]

    assert summary["twr_pct"] == pytest.approx(100.0, rel=1e-6)


def test_time_weighted_return_ignores_contributions(doubling_prices):
    """The headline return must not move when the funding schedule changes."""
    held = analyse(run(doubling_prices))["summary"]
    funded = analyse(run(doubling_prices, monthly=500))["summary"]

    assert held["twr_pct"] == pytest.approx(funded["twr_pct"], rel=1e-6)


def test_money_weighted_return_does_react_to_contributions(doubling_prices):
    """...whereas the investor's own return absolutely does, because money added
    late in a rising market earns less of the rise."""
    held = analyse(run(doubling_prices))["summary"]
    funded = analyse(run(doubling_prices, monthly=500))["summary"]

    assert held["money_weighted_pct"] != pytest.approx(funded["money_weighted_pct"])


def test_years_covers_the_span_of_the_prices(doubling_prices):
    summary = analyse(run(doubling_prices))["summary"]
    span = (doubling_prices.index[-1] - doubling_prices.index[0]).days / 365.25

    assert summary["years"] == pytest.approx(span, rel=1e-3)


def test_annualised_return_of_a_one_year_doubling_is_about_a_hundred(doubling_prices):
    summary = analyse(run(doubling_prices))["summary"]

    assert summary["cagr_pct"] == pytest.approx(100.0, rel=0.02)


def test_a_flat_portfolio_reports_zeros(flat_prices):
    summary = analyse(run(flat_prices))["summary"]

    assert summary["twr_pct"] == pytest.approx(0.0)
    assert summary["volatility_pct"] == pytest.approx(0.0)
    assert summary["max_drawdown_pct"] == pytest.approx(0.0)


def test_drawdown_dates_are_reported_as_strings_or_none(crash_and_recover):
    summary = analyse(run(crash_and_recover))["summary"]

    for key in ("drawdown_peak", "drawdown_trough", "drawdown_recovery"):
        assert key in summary
        assert summary[key] is None or isinstance(summary[key], str)


def test_the_worst_drawdown_is_found(crash_and_recover):
    """The fixture peaks at 120 and troughs at 60."""
    summary = analyse(run(crash_and_recover))["summary"]

    assert summary["max_drawdown_pct"] == pytest.approx(-50.0, rel=1e-3)


def test_every_summary_value_is_json_safe(two_assets):
    """NaN and numpy scalars both break jsonify, so neither may reach the API."""
    summary = analyse(run(two_assets, [0.5, 0.5], monthly=100))["summary"]

    for key, value in summary.items():
        if value is None or isinstance(value, str):
            continue
        assert type(value) is float, f"{key} is {type(value)}"
        assert not np.isnan(value), f"{key} is NaN"


# --------------------------- beta and alpha ---------------------------

def test_a_portfolio_measured_against_itself_has_a_beta_of_one(two_assets):
    result = run(two_assets, [0.5, 0.5])

    beta, _ = beta_alpha(result.returns, result.returns)

    assert beta == pytest.approx(1.0)


def test_a_portfolio_measured_against_itself_has_no_alpha(two_assets):
    result = run(two_assets, [0.5, 0.5])

    _, alpha = beta_alpha(result.returns, result.returns)

    assert alpha == pytest.approx(0.0, abs=1e-6)


def test_a_portfolio_that_moves_twice_as_hard_has_a_beta_of_two():
    index = pd.bdate_range("2024-01-01", periods=200)
    moves = np.resize([0.01, -0.008, 0.012, -0.005], 200)
    market = pd.Series(moves, index=index)

    beta, _ = beta_alpha(market * 2, market)

    assert beta == pytest.approx(2.0)


def test_a_flat_portfolio_has_no_beta_to_a_moving_market():
    index = pd.bdate_range("2024-01-01", periods=200)
    market = pd.Series(np.resize([0.01, -0.008, 0.012, -0.005], 200), index=index)
    flat = pd.Series(0.0, index=index)

    beta, _ = beta_alpha(flat, market)

    assert beta == pytest.approx(0.0)


def test_tracking_error_against_itself_is_zero(two_assets):
    result = run(two_assets, [0.5, 0.5])

    assert tracking_error(result.returns, result.returns) == pytest.approx(0.0)


def test_tracking_error_grows_as_the_portfolio_diverges():
    index = pd.bdate_range("2024-01-01", periods=200)
    market = pd.Series(np.resize([0.01, -0.008, 0.012, -0.005], 200), index=index)
    noise = pd.Series(np.resize([0.004, -0.004], 200), index=index)

    close = tracking_error(market + noise * 0.1, market)
    far = tracking_error(market + noise, market)

    assert far > close


# --------------------------- the benchmark ---------------------------

def test_without_a_benchmark_the_comparison_fields_are_absent(two_assets):
    bundle = analyse(run(two_assets, [0.5, 0.5]))

    assert bundle.get("benchmark_summary") is None


def test_with_a_benchmark_its_summary_comes_back(two_assets, doubling_prices):
    portfolio = run(two_assets, [0.5, 0.5])
    benchmark = run(doubling_prices.iloc[: len(two_assets)])

    bundle = analyse(portfolio, benchmark=benchmark)

    assert bundle["benchmark_summary"]["final_value"] > 0
    assert "twr_pct" in bundle["benchmark_summary"]


def test_beta_and_alpha_only_appear_with_a_benchmark(two_assets, doubling_prices):
    portfolio = run(two_assets, [0.5, 0.5])
    benchmark = run(doubling_prices.iloc[: len(two_assets)])

    without = analyse(portfolio)["summary"]
    with_bench = analyse(portfolio, benchmark=benchmark)["summary"]

    assert without.get("beta") is None
    assert with_bench["beta"] is not None
    assert with_bench["alpha_pct"] is not None


def test_a_portfolio_benchmarked_against_itself_has_beta_one(two_assets):
    result = run(two_assets, [0.5, 0.5])

    summary = analyse(result, benchmark=result)["summary"]

    assert summary["beta"] == pytest.approx(1.0)
    assert summary["alpha_pct"] == pytest.approx(0.0, abs=1e-6)


# --------------------------- correlation ---------------------------

def test_correlation_of_a_single_holding_is_not_reported(flat_prices):
    bundle = analyse(run(flat_prices))

    correlation = bundle.get("correlation")
    assert correlation is None or correlation.shape == (1, 1)


def test_correlation_is_square_and_named(two_assets):
    correlation = analyse(run(two_assets, [0.5, 0.5]))["correlation"]

    assert correlation.shape == (2, 2)
    assert list(correlation.columns) == list(two_assets.columns)


def test_an_asset_correlates_perfectly_with_itself(two_assets):
    correlation = analyse(run(two_assets, [0.5, 0.5]))["correlation"]

    assert np.allclose(np.diag(correlation.to_numpy()), 1.0)


def test_identical_assets_correlate_at_one():
    moves = [100.0 * (1.001) ** i for i in range(120)]
    prices = price_frame({"A": moves, "B": moves})

    correlation = analyse(run(prices, [0.5, 0.5]))["correlation"]

    assert correlation.loc["A", "B"] == pytest.approx(1.0)


# --------------------------- calendar years ---------------------------

def test_one_row_per_calendar_year():
    prices = price_frame({"X": [100.0 * 1.0005**i for i in range(600)]})

    years = analyse(run(prices))["annual_returns"]

    assert [row["year"] for row in years] == sorted({d.year for d in prices.index})


def test_annual_returns_carry_the_benchmark_when_there_is_one(two_assets):
    portfolio = run(two_assets, [0.5, 0.5])

    years = analyse(portfolio, benchmark=portfolio)["annual_returns"]

    assert all(row["benchmark"] is not None for row in years)


def test_annual_returns_are_none_for_the_benchmark_when_absent(two_assets):
    years = analyse(run(two_assets, [0.5, 0.5]))["annual_returns"]

    assert all(row.get("benchmark") is None for row in years)


def test_a_flat_year_returns_zero(flat_prices):
    years = analyse(run(flat_prices))["annual_returns"]

    assert all(row["portfolio"] == pytest.approx(0.0) for row in years)


# --------------------------- the drawdown series ---------------------------

def test_the_drawdown_series_is_returned_for_charting(crash_and_recover):
    bundle = analyse(run(crash_and_recover))

    assert len(bundle["drawdown"]) == len(crash_and_recover)
    assert bundle["drawdown"].max() <= 0
    assert bundle["drawdown"].min() == pytest.approx(-50.0, rel=1e-3)
