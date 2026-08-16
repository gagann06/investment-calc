"""Tests for the money-weighted return (XIRR).

Time-weighted return answers "how did the investment do". This answers "how did
*I* do" - it accounts for when money was actually committed, so a good fund
bought mostly at the top still reports a poor personal return.

There is no closed form for it. The rate is whatever makes the discounted value
of every cashflow sum to zero, and that has to be solved numerically.
"""

import pandas as pd
import pytest

from portfolio.metrics import xirr


def day(offset):
    return pd.Timestamp("2020-01-01") + pd.Timedelta(days=offset)


# --------------------------- single investment ---------------------------

def test_doubling_in_one_year_is_one_hundred_percent():
    assert xirr({day(0): 1000.0}, 2000.0, day(365)) == pytest.approx(100.0, abs=0.1)


def test_no_gain_is_no_return():
    assert xirr({day(0): 1000.0}, 1000.0, day(365)) == pytest.approx(0.0, abs=0.01)


def test_halving_in_one_year_is_minus_fifty_percent():
    assert xirr({day(0): 1000.0}, 500.0, day(365)) == pytest.approx(-50.0, abs=0.1)


def test_doubling_over_two_years_annualises_to_about_forty_one_percent():
    """Not 50% - the growth compounds, so the yearly rate is sqrt(2) - 1."""
    result = xirr({day(0): 1000.0}, 2000.0, day(730))

    assert result == pytest.approx((2 ** 0.5 - 1) * 100, abs=0.2)


def test_a_gain_over_half_a_year_annualises_upwards():
    """Earning 20% in six months is far more than 20% a year."""
    result = xirr({day(0): 1000.0}, 1200.0, day(182))

    assert result > 40.0


# --------------------------- the point of the measure ---------------------------

def test_money_added_late_earns_less_of_the_rise():
    """Two investors, same asset, same finish. The one who committed capital
    earlier earned more on it, and the measure has to show that."""
    early = xirr({day(0): 2000.0}, 3000.0, day(365))
    late = xirr({day(0): 1000.0, day(300): 1000.0}, 3000.0, day(365))

    assert late > early, "the late investor's money worked for less time"


def test_contributions_are_discounted_by_how_long_they_were_invested():
    steady = xirr({day(0): 1000.0, day(180): 1000.0}, 2200.0, day(365))

    assert -100.0 < steady < 100.0


def test_the_order_of_the_cashflows_does_not_matter():
    forwards = xirr({day(0): 1000.0, day(180): 500.0}, 1800.0, day(365))
    backwards = xirr({day(180): 500.0, day(0): 1000.0}, 1800.0, day(365))

    assert forwards == pytest.approx(backwards)


# --------------------------- losses ---------------------------

def test_losing_money_gives_a_negative_rate():
    assert xirr({day(0): 1000.0, day(180): 1000.0}, 1500.0, day(365)) < 0


def test_an_almost_total_loss_approaches_minus_one_hundred():
    result = xirr({day(0): 10_000.0}, 1.0, day(365))

    assert result < -99.0


# --------------------------- degenerate input ---------------------------

def test_a_zero_final_value_does_not_raise():
    assert isinstance(xirr({day(0): 1000.0}, 0.0, day(365)), float)


def test_no_cashflows_returns_zero():
    assert xirr({}, 0.0, day(365)) == pytest.approx(0.0)


def test_a_same_day_exit_does_not_raise():
    assert isinstance(xirr({day(0): 1000.0}, 1000.0, day(0)), float)


def test_the_result_is_a_plain_float():
    assert type(xirr({day(0): 1000.0}, 1500.0, day(365))) is float
