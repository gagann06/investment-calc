"""Tests for the data layer.

`yfinance.download` is replaced in every test here. Hitting the real API would
make the suite slow, flaky and dependent on the market being open, and it would
test Yahoo rather than this code.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import trading_days, yf_frame
from portfolio import data
from portfolio.data import DataError, fetch_prices


@pytest.fixture
def fake_download(monkeypatch):
    """Install a stand-in for yf.download and record how it was called."""
    calls = []

    def install(response):
        def fake(*args, **kwargs):
            calls.append(kwargs)
            if isinstance(response, Exception):
                raise response
            return response

        monkeypatch.setattr(data.yf, "download", fake)
        return calls

    return install


# --------------------------- shapes yfinance returns ---------------------------

def test_multi_ticker_gives_one_column_per_ticker(fake_download):
    fake_download(yf_frame({"AAA": [1.0] * 10, "BBB": [2.0] * 10}))

    prices = fetch_prices(["AAA", "BBB"], "2024-01-01", "2024-02-01")

    assert list(prices.columns) == ["AAA", "BBB"]
    assert len(prices) == 10


def test_columns_come_back_in_the_order_requested(fake_download):
    fake_download(yf_frame({"AAA": [1.0] * 10, "BBB": [2.0] * 10}))

    prices = fetch_prices(["BBB", "AAA"], "2024-01-01", "2024-02-01")

    assert list(prices.columns) == ["BBB", "AAA"]


def test_single_ticker_column_keeps_its_name(fake_download):
    fake_download(yf_frame({"AAA": [1.0] * 10}))

    prices = fetch_prices(["AAA"], "2024-01-01", "2024-02-01")

    assert list(prices.columns) == ["AAA"]


def test_single_ticker_flat_layout_gets_a_name(fake_download):
    """Older yfinance returned plain OHLCV columns with the ticker nowhere in them."""
    fake_download(yf_frame({"AAA": [1.0] * 10}, multiindex=False))

    prices = fetch_prices(["AAA"], "2024-01-01", "2024-02-01")

    assert list(prices.columns) == ["AAA"]


def test_close_is_found_when_the_ticker_level_comes_first(fake_download):
    """group_by='ticker' nests the other way round; the level must be located, not assumed."""
    fake_download(yf_frame({"AAA": [1.0] * 10, "BBB": [2.0] * 10}, ticker_first=True))

    prices = fetch_prices(["AAA", "BBB"], "2024-01-01", "2024-02-01")

    assert list(prices.columns) == ["AAA", "BBB"]


def test_it_reads_close_and_not_some_other_field(fake_download):
    """The fixture offsets Open/High/Low/Volume by +500, so picking one is visible."""
    fake_download(yf_frame({"AAA": [7.0] * 10}))

    prices = fetch_prices(["AAA"], "2024-01-01", "2024-02-01")

    assert prices["AAA"].iloc[0] == pytest.approx(7.0)


# --------------------------- ticker handling ---------------------------

@pytest.mark.parametrize(
    "given, expected",
    [
        (["aaa", "bbb"], ["AAA", "BBB"]),          # lower-cased
        (["  AAA  ", "BBB"], ["AAA", "BBB"]),      # padded
        (["AAA", "AAA", "BBB"], ["AAA", "BBB"]),   # duplicated
        (["aaa", "AAA", "BBB"], ["AAA", "BBB"]),   # duplicated after casing
        (["AAA", "aaa", "BBB"], ["AAA", "BBB"]),   # ...and in the other order
        (["AAA", "  AAA  ", "BBB"], ["AAA", "BBB"]),  # duplicated after stripping
        (["AAA", "", "BBB"], ["AAA", "BBB"]),      # blank entry
        (["AAA", "   ", "BBB"], ["AAA", "BBB"]),   # whitespace-only entry
    ],
)
def test_tickers_are_normalised(fake_download, given, expected):
    fake_download(yf_frame({"AAA": [1.0] * 10, "BBB": [2.0] * 10}))

    prices = fetch_prices(given, "2024-01-01", "2024-02-01")

    assert list(prices.columns) == expected


# --------------------------- alignment ---------------------------

def test_a_gap_in_one_asset_is_forward_filled(fake_download):
    """Different exchanges close on different days; a holiday is not missing data."""
    raw = yf_frame({"AAA": [10.0] * 10, "BBB": [20.0] * 10})
    raw.loc[raw.index[4], ("Close", "BBB")] = np.nan

    fake_download(raw)
    prices = fetch_prices(["AAA", "BBB"], "2024-01-01", "2024-02-01")

    assert len(prices) == 10, "the row should be kept, not dropped"
    assert prices["BBB"].iloc[4] == pytest.approx(20.0)
    assert not prices.isna().any().any()


def test_rows_before_every_asset_has_data_are_dropped(fake_download):
    """A fund that listed later cannot be held before it existed."""
    raw = yf_frame({"AAA": [10.0] * 10, "BBB": [20.0] * 10})
    raw.loc[raw.index[:3], ("Close", "BBB")] = np.nan

    fake_download(raw)
    prices = fetch_prices(["AAA", "BBB"], "2024-01-01", "2024-02-01")

    assert len(prices) == 7, "leading NaNs survive an ffill and must be trimmed"
    assert not prices.isna().any().any()


# --------------------------- failure paths ---------------------------

def test_no_tickers_raises():
    with pytest.raises(DataError):
        fetch_prices([], "2024-01-01", "2024-02-01")


def test_empty_response_raises(fake_download):
    fake_download(pd.DataFrame())

    with pytest.raises(DataError):
        fetch_prices(["AAA"], "2024-01-01", "2024-02-01")


def test_unknown_ticker_is_named_in_the_error(fake_download):
    raw = yf_frame({"AAA": [10.0] * 10, "GHOST": [np.nan] * 10})
    fake_download(raw)

    with pytest.raises(DataError, match="GHOST"):
        fetch_prices(["AAA", "GHOST"], "2024-01-01", "2024-02-01")


def test_a_single_row_raises(fake_download):
    """One price is a quote, not a history: no return can be computed from it."""
    fake_download(yf_frame({"AAA": [10.0]}))

    with pytest.raises(DataError):
        fetch_prices(["AAA"], "2024-01-01", "2024-01-02")


def test_provider_failure_becomes_a_dataerror(fake_download):
    fake_download(ConnectionError("network is down"))

    with pytest.raises(DataError):
        fetch_prices(["AAA"], "2024-01-01", "2024-02-01")


# --------------------------- adjustment ---------------------------

def test_adjusted_prices_are_requested(fake_download):
    """Without auto_adjust every dividend silently vanishes from the return."""
    calls = fake_download(yf_frame({"AAA": [10.0] * 10}))

    fetch_prices(["AAA"], "2024-01-01", "2024-02-01")

    assert calls[0].get("auto_adjust") is True


# --------------------------- output contract ---------------------------

def test_index_is_datetime_and_values_are_float(fake_download):
    fake_download(yf_frame({"AAA": [10.0] * 10}))

    prices = fetch_prices(["AAA"], "2024-01-01", "2024-02-01")

    assert isinstance(prices.index, pd.DatetimeIndex)
    assert all(dtype == "float64" for dtype in prices.dtypes)
