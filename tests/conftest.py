"""Shared fixtures.

Every price frame here is synthetic and hand-checkable. Tests that assert on
real market data would fail whenever the market moved, which makes them useless
as a signal, so the only test that touches yfinance is the one that fakes it.
"""

import numpy as np
import pandas as pd
import pytest

FIELDS = ("Open", "High", "Low", "Close", "Volume")


def trading_days(periods, start="2024-01-01"):
    """A business-day index, standing in for a trading calendar."""
    return pd.bdate_range(start=start, periods=periods)


def price_frame(closes, tickers=None, start="2024-01-01", index=None):
    """Build an adjusted-close frame in the shape `fetch_prices` should return.

    `closes` is a 1-D sequence for one asset, or a dict of ticker -> sequence.
    """
    if not isinstance(closes, dict):
        closes = {(tickers[0] if tickers else "AAA"): closes}

    length = len(next(iter(closes.values())))
    idx = index if index is not None else trading_days(length, start)

    return pd.DataFrame(
        {ticker: np.asarray(values, dtype="float64") for ticker, values in closes.items()},
        index=idx,
    )


def yf_frame(closes, multiindex=True, ticker_first=False, start="2024-01-01"):
    """Build a frame shaped like a raw `yfinance.download` response.

    yfinance has shipped several column layouts across versions, so the data
    layer has to cope with more than one. This produces them on demand.
    """
    if not isinstance(closes, dict):
        closes = {"AAA": closes}

    tickers = list(closes)
    length = len(next(iter(closes.values())))
    idx = trading_days(length, start)

    # Non-Close fields are deliberately offset, so a data layer that grabs the
    # wrong field produces obviously wrong numbers rather than passing by luck.
    def field_values(series, field):
        return series if field == "Close" else series + 500.0

    if not multiindex:
        # Legacy single-ticker layout: plain OHLCV columns, no ticker anywhere.
        assert len(tickers) == 1, "the flat layout only ever held one ticker"
        series = np.asarray(closes[tickers[0]], dtype="float64")
        return pd.DataFrame(
            {field: field_values(series, field) for field in FIELDS}, index=idx
        )

    levels = [tickers, FIELDS] if ticker_first else [FIELDS, tickers]
    names = ["Ticker", "Price"] if ticker_first else ["Price", "Ticker"]
    columns = pd.MultiIndex.from_product(levels, names=names)

    frame = pd.DataFrame(index=idx, columns=columns, dtype="float64")
    for ticker in tickers:
        series = np.asarray(closes[ticker], dtype="float64")
        for field in FIELDS:
            key = (ticker, field) if ticker_first else (field, ticker)
            frame[key] = field_values(series, field)
    return frame


@pytest.fixture
def flat_prices():
    """One asset, unchanging at 100. Every return metric should be zero."""
    return price_frame([100.0] * 60, tickers=["FLAT"])


@pytest.fixture
def doubling_prices():
    """One asset compounding smoothly to exactly 2x over 252 trading days.

    252 days is one trading year, so the annualised return here should come out
    at 100% almost exactly - which makes it a good check on annualisation.
    """
    daily = 2.0 ** (1 / 252)
    closes = [100.0 * daily**i for i in range(253)]
    return price_frame(closes, tickers=["GROW"])


@pytest.fixture
def two_assets():
    """Two assets drifting apart, so rebalancing has something to correct.

    A rises 50%, B falls 20%. An unrebalanced 50/50 split ends badly skewed
    towards A; a rebalanced one keeps selling A to buy B.
    """
    length = 120
    rise = [100.0 * (1.5 ** (i / (length - 1))) for i in range(length)]
    fall = [100.0 * (0.8 ** (i / (length - 1))) for i in range(length)]
    return price_frame({"RISE": rise, "FALL": fall})


@pytest.fixture
def crash_and_recover():
    """Rises to 120, falls to 60, recovers past the old peak to 130.

    Maximum drawdown is -50% exactly, from the 120 peak to the 60 trough, and
    there is a genuine recovery date afterwards.
    """
    up = list(np.linspace(100, 120, 21))
    down = list(np.linspace(120, 60, 61))[1:]
    back = list(np.linspace(60, 130, 71))[1:]
    return price_frame(up + down + back, tickers=["CYCLE"])
