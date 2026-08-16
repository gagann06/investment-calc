"""Price data retrieval.

Prices here are *adjusted* closes: splits and dividends are folded in. Raw
closes understate the return of anything that pays a dividend, which over a long
window is most of the gap between an index's price return and its total return.

This is the only module that touches the network, which is what lets the rest of
the engine be tested against hand-written price frames.
"""

import yfinance as yf
import pandas as pd


class DataError(Exception):
    """Raised when usable price data cannot be retrieved."""


def fetch_prices(tickers, start_date, end_date):
    """Return a DataFrame of adjusted closes, one column per ticker.

    Everything downstream relies on the guarantees made here, so they are worth
    stating plainly. The returned frame:

    * is indexed by date, ascending, as a DatetimeIndex
    * has one float64 column per requested ticker, named after it, **in the
      order the caller asked for** rather than the order the provider chose
    * contains no NaN: gaps from mismatched exchange holidays are forward-filled,
      and dates before every ticker had listed are trimmed off the front
    * has at least two rows, since a single price cannot produce a return

    Anything that would violate those raises `DataError` rather than returning
    a partial frame or None, so a caller never has to check before using it.
    """
    tickers_normalised = []
 
    for t in tickers:
        clean = str(t).strip().upper()
        if not clean:
            continue
        if clean in tickers_normalised:
            continue
        tickers_normalised.append(clean)

    if not tickers_normalised:
        raise DataError("No tickers supplied.")

    try:
        raw = yf.download(tickers_normalised, start=start_date, end=end_date,
                auto_adjust=True, progress=False)
    except Exception as e:
        raise DataError(f"Could not reach the price data provider: {e}") from e

    if raw is None or raw.empty:
        raise DataError("No ticker data received.")

    close = None

    if isinstance(raw.columns, pd.MultiIndex):
        for lvl in range(raw.columns.nlevels):
            if "Close" in raw.columns.get_level_values(lvl):
                close = raw.xs("Close", axis=1, level=lvl)
                break
    elif "Close" in raw.columns:
        close = raw[["Close"]]

    if close is None:
        raise DataError("Price data contained no Close column.")

    if len(close.columns) == 1 and len(tickers_normalised) == 1:
        close.columns = tickers_normalised

    missing = [
        t for t in tickers_normalised
        if t not in close.columns or close[t].isna().all()
        ]
    if missing:
        raise DataError(f"No price data found for: {', '.join(missing)}")

    close = close[tickers_normalised]
    close = close.ffill().dropna()

    if len(close) < 2:
        raise DataError("Fewer than two days of overlapping data for these tickers. If one of them listed recently, try a later start date.")

    close.index = pd.to_datetime(close.index)
    close = close.astype("float64")

    return close