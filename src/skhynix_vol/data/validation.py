"""Validation and normalization for daily price observations."""
from __future__ import annotations

import pandas as pd

REQUIRED_PRICE_COLUMNS = ("open", "high", "low", "close", "adj_close", "volume")


def normalize_prices(frame: pd.DataFrame, ticker: str = "000660.KS", source: str = "yfinance") -> pd.DataFrame:
    """Return a validated, consistently named daily price table.

    Dates are normalized to exchange date-only sessions. Missing dates are not
    filled because weekends and KRX holidays are not missing observations.
    """
    if frame is None or frame.empty:
        raise ValueError("Price data is empty")
    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [str(col[0]) for col in data.columns]
    data = data.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close",
        "Adj Close": "adj_close", "Adj_Close": "adj_close", "Volume": "volume",
        "Dividends": "dividends", "Stock Splits": "stock_splits"})
    if "adj_close" not in data and "close" in data:
        data["adj_close"] = data["close"]
    missing = set(REQUIRED_PRICE_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required price columns: {', '.join(sorted(missing))}")
    idx = pd.to_datetime(data.index, errors="coerce")
    if idx.isna().any():
        raise ValueError("Price data contains invalid dates")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("Asia/Seoul").tz_localize(None)
    data.index = idx.normalize()
    data.index.name = "date"
    data = data.sort_index()
    if data.index.has_duplicates:
        raise ValueError("Price data contains duplicate dates")
    for col in REQUIRED_PRICE_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    if data[list(REQUIRED_PRICE_COLUMNS)].isna().any().any():
        raise ValueError("Price data contains missing or non-numeric required values")
    if (data[["open", "high", "low", "close", "adj_close"]] <= 0).any().any():
        raise ValueError("Price data contains non-positive prices")
    if (data["volume"] < 0).any():
        raise ValueError("Price data contains negative volume")
    # Preserve provider OHLC verbatim, but make internally inconsistent bars
    # explicit. Adjusted-close return research can continue; OHLC-derived
    # estimators must filter these rows or resolve them against another source.
    bad_range = ((data["high"] < data[["open", "close", "low"]].max(axis=1)) |
                 (data["low"] > data[["open", "close", "high"]].min(axis=1)))
    for col in ("dividends", "stock_splits"):
        if col not in data:
            data[col] = 0.0
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)
    out = data.loc[:, [*REQUIRED_PRICE_COLUMNS, "dividends", "stock_splits"]].copy()
    out.insert(0, "ticker", ticker)
    out["ohlc_quality_flag"] = bad_range.map({True: "ohlc_range_inconsistent", False: "ok"}).to_numpy()
    out["source"] = source
    out["retrieval_timestamp"] = pd.Timestamp.now(tz="UTC")
    return out.reset_index()
