"""Daily SK Hynix price downloads and durable storage."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from skhynix_vol.data.validation import normalize_prices


def download_prices(start: str = "2015-01-01", end: str | None = None,
                    ticker: str = "000660.KS") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download Yahoo Finance OHLCV and actions; return raw and normalized tables."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install project dependencies to use Yahoo Finance downloads") from exc
    end = end or pd.Timestamp.now(tz="Asia/Seoul").date().isoformat()
    try:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=False, actions=True,
                          progress=False)
    except Exception as exc:
        raise RuntimeError(f"Yahoo Finance download failed for {ticker}: {exc}") from exc
    if raw is None or raw.empty:
        raise RuntimeError(f"Yahoo Finance returned no observations for {ticker} ({start} to {end})")
    return raw, normalize_prices(raw, ticker=ticker, source="Yahoo Finance (yfinance)")


def save_price_data(raw: pd.DataFrame, normalized: pd.DataFrame,
                    raw_dir: str | Path = "data/raw",
                    processed_dir: str | Path = "data/processed") -> tuple[Path, Path]:
    """Persist original provider response and normalized schema as Parquet."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = Path(raw_dir) / f"prices_000660_KS_{stamp}.parquet"
    processed_path = Path(processed_dir) / "daily_prices.parquet"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(raw_path)
    normalized.to_parquet(processed_path, index=False)
    return raw_path, processed_path
