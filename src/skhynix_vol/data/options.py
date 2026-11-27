"""Import and normalize historical option quote files supplied by a provider."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_OPTION_COLUMNS = ("date", "underlying", "expiry", "option_type", "strike",
                           "bid", "ask", "underlying_close", "implied_volatility")


def normalize_option_quotes(frame: pd.DataFrame, iv_units: str = "decimal",
                            source: str = "user_file") -> pd.DataFrame:
    """Validate provider quotes; normalize IV to a decimal without fabricating data.

    `iv_units` must be explicitly `decimal` or `percent`. Forward prices remain
    missing unless supplied by the source; callers may request the documented
    spot/carry approximation later when constructing a surface.
    """
    if iv_units not in {"decimal", "percent"}:
        raise ValueError("iv_units must be 'decimal' or 'percent'")
    data = frame.copy()
    if data.empty:
        raise ValueError("Option quote file is empty")
    data.columns = [str(c).strip().lower().replace(" ", "_") for c in data.columns]
    aliases = {"expiration": "expiry", "expiration_date": "expiry", "type": "option_type",
               "iv": "implied_volatility", "spot": "underlying_close", "openinterest": "open_interest",
               "risk_free": "risk_free_rate", "forward": "forward_price"}
    data = data.rename(columns={c: aliases.get(c, c) for c in data.columns})
    missing = set(REQUIRED_OPTION_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Option quotes missing required columns: {', '.join(sorted(missing))}")
    for col in ("date", "expiry"):
        parsed = pd.to_datetime(data[col], errors="coerce")
        if parsed.dt.tz is not None:
            parsed = parsed.dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
        data[col] = parsed.dt.normalize()
    if data[["date", "expiry"]].isna().any().any():
        raise ValueError("Option quotes contain invalid observation or expiry dates")
    data["option_type"] = data.option_type.astype(str).str.lower().str.strip()
    data["option_type"] = data.option_type.replace({"c": "call", "p": "put"})
    if not data.option_type.isin(["call", "put"]).all():
        raise ValueError("option_type must contain call/put or C/P")
    numeric = ["strike", "bid", "ask", "underlying_close", "implied_volatility"]
    for col in numeric:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    if data[numeric].isna().any().any():
        raise ValueError("Required option quote fields contain missing or non-numeric values")
    if iv_units == "percent":
        data["implied_volatility"] /= 100.0
    if (data[["strike", "underlying_close"]] <= 0).any().any():
        raise ValueError("Strike and underlying_close must be positive")
    if (data[["bid", "ask"]] < 0).any().any():
        raise ValueError("Option bid and ask cannot be negative")
    if ((data.implied_volatility <= 0) | (data.implied_volatility > 5.0)).any():
        raise ValueError("Implied volatility must be in (0, 5]; if source IV is percent, pass iv_units='percent'")
    for optional in ("risk_free_rate", "dividend_yield"):
        if optional in data:
            data[optional] = pd.to_numeric(data[optional], errors="coerce")
    if "quote_timestamp" in data:
        data["quote_timestamp"] = pd.to_datetime(data.quote_timestamp, errors="coerce", utc=True)
    if (data.expiry <= data.date).any():
        raise ValueError("Option expiry must be after its observation date")
    data["quote_quality_flag"] = np.where(data.ask < data.bid, "crossed_market", "ok")
    data["dte"] = (data.expiry - data.date).dt.days
    data["tenor_years"] = data.dte / 365.0
    data["bid_ask_spread"] = data.ask - data.bid
    data["relative_spread"] = np.where((data.ask + data.bid) > 0,
        data.bid_ask_spread / ((data.ask + data.bid) / 2.0), np.nan)
    data["source"] = source
    data["retrieval_timestamp"] = pd.Timestamp.now(tz="UTC")
    for col, default in (("last", np.nan), ("volume", np.nan), ("open_interest", np.nan),
                         ("risk_free_rate", np.nan), ("dividend_yield", np.nan),
                         ("forward_price", np.nan), ("contract_multiplier", np.nan)):
        if col not in data:
            data[col] = default
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["log_forward_moneyness"] = np.nan
    data["iv_units"] = "decimal"
    return data.sort_values(["date", "expiry", "option_type", "strike"]).reset_index(drop=True)


def attach_krx_single_stock_option_spec(quotes: pd.DataFrame,
                                        underlying: str = "000660") -> pd.DataFrame:
    """Attach KRX SK hynix product facts and validate supplied multipliers."""
    from skhynix_vol.data.instruments import instrument_spec
    spec = instrument_spec(underlying)
    data = quotes.copy()
    if "underlying" in data:
        names = data.underlying.astype(str).str.lower()
        known = names.str.contains(r"000660|sk\s*hynix|^660$", regex=True)
        if not known.all():
            raise ValueError("KRX SK hynix specifications can only be attached to 000660 / SK hynix quotes")
    multiplier = pd.to_numeric(data.get("contract_multiplier", pd.Series(np.nan, index=data.index)), errors="coerce")
    conflict = multiplier.notna() & ~np.isclose(multiplier, float(spec["contract_multiplier"]))
    if conflict.any():
        raise ValueError("KRX SK hynix single-stock options use a 10-share contract multiplier; source conflicts found")
    data["contract_multiplier"] = multiplier.fillna(spec["contract_multiplier"])
    data["exchange"] = spec["exchange"]
    data["exchange_product"] = spec["product"]
    data["exercise_style"] = spec["exercise_style"]
    data["settlement_type"] = spec["settlement_type"]
    data["last_trading_day_rule"] = spec["last_trading_day"]
    data["final_settlement_lag_sessions"] = spec["final_settlement_lag_sessions"]
    data["exercise_settlement_price_rule"] = spec["exercise_settlement_price"]
    return data


def load_option_file(path: str | Path, iv_units: str = "decimal",
                     source: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load CSV/Parquet input and return untouched source data plus normalized quotes."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        raw = pd.read_csv(path)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        raw = pd.read_parquet(path)
    else:
        raise ValueError("Options input must be CSV or Parquet")
    source_label = source.strip() if source and source.strip() else f"file:{path.name}"
    digest_builder = hashlib.sha256()
    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest_builder.update(block)
    digest = digest_builder.hexdigest()
    normalized = normalize_option_quotes(raw, iv_units=iv_units, source=source_label)
    normalized["source_file"] = path.name
    normalized["source_sha256"] = digest
    return raw, normalized


def save_option_quotes(raw: pd.DataFrame, normalized: pd.DataFrame,
                       raw_dir: str | Path = "data/raw",
                       processed_dir: str | Path = "data/processed") -> tuple[Path, Path]:
    """Store provider input separately from normalized quotes as Parquet."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = Path(raw_dir) / f"option_quotes_{stamp}.parquet"
    processed_path = Path(processed_dir) / "option_quotes.parquet"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(raw_path, index=False)
    normalized.to_parquet(processed_path, index=False)
    return raw_path, processed_path
