"""Coverage and integrity summaries for imported option quote chains."""
from __future__ import annotations

import numpy as np
import pandas as pd


def option_chain_quality(quotes: pd.DataFrame) -> pd.DataFrame:
    """Summarize quote coverage and basic quality by observation date/expiry.

    The report is descriptive: it does not silently drop quotes or declare a
    chain suitable for trading. Spreads are reported as fractions of midpoint.
    """
    required = {"date", "expiry", "option_type", "strike", "bid", "ask"}
    missing = required - set(quotes.columns)
    if missing:
        raise ValueError(f"Quotes missing chain-quality fields: {', '.join(sorted(missing))}")
    data = quotes.copy()
    data["date"] = pd.to_datetime(data.date, errors="coerce").dt.normalize()
    data["expiry"] = pd.to_datetime(data.expiry, errors="coerce").dt.normalize()
    data["option_type"] = data.option_type.astype(str).str.lower().replace({"c": "call", "p": "put"})
    for field in ("strike", "bid", "ask"):
        data[field] = pd.to_numeric(data[field], errors="coerce")
    data["mid"] = (data.bid + data.ask) / 2
    data["relative_spread"] = np.where(data.mid > 0, (data.ask - data.bid) / data.mid, np.nan)
    data["crossed"] = data.ask < data.bid
    data["zero_bid"] = data.bid.eq(0)
    data["invalid_price"] = (data[["strike", "bid", "ask"]].isna().any(axis=1)
                              | data.strike.le(0) | data.bid.lt(0) | data.ask.lt(0))
    group_fields = ["date", "expiry"]
    rows = []
    for (date, expiry), group in data.dropna(subset=group_fields).groupby(group_fields, sort=True):
        calls = group[group.option_type.eq("call")]
        puts = group[group.option_type.eq("put")]
        call_strikes = set(calls.strike.dropna())
        put_strikes = set(puts.strike.dropna())
        matched = call_strikes & put_strikes
        row = {
            "date": date,
            "expiry": expiry,
            "dte": int((expiry - date).days),
            "quote_count": int(len(group)),
            "call_count": int(len(calls)),
            "put_count": int(len(puts)),
            "unique_strikes": int(group.strike.nunique()),
            "matched_call_put_strikes": int(len(matched)),
            "duplicate_contract_rows": int(group.duplicated(["option_type", "strike"], keep=False).sum()),
            "crossed_quote_rows": int(group.crossed.sum()),
            "zero_bid_rows": int(group.zero_bid.sum()),
            "invalid_price_rows": int(group.invalid_price.sum()),
            "median_relative_spread": float(group.relative_spread.replace([np.inf, -np.inf], np.nan).median()),
            "p90_relative_spread": float(group.relative_spread.replace([np.inf, -np.inf], np.nan).quantile(.9)),
        }
        if "forward_price" in group:
            row["quotes_with_forward"] = int(pd.to_numeric(group.forward_price, errors="coerce").notna().sum())
        if "risk_free_rate" in group:
            row["quotes_with_rate"] = int(pd.to_numeric(group.risk_free_rate, errors="coerce").notna().sum())
        if "quote_timestamp" in group:
            timestamps = pd.to_datetime(group.quote_timestamp, errors="coerce", utc=True).dropna()
            row["timestamped_quotes"] = int(len(timestamps))
            row["quote_time_span_seconds"] = float((timestamps.max() - timestamps.min()).total_seconds()) if len(timestamps) else np.nan
        for field in ("source", "source_file", "source_sha256", "synthetic_data_flag"):
            if field in group:
                values = group[field].dropna().astype(str).unique()
                row[field] = values[0] if len(values) == 1 else ("mixed" if len(values) else None)
        rows.append(row)
    return pd.DataFrame(rows)
