"""Quote-level static-arbitrage diagnostics for European option snapshots."""
from __future__ import annotations

import numpy as np
import pandas as pd


def static_arbitrage_diagnostics(quotes: pd.DataFrame) -> pd.DataFrame:
    """Summarize vertical monotonicity, call convexity, and parity checks.

    Violations are flagged from quoted bid/ask intervals for call monotonicity
    and put-call parity. Convexity uses midpoint call prices across ordered
    strikes. This is a diagnostic, not an arbitrage-free surface projection.
    """
    required = {"date", "expiry", "option_type", "strike", "bid", "ask"}
    missing = required - set(quotes.columns)
    if missing:
        raise ValueError(f"Quotes missing static-arbitrage fields: {', '.join(sorted(missing))}")
    q = quotes.copy()
    q["date"] = pd.to_datetime(q.date, errors="coerce").dt.normalize()
    q["expiry"] = pd.to_datetime(q.expiry, errors="coerce").dt.normalize()
    q["option_type"] = q.option_type.astype(str).str.lower().replace({"c": "call", "p": "put"})
    for column in ("strike", "bid", "ask"):
        q[column] = pd.to_numeric(q[column], errors="coerce")
    q = q.dropna(subset=["date", "expiry", "strike", "bid", "ask"])
    if "quote_quality_flag" in q:
        q = q[q.quote_quality_flag.eq("ok")]
    group_fields = [c for c in ("date", "expiry") if c in q]
    rows = []
    for (date, expiry), group in q.groupby(group_fields, sort=True):
        calls = group[group.option_type.eq("call")].sort_values("strike").drop_duplicates("strike", keep="last")
        puts = group[group.option_type.eq("put")].drop_duplicates("strike", keep="last")
        monotonic_violations = 0
        convexity_violations = 0
        parity_violations = 0
        if len(calls) >= 2:
            c = calls[["strike", "bid", "ask"]].to_numpy(float)
            # A lower-strike call must be worth at least as much as a higher-
            # strike call. Flag when the higher-strike bid exceeds the lower
            # strike ask, allowing execution of a positive-cost-free reversal.
            monotonic_violations = int(np.sum(c[1:, 1] > c[:-1, 2]))
        if len(calls) >= 3:
            c = calls[["strike", "bid", "ask"]].copy()
            c["mid"] = (c.bid + c.ask) / 2.0
            strikes = c.strike.to_numpy(float)
            mids = c.mid.to_numpy(float)
            slopes = np.diff(mids) / np.diff(strikes)
            convexity_violations = int(np.sum(np.diff(slopes) < -1e-8))
        matched = calls.merge(puts, on="strike", suffixes=("_call", "_put"))
        if len(matched) and {"forward_price", "risk_free_rate"}.issubset(group.columns):
            forwards = pd.to_numeric(group.forward_price, errors="coerce").dropna()
            rates = pd.to_numeric(group.risk_free_rate, errors="coerce").dropna()
            years = (expiry - date).days / 365.0
            if len(forwards) and len(rates) and years > 0:
                forward, rate = float(forwards.median()), float(rates.median())
                expected = np.exp(-rate * years) * (forward - matched.strike.to_numpy(float))
                call_mid = (matched.bid_call.to_numpy(float) + matched.ask_call.to_numpy(float)) / 2.0
                put_mid = (matched.bid_put.to_numpy(float) + matched.ask_put.to_numpy(float)) / 2.0
                allowance = ((matched.ask_call - matched.bid_call)
                             + (matched.ask_put - matched.bid_put)).to_numpy(float) / 2.0
                parity_violations = int(np.sum(np.abs(call_mid - put_mid - expected) > allowance + 1e-8))
        row = {"date": date, "expiry": expiry, "n_calls": len(calls), "n_puts": len(puts),
            "n_matched_strikes": len(matched),
            "vertical_monotonicity_violations": monotonic_violations,
            "call_convexity_violations": convexity_violations,
            "put_call_parity_violations": parity_violations}
        row["arbitrage_flag"] = any(row[name] > 0 for name in (
            "vertical_monotonicity_violations", "call_convexity_violations",
            "put_call_parity_violations"))
        for field in ("data_origin", "synthetic_data_flag"):
            if field in group:
                values = group[field].dropna().unique()
                row[field] = values[0] if len(values) == 1 else ("mixed" if len(values) else None)
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        columns = ["date", "expiry", "n_calls", "n_puts", "n_matched_strikes",
            "vertical_monotonicity_violations", "call_convexity_violations",
            "put_call_parity_violations", "arbitrage_flag"]
        columns.extend(field for field in ("data_origin", "synthetic_data_flag") if field in quotes)
        return pd.DataFrame(columns=columns)
    return result
