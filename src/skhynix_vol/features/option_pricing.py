"""European option quote checks and implied-volatility inversion."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def european_option_price(forward: float, strike: float, years: float, rate: float,
                          volatility: float, option_type: str) -> float:
    """Black-76 European option value per underlying share."""
    if min(forward, strike, years, volatility) <= 0 or not np.isfinite([forward, strike, years, rate, volatility]).all():
        raise ValueError("Forward, strike, tenor, and volatility must be finite and positive")
    root_t = math.sqrt(years)
    d1 = (math.log(forward / strike) + 0.5 * volatility * volatility * years) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    kind = option_type.lower()
    if kind == "call":
        undiscounted = forward * _cdf(d1) - strike * _cdf(d2)
    elif kind == "put":
        undiscounted = strike * _cdf(-d2) - forward * _cdf(-d1)
    else:
        raise ValueError("option_type must be call or put")
    return math.exp(-rate * years) * undiscounted


def implied_volatility_from_price(price: float, forward: float, strike: float,
                                 years: float, rate: float, option_type: str,
                                 tolerance: float = 1e-9) -> float:
    """Invert a European option premium by bisection; return decimal IV."""
    if not np.isfinite([price, forward, strike, years, rate]).all() or price < 0:
        raise ValueError("Option pricing inputs must be finite and price nonnegative")
    discount = math.exp(-rate * years)
    kind = option_type.lower()
    intrinsic = discount * max(forward - strike, 0.0) if kind == "call" else discount * max(strike - forward, 0.0)
    upper = discount * (forward if kind == "call" else strike)
    if kind not in {"call", "put"}:
        raise ValueError("option_type must be call or put")
    if price < intrinsic - tolerance or price >= upper:
        raise ValueError("Option premium violates European no-arbitrage bounds")
    low, high = 1e-6, 5.0
    if european_option_price(forward, strike, years, rate, high, kind) < price:
        raise ValueError("Premium implies volatility above the supported 500% cap")
    for _ in range(100):
        mid = (low + high) / 2.0
        model_price = european_option_price(forward, strike, years, rate, mid, kind)
        if abs(model_price - price) <= tolerance:
            return mid
        if model_price < price:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def validate_option_iv(quotes: pd.DataFrame, iv_tolerance: float = 0.05) -> pd.DataFrame:
    """Reprice quote midpoints and flag missing inputs, bounds, and IV mismatch.

    The inversion requires a source forward and risk-free rate. If forward is
    absent, it is derived from spot, rate, and dividend yield. The provider IV
    is preserved; calculated values and comparison flags are additive.
    """
    required = {"date", "expiry", "option_type", "strike", "bid", "ask", "underlying_close", "implied_volatility"}
    if not required.issubset(quotes.columns):
        raise ValueError(f"Quotes missing IV-validation fields: {', '.join(sorted(required - set(quotes.columns)))}")
    out = quotes.copy()
    out["date"] = pd.to_datetime(out.date, errors="coerce").dt.normalize()
    out["expiry"] = pd.to_datetime(out.expiry, errors="coerce").dt.normalize()
    out["tenor_years"] = (out.expiry - out.date).dt.days / 365.0
    out["mid"] = (pd.to_numeric(out.bid, errors="coerce") + pd.to_numeric(out.ask, errors="coerce")) / 2.0
    out["calculated_implied_volatility"] = np.nan
    out["iv_validation_status"] = "missing_carry_inputs"
    out["iv_consistency_error"] = np.nan
    for idx, row in out.iterrows():
        if row.get("quote_quality_flag", "ok") != "ok":
            out.at[idx, "iv_validation_status"] = "invalid_quote"
            continue
        rate = pd.to_numeric(pd.Series([row.get("risk_free_rate")]), errors="coerce").iloc[0]
        forward = pd.to_numeric(pd.Series([row.get("forward_price")]), errors="coerce").iloc[0]
        spot = pd.to_numeric(pd.Series([row.get("underlying_close")]), errors="coerce").iloc[0]
        dividend = pd.to_numeric(pd.Series([row.get("dividend_yield")]), errors="coerce").iloc[0]
        if not np.isfinite(rate):
            continue
        if not np.isfinite(forward):
            if not np.isfinite(spot) or not np.isfinite(dividend):
                continue
            forward = spot * np.exp((rate - dividend) * row.tenor_years)
        try:
            calc = implied_volatility_from_price(float(row.mid), float(forward), float(row.strike),
                float(row.tenor_years), float(rate), str(row.option_type))
        except (ValueError, TypeError, OverflowError):
            out.at[idx, "iv_validation_status"] = "outside_model_bounds"
            continue
        out.at[idx, "calculated_implied_volatility"] = calc
        source_iv = pd.to_numeric(pd.Series([row.implied_volatility]), errors="coerce").iloc[0]
        if np.isfinite(source_iv):
            err = calc - float(source_iv)
            out.at[idx, "iv_consistency_error"] = err
            out.at[idx, "iv_validation_status"] = "consistent" if abs(err) <= iv_tolerance else "iv_mismatch"
        else:
            out.at[idx, "iv_validation_status"] = "calculated_source_iv_missing"
    return out
