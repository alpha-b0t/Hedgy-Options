"""European option quote checks and implied-volatility inversion."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.special import ndtr


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

    # Vectorized Black-76 inversion keeps full historical chains practical to
    # validate. Bounds and input checks match implied_volatility_from_price.
    n = len(out)
    as_numeric = lambda name: pd.to_numeric(out.get(name, pd.Series(np.nan, index=out.index)), errors="coerce").to_numpy(float)
    rate = as_numeric("risk_free_rate")
    forward = as_numeric("forward_price")
    spot = as_numeric("underlying_close")
    dividend = as_numeric("dividend_yield")
    strike = as_numeric("strike")
    tenor = out.tenor_years.to_numpy(float)
    mid = out.mid.to_numpy(float)
    kind = out.option_type.astype(str).str.lower().to_numpy()
    is_call = kind == "call"
    has_source_forward = np.isfinite(forward)
    can_derive_forward = (~has_source_forward & np.isfinite(spot) & np.isfinite(dividend)
                          & np.isfinite(rate) & np.isfinite(tenor))
    used_forward = forward.copy()
    used_forward[can_derive_forward] = spot[can_derive_forward] * np.exp(
        (rate[can_derive_forward] - dividend[can_derive_forward]) * tenor[can_derive_forward])
    has_carry = np.isfinite(rate) & (has_source_forward | can_derive_forward)
    quality_ok = (out.get("quote_quality_flag", pd.Series("ok", index=out.index))
                  .fillna("invalid_quote").eq("ok").to_numpy())
    finite_inputs = (np.isfinite(mid) & np.isfinite(used_forward) & np.isfinite(strike)
                     & np.isfinite(tenor) & np.isfinite(rate))
    discount = np.exp(-np.where(np.isfinite(rate), rate, 0.0) * np.where(np.isfinite(tenor), tenor, 0.0))
    intrinsic = discount * np.maximum(np.where(is_call, used_forward - strike, strike - used_forward), 0.0)
    upper = discount * np.where(is_call, used_forward, strike)
    in_bounds = (finite_inputs & (used_forward > 0) & (strike > 0) & (tenor > 0)
                 & (mid >= intrinsic - 1e-9) & (mid < upper))
    valid = quality_ok & has_carry & in_bounds & np.isin(kind, ["call", "put"])
    statuses = np.full(n, "missing_carry_inputs", dtype=object)
    statuses[~quality_ok] = "invalid_quote"
    statuses[quality_ok & has_carry & ~in_bounds] = "outside_model_bounds"
    calculated = np.full(n, np.nan)
    if valid.any():
        f, k, t, r, price, call = (used_forward[valid], strike[valid], tenor[valid],
                                   rate[valid], mid[valid], is_call[valid])
        low = np.full(len(f), 1e-6)
        high = np.full(len(f), 5.0)
        discount_valid = np.exp(-r * t)
        for _ in range(80):
            vol = (low + high) / 2.0
            root_t = np.sqrt(t)
            d1 = (np.log(f / k) + 0.5 * vol * vol * t) / (vol * root_t)
            d2 = d1 - vol * root_t
            call_price = discount_valid * (f * ndtr(d1) - k * ndtr(d2))
            put_price = discount_valid * (k * ndtr(-d2) - f * ndtr(-d1))
            model_price = np.where(call, call_price, put_price)
            below = model_price < price
            low = np.where(below, vol, low)
            high = np.where(below, high, vol)
        calculated[valid] = (low + high) / 2.0
    out["calculated_implied_volatility"] = calculated
    source_iv = as_numeric("implied_volatility")
    comparable = valid & np.isfinite(source_iv)
    errors = calculated - source_iv
    out["iv_consistency_error"] = np.where(comparable, errors, np.nan)
    statuses[valid & ~np.isfinite(source_iv)] = "calculated_source_iv_missing"
    statuses[comparable] = np.where(np.abs(errors[comparable]) <= iv_tolerance,
                                    "consistent", "iv_mismatch")
    out["iv_validation_status"] = statuses
    return out
