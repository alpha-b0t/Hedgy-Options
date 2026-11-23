"""Clearly synthetic option quotes for exercising the research pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd

from skhynix_vol.features.option_pricing import european_option_price


def generate_synthetic_option_quotes(prices: pd.DataFrame, days: int = 65,
                                     underlying: str = "000660") -> pd.DataFrame:
    """Create labeled Black-76 quotes for demos; never suitable for research.

    Uses recent close prices only to give the generated quotes plausible scales.
    All premiums, IVs, forwards, and expiries are simulated. Expiries use a
    weekly calendar so several dates share contracts for interface checks.
    """
    if days < 10:
        raise ValueError("days must be at least 10 to demonstrate chronological IV evaluation")
    required = {"date"}
    if not required.issubset(prices.columns):
        raise ValueError("prices must include date and close (or adj_close)")
    price_column = "close" if "close" in prices else "adj_close" if "adj_close" in prices else None
    if price_column is None:
        raise ValueError("prices must include close or adj_close")
    history = prices[["date", price_column]].copy()
    history["date"] = pd.to_datetime(history.date, errors="coerce").dt.normalize()
    history[price_column] = pd.to_numeric(history[price_column], errors="coerce")
    history = history.dropna().sort_values("date").tail(days)
    if len(history) < 10 or (history[price_column] <= 0).any():
        raise ValueError("Need at least 10 rows of positive underlying closes")

    last_date = history.date.max()
    first_expiry = history.date.min() + pd.Timedelta(days=1)
    expiries = pd.date_range(first_expiry, last_date + pd.Timedelta(days=372), freq="W-THU")
    strike_offsets = np.linspace(-0.35, 0.35, 9)
    # Keep contract strikes fixed across observation dates so the quote-based
    # backtest can match the same contract at entry and exit.
    reference_spot = float(history[price_column].median())
    rate, dividend_yield = 0.03, 0.005
    rows: list[dict[str, object]] = []
    for date_index, observation in enumerate(history.itertuples(index=False)):
        date = pd.Timestamp(observation.date)
        spot = float(getattr(observation, price_column))
        active_expiries = expiries[(expiries > date) & (expiries <= date + pd.Timedelta(days=372))]
        for expiry in active_expiries:
            dte = int((expiry - date).days)
            tenor = dte / 365.0
            forward = spot * np.exp((rate - dividend_yield) * tenor)
            for offset in strike_offsets:
                strike = reference_spot * np.exp(float(offset))
                log_moneyness = np.log(strike / forward)
                # Smooth, mildly skewed, slowly changing synthetic smile.
                base_iv = (0.31 + 0.035 * np.sqrt(tenor) + 0.16 * max(-log_moneyness, 0.0)
                           + 0.11 * log_moneyness**2 + 0.018 * np.sin(date_index / 7.0))
                for option_type in ("call", "put"):
                    premium = european_option_price(forward, strike, tenor, rate,
                                                    base_iv, option_type)
                    half_spread = max(2.0, premium * 0.004)
                    rows.append({
                        "date": date,
                        "underlying": underlying,
                        "expiry": expiry,
                        "option_type": option_type,
                        "strike": strike,
                        "bid": max(0.0, premium - half_spread),
                        "ask": premium + half_spread,
                        "underlying_close": spot,
                        "forward_price": forward,
                        "risk_free_rate": rate,
                        "dividend_yield": dividend_yield,
                        "implied_volatility": base_iv,
                        "contract_multiplier": 10,
                        "data_origin": "SYNTHETIC_DEMO_DO_NOT_USE_FOR_RESEARCH",
                        "synthetic_data_flag": True,
                    })
    return pd.DataFrame(rows)
