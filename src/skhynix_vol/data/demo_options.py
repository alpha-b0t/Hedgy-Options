"""Clearly synthetic option quotes for exercising the research pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd

from skhynix_vol.features.option_pricing import european_option_price


def generate_synthetic_prices(sessions: int = 3024, seed: int = 7,
                              start: str = "2014-01-02",
                              initial_price: float = 150_000.0) -> pd.DataFrame:
    """Generate a deterministic, regime-switching price path for demos only."""
    if sessions < 500 or initial_price <= 0:
        raise ValueError("sessions must be at least 500 and initial_price must be positive")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=sessions)
    # A persistent volatility state makes the exercise less degenerate than
    # constant-volatility Brownian motion while remaining reproducible.
    regime = np.zeros(sessions)
    shocks = rng.standard_normal(sessions)
    for i in range(1, sessions):
        regime[i] = 0.94 * regime[i - 1] + 0.34 * shocks[i - 1]
    daily_vol = np.clip(0.016 * np.exp(0.35 * regime), 0.006, 0.065)
    returns = 0.0002 + daily_vol * shocks
    close = initial_price * np.exp(np.cumsum(returns))
    overnight = rng.normal(0.0, 0.002, sessions)
    open_price = close * np.exp(-returns + overnight)
    intraday_range = np.abs(rng.normal(0.008, 0.004, sessions))
    high = np.maximum(open_price, close) * (1.0 + intraday_range)
    low = np.minimum(open_price, close) * (1.0 - intraday_range)
    volume = rng.lognormal(mean=15.0, sigma=0.45, size=sessions).astype("int64")
    return pd.DataFrame({
        "date": dates,
        "ticker": "SYNTHETIC_000660",
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "adj_close": close,
        "volume": volume,
        "dividends": 0.0,
        "stock_splits": 0.0,
        "ohlc_quality_flag": "synthetic",
        "source": "SYNTHETIC_DEMO_DO_NOT_USE_FOR_RESEARCH",
        "synthetic_data_flag": True,
    })


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
                        "volume": int(250 + 20 * date_index + 100 * abs(offset)),
                        "open_interest": int(1500 + 30 * date_index + 150 * abs(offset)),
                        "contract_multiplier": 10,
                        "data_origin": "SYNTHETIC_DEMO_DO_NOT_USE_FOR_RESEARCH",
                        "synthetic_data_flag": True,
                    })
    return pd.DataFrame(rows)
