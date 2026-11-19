"""Realized-volatility features and forward targets."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_realized_dataset(prices: pd.DataFrame, annualization_days: int = 252,
                           ewma_lambda: float = 0.94) -> pd.DataFrame:
    """Create daily log returns, trailing volatility features, and forward labels.

    Rolling RV is the square root of the mean daily squared adjusted-close log
    returns, annualized by 252. EWMA variance follows v_t=lambda*v_(t-1)+
    (1-lambda)*r_t^2. The h-day target uses only returns t+1 through t+h.
    """
    if annualization_days <= 0 or not 0 < ewma_lambda < 1:
        raise ValueError("annualization_days must be positive and ewma_lambda must be in (0, 1)")
    data = prices.copy()
    if "date" in data.columns:
        data = data.set_index("date")
    price_col = "adj_close" if "adj_close" in data else "Adj Close" if "Adj Close" in data else None
    if price_col is None:
        raise ValueError("Prices must contain adj_close")
    close = pd.to_numeric(data[price_col], errors="coerce")
    if close.isna().any() or (close <= 0).any():
        raise ValueError("Adjusted closes must be positive numeric values")
    result = pd.DataFrame(index=data.index)
    result["return_1d"] = np.log(close / close.shift(1))
    result["abs_return_1d"] = result["return_1d"].abs()
    for window in (1, 5, 20, 60):
        result[f"rv_{window}"] = np.sqrt(result["return_1d"].pow(2).rolling(window, min_periods=window).mean() * annualization_days)
    ewma_var = result["return_1d"].pow(2).ewm(alpha=1.0 - ewma_lambda, adjust=False, min_periods=1).mean()
    result["ewma_vol_20"] = np.sqrt(ewma_var * annualization_days)
    if "volume" in data:
        volume_change = pd.to_numeric(data["volume"], errors="coerce").pct_change()
        result["volume_change_1d"] = volume_change.replace([np.inf, -np.inf], np.nan)
    else:
        result["volume_change_1d"] = np.nan
    result["drawdown_20"] = close / close.rolling(20, min_periods=20).max() - 1.0
    squared = result["return_1d"].pow(2)
    for horizon in (5, 20, 60):
        future_sum = squared.shift(-1).iloc[::-1].rolling(horizon, min_periods=horizon).sum().iloc[::-1]
        result[f"target_rv_{horizon}d"] = np.sqrt(future_sum * annualization_days / horizon)
    result.index.name = "date"
    return result.reset_index()
