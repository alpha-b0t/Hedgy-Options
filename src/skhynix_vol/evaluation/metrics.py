"""Forecast metrics, including QLIKE on the variance scale."""
from __future__ import annotations

import numpy as np


def forecast_metrics(actual_vol: np.ndarray, predicted_vol: np.ndarray) -> dict[str, float]:
    """Calculate scale-aware forecast metrics; QLIKE is computed on variance.

    QLIKE = y/p - log(y/p) - 1, where y is realized variance and p is
    forecast variance. Non-positive forecasts are floored for numerical safety.
    """
    actual = np.asarray(actual_vol, dtype=float)
    forecast = np.asarray(predicted_vol, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(forecast)
    actual, forecast = actual[valid], forecast[valid]
    if not len(actual):
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "qlike": np.nan,
                "r2": np.nan, "directional_accuracy": np.nan}
    if (actual < 0).any():
        raise ValueError("Realized volatility observations cannot be negative")
    error = forecast - actual
    y_var = np.maximum(actual ** 2, 1e-12)
    p_var = np.maximum(forecast ** 2, 1e-12)
    qlike = y_var / p_var - np.log(y_var / p_var) - 1.0
    if len(actual) > 1 and np.var(actual) > 0:
        r2 = 1.0 - float(np.sum(error ** 2) / np.sum((actual - actual.mean()) ** 2))
        direction = float(np.mean(np.sign(np.diff(forecast)) == np.sign(np.diff(actual))))
    else:
        r2, direction = np.nan, np.nan
    return {"n": int(len(actual)), "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))), "qlike": float(np.mean(qlike)),
            "r2": r2, "directional_accuracy": direction}
