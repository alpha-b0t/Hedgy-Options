"""HAR-RV forecast using trailing realized volatility at several scales."""
from __future__ import annotations

import numpy as np
import pandas as pd

HAR_FEATURES = ("rv_1", "rv_5", "rv_20", "rv_60")


def fit_har_rv(train: pd.DataFrame, target_column: str) -> np.ndarray:
    """Fit an OLS HAR-RV model and return coefficients including intercept.

    This transparent baseline regresses forward realized volatility on its
    daily, weekly, monthly, and quarterly trailing volatility measures. The
    caller is responsible for chronological splitting and label purging.
    """
    required = [*HAR_FEATURES, target_column]
    sample = train.loc[:, required].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) <= len(HAR_FEATURES):
        raise ValueError("Insufficient complete observations to fit HAR-RV")
    x = np.column_stack([np.ones(len(sample)), sample.loc[:, HAR_FEATURES].to_numpy(float)])
    y = sample[target_column].to_numpy(float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return beta


def predict_har_rv(frame: pd.DataFrame, coefficients: np.ndarray) -> np.ndarray:
    """Predict forward RV from trailing RV features."""
    x = np.column_stack([np.ones(len(frame)), frame.loc[:, HAR_FEATURES].to_numpy(float)])
    return x @ coefficients
