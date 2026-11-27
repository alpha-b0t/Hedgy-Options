"""Leakage-safe LSTM forecasting and evaluation for IV surfaces."""
from __future__ import annotations

import numpy as np
import pandas as pd

from skhynix_vol.evaluation.iv_walk_forward import _iv_metrics
from skhynix_vol.models.iv_lstm import fit_predict_iv_lstm


def _training_grid(surface: pd.DataFrame, min_cell_coverage: float,
                   min_row_coverage: float) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    grid = surface.pivot_table(index="date",
        columns=["log_forward_moneyness", "tenor_days"],
        values="implied_volatility", aggfunc="mean").sort_index()
    coverage = grid.notna().mean(axis=0)
    active = coverage[coverage >= min_cell_coverage].index
    grid = grid.loc[:, active]
    row_mask = grid.notna().mean(axis=1) >= min_row_coverage
    grid = grid.loc[row_mask]
    means = grid.mean(axis=0)
    matrix = grid.fillna(means).to_numpy(float)
    return grid, coverage.loc[active], matrix


def walk_forward_iv_lstm(surface: pd.DataFrame, min_training_dates: int = 30,
                         sequence_length: int = 5, min_cell_coverage: float = 0.60,
                         min_row_coverage: float = 0.30,
                         max_epochs: int = 60, seed: int = 42
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-step walk-forward IV predictions from preceding surface snapshots.

    Cell selection, coverage filtering, and mean imputation are recalculated
    inside each training fold. Test surfaces never contribute to preprocessing
    or model fitting.
    """
    required = {"date", "log_forward_moneyness", "tenor_days", "implied_volatility"}
    if not required.issubset(surface.columns):
        raise ValueError(f"Surface missing fields: {', '.join(sorted(required - set(surface.columns)))}")
    if min_training_dates < sequence_length + 9:
        raise ValueError("min_training_dates must be at least sequence_length + 9")
    data = surface.copy()
    data["date"] = pd.to_datetime(data.date, errors="coerce").dt.normalize()
    if data.date.isna().any():
        raise ValueError("Surface contains invalid dates")
    dates = sorted(data.date.unique())
    prediction_frames, metric_rows = [], []
    for date_index in range(min_training_dates, len(dates)):
        target_date = pd.Timestamp(dates[date_index])
        train_data = data[data.date < target_date]
        grid, coverage, matrix = _training_grid(train_data, min_cell_coverage, min_row_coverage)
        if len(grid) < min_training_dates or matrix.shape[1] < 3:
            continue
        forecast = fit_predict_iv_lstm(matrix, sequence_length=sequence_length,
                                       max_epochs=max_epochs, seed=seed)
        if not np.isfinite(forecast).all():
            continue
        cells = pd.DataFrame({"log_forward_moneyness": [float(c[0]) for c in grid.columns],
            "tenor_days": [int(c[1]) for c in grid.columns],
            "forecast": forecast, "historical_cell_coverage": coverage.to_numpy(float)})
        actual = data[data.date.eq(target_date)][
            ["log_forward_moneyness", "tenor_days", "implied_volatility"]]
        scored = cells.merge(actual, on=["log_forward_moneyness", "tenor_days"], how="inner")
        if scored.empty:
            continue
        scored = scored.rename(columns={"implied_volatility": "actual_iv"})
        scored["date"] = target_date
        scored["as_of_date"] = pd.Timestamp(grid.index[-1])
        scored["model"] = "LSTM_SEQUENCE_TO_SURFACE"
        scored["n_training_dates"] = len(grid)
        scored["sequence_length"] = sequence_length
        if "forward_method" in train_data:
            methods = train_data.forward_method.dropna().unique()
            scored["forward_method"] = methods[0] if len(methods) == 1 else "mixed"
        for field in ("data_origin", "synthetic_data_flag"):
            if field in train_data:
                values = train_data[field].dropna().astype(str).unique()
                scored[field] = values[0] if len(values) == 1 else "mixed"
        metric = {"date": target_date, "as_of_date": pd.Timestamp(grid.index[-1]),
            "model": "LSTM_SEQUENCE_TO_SURFACE", "n_training_dates": len(grid),
            "sequence_length": sequence_length, **_iv_metrics(scored)}
        for field in ("data_origin", "synthetic_data_flag", "forward_method"):
            if field in scored:
                metric[field] = scored[field].iloc[0]
        prediction_frames.append(scored)
        metric_rows.append(metric)
    predictions = (pd.concat(prediction_frames, ignore_index=True) if prediction_frames else
        pd.DataFrame(columns=["date", "as_of_date", "model", "log_forward_moneyness",
            "tenor_days", "forecast", "actual_iv", "n_training_dates", "sequence_length"]))
    return predictions, pd.DataFrame(metric_rows)


def forecast_next_iv_lstm(surface: pd.DataFrame, sequence_length: int = 5,
                          min_cell_coverage: float = 0.60, min_row_coverage: float = 0.30,
                          max_epochs: int = 60, seed: int = 42) -> pd.DataFrame:
    """Fit on the supplied surface history and forecast its next observation."""
    required = {"date", "log_forward_moneyness", "tenor_days", "implied_volatility"}
    if not required.issubset(surface.columns):
        raise ValueError(f"Surface missing fields: {', '.join(sorted(required - set(surface.columns)))}")
    data = surface.copy()
    data["date"] = pd.to_datetime(data.date, errors="coerce").dt.normalize()
    if data.date.isna().any():
        raise ValueError("Surface contains invalid dates")
    grid, coverage, matrix = _training_grid(data, min_cell_coverage, min_row_coverage)
    if len(grid) < sequence_length + 9 or matrix.shape[1] < 3:
        raise ValueError("Insufficient surface history for the selected LSTM sequence length")
    forecast = fit_predict_iv_lstm(matrix, sequence_length=sequence_length,
                                   max_epochs=max_epochs, seed=seed)
    if not np.isfinite(forecast).all():
        raise ValueError("Insufficient sequence samples for IV LSTM forecast")
    result = pd.DataFrame({
        "log_forward_moneyness": [float(c[0]) for c in grid.columns],
        "tenor_days": [int(c[1]) for c in grid.columns],
        "implied_volatility_forecast": forecast,
        "historical_cell_coverage": coverage.to_numpy(float),
    })
    result["as_of_date"] = pd.Timestamp(grid.index[-1])
    result["n_training_dates"] = len(grid)
    result["sequence_length"] = sequence_length
    result["model"] = "LSTM_SEQUENCE_TO_SURFACE"
    if "forward_method" in data:
        methods = data.forward_method.dropna().unique()
        result["forward_method"] = methods[0] if len(methods) == 1 else "mixed"
    for field in ("data_origin", "synthetic_data_flag"):
        if field in data:
            values = data[field].dropna().astype(str).unique()
            result[field] = values[0] if len(values) == 1 else "mixed"
    return result
