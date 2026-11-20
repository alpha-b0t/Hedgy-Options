"""Expanding annual out-of-sample evaluation for GARCH-family models."""
from __future__ import annotations

import numpy as np
import pandas as pd

from skhynix_vol.evaluation.metrics import forecast_metrics
from skhynix_vol.models.garch import (MODEL_TYPES, conditional_variance_path,
    fit_garch_model, forecast_annualized_rv)


def walk_forward_garch(data: pd.DataFrame, horizons: tuple[int, ...] = (5, 20, 60),
                       min_train_years: int = 5, egarch_simulations: int = 512,
                       seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit each GARCH family once per annual fold and filter forward sequentially.

    Model parameters use only returns dated before the test year. During a test
    year, parameters remain fixed while conditional variances update from each
    newly observed return. EGARCH multi-step forecasts use seeded simulation.
    """
    frame = data.copy()
    if not {"date", "return_1d"}.issubset(frame.columns):
        raise ValueError("data must contain date and return_1d")
    frame["date"] = pd.to_datetime(frame.date)
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame.date.isna().any() or frame.date.duplicated().any():
        raise ValueError("Evaluation data must have valid, unique dates")
    years = sorted(frame.date.dt.year.unique())
    predictions = []
    return_full = frame.return_1d
    for year in years:
        if year - years[0] < min_train_years:
            continue
        test_mask = frame.date.dt.year.eq(year)
        test_start = int(np.flatnonzero(test_mask.to_numpy())[0])
        train_returns = frame.return_1d.iloc[:test_start].dropna()
        if len(train_returns) < 250:
            continue
        test_indices = np.flatnonzero(test_mask.to_numpy())
        for model_type in MODEL_TYPES:
            try:
                fitted = fit_garch_model(train_returns, model_type)
                variance = conditional_variance_path(return_full, fitted)
            except (ValueError, RuntimeError) as exc:
                # A failed optimizer is recorded by omission for this fold/model;
                # other models and years remain usable.
                continue
            for idx in test_indices:
                row = frame.iloc[idx]
                for horizon in horizons:
                    target = f"target_rv_{horizon}d"
                    if target not in frame:
                        raise ValueError(f"Missing target column {target}")
                    if pd.isna(row.get(target)) or pd.isna(row.return_1d):
                        continue
                    forecast = forecast_annualized_rv(return_full, variance, idx, horizon,
                        fitted, simulations=egarch_simulations,
                        seed=seed + idx * 31 + horizon)
                    predictions.append({"date": row.date, "horizon": horizon,
                        "model": model_type, "actual_rv": float(row[target]),
                        "forecast_rv": forecast, "test_year": year,
                        "training_start": frame.date.iloc[0],
                        "training_end": frame.date.iloc[test_start - 1],
                        "n_train": fitted.n_observations,
                        "convergence_flag": fitted.convergence_flag,
                        "omega": fitted.omega, "alpha": fitted.alpha,
                        "beta": fitted.beta, "gamma": fitted.gamma})
    pred = pd.DataFrame(predictions)
    if pred.empty:
        return pred, pd.DataFrame(columns=["horizon", "model", "test_year", "n", "mae", "rmse", "qlike", "r2", "directional_accuracy"])
    metrics = []
    for (horizon, model, year), group in pred.groupby(["horizon", "model", "test_year"]):
        metrics.append({"horizon": horizon, "model": model, "test_year": year,
                        "train_end": group.training_end.iloc[0], "n_train": group.n_train.iloc[0],
                        **forecast_metrics(group.actual_rv.to_numpy(), group.forecast_rv.to_numpy())})
    return pred, pd.DataFrame(metrics)


def forecast_latest_garch(data: pd.DataFrame, horizons: tuple[int, ...] = (5, 20, 60),
                          egarch_simulations: int = 512, seed: int = 42) -> pd.DataFrame:
    """Fit GARCH-family models on available returns and forecast from latest close."""
    frame = data.copy().sort_values("date").reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame.date)
    train = frame.return_1d.dropna()
    if len(train) < 250:
        raise ValueError("At least 250 daily returns are required for GARCH forecasts")
    rows = []
    for model_type in MODEL_TYPES:
        fitted = fit_garch_model(train, model_type)
        variance = conditional_variance_path(frame.return_1d, fitted)
        latest_index = len(frame) - 1
        for horizon in horizons:
            forecast = forecast_annualized_rv(frame.return_1d, variance, latest_index, horizon,
                fitted, simulations=egarch_simulations, seed=seed + horizon)
            rows.append({"as_of_date": frame.date.iloc[-1], "horizon_sessions": horizon,
                "model": model_type, "forecast_rv": forecast,
                "training_start": frame.loc[frame.return_1d.notna(), "date"].iloc[0],
                "training_end": frame.date.iloc[-1], "n_train": fitted.n_observations,
                "convergence_flag": fitted.convergence_flag, "omega": fitted.omega,
                "alpha": fitted.alpha, "beta": fitted.beta, "gamma": fitted.gamma})
    return pd.DataFrame(rows)
