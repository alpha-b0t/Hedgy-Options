"""Expanding-window RV evaluation for machine-learning model baselines."""
from __future__ import annotations

import numpy as np
import pandas as pd

from skhynix_vol.evaluation.metrics import forecast_metrics
from skhynix_vol.models.ml import ML_FEATURES, fit_predict_ml, make_ml_models


def walk_forward_ml(data: pd.DataFrame, horizons: tuple[int, ...] = (5, 20, 60),
                    min_train_years: int = 5, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate RF, histogram boosting, SVR, and available optional boosters.

    Each calendar-year fold uses earlier dates only. Training rows are purged
    by each target's forward horizon, and every model/preprocessor is fit inside
    the fold. Hyperparameters are fixed in advance; no random split or tuning on
    the reported test folds is performed.
    """
    frame = data.copy()
    if "date" not in frame:
        raise ValueError("data must contain date")
    frame["date"] = pd.to_datetime(frame.date)
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame.date.isna().any() or frame.date.duplicated().any():
        raise ValueError("Evaluation data must have valid, unique dates")
    missing_features = set(ML_FEATURES) - set(frame.columns)
    if missing_features:
        raise ValueError(f"Missing ML features: {', '.join(sorted(missing_features))}")
    years = sorted(frame.date.dt.year.unique())
    rows = []
    for horizon in horizons:
        target = f"target_rv_{horizon}d"
        if target not in frame:
            raise ValueError(f"Missing target column {target}")
        for year in years:
            if year - years[0] < min_train_years:
                continue
            mask = frame.date.dt.year.eq(year)
            test_start = int(np.flatnonzero(mask.to_numpy())[0])
            train = frame.iloc[:max(0, test_start - horizon)].copy()
            test = frame.loc[mask].copy()
            train = train.dropna(subset=[target, *ML_FEATURES])
            test = test.dropna(subset=[target, *ML_FEATURES])
            if len(train) < 50 or test.empty:
                continue
            forecasts = fit_predict_ml(train, test, target, seed=seed)
            for model_name, values in forecasts.items():
                if len(values) != len(test):
                    continue
                for date, actual, predicted in zip(test.date, test[target], values):
                    rows.append({"date": date, "horizon": horizon, "model": model_name,
                        "actual_rv": float(actual), "forecast_rv": float(predicted),
                        "test_year": year, "training_start": train.date.min(),
                        "training_end": train.date.max(), "n_train": len(train),
                        "seed": seed})
    pred = pd.DataFrame(rows)
    if pred.empty:
        return pred, pd.DataFrame(columns=["horizon", "model", "test_year", "n", "mae", "rmse", "qlike", "r2", "directional_accuracy"])
    metrics = []
    for (horizon, model, year), group in pred.groupby(["horizon", "model", "test_year"]):
        metrics.append({"horizon": horizon, "model": model, "test_year": year,
            "training_start": group.training_start.iloc[0], "training_end": group.training_end.iloc[0],
            "n_train": int(group.n_train.iloc[0]), "seed": seed,
            **forecast_metrics(group.actual_rv.to_numpy(), group.forecast_rv.to_numpy())})
    return pred, pd.DataFrame(metrics)


def forecast_latest_ml(data: pd.DataFrame, horizons: tuple[int, ...] = (5, 20, 60),
                       seed: int = 42) -> pd.DataFrame:
    """Fit each ML model on complete historical labels and forecast the latest close."""
    frame = data.copy()
    frame["date"] = pd.to_datetime(frame.date)
    frame = frame.sort_values("date").reset_index(drop=True)
    latest = frame.iloc[[-1]].copy()
    if latest[list(ML_FEATURES)].isna().to_numpy().any():
        raise ValueError("Latest row is missing one or more ML features")
    rows = []
    model_names = list(make_ml_models(seed))
    for horizon in horizons:
        target = f"target_rv_{horizon}d"
        train = frame.dropna(subset=[target, *ML_FEATURES])
        forecasts = fit_predict_ml(train, latest, target, seed=seed)
        for model_name in model_names:
            if model_name not in forecasts:
                continue
            rows.append({"as_of_date": latest.date.iloc[0], "horizon_sessions": horizon,
                "model": model_name, "forecast_rv": float(forecasts[model_name][0]),
                "training_start": train.date.min(), "training_end": train.date.max(),
                "n_train": len(train), "seed": seed})
    return pd.DataFrame(rows)
