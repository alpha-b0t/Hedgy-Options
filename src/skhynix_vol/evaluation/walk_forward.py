"""Expanding-window annual walk-forward evaluation with overlap purging."""
from __future__ import annotations

import numpy as np
import pandas as pd

from skhynix_vol.evaluation.metrics import forecast_metrics
from skhynix_vol.models.har import fit_har_rv, predict_har_rv


def walk_forward_rv(data: pd.DataFrame, horizons: tuple[int, ...] = (5, 20, 60),
                    min_train_years: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate rolling, EWMA, and HAR forecasts in expanding annual folds.

    Forecasts for each calendar-year test fold are fitted only on earlier
    observations. Training labels whose forward window touches the test period
    are purged using the horizon in trading-session rows. Scaling and all model
    fits are therefore confined to each training fold.
    """
    frame = data.copy()
    if "date" not in frame:
        raise ValueError("data must contain a date column")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame.date.isna().any() or frame.date.duplicated().any():
        raise ValueError("Evaluation data must have valid, unique dates")
    if min_train_years < 1 or any(h < 1 for h in horizons):
        raise ValueError("min_train_years and forecast horizons must be positive")
    years = sorted(frame.date.dt.year.unique())
    predictions: list[pd.DataFrame] = []
    for horizon in horizons:
        target = f"target_rv_{horizon}d"
        if target not in frame:
            raise ValueError(f"Missing target column {target}")
        for year in years:
            if year - years[0] < min_train_years:
                continue
            test_mask = frame.date.dt.year.eq(year)
            if not test_mask.any():
                continue
            test_start = int(np.flatnonzero(test_mask.to_numpy())[0])
            test = frame.loc[test_mask].copy()
            # A training label at row i uses returns through i+h. Require that
            # its endpoint precedes the first test row.
            train = frame.iloc[:max(0, test_start - horizon)].copy()
            train = train.dropna(subset=[target, "rv_1", "rv_5", "rv_20", "rv_60"])
            test = test.dropna(subset=[target, "rv_1", "rv_5", "rv_20", "rv_60"])
            if len(train) <= 4 or test.empty:
                continue
            beta = fit_har_rv(train, target)
            har_pred = np.maximum(predict_har_rv(test, beta), 0.0)
            out = pd.DataFrame({"date": test.date, "horizon": horizon,
                "actual_rv": test[target].to_numpy(),
                "rolling_rv": test[f"rv_{horizon}"].to_numpy(),
                "ewma_vol": test["ewma_vol_20"].to_numpy(),
                "har_rv": har_pred, "test_year": year,
                "train_start": train.date.min(), "train_end": train.date.max(),
                "n_train": len(train), "n_test": len(test),
                "har_intercept": beta[0], "har_beta_rv_1": beta[1],
                "har_beta_rv_5": beta[2], "har_beta_rv_20": beta[3],
                "har_beta_rv_60": beta[4]})
            predictions.append(out)
    if not predictions:
        empty = pd.DataFrame(columns=["date", "horizon", "actual_rv", "rolling_rv", "ewma_vol", "har_rv", "test_year"])
        return empty, pd.DataFrame(columns=["horizon", "model", "test_year", "n", "mae", "rmse", "qlike", "r2", "directional_accuracy"])
    pred = pd.concat(predictions, ignore_index=True)
    metric_rows = []
    for (horizon, year), group in pred.groupby(["horizon", "test_year"]):
        for model in ("rolling_rv", "ewma_vol", "har_rv"):
            metric_rows.append({"horizon": horizon, "model": model, "test_year": year,
                                "train_start": group.train_start.iloc[0],
                                "train_end": group.train_end.iloc[0],
                                "n_train": int(group.n_train.iloc[0]),
                                **forecast_metrics(group.actual_rv.to_numpy(), group[model].to_numpy())})
    return pred, pd.DataFrame(metric_rows)
