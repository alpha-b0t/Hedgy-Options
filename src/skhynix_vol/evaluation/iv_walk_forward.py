"""Leakage-safe one-step-ahead evaluation of implied-volatility surfaces."""
from __future__ import annotations

import numpy as np
import pandas as pd

from skhynix_vol.features.implied import forecast_next_iv_surface


def walk_forward_iv(surface: pd.DataFrame, min_training_dates: int = 8,
                    min_cell_coverage: float = 0.60,
                    min_row_coverage: float = 0.30,
                    n_components: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare persistence, training mean, and PCA AR(1) factor-change forecasts.

    Each forecast uses surfaces strictly earlier than the target observation.
    PCA, cell selection, and imputation are refit using only that training
    window. Unsupported target cells are omitted instead of extrapolated.
    """
    required = {"date", "log_forward_moneyness", "tenor_days", "implied_volatility"}
    if not required.issubset(surface.columns):
        raise ValueError("surface must contain date, moneyness, tenor, and IV columns")
    if min_training_dates < 8:
        raise ValueError("min_training_dates must be at least 8 for the PCA factor-change model")
    data = surface.copy()
    data["date"] = pd.to_datetime(data.date).dt.normalize()
    if data.date.isna().any():
        raise ValueError("Surface contains invalid dates")
    dates = sorted(data.date.unique())
    prediction_frames, metric_rows = [], []
    for date_index in range(min_training_dates, len(dates)):
        target_date = pd.Timestamp(dates[date_index])
        train = data[data.date < target_date]
        actual = data[data.date.eq(target_date)][["log_forward_moneyness", "tenor_days", "implied_volatility"]]
        if actual.empty:
            continue
        try:
            pca = forecast_next_iv_surface(train, n_components=n_components,
                min_cell_coverage=min_cell_coverage, min_row_coverage=min_row_coverage)
        except ValueError:
            continue
        active_keys = pca[["log_forward_moneyness", "tenor_days"]]
        training_grid = train.pivot_table(index="date", columns=["log_forward_moneyness", "tenor_days"],
            values="implied_volatility", aggfunc="mean").sort_index()
        coverage = training_grid.notna().mean(axis=0)
        active = coverage[coverage >= min_cell_coverage].index
        training_grid = training_grid.loc[:, active]
        training_grid = training_grid.loc[training_grid.notna().mean(axis=1) >= min_row_coverage]
        means = training_grid.mean(axis=0)
        last_values = training_grid.ffill().iloc[-1].fillna(means)
        baseline = pd.DataFrame({"log_forward_moneyness": [key[0] for key in active],
            "tenor_days": [int(key[1]) for key in active], "historical_mean": means.to_numpy(float),
            "persistence": last_values.to_numpy(float)})
        # Retain exactly the active keys represented by the PCA training fit.
        baseline = active_keys.merge(baseline, on=["log_forward_moneyness", "tenor_days"], how="inner")
        model_predictions = {
            "PCA_AR1_FACTOR_CHANGES": pca.rename(columns={"implied_volatility_forecast": "forecast"})[
                ["log_forward_moneyness", "tenor_days", "forecast"]],
            "persistence": baseline.rename(columns={"persistence": "forecast"})[
                ["log_forward_moneyness", "tenor_days", "forecast"]],
            "historical_mean": baseline.rename(columns={"historical_mean": "forecast"})[
                ["log_forward_moneyness", "tenor_days", "forecast"]],
        }
        for model, prediction in model_predictions.items():
            scored = prediction.merge(actual, on=["log_forward_moneyness", "tenor_days"], how="inner")
            if scored.empty:
                continue
            scored = scored.rename(columns={"implied_volatility": "actual_iv"})
            scored["date"] = target_date
            scored["as_of_date"] = pd.Timestamp(train.date.max())
            scored["model"] = model
            scored["n_training_dates"] = train.date.nunique()
            if "forward_method" in train:
                methods = train.forward_method.dropna().unique()
                scored["forward_method"] = methods[0] if len(methods) == 1 else "mixed"
            prediction_frames.append(scored)
            metric_rows.append({"date": target_date, "as_of_date": pd.Timestamp(train.date.max()),
                "model": model, "n_training_dates": train.date.nunique(), **_iv_metrics(scored)})
    if not prediction_frames:
        predictions = pd.DataFrame(columns=["date", "as_of_date", "model", "log_forward_moneyness",
            "tenor_days", "forecast", "actual_iv", "n_training_dates"])
    else:
        predictions = pd.concat(prediction_frames, ignore_index=True)
    return predictions, pd.DataFrame(metric_rows)


def _iv_metrics(scored: pd.DataFrame) -> dict[str, float | int]:
    error = scored.forecast.to_numpy(float) - scored.actual_iv.to_numpy(float)
    atm = scored[np.isclose(scored.log_forward_moneyness, 0.0, atol=0.03)]
    wide = scored.pivot_table(index="tenor_days", columns="log_forward_moneyness",
                              values=["forecast", "actual_iv"], aggfunc="mean")
    skew_errors = {"downside_skew_mae": np.nan, "upside_skew_mae": np.nan}
    if not wide.empty:
        k_values = np.array(sorted(scored.log_forward_moneyness.unique()), dtype=float)
        nearest = {target: float(k_values[np.argmin(np.abs(k_values - target))]) for target in (-0.25, 0.0, 0.25)}
        if all(abs(nearest[t] - t) <= 0.06 for t in nearest):
            try:
                actual_atm = wide[("actual_iv", nearest[0.0])]
                forecast_atm = wide[("forecast", nearest[0.0])]
                down_error = (wide[("forecast", nearest[-0.25])] - forecast_atm) - (
                    wide[("actual_iv", nearest[-0.25])] - actual_atm)
                down_error = down_error.abs().dropna()
                up_error = (wide[("forecast", nearest[0.25])] - forecast_atm) - (
                    wide[("actual_iv", nearest[0.25])] - actual_atm)
                up_error = up_error.abs().dropna()
                skew_errors = {"downside_skew_mae": float(down_error.mean()) if len(down_error) else np.nan,
                               "upside_skew_mae": float(up_error.mean()) if len(up_error) else np.nan}
            except KeyError:
                pass
    return {"n_grid_points": int(len(scored)), "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "atm_mae": float(np.mean(np.abs(atm.forecast - atm.actual_iv))) if len(atm) else np.nan,
        **skew_errors}
