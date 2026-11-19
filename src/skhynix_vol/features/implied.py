"""Coverage-aware implied-volatility surface construction and PCA forecasts."""
from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_MONEYNESS = np.round(np.linspace(-0.30, 0.30, 13), 2)
DEFAULT_TENOR_DAYS = np.array([7, 14, 30, 60, 90, 180, 365])


def build_iv_surface(quotes: pd.DataFrame, moneyness_grid: np.ndarray = DEFAULT_MONEYNESS,
                     tenor_grid_days: np.ndarray = DEFAULT_TENOR_DAYS,
                     spot_forward_approximation: bool = False) -> pd.DataFrame:
    """Interpolate daily quote points only inside observed strike/tenor support.

    IV is decimal. A forward supplied by the source is preferred. If absent,
    `spot_forward_approximation=True` explicitly uses F=S*exp((r-q)*T), with
    missing rate/dividend yield set to zero and marked in the output.
    """
    required = {"date", "expiry", "option_type", "strike", "underlying_close", "implied_volatility"}
    if not required.issubset(quotes.columns):
        raise ValueError(f"Quotes missing surface fields: {', '.join(sorted(required - set(quotes.columns)))}")
    q = quotes.copy()
    q["date"] = pd.to_datetime(q.date).dt.normalize()
    q["expiry"] = pd.to_datetime(q.expiry).dt.normalize()
    q["dte"] = (q.expiry - q.date).dt.days
    q["tenor_years"] = q.dte / 365.0
    supplied_forward = "forward_price" in q and q.forward_price.notna().any()
    if supplied_forward:
        q["forward_price_used"] = pd.to_numeric(q.forward_price, errors="coerce")
        forward_method = "source_forward"
    elif spot_forward_approximation:
        rate_source = q["risk_free_rate"] if "risk_free_rate" in q else pd.Series(0.0, index=q.index)
        div_source = q["dividend_yield"] if "dividend_yield" in q else pd.Series(0.0, index=q.index)
        rate = pd.to_numeric(rate_source, errors="coerce").fillna(0.0)
        div = pd.to_numeric(div_source, errors="coerce").fillna(0.0)
        q["forward_price_used"] = q.underlying_close * np.exp((rate - div) * q.tenor_years)
        forward_method = "spot_carry_approximation"
    else:
        raise ValueError("No forward_price supplied; explicitly enable the spot/carry approximation to proceed")
    q = q[(q.quote_quality_flag == "ok") if "quote_quality_flag" in q else np.ones(len(q), dtype=bool)].copy()
    q = q[(q.dte > 0) & (q.forward_price_used > 0) & (q.implied_volatility > 0)]
    q["log_forward_moneyness"] = np.log(q.strike / q.forward_price_used)
    k_grid, d_grid = np.asarray(moneyness_grid, float), np.asarray(tenor_grid_days, float)
    if not len(q):
        raise ValueError("No valid option quotes remain for surface construction")
    rows = []
    for date, daily in q.groupby("date"):
        coords = daily[["log_forward_moneyness", "dte"]].to_numpy(float)
        values = daily.implied_volatility.to_numpy(float)
        # Calls and puts may quote the same coordinate. Median avoids giving
        # one side extra weight and is robust to occasional crossed markets.
        points = pd.DataFrame({"k": coords[:, 0], "dte": coords[:, 1], "iv": values}).groupby(["k", "dte"], as_index=False).iv.median()
        coords = points[["k", "dte"]].to_numpy(float)
        if len(coords) < 3:
            continue
        iv_grid = _bracketed_bilinear(points, k_grid, d_grid)
        for i, k in enumerate(k_grid):
            for j, dte in enumerate(d_grid):
                iv = float(iv_grid[i, j])
                if np.isfinite(iv):
                    rows.append({"date": date, "log_forward_moneyness": float(k),
                        "tenor_days": int(dte), "tenor_years": float(dte / 365.0),
                        "implied_volatility": iv, "n_quote_points": int(len(coords)),
                        "forward_method": forward_method})
    surface = pd.DataFrame(rows)
    if surface.empty:
        raise ValueError("No grid points are supported inside observed quote coverage")
    return surface.sort_values(["date", "tenor_days", "log_forward_moneyness"]).reset_index(drop=True)


def _bracketed_bilinear(points: pd.DataFrame, k_grid: np.ndarray,
                        tenor_grid_days: np.ndarray) -> np.ndarray:
    """Interpolate along strike then tenor, returning NaN outside quote support."""
    if "dte" not in points:
        raise ValueError("Internal IV interpolation points require dte")
    surface = np.full((len(k_grid), len(tenor_grid_days)), np.nan, dtype=float)
    smiles = {}
    for tenor, group in points.groupby("dte"):
        group = group.sort_values("k")
        smiles[int(tenor)] = (group.k.to_numpy(float), group.iv.to_numpy(float))
    available = np.array(sorted(smiles), dtype=int)
    for j, target_tenor in enumerate(tenor_grid_days):
        lower_candidates = available[available <= target_tenor]
        upper_candidates = available[available >= target_tenor]
        if not len(lower_candidates) or not len(upper_candidates):
            continue
        low_t, high_t = int(lower_candidates[-1]), int(upper_candidates[0])
        low_k, low_iv = smiles[low_t]
        if low_t == high_t:
            supported = (k_grid >= low_k.min()) & (k_grid <= low_k.max())
            surface[supported, j] = np.interp(k_grid[supported], low_k, low_iv)
            continue
        high_k, high_iv = smiles[high_t]
        common_min, common_max = max(low_k.min(), high_k.min()), min(low_k.max(), high_k.max())
        supported = (k_grid >= common_min) & (k_grid <= common_max)
        if not supported.any() or high_t == low_t:
            continue
        iv_low = np.interp(k_grid[supported], low_k, low_iv)
        iv_high = np.interp(k_grid[supported], high_k, high_iv)
        weight = (target_tenor - low_t) / (high_t - low_t)
        surface[supported, j] = (1.0 - weight) * iv_low + weight * iv_high
    return surface


def forecast_next_iv_surface(surface: pd.DataFrame, n_components: int = 3,
                             min_cell_coverage: float = 0.60,
                             min_row_coverage: float = 0.30) -> pd.DataFrame:
    """Forecast next-session surface with training-only PCA plus VAR(1) factors.

    Only grid cells observed in at least `min_cell_coverage` of training days
    are retained. Missing training cells are mean-imputed using training-only
    means; cells outside that historical support remain absent from the output.
    """
    required = {"date", "log_forward_moneyness", "tenor_days", "implied_volatility"}
    if not required.issubset(surface.columns):
        raise ValueError("Surface data is missing required grid columns")
    data = surface.copy()
    data["date"] = pd.to_datetime(data.date)
    grid = data.pivot_table(index="date", columns=["log_forward_moneyness", "tenor_days"],
                            values="implied_volatility", aggfunc="mean").sort_index()
    coverage = grid.notna().mean(axis=0)
    active = coverage[coverage >= min_cell_coverage].index
    grid = grid.loc[:, active]
    row_mask = grid.notna().mean(axis=1) >= min_row_coverage
    grid = grid.loc[row_mask]
    if len(grid) < 8 or grid.shape[1] < 3:
        raise ValueError("Insufficient historical surface coverage for PCA forecast (need 8 dates and 3 supported cells)")
    means = grid.mean(axis=0)
    x = grid.fillna(means).to_numpy(float)
    center = x.mean(axis=0)
    centered = x - center
    u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    components_count = min(int(n_components), len(singular), grid.shape[1])
    if components_count < 1:
        raise ValueError("Unable to extract surface factors")
    components = vt[:components_count]
    scores = u[:, :components_count] * singular[:components_count]
    if len(scores) < 3:
        raise ValueError("At least three complete surface observations are required")
    design = np.column_stack([np.ones(len(scores) - 1), scores[:-1]])
    coefficients, *_ = np.linalg.lstsq(design, scores[1:], rcond=None)
    next_scores = np.r_[1.0, scores[-1]] @ coefficients
    prediction = center + next_scores @ components
    out = pd.DataFrame({"log_forward_moneyness": [c[0] for c in active],
        "tenor_days": [int(c[1]) for c in active], "implied_volatility_forecast": np.maximum(prediction, 0.0),
        "historical_cell_coverage": coverage.loc[active].to_numpy(float)})
    out["as_of_date"] = grid.index[-1]
    out["training_start"] = grid.index[0]
    out["training_end"] = grid.index[-1]
    out["n_training_dates"] = len(grid)
    out["n_components"] = components_count
    out["model"] = "PCA_VAR1"
    if "forward_method" in data:
        methods = data.forward_method.dropna().unique()
        out["forward_method"] = methods[0] if len(methods) == 1 else "mixed"
    return out
