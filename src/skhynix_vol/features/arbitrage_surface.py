"""Project IV-grid call prices onto static-arbitrage constraints."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, minimize

from skhynix_vol.features.option_pricing import (
    european_option_price, implied_volatility_from_price,
)


def project_iv_surface(surface: pd.DataFrame) -> pd.DataFrame:
    """Project each date's normalized call-price grid onto static constraints.

    Uses forward-normalized Black-76 calls (F=1, r=0). Constraints enforce
    decreasing and convex calls across strikes, intrinsic/upper price bounds,
    and nondecreasing normalized call prices across listed tenors at common
    moneyness. The result is a research diagnostic/projection; it does not
    model discrete dividends, American exercise, or quote uncertainty.
    """
    required = {"date", "log_forward_moneyness", "tenor_days", "implied_volatility"}
    if not required.issubset(surface.columns):
        raise ValueError(f"Surface missing fields: {', '.join(sorted(required - set(surface.columns)))}")
    data = surface.copy()
    data["date"] = pd.to_datetime(data.date, errors="coerce").dt.normalize()
    for column in ("log_forward_moneyness", "tenor_days", "implied_volatility"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["date", "log_forward_moneyness", "tenor_days", "implied_volatility"])
    rows = []
    for date, daily in data.groupby("date", sort=True):
        daily = daily.sort_values(["tenor_days", "log_forward_moneyness"]).reset_index(drop=True)
        moneyness = daily.log_forward_moneyness.to_numpy(float)
        tenor_days = daily.tenor_days.to_numpy(float)
        years = tenor_days / 365.0
        strikes = np.exp(moneyness)
        original_iv = daily.implied_volatility.to_numpy(float)
        if (years <= 0).any() or (original_iv <= 0).any():
            raise ValueError("Tenors and IV values must be positive")
        prices = np.array([european_option_price(1.0, k, t, 0.0, iv, "call")
                           for k, t, iv in zip(strikes, years, original_iv)])
        constraints = []
        groups = daily.groupby("tenor_days", sort=True).indices
        for positions in groups.values():
            positions = np.asarray(positions, dtype=int)
            positions = positions[np.argsort(moneyness[positions])]
            for left, right in zip(positions[:-1], positions[1:]):
                row = np.zeros(len(daily)); row[left] = 1.0; row[right] = -1.0
                constraints.append(row)
            for left, middle, right in zip(positions[:-2], positions[1:-1], positions[2:]):
                k0, k1, k2 = strikes[[left, middle, right]]
                row = np.zeros(len(daily))
                row[left] = 1.0 / (k1 - k0)
                row[middle] = -(1.0 / (k1 - k0) + 1.0 / (k2 - k1))
                row[right] = 1.0 / (k2 - k1)
                constraints.append(row)
        tenors = sorted(groups)
        for shorter, longer in zip(tenors[:-1], tenors[1:]):
            short_positions = groups[shorter]
            long_positions = groups[longer]
            short_lookup = {moneyness[i]: int(i) for i in short_positions}
            long_lookup = {moneyness[i]: int(i) for i in long_positions}
            for k in short_lookup.keys() & long_lookup.keys():
                row = np.zeros(len(daily))
                row[long_lookup[k]] = 1.0
                row[short_lookup[k]] = -1.0
                constraints.append(row)
        matrix = np.vstack(constraints) if constraints else np.zeros((0, len(daily)))
        lower_intrinsic = np.maximum(1.0 - strikes, 0.0)
        objective = lambda values: 0.5 * float(np.sum((values - prices) ** 2))
        gradient = lambda values: values - prices
        result = minimize(objective, np.clip(prices, lower_intrinsic + 1e-10, 1.0 - 1e-10),
            jac=gradient, method="SLSQP", bounds=Bounds(lower_intrinsic, np.ones(len(daily))),
            constraints=[LinearConstraint(matrix, 0.0, np.inf)] if len(matrix) else [],
            options={"maxiter": 1000, "ftol": 1e-12})
        projected = result.x if result.success else prices
        projected = np.clip(projected, lower_intrinsic + 1e-10, 1.0 - 1e-10)
        projected_iv = np.array([implied_volatility_from_price(float(price), 1.0, float(k),
            float(t), 0.0, "call") for price, k, t in zip(projected, strikes, years)])
        for i, original in daily.iterrows():
            row = original.to_dict()
            row.update({"call_price_normalized_original": float(prices[i]),
                "call_price_normalized_projected": float(projected[i]),
                "implied_volatility_original": float(original_iv[i]),
                "implied_volatility_projected": float(projected_iv[i]),
                "absolute_iv_adjustment": float(abs(projected_iv[i] - original_iv[i])),
                "projection_status": "optimal" if result.success else "solver_failed_original_kept"})
            rows.append(row)
    return pd.DataFrame(rows)
