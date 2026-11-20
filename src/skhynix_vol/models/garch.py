"""GARCH, EGARCH, and GJR-GARCH estimation and horizon forecasting.

Model estimation uses the `arch` package with zero conditional mean and Gaussian
innovations. Returns are scaled to percent during estimation for numerical
stability. Forecasts are converted back to annualized decimal volatility.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MODEL_TYPES = ("garch", "egarch", "gjr_garch")


@dataclass(frozen=True)
class GarchFit:
    """Estimated parameters and initialization state for one volatility model."""
    model_type: str
    omega: float
    alpha: float
    beta: float
    gamma: float
    initial_variance: float
    convergence_flag: int
    n_observations: int


def fit_garch_model(returns: pd.Series, model_type: str) -> GarchFit:
    """Estimate a standard Gaussian GARCH-family model on daily decimal returns."""
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}")
    clean = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 250:
        raise ValueError("At least 250 daily returns are required to fit a GARCH model")
    try:
        from arch import arch_model
    except ImportError as exc:
        raise RuntimeError("Install the project dependencies, including arch, to fit GARCH models") from exc
    if model_type == "egarch":
        vol, p, o, q = "EGARCH", 1, 1, 1
    elif model_type == "gjr_garch":
        vol, p, o, q = "GARCH", 1, 1, 1
    else:
        vol, p, o, q = "GARCH", 1, 0, 1
    model = arch_model(clean * 100.0, mean="Zero", vol=vol, p=p, o=o, q=q,
                       dist="normal", rescale=False)
    starting_values = None
    if model_type == "egarch":
        sample_variance = float(np.var(clean.to_numpy(float) * 100.0))
        starting_values = np.array([
            0.10 * np.log(max(sample_variance, 1e-8)), 0.10, 0.0, 0.90
        ])
    result = model.fit(disp="off", show_warning=False, options={"maxiter": 1000},
                        starting_values=starting_values)
    params = result.params
    omega = float(params["omega"])
    alpha = float(params.get("alpha[1]", 0.0))
    beta = float(params.get("beta[1]", 0.0))
    gamma = float(params.get("gamma[1]", 0.0)) if model_type != "garch" else 0.0
    initial_variance = float(np.asarray(result.conditional_volatility)[0] ** 2)
    if not np.isfinite([omega, alpha, beta, gamma, initial_variance]).all() or initial_variance <= 0:
        raise RuntimeError(f"{model_type} fit returned invalid parameters")
    return GarchFit(model_type, omega, alpha, beta, gamma, initial_variance,
                    int(result.convergence_flag), len(clean))


def conditional_variance_path(returns: pd.Series | np.ndarray, fit: GarchFit) -> np.ndarray:
    """Filter percent-squared conditional variances through observed returns."""
    eps = np.asarray(pd.to_numeric(pd.Series(returns), errors="coerce"), dtype=float) * 100.0
    eps = np.nan_to_num(eps, nan=0.0, posinf=0.0, neginf=0.0)
    if len(eps) == 0:
        return np.array([], dtype=float)
    h = np.empty(len(eps), dtype=float)
    h[0] = fit.initial_variance
    for t in range(1, len(eps)):
        prev_h, prev_eps = max(h[t - 1], 1e-12), eps[t - 1]
        if fit.model_type == "egarch":
            z = prev_eps / np.sqrt(prev_h)
            log_h = (fit.omega + fit.beta * np.log(prev_h) +
                     fit.alpha * (abs(z) - np.sqrt(2.0 / np.pi)) + fit.gamma * z)
            h[t] = np.exp(np.clip(log_h, -30.0, 30.0))
        else:
            asymmetry = fit.gamma * (prev_eps < 0) if fit.model_type == "gjr_garch" else 0.0
            h[t] = fit.omega + (fit.alpha + asymmetry) * prev_eps ** 2 + fit.beta * prev_h
            h[t] = max(h[t], 1e-12)
    return h


def forecast_annualized_rv(returns: pd.Series | np.ndarray, variance_path: np.ndarray,
                           as_of_index: int, horizon: int, fit: GarchFit,
                           simulations: int = 512, seed: int = 42,
                           annualization_days: int = 252) -> float:
    """Forecast annualized forward RV from an as-of close.

    GARCH and GJR-GARCH use their analytic expected variance recursion.
    EGARCH uses seeded Monte Carlo because its multi-step variance expectation
    is not closed form. Return values are decimal annualized volatility.
    """
    if horizon < 1 or as_of_index < 0 or as_of_index >= len(variance_path):
        raise ValueError("Invalid horizon or as-of index")
    eps = np.asarray(pd.to_numeric(pd.Series(returns), errors="coerce"), dtype=float) * 100.0
    if as_of_index >= len(eps):
        raise ValueError("The as-of return is missing")
    prev_h = max(float(variance_path[as_of_index]), 1e-12)
    prev_eps = float(eps[as_of_index])
    if fit.model_type == "egarch":
        expected_variances = _egarch_expected_variances(prev_h, prev_eps, horizon, fit,
                                                        simulations=simulations, seed=seed)
    else:
        asymmetry = fit.gamma * (prev_eps < 0) if fit.model_type == "gjr_garch" else 0.0
        persistence = fit.alpha + fit.beta + (0.5 * fit.gamma if fit.model_type == "gjr_garch" else 0.0)
        next_variance = fit.omega + (fit.alpha + asymmetry) * prev_eps ** 2 + fit.beta * prev_h
        expected_variances = np.empty(horizon, dtype=float)
        expected_variances[0] = max(next_variance, 1e-12)
        for step in range(1, horizon):
            expected_variances[step] = max(fit.omega + persistence * expected_variances[step - 1], 1e-12)
    return float(np.sqrt(annualization_days / horizon * expected_variances.sum() / 10_000.0))


def _egarch_expected_variances(last_variance: float, last_return: float, horizon: int,
                               fit: GarchFit, simulations: int, seed: int) -> np.ndarray:
    if simulations < 100:
        raise ValueError("EGARCH forecasts require at least 100 simulation paths")
    rng = np.random.default_rng(seed)
    h_prev = np.full(simulations, max(last_variance, 1e-12), dtype=float)
    eps_prev = np.full(simulations, last_return, dtype=float)
    next_z = eps_prev / np.sqrt(h_prev)
    log_h = (fit.omega + fit.beta * np.log(h_prev) +
             fit.alpha * (np.abs(next_z) - np.sqrt(2.0 / np.pi)) + fit.gamma * next_z)
    h = np.exp(np.clip(log_h, -30.0, 30.0))
    means = np.empty(horizon, dtype=float)
    means[0] = float(h.mean())
    for step in range(1, horizon):
        z = rng.standard_normal(simulations)
        log_h = (fit.omega + fit.beta * np.log(h) +
                 fit.alpha * (np.abs(z) - np.sqrt(2.0 / np.pi)) + fit.gamma * z)
        h = np.exp(np.clip(log_h, -30.0, 30.0))
        means[step] = float(h.mean())
    return np.maximum(means, 1e-12)
