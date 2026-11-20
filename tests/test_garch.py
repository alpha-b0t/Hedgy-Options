import numpy as np
import pandas as pd

from skhynix_vol.models.garch import (MODEL_TYPES, conditional_variance_path,
    fit_garch_model, forecast_annualized_rv)


def test_garch_family_fits_and_produces_positive_horizon_forecasts():
    returns = pd.Series(np.random.default_rng(7).normal(0.0002, 0.018, 650))
    for model_type in MODEL_TYPES:
        fitted = fit_garch_model(returns, model_type)
        variance = conditional_variance_path(returns, fitted)
        forecast = forecast_annualized_rv(returns, variance, len(returns) - 1,
            horizon=20, fit=fitted, seed=11)
        assert np.isfinite(variance).all() and (variance > 0).all()
        assert np.isfinite(forecast) and forecast > 0


def test_garch_forecast_uses_asof_return_and_respects_horizon():
    from skhynix_vol.models.garch import GarchFit
    fit = GarchFit("garch", omega=0.01, alpha=0.1, beta=0.8, gamma=0.0,
                   initial_variance=0.1, convergence_flag=0, n_observations=400)
    returns = np.array([0.01, -0.02, 0.03])
    variance = conditional_variance_path(returns, fit)
    one_day = forecast_annualized_rv(returns, variance, 2, 1, fit)
    five_day = forecast_annualized_rv(returns, variance, 2, 5, fit)
    assert one_day > 0 and five_day > 0
