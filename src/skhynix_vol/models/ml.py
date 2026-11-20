"""Leakage-safe tree and kernel regressors for realized-volatility targets."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

ML_FEATURES = ("rv_1", "rv_5", "rv_20", "rv_60", "ewma_vol_20",
               "abs_return_1d", "volume_change_1d", "drawdown_20")


def make_ml_models(seed: int = 42) -> dict[str, Any]:
    """Create deterministic benchmark regressors; optional boosters join if installed."""
    try:
        from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
    except ImportError as exc:
        raise RuntimeError("Install scikit-learn to run the Phase 3 RV models") from exc
    models = {
        "random_forest": RandomForestRegressor(n_estimators=300, max_features=0.8,
            min_samples_leaf=5, random_state=seed, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=250,
            learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
            random_state=seed),
        "svr_rbf": make_pipeline(StandardScaler(), SVR(C=3.0, epsilon=0.02, kernel="rbf")),
    }
    try:
        from xgboost import XGBRegressor
        models["xgboost"] = XGBRegressor(n_estimators=300, max_depth=3,
            learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, objective="reg:squarederror", random_state=seed,
            n_jobs=-1, verbosity=0, tree_method="hist")
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor
        models["lightgbm"] = LGBMRegressor(n_estimators=300, max_depth=-1,
            num_leaves=15, learning_rate=0.03, reg_lambda=1.0,
            random_state=seed, verbosity=-1, n_jobs=-1)
    except ImportError:
        pass
    return models


def fit_predict_ml(train: pd.DataFrame, test: pd.DataFrame, target: str,
                   seed: int = 42) -> dict[str, np.ndarray]:
    """Fit fresh fold-local models and return nonnegative volatility forecasts."""
    train = train.replace([np.inf, -np.inf], np.nan).dropna(subset=[*ML_FEATURES, target])
    test = test.replace([np.inf, -np.inf], np.nan).dropna(subset=list(ML_FEATURES))
    if len(train) < 50 or test.empty:
        return {}
    x_train, y_train = train.loc[:, ML_FEATURES], train[target].to_numpy(float)
    x_test = test.loc[:, ML_FEATURES]
    output = {}
    for name, model in make_ml_models(seed).items():
        model.fit(x_train, y_train)
        output[name] = np.maximum(np.asarray(model.predict(x_test), dtype=float), 0.0)
    return output
