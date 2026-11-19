import numpy as np
import pandas as pd
import pytest

from skhynix_vol.evaluation.metrics import forecast_metrics
from skhynix_vol.evaluation.walk_forward import walk_forward_rv


def test_qlike_is_computed_from_variances():
    metrics = forecast_metrics(np.array([0.2]), np.array([0.1]))
    ratio = (0.2 ** 2) / (0.1 ** 2)
    assert metrics["qlike"] == pytest.approx(ratio - np.log(ratio) - 1)


def test_walk_forward_is_chronological_and_produces_all_baselines():
    dates = pd.bdate_range("2015-01-01", "2022-12-31")
    t = np.arange(len(dates))
    rv = 0.15 + 0.02 * np.sin(t / 30)
    data = pd.DataFrame({"date": dates, "rv_1": rv, "rv_5": rv * 1.01,
        "rv_20": rv * 1.02, "rv_60": rv * 1.03, "ewma_vol_20": rv * 1.02,
        "target_rv_5d": rv * 1.04, "target_rv_20d": rv * 1.05,
        "target_rv_60d": rv * 1.06})
    predictions, metrics = walk_forward_rv(data, min_train_years=5)
    assert not predictions.empty
    assert set(metrics.model) == {"rolling_rv", "ewma_vol", "har_rv"}
    assert predictions.groupby("horizon").test_year.min().min() >= 2020
    assert predictions.actual_rv.notna().all()
    first_fold = predictions[predictions.test_year.eq(2020)]
    for row in first_fold.itertuples():
        assert row.train_end < row.date
        assert row.n_train > 0
