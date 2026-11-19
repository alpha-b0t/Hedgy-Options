import numpy as np
import pandas as pd
import pytest

from skhynix_vol.features.realized import build_realized_dataset


def prices_from_returns(returns):
    close = 100 * np.exp(np.r_[0.0, np.cumsum(returns)])
    return pd.DataFrame({"date": pd.date_range("2024-01-01", periods=len(close)), "adj_close": close})


def test_log_returns_and_one_day_realized_volatility():
    returns = np.array([0.01, -0.02, 0.03])
    result = build_realized_dataset(prices_from_returns(returns))
    np.testing.assert_allclose(result.return_1d.iloc[1:], returns)
    assert result.rv_1.iloc[1] == pytest.approx(abs(returns[0]) * np.sqrt(252))


def test_trailing_volatility_uses_only_current_and_prior_sessions():
    returns = np.array([0.01, 0.02, -0.01, 0.0, 0.03])
    result = build_realized_dataset(prices_from_returns(returns))
    expected = np.sqrt(np.mean(returns[-5:] ** 2) * 252)
    assert result.rv_5.iloc[-1] == pytest.approx(expected)


@pytest.mark.parametrize("horizon", [5, 20, 60])
def test_forward_target_alignment_and_trailing_missing_labels(horizon):
    returns = np.arange(1, horizon + 2, dtype=float) / 1000
    result = build_realized_dataset(prices_from_returns(returns))
    expected = np.sqrt(np.mean(returns[1:horizon + 1] ** 2) * 252)
    assert result[f"target_rv_{horizon}d"].iloc[1] == pytest.approx(expected)
    assert result[f"target_rv_{horizon}d"].iloc[-1:].isna().all()


def test_input_validation():
    with pytest.raises(ValueError, match="positive numeric"):
        build_realized_dataset(pd.DataFrame({"adj_close": [10.0, 0.0]}))


def test_zero_prior_volume_yields_missing_change_not_infinity():
    prices = prices_from_returns([0.01, -0.01])
    prices["volume"] = [0, 100, 150]
    result = build_realized_dataset(prices)
    assert np.isnan(result.volume_change_1d.iloc[1])
    assert np.isfinite(result.volume_change_1d.iloc[2])
