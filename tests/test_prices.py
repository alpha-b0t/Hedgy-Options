import pandas as pd
import pytest

from skhynix_vol.data.validation import normalize_prices


def sample_prices():
    return pd.DataFrame({"Open": [10.0, 11.0], "High": [11.0, 12.0], "Low": [9.0, 10.0],
        "Close": [10.5, 11.5], "Adj Close": [10.5, 11.5], "Volume": [100, 120]},
        index=pd.to_datetime(["2024-01-02", "2024-01-04"]))


def test_normalizes_schema_and_preserves_observed_sessions():
    result = normalize_prices(sample_prices())
    assert result.date.tolist() == list(pd.to_datetime(["2024-01-02", "2024-01-04"]))
    assert result.ticker.eq("000660.KS").all()
    assert result.dividends.eq(0).all()
    assert {"source", "retrieval_timestamp"}.issubset(result.columns)


def test_flags_invalid_ohlc_without_changing_source_values():
    frame = sample_prices()
    frame.loc[frame.index[0], "High"] = 8
    result = normalize_prices(frame)
    assert result.loc[0, "ohlc_quality_flag"] == "ohlc_range_inconsistent"
    assert result.loc[0, "high"] == 8


def test_rejects_duplicate_dates():
    frame = sample_prices()
    frame.index = pd.to_datetime(["2024-01-02", "2024-01-02"])
    with pytest.raises(ValueError, match="duplicate"):
        normalize_prices(frame)
