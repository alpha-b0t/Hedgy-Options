import numpy as np
import pandas as pd
import pytest

from skhynix_vol.data.options import normalize_option_quotes
from skhynix_vol.evaluation.walk_forward import forecast_latest_rv
from skhynix_vol.features.implied import build_iv_surface, forecast_next_iv_surface
from skhynix_vol.strategies.backtest import backtest_straddles
from skhynix_vol.visualization.iv_charts import plot_iv_surface


def test_latest_rv_forecast_is_forward_only_and_exportable():
    returns = np.random.default_rng(4).normal(0, 0.02, 450)
    close = 100 * np.exp(np.r_[0, np.cumsum(returns)])
    from skhynix_vol.features.realized import build_realized_dataset
    data = build_realized_dataset(pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=len(close)), "adj_close": close}))
    result = forecast_latest_rv(data)
    assert result.horizon_sessions.tolist() == [5, 20, 60]
    assert (result.training_end < result.as_of_date).all()
    assert result.har_rv_forecast.ge(0).all()


def surface_quotes(days=12):
    records = []
    dates = pd.bdate_range("2025-01-02", periods=days)
    for date_idx, date in enumerate(dates):
        for dte in (14, 30, 60, 90):
            for strike in (70, 85, 100, 115, 130):
                k = np.log(strike / 100)
                iv = 0.24 + 0.22 * k**2 + 0.01 * np.sqrt(dte / 365) + date_idx * 0.001
                records.append({"date": date, "underlying": "000660.KS", "expiry": date + pd.Timedelta(days=dte),
                    "option_type": "call", "strike": strike, "bid": 1.0, "ask": 1.1,
                    "underlying_close": 100.0, "implied_volatility": iv, "forward_price": 100.0,
                    "risk_free_rate": 0.0, "dividend_yield": 0.0, "contract_multiplier": 100})
    return normalize_option_quotes(pd.DataFrame(records), iv_units="decimal")


def test_surface_forecast_and_image_keep_supported_cells(tmp_path):
    quotes = surface_quotes()
    surface = build_iv_surface(quotes)
    assert surface.implied_volatility.notna().all()
    assert set(surface.forward_method) == {"source_forward"}
    assert surface.tenor_days.min() == 14 and surface.tenor_days.max() == 90
    forecast = forecast_next_iv_surface(surface)
    assert forecast.n_training_dates.nunique() == 1
    assert forecast.implied_volatility_forecast.ge(0).all()
    image = plot_iv_surface(forecast, tmp_path / "iv.png", value_column="implied_volatility_forecast")
    assert image.exists() and image.stat().st_size > 0


def test_surface_requires_explicit_forward_approximation():
    quotes = surface_quotes().drop(columns="forward_price")
    with pytest.raises(ValueError, match="explicitly enable"):
        build_iv_surface(quotes)
    assert not build_iv_surface(quotes, spot_forward_approximation=True).empty


def test_option_iv_percent_units_normalize_to_decimal():
    quotes = surface_quotes(days=1).head(1).copy()
    quotes["implied_volatility"] = 25.0
    normalized = normalize_option_quotes(quotes, iv_units="percent")
    assert normalized.implied_volatility.iloc[0] == pytest.approx(0.25)


def test_quote_backtest_applies_bid_ask_and_leg_fees():
    dates = pd.bdate_range("2025-01-02", periods=90)
    expiry = pd.Timestamp("2025-05-30")
    quote_rows, pred_rows = [], []
    for date in dates:
        if expiry > date:
            for side in ("call", "put"):
                quote_rows.append({"date": date, "expiry": expiry, "option_type": side, "strike": 100,
                    "bid": 1.0, "ask": 1.1, "implied_volatility": 0.2,
                    "underlying_close": 100, "contract_multiplier": 100, "quote_quality_flag": "ok"})
        pred_rows.append({"date": date, "horizon": 5, "har_rv": 0.4})
    quotes = pd.DataFrame(quote_rows)
    prices = pd.DataFrame({"date": dates, "close": 100.0})
    trades, summary = backtest_straddles(quotes, pd.DataFrame(pred_rows), prices, horizon=5,
        min_vol_edge=0.02, fee_per_contract=0.5)
    assert not trades.empty
    assert trades.side.eq("long").all()
    assert (trades.net_pnl_krw < trades.gross_pnl_krw).all()
    assert summary.trade_count.iloc[0] == len(trades)
    assert summary.total_fees_krw.iloc[0] == pytest.approx(len(trades) * 2.0)


def test_empty_backtest_keeps_export_schema():
    trades, summary = backtest_straddles(pd.DataFrame(columns=["date", "expiry", "option_type", "strike", "bid", "ask", "implied_volatility", "underlying_close"]),
        pd.DataFrame(columns=["date", "horizon", "har_rv"]), pd.DataFrame({"date": pd.to_datetime([])}), horizon=5)
    assert "net_pnl_krw" in trades
    assert summary.trade_count.iloc[0] == 0
