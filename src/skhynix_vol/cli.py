"""Command-line entry point for the initial research pipeline."""
import argparse
import json
from pathlib import Path

import pandas as pd

from skhynix_vol.data.prices import download_prices, save_price_data
from skhynix_vol.data.options import (load_option_file, save_option_quotes,
    attach_krx_single_stock_option_spec, normalize_option_quotes)
from skhynix_vol.data.demo_options import generate_synthetic_option_quotes
from skhynix_vol.data.option_quality import option_chain_quality
from skhynix_vol.data.instruments import instrument_spec
from skhynix_vol.evaluation.walk_forward import forecast_latest_rv, walk_forward_rv
from skhynix_vol.evaluation.iv_walk_forward import walk_forward_iv
from skhynix_vol.evaluation.demo_workflow import run_synthetic_research_demo
from skhynix_vol.evaluation.dl_iv_walk_forward import (
    walk_forward_iv_lstm, forecast_next_iv_lstm,
)
from skhynix_vol.evaluation.garch_walk_forward import forecast_latest_garch, walk_forward_garch
from skhynix_vol.evaluation.ml_walk_forward import forecast_latest_ml, walk_forward_ml
from skhynix_vol.models.lstm import lstm_available
from skhynix_vol.features.implied import (build_iv_surface, forecast_next_iv_surface,
    iv_surface_pca_factors)
from skhynix_vol.features.option_pricing import validate_option_iv
from skhynix_vol.features.arbitrage import static_arbitrage_diagnostics
from skhynix_vol.features.arbitrage_surface import project_iv_surface
from skhynix_vol.features.realized import build_realized_dataset
from skhynix_vol.strategies.backtest import backtest_straddles


def run(start: str = "2015-01-01", end: str | None = None) -> None:
    from skhynix_vol.visualization.charts import create_exploratory_charts
    raw, prices = download_prices(start=start, end=end)
    raw_path, price_path = save_price_data(raw, prices)
    features = build_realized_dataset(prices)
    target_columns = [col for col in features if col.startswith("target_rv_")]
    feature_path = "data/processed/rv_features.parquet"
    target_path = "data/processed/rv_targets.parquet"
    features.drop(columns=target_columns).to_parquet(feature_path, index=False)
    features.loc[:, ["date", *target_columns]].to_parquet(target_path, index=False)
    charts = create_exploratory_charts(prices, features)
    print(f"Retrieved {len(prices):,} sessions: {prices.date.min().date()} to {prices.date.max().date()}")
    flagged = prices.loc[prices.ohlc_quality_flag.ne("ok"), "date"]
    if len(flagged):
        dates = ", ".join(pd.to_datetime(flagged).dt.strftime("%Y-%m-%d"))
        print(f"Data quality: {len(flagged)} OHLC range inconsistencies flagged and retained: {dates}")
    print(f"Saved raw: {raw_path}\nSaved processed prices: {price_path}")
    print(f"Saved features: {feature_path}\nSaved targets: {target_path}")
    print("Charts: " + ", ".join(str(path) for path in charts))


def evaluate() -> None:
    """Run the expanding-window RV baselines from saved processed prices."""
    path = Path("data/processed/daily_prices.parquet")
    if not path.exists():
        raise FileNotFoundError("Run `skhynix-vol download` before `skhynix-vol evaluate`")
    prices = pd.read_parquet(path)
    features = build_realized_dataset(prices)
    predictions, metrics = walk_forward_rv(features)
    garch_predictions, garch_metrics = walk_forward_garch(features)
    out = Path("reports"); out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out / "rv_walk_forward_predictions.csv", index=False)
    metrics.to_csv(out / "rv_walk_forward_metrics.csv", index=False)
    garch_predictions.to_csv(out / "rv_garch_walk_forward_predictions.csv", index=False)
    garch_metrics.to_csv(out / "rv_garch_walk_forward_metrics.csv", index=False)
    metadata = {
        "ticker": "000660.KS",
        "source": "Yahoo Finance (yfinance), as recorded in processed daily prices",
        "observation_count": int(len(prices)),
        "data_start": str(pd.to_datetime(prices.date).min().date()) if len(prices) else None,
        "data_end": str(pd.to_datetime(prices.date).max().date()) if len(prices) else None,
        "feature_target_definition": "adjusted-close log returns; 252-session annualization; forward returns t+1..t+h",
        "evaluation": {"folds": "expanding calendar-year test folds", "min_train_years": 5,
                       "purge_sessions": [5, 20, 60], "models": ["rolling_rv", "ewma_vol", "har_rv"],
                       "har_features": ["rv_1", "rv_5", "rv_20", "rv_60"],
                       "ewma_lambda": 0.94, "garch_models": ["garch", "egarch", "gjr_garch"],
                       "metrics": ["mae", "rmse", "qlike_variance_scale", "r2", "directional_accuracy"]},
    }
    (out / "rv_walk_forward_run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if metrics.empty:
        print("No walk-forward folds available; check the price history length.")
    else:
        summary = metrics.groupby(["horizon", "model"])[["mae", "rmse", "qlike"]].mean()
        print(summary.to_string())
        if not garch_metrics.empty:
            print("\nGARCH family (annual-fold mean):")
            print(garch_metrics.groupby(["horizon", "model"])[["mae", "rmse", "qlike"]].mean().to_string())
        print("Saved annual fold predictions and metrics under reports/.")


def evaluate_ml() -> None:
    """Run fold-local ML models and save their out-of-sample forecasts."""
    path = Path("data/processed/daily_prices.parquet")
    if not path.exists():
        raise FileNotFoundError("Run `skhynix-vol download` before `skhynix-vol evaluate-ml`")
    features = build_realized_dataset(pd.read_parquet(path))
    predictions, metrics = walk_forward_ml(features)
    out = Path("reports"); out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out / "rv_ml_walk_forward_predictions.csv", index=False)
    metrics.to_csv(out / "rv_ml_walk_forward_metrics.csv", index=False)
    if not lstm_available():
        print("LSTM skipped; install the optional dependency with `python -m pip install -e '.[ml-deep]'`.")
    if metrics.empty:
        print("No ML walk-forward folds were available.")
    else:
        print(metrics.groupby(["horizon", "model"])[["mae", "rmse", "qlike"]].mean().to_string())
    print("Saved ML fold predictions and metrics under reports/.")


def forecast_rv() -> None:
    """Write latest-close 5/20/60 session RV forecasts."""
    path = Path("data/processed/daily_prices.parquet")
    if not path.exists():
        raise FileNotFoundError("Run `skhynix-vol download` before `skhynix-vol forecast-rv`")
    prices = pd.read_parquet(path)
    dataset = build_realized_dataset(prices)
    forecasts = forecast_latest_rv(dataset)
    garch_forecasts = forecast_latest_garch(dataset)
    out = Path("reports"); out.mkdir(parents=True, exist_ok=True)
    target = out / "rv_forecasts_latest.csv"
    forecasts.to_csv(target, index=False)
    garch_target = out / "rv_garch_forecasts_latest.csv"
    garch_forecasts.to_csv(garch_target, index=False)
    print(f"As-of {forecasts.as_of_date.iloc[0].date()}:\n{forecasts[['horizon_sessions','har_rv_forecast','rolling_rv_forecast','ewma_vol_forecast']].to_string(index=False)}")
    print(f"Saved forecasts to {target}")
    print("Latest GARCH-family forecasts:")
    print(garch_forecasts[["horizon_sessions", "model", "forecast_rv"]].to_string(index=False))
    print(f"Saved GARCH forecasts to {garch_target}")


def forecast_ml() -> None:
    path = Path("data/processed/daily_prices.parquet")
    if not path.exists():
        raise FileNotFoundError("Run `skhynix-vol download` before `skhynix-vol forecast-ml`")
    features = build_realized_dataset(pd.read_parquet(path))
    forecasts = forecast_latest_ml(features)
    if not lstm_available():
        print("LSTM skipped; install the optional dependency with `python -m pip install -e '.[ml-deep]'`.")
    out = Path("reports"); out.mkdir(parents=True, exist_ok=True)
    target = out / "rv_ml_forecasts_latest.csv"
    forecasts.to_csv(target, index=False)
    print(forecasts[["horizon_sessions", "model", "forecast_rv"]].to_string(index=False))
    print(f"Saved latest ML forecasts to {target}")


def import_options(path: str, iv_units: str, source: str | None = None) -> None:
    raw, normalized = load_option_file(path, iv_units=iv_units, source=source)
    if ("synthetic_data_flag" in normalized and normalized.synthetic_data_flag.fillna(False)
            .astype(str).str.lower().isin({"true", "1", "yes"}).any()):
        raise ValueError("Synthetic demo quotes cannot be imported into data/processed; use them directly with demo workflows")
    normalized = attach_krx_single_stock_option_spec(normalized)
    normalized = validate_option_iv(normalized)
    arbitrage_report = static_arbitrage_diagnostics(normalized)
    raw_path, processed_path = save_option_quotes(raw, normalized)
    out = Path("reports"); out.mkdir(parents=True, exist_ok=True)
    normalized.to_csv("reports/option_quote_iv_validation.csv", index=False)
    arbitrage_report.to_csv("reports/option_static_arbitrage_diagnostics.csv", index=False)
    option_chain_quality(normalized).to_csv(out / "option_chain_quality.csv", index=False)
    (out / "option_import_manifest.json").write_text(json.dumps({
        "source": normalized.source.iloc[0] if len(normalized) else source or f"file:{Path(path).name}",
        "source_file": Path(path).name,
        "source_sha256": normalized.source_sha256.iloc[0],
        "iv_units_input": iv_units,
        "quote_count": int(len(normalized)),
        "observation_start": str(normalized.date.min().date()) if len(normalized) else None,
        "observation_end": str(normalized.date.max().date()) if len(normalized) else None,
        "expiry_start": str(normalized.expiry.min().date()) if len(normalized) else None,
        "expiry_end": str(normalized.expiry.max().date()) if len(normalized) else None,
        "synthetic_data": bool(normalized.synthetic_data_flag.fillna(False).astype(str).str.lower()
                                .isin({"true", "1", "yes"}).any()) if "synthetic_data_flag" in normalized else False,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Imported {len(normalized):,} option quotes from {path}")
    print(f"Recorded source: {normalized.source.iloc[0]}")
    print(f"Observation dates: {normalized.date.min().date()} to {normalized.date.max().date()}")
    print(f"Crossed-market rows flagged: {(normalized.quote_quality_flag != 'ok').sum()}")
    print("IV validation statuses:")
    print(normalized.iv_validation_status.value_counts(dropna=False).to_string())
    spec = instrument_spec()
    print(f"Instrument: {spec['exchange']} {spec['product']}; European exercise; cash settled; multiplier {spec['contract_multiplier']} shares")
    print(f"Saved raw: {raw_path}\nSaved normalized: {processed_path}")


def generate_demo_options(prices_path: str, output_path: str, days: int) -> None:
    """Write explicitly synthetic option quotes outside processed market data."""
    source = Path(prices_path)
    if not source.exists():
        raise FileNotFoundError(f"Price history not found: {source}; run download first")
    prices = pd.read_parquet(source) if source.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(source)
    quotes = generate_synthetic_option_quotes(prices, days=days)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    quotes.to_csv(target, index=False)
    print(f"Generated {len(quotes):,} SYNTHETIC quote rows at {target}")
    print("These are simulated Black-76 premiums and IVs; do not use for model performance or trading claims.")


def run_demo_research(output_dir: str, reports_dir: str, sessions: int,
                      quote_days: int, seed: int) -> None:
    result = run_synthetic_research_demo(output_dir=output_dir, reports_dir=reports_dir,
        sessions=sessions, quote_days=quote_days, seed=seed)
    print("Completed offline SYNTHETIC research workflow (not market evidence).")
    for key, value in result.items():
        print(f"{key}: {value}")


def validate_options_file(path: str, iv_units: str, source: str | None = None) -> None:
    """Create quote-quality and provider-IV consistency diagnostics."""
    raw, normalized = load_option_file(path, iv_units=iv_units, source=source)
    normalized = attach_krx_single_stock_option_spec(normalized)
    report = validate_option_iv(normalized)
    arbitrage_report = static_arbitrage_diagnostics(normalized)
    synthetic = (normalized.synthetic_data_flag.fillna(False).astype(str).str.lower()
                 .isin({"true", "1", "yes"}).any()) if "synthetic_data_flag" in normalized else False
    suffix = "_synthetic_demo" if synthetic else ""
    out = Path("reports"); out.mkdir(parents=True, exist_ok=True)
    target = out / f"option_quote_iv_validation{suffix}.csv"
    report.to_csv(target, index=False)
    arb_target = out / f"option_static_arbitrage_diagnostics{suffix}.csv"
    arbitrage_report.to_csv(arb_target, index=False)
    quality_target = out / f"option_chain_quality{suffix}.csv"
    option_chain_quality(report).to_csv(quality_target, index=False)
    print(report.iv_validation_status.value_counts(dropna=False).to_string())
    print(f"Saved {len(report):,} quote diagnostics to {target}; raw source rows: {len(raw):,}")
    print(f"Saved chain coverage diagnostics to {quality_target}")
    print(f"Static-arbitrage flagged expiry groups: {int(arbitrage_report.arbitrage_flag.sum())}; saved to {arb_target}")


def make_iv_surface(quote_path: str, spot_forward_approximation: bool,
                    as_of: str | None = None, forecast: bool = False,
                    n_components: int = 3) -> None:
    from skhynix_vol.visualization.iv_charts import plot_iv_surface
    path = Path(quote_path)
    if not path.exists():
        raise FileNotFoundError(f"Options file not found: {path}; import provider quotes first")
    quotes = pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)
    if "quote_quality_flag" not in quotes:
        from skhynix_vol.data.options import normalize_option_quotes
        quotes = normalize_option_quotes(quotes, iv_units="decimal", source=f"file:{path.name}")
    synthetic = (quotes.synthetic_data_flag.fillna(False).astype(str).str.lower()
                 .isin({"true", "1", "yes"}).any()) if "synthetic_data_flag" in quotes else False
    suffix = "_synthetic_demo" if synthetic else ""
    surface = build_iv_surface(quotes, spot_forward_approximation=spot_forward_approximation)
    surface_path = Path("data/demo_synthetic/iv_surface.parquet") if synthetic else Path("data/processed/iv_surface.parquet")
    surface_path.parent.mkdir(parents=True, exist_ok=True)
    surface.to_parquet(surface_path, index=False)
    out = Path("reports"); out.mkdir(parents=True, exist_ok=True)
    if as_of is not None:
        as_date = pd.Timestamp(as_of).normalize()
        surface = surface[surface.date.eq(as_date)]
        if surface.empty:
            raise ValueError(f"No supported IV surface grid for {as_date.date()}")
    if forecast:
        prediction = forecast_next_iv_surface(surface, n_components=n_components)
        prediction_path = out / f"iv_surface_forecast{suffix}.csv"
        prediction.to_csv(prediction_path, index=False)
        factor_columns = [c for c in prediction if c.endswith("_change_forecast") or c.endswith("_score_asof") or c.endswith("_score_forecast")]
        factor_forecast = prediction[["as_of_date", "training_start", "training_end", "n_training_dates", "n_components", "model", *factor_columns]].drop_duplicates()
        factor_path = out / f"iv_surface_factor_change_forecast{suffix}.csv"
        factor_forecast.to_csv(factor_path, index=False)
        title = "SYNTHETIC DEMO — SK Hynix next-session IV surface forecast" if synthetic else "SK Hynix next-session IV surface forecast"
        image_path = plot_iv_surface(prediction, out / f"iv_surface_forecast{suffix}.png",
            value_column="implied_volatility_forecast", title=title)
        print(f"Saved {len(prediction)} forecast grid cells to {prediction_path}")
        print(f"Saved PCA factor change forecast to {factor_path}")
        print(f"Saved forecast image: {image_path}")
    else:
        title = "SYNTHETIC DEMO — SK Hynix implied-volatility surface" if synthetic else "SK Hynix implied-volatility surface"
        image_path = plot_iv_surface(surface, out / f"iv_surface{suffix}_{surface.date.max():%Y%m%d}.png", title=title)
        print(f"Saved supported surface points: {surface_path}")
        print(f"Saved image: {image_path}")


def evaluate_iv_surface(quote_path: str, spot_forward_approximation: bool,
                        n_components: int, min_training_dates: int) -> None:
    path = Path(quote_path)
    if not path.exists():
        raise FileNotFoundError(f"Options file not found: {path}; import provider quotes first")
    quotes = pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)
    if "quote_quality_flag" not in quotes:
        from skhynix_vol.data.options import normalize_option_quotes
        quotes = normalize_option_quotes(quotes, iv_units="decimal", source=f"file:{path.name}")
    synthetic = (quotes.synthetic_data_flag.fillna(False).astype(str).str.lower()
                 .isin({"true", "1", "yes"}).any()) if "synthetic_data_flag" in quotes else False
    suffix = "_synthetic_demo" if synthetic else ""
    surface = build_iv_surface(quotes, spot_forward_approximation=spot_forward_approximation)
    predictions, metrics = walk_forward_iv(surface, n_components=n_components,
                                            min_training_dates=min_training_dates)
    out = Path("reports"); out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out / f"iv_surface_walk_forward_predictions{suffix}.csv", index=False)
    metrics.to_csv(out / f"iv_surface_walk_forward_metrics{suffix}.csv", index=False)
    if metrics.empty:
        print("No IV walk-forward folds met the minimum coverage requirements.")
    else:
        summary_cols = ["mae", "rmse", "atm_mae", "downside_skew_mae", "upside_skew_mae"]
        print(metrics.groupby("model")[summary_cols].mean().to_string())
    print(f"Saved IV out-of-sample predictions and metrics under reports/ (synthetic={synthetic}).")


def evaluate_iv_surface_dl(quote_path: str, spot_forward_approximation: bool,
                           min_training_dates: int, sequence_length: int,
                           max_epochs: int, seed: int) -> None:
    """Evaluate the fold-local LSTM IV-surface forecaster."""
    path = Path(quote_path)
    if not path.exists():
        raise FileNotFoundError(f"Options file not found: {path}")
    quotes = pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)
    if "quote_quality_flag" not in quotes:
        from skhynix_vol.data.options import normalize_option_quotes
        quotes = normalize_option_quotes(quotes, iv_units="decimal", source=f"file:{path.name}")
    synthetic = (quotes.synthetic_data_flag.fillna(False).astype(str).str.lower()
                 .isin({"true", "1", "yes"}).any()) if "synthetic_data_flag" in quotes else False
    suffix = "_synthetic_demo" if synthetic else ""
    surface = build_iv_surface(quotes, spot_forward_approximation=spot_forward_approximation)
    predictions, metrics = walk_forward_iv_lstm(surface,
        min_training_dates=min_training_dates, sequence_length=sequence_length,
        max_epochs=max_epochs, seed=seed)
    out = Path("reports"); out.mkdir(parents=True, exist_ok=True)
    pred_path = out / f"iv_surface_lstm_walk_forward_predictions{suffix}.csv"
    metric_path = out / f"iv_surface_lstm_walk_forward_metrics{suffix}.csv"
    predictions.to_csv(pred_path, index=False)
    metrics.to_csv(metric_path, index=False)
    if metrics.empty:
        print("No LSTM IV folds met the historical coverage requirements.")
    else:
        print(metrics[["date", "n_training_dates", "n_grid_points", "mae", "rmse"]].to_string(index=False))
    print(f"Saved {len(predictions):,} LSTM IV predictions to {pred_path}")
    print(f"Saved LSTM IV metrics to {metric_path}")


def forecast_iv_surface_dl(quote_path: str, spot_forward_approximation: bool,
                           sequence_length: int, max_epochs: int, seed: int) -> None:
    """Fit an LSTM on all prior surfaces and export the next-session forecast."""
    from skhynix_vol.visualization.iv_charts import plot_iv_surface
    path = Path(quote_path)
    if not path.exists():
        raise FileNotFoundError(f"Options file not found: {path}")
    quotes = pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)
    if "quote_quality_flag" not in quotes:
        from skhynix_vol.data.options import normalize_option_quotes
        quotes = normalize_option_quotes(quotes, iv_units="decimal", source=f"file:{path.name}")
    synthetic = (quotes.synthetic_data_flag.fillna(False).astype(str).str.lower()
                 .isin({"true", "1", "yes"}).any()) if "synthetic_data_flag" in quotes else False
    suffix = "_synthetic_demo" if synthetic else ""
    surface = build_iv_surface(quotes, spot_forward_approximation=spot_forward_approximation)
    forecast = forecast_next_iv_lstm(surface, sequence_length=sequence_length,
                                     max_epochs=max_epochs, seed=seed)
    out = Path("reports"); out.mkdir(parents=True, exist_ok=True)
    target = out / f"iv_surface_lstm_forecast{suffix}.csv"
    forecast.to_csv(target, index=False)
    image = plot_iv_surface(forecast, out / f"iv_surface_lstm_forecast{suffix}.png",
        value_column="implied_volatility_forecast",
        title="SYNTHETIC DEMO — LSTM IV surface forecast" if synthetic
              else "SK Hynix LSTM IV surface forecast")
    print(f"Saved {len(forecast):,} forecast cells to {target}")
    print(f"Saved forecast image to {image}")


def project_iv(quote_path: str, spot_forward_approximation: bool) -> None:
    """Project supported call prices onto static-arbitrage constraints."""
    path = Path(quote_path)
    if not path.exists():
        raise FileNotFoundError(f"Options file not found: {path}")
    quotes = pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)
    if "quote_quality_flag" not in quotes:
        quotes = normalize_option_quotes(quotes, iv_units="decimal", source=f"file:{path.name}")
    synthetic = (quotes.synthetic_data_flag.fillna(False).astype(str).str.lower()
                 .isin({"true", "1", "yes"}).any()) if "synthetic_data_flag" in quotes else False
    suffix = "_synthetic_demo" if synthetic else ""
    surface = build_iv_surface(quotes, spot_forward_approximation=spot_forward_approximation)
    projected = project_iv_surface(surface)
    out = Path("reports"); out.mkdir(parents=True, exist_ok=True)
    target = out / f"iv_surface_arbitrage_projection{suffix}.csv"
    projected.to_csv(target, index=False)
    failed = projected.loc[projected.projection_status.ne("optimal"), "date"].nunique()
    print(f"Projected {len(projected):,} surface cells; solver-failed dates: {failed}; max IV adjustment: {projected.absolute_iv_adjustment.max():.6f}")
    print(f"Saved projected surface to {target}")


def export_iv_factors(quote_path: str, spot_forward_approximation: bool,
                      n_components: int) -> None:
    """Export historical PCA factor scores and surface loadings."""
    path = Path(quote_path)
    if not path.exists():
        raise FileNotFoundError(f"Options file not found: {path}; import provider quotes first")
    quotes = pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)
    if "quote_quality_flag" not in quotes:
        quotes = normalize_option_quotes(quotes, iv_units="decimal", source=f"file:{path.name}")
    synthetic = (quotes.synthetic_data_flag.fillna(False).astype(str).str.lower()
                 .isin({"true", "1", "yes"}).any()) if "synthetic_data_flag" in quotes else False
    suffix = "_synthetic_demo" if synthetic else ""
    surface = build_iv_surface(quotes, spot_forward_approximation=spot_forward_approximation)
    scores, loadings = iv_surface_pca_factors(surface, n_components=n_components)
    out = Path("reports"); out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / f"iv_surface_pca_scores{suffix}.csv", index=False)
    loadings.to_csv(out / f"iv_surface_pca_loadings{suffix}.csv", index=False)
    print(f"Saved PCA factors for {scores.date.nunique()} dates and {loadings.component.nunique()} components under reports/.")


def backtest(quotes_path: str, predictions_path: str, prices_path: str,
             horizons: list[int], min_vol_edge: float, fee_per_contract: float,
             contract_multiplier: float | None, hedge_mode: str = "none",
             option_slippage_bps: float = 0.0, hedge_slippage_bps: float = 0.0,
             hedge_fee_per_share: float = 0.0, max_relative_spread: float | None = None,
             min_option_volume: float = 0.0, min_open_interest: float = 0.0,
             hedge_risk_free_rate: float | None = None,
             hedge_dividend_yield: float | None = None, contracts: int = 1,
             max_volume_participation: float | None = None,
             max_open_interest_participation: float | None = None,
             max_cumulative_loss_krw: float | None = None,
             short_margin_per_contract_krw: float | None = None,
             capital_base_krw: float | None = None,
             annual_financing_rate: float = 0.0,
             annual_stock_borrow_rate: float = 0.0,
             max_cumulative_drawdown_krw: float | None = None,
             exercise_fee_per_contract: float = 0.0,
             margin_schedule_path: str | None = None) -> None:
    def read_table(path: str) -> pd.DataFrame:
        return pd.read_parquet(path) if Path(path).suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)
    quotes, predictions, prices = map(read_table, [quotes_path, predictions_path, prices_path])
    margin_schedule = read_table(margin_schedule_path) if margin_schedule_path else None
    all_trades, all_summaries = [], []
    for horizon in horizons:
        trades, summary = backtest_straddles(quotes, predictions, prices, horizon=horizon,
            min_vol_edge=min_vol_edge, fee_per_contract=fee_per_contract,
            contract_multiplier=contract_multiplier, hedge_mode=hedge_mode,
            option_slippage_bps=option_slippage_bps, hedge_slippage_bps=hedge_slippage_bps,
            hedge_fee_per_share=hedge_fee_per_share, max_relative_spread=max_relative_spread,
            min_option_volume=min_option_volume, min_open_interest=min_open_interest,
            hedge_risk_free_rate=hedge_risk_free_rate,
            hedge_dividend_yield=hedge_dividend_yield, contracts=contracts,
            max_volume_participation=max_volume_participation,
            max_open_interest_participation=max_open_interest_participation,
            max_cumulative_loss_krw=max_cumulative_loss_krw,
            short_margin_per_contract_krw=short_margin_per_contract_krw,
            capital_base_krw=capital_base_krw,
            annual_financing_rate=annual_financing_rate,
            annual_stock_borrow_rate=annual_stock_borrow_rate,
            max_cumulative_drawdown_krw=max_cumulative_drawdown_krw,
            exercise_fee_per_contract=exercise_fee_per_contract,
            margin_schedule=margin_schedule)
        all_trades.append(trades); all_summaries.append(summary)
    out = Path("reports"); out.mkdir(parents=True, exist_ok=True)
    trade_frame = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    summary_frame = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    synthetic = (quotes.synthetic_data_flag.fillna(False).astype(str).str.lower()
                 .isin({"true", "1", "yes"}).any()) if "synthetic_data_flag" in quotes else False
    suffix = "_synthetic_demo" if synthetic else ""
    trade_frame.to_csv(out / f"straddle_backtest_trades{suffix}.csv", index=False)
    summary_frame.to_csv(out / f"straddle_backtest_summary{suffix}.csv", index=False)
    summary_frame.to_json(out / f"straddle_backtest_summary{suffix}.json", orient="records", indent=2, date_format="iso")
    if summary_frame.trade_count.sum() == 0:
        print("No eligible straddle trades. Confirm quote coverage, contract multipliers, and forecast dates.")
    else:
        print(summary_frame.to_string(index=False))
    print(f"Saved trade ledger and summary under reports/ (synthetic={synthetic}).")


def main() -> None:
    parser = argparse.ArgumentParser(description="SK Hynix price and realized-volatility research pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download prices and construct RV data")
    download.add_argument("--start", default="2015-01-01", help="First requested date (inclusive)")
    download.add_argument("--end", default=None, help="Last requested date (exclusive; default today)")
    commands.add_parser("evaluate", help="Run expanding-window RV baseline evaluation")
    commands.add_parser("forecast-rv", help="Fit RV baselines and export current forward forecasts")
    commands.add_parser("evaluate-ml", help="Run chronological walk-forward ML model comparison")
    commands.add_parser("forecast-ml", help="Fit ML models and export current forward forecasts")
    demo = commands.add_parser("generate-demo-options", help="Generate removable, clearly labeled synthetic options data")
    demo.add_argument("--prices", default="data/processed/daily_prices.parquet")
    demo.add_argument("--out", default="data/demo_synthetic/options_quotes.csv")
    demo.add_argument("--days", type=int, default=65, help="Recent underlying sessions to simulate")
    demo_run = commands.add_parser("demo-research", help="Run the complete offline workflow on generated data")
    demo_run.add_argument("--out-dir", default="data/demo_synthetic")
    demo_run.add_argument("--reports-dir", default="reports")
    demo_run.add_argument("--sessions", type=int, default=3024, help="Synthetic underlying sessions")
    demo_run.add_argument("--quote-days", type=int, default=126, help="Recent sessions of synthetic option chains")
    demo_run.add_argument("--seed", type=int, default=7)
    imp = commands.add_parser("import-options", help="Import a provider-supplied options CSV or Parquet")
    imp.add_argument("path")
    imp.add_argument("--iv-units", choices=["decimal", "percent"], default="decimal")
    validate = commands.add_parser("validate-options", help="Validate quotes and independently reprice provider IV")
    validate.add_argument("path")
    validate.add_argument("--iv-units", choices=["decimal", "percent"], default="decimal")
    imp.add_argument("--source", default=None,
                     help="Provider or dataset label retained in normalized quote data")
    validate.add_argument("--source", default=None,
                          help="Provider or dataset label retained in validation output")
    iv = commands.add_parser("iv-surface", help="Build and plot coverage-supported daily IV surfaces")
    iv.add_argument("--quotes", default="data/processed/option_quotes.parquet")
    iv.add_argument("--as-of", default=None, help="Observation date to plot")
    iv.add_argument("--spot-forward-approximation", action="store_true", help="Use F=S*exp((r-q)T), with missing carry rates set to zero")
    ivf = commands.add_parser("forecast-iv", help="Forecast and export the next-session PCA IV surface")
    ivf.add_argument("--quotes", default="data/processed/option_quotes.parquet")
    ivf.add_argument("--spot-forward-approximation", action="store_true", help="Use documented spot/carry approximation when forward is absent")
    ivf.add_argument("--n-components", type=int, default=3)
    ive = commands.add_parser("evaluate-iv", help="Run leakage-safe next-session IV surface evaluation")
    ive.add_argument("--quotes", default="data/processed/option_quotes.parquet")
    ive.add_argument("--spot-forward-approximation", action="store_true")
    ive.add_argument("--n-components", type=int, default=3)
    ive.add_argument("--min-training-dates", type=int, default=8)
    iv_dl_eval = commands.add_parser("evaluate-iv-dl", help="Run leakage-safe LSTM IV-surface evaluation")
    iv_dl_eval.add_argument("--quotes", default="data/processed/option_quotes.parquet")
    iv_dl_eval.add_argument("--spot-forward-approximation", action="store_true")
    iv_dl_eval.add_argument("--min-training-dates", type=int, default=30)
    iv_dl_eval.add_argument("--sequence-length", type=int, default=5)
    iv_dl_eval.add_argument("--max-epochs", type=int, default=60)
    iv_dl_eval.add_argument("--seed", type=int, default=42)
    iv_dl_forecast = commands.add_parser("forecast-iv-dl", help="Forecast the next IV surface using an LSTM")
    iv_dl_forecast.add_argument("--quotes", default="data/processed/option_quotes.parquet")
    iv_dl_forecast.add_argument("--spot-forward-approximation", action="store_true")
    iv_dl_forecast.add_argument("--sequence-length", type=int, default=5)
    iv_dl_forecast.add_argument("--max-epochs", type=int, default=60)
    iv_dl_forecast.add_argument("--seed", type=int, default=42)
    projection = commands.add_parser("project-iv", help="Project IV grid onto static-arbitrage constraints")
    projection.add_argument("--quotes", default="data/processed/option_quotes.parquet")
    projection.add_argument("--spot-forward-approximation", action="store_true")
    ivfactors = commands.add_parser("iv-factors", help="Export historical IV PCA factor scores and loadings")
    ivfactors.add_argument("--quotes", default="data/processed/option_quotes.parquet")
    ivfactors.add_argument("--spot-forward-approximation", action="store_true")
    ivfactors.add_argument("--n-components", type=int, default=3)
    bt = commands.add_parser("backtest", help="Simulate historical quote-based long/short straddles")
    bt.add_argument("--quotes", default="data/processed/option_quotes.parquet")
    bt.add_argument("--forecasts", default="reports/rv_walk_forward_predictions.csv")
    bt.add_argument("--prices", default="data/processed/daily_prices.parquet")
    bt.add_argument("--horizons", default="5,20,60")
    bt.add_argument("--min-vol-edge", type=float, default=0.02)
    bt.add_argument("--fee-per-contract", type=float, default=0.0,
                    help="KRW fee per option contract per execution leg")
    bt.add_argument("--contract-multiplier", type=float, default=None,
                    help="Use only if the quote file omits the verified contract multiplier")
    bt.add_argument("--hedge-mode", choices=["none", "daily_delta"], default="none",
                    help="Optionally rebalance the underlying hedge at each close")
    bt.add_argument("--option-slippage-bps", type=float, default=0.0,
                    help="Additional option premium slippage per execution side, in basis points")
    bt.add_argument("--hedge-slippage-bps", type=float, default=0.0,
                    help="Underlying hedge slippage per rebalance, in basis points of traded notional")
    bt.add_argument("--hedge-fee-per-share", type=float, default=0.0,
                    help="Underlying hedge commission in KRW per share traded")
    bt.add_argument("--max-relative-spread", type=float, default=None,
                    help="Skip entry/exit quotes whose bid/ask spread over midpoint exceeds this fraction")
    bt.add_argument("--min-option-volume", type=float, default=0.0,
                    help="Minimum reported volume for both option legs")
    bt.add_argument("--min-open-interest", type=float, default=0.0,
                    help="Minimum reported open interest for both option legs")
    bt.add_argument("--hedge-risk-free-rate", type=float, default=None,
                    help="Explicit fallback rate for delta calculation when quotes omit it")
    bt.add_argument("--hedge-dividend-yield", type=float, default=None,
                    help="Explicit fallback dividend yield when quotes omit carry inputs")
    bt.add_argument("--contracts", type=int, default=1,
                    help="Fixed number of straddles per signal")
    bt.add_argument("--max-volume-participation", type=float, default=None,
                    help="Maximum contracts as a fraction of reported option volume (0, 1]")
    bt.add_argument("--max-open-interest-participation", type=float, default=None,
                    help="Maximum contracts as a fraction of reported open interest (0, 1]")
    bt.add_argument("--max-cumulative-loss-krw", type=float, default=None,
                    help="Halt new entries once realized net P&L reaches this loss budget")
    bt.add_argument("--max-cumulative-drawdown-krw", type=float, default=None,
                    help="Halt new entries when realized P&L falls this far below its previous peak")
    bt.add_argument("--short-margin-per-contract-krw", type=float, default=None,
                    help="User-supplied short-option margin proxy; this is not a KRX margin calculation")
    bt.add_argument("--capital-base-krw", type=float, default=None,
                    help="Capital budget used to skip trades whose estimated requirement exceeds the base")
    bt.add_argument("--annual-financing-rate", type=float, default=0.0,
                    help="Annual carry charge on long premium, posted short margin, and long hedge notional")
    bt.add_argument("--annual-stock-borrow-rate", type=float, default=0.0,
                    help="Annual borrow cost applied to time-weighted short hedge notional")
    bt.add_argument("--exercise-fee-per-contract", type=float, default=0.0,
                    help="User-supplied fee for each exercised option contract at expiry")
    bt.add_argument("--margin-schedule", default=None,
                    help="CSV/Parquet of position-specific daily KRX/member margin requirements")
    args = parser.parse_args()
    if args.command == "download":
        run(start=args.start, end=args.end)
    elif args.command == "evaluate":
        evaluate()
    elif args.command == "forecast-rv":
        forecast_rv()
    elif args.command == "evaluate-ml":
        evaluate_ml()
    elif args.command == "forecast-ml":
        forecast_ml()
    elif args.command == "generate-demo-options":
        generate_demo_options(args.prices, args.out, args.days)
    elif args.command == "demo-research":
        run_demo_research(args.out_dir, args.reports_dir, args.sessions,
                          args.quote_days, args.seed)
    elif args.command == "import-options":
        import_options(args.path, args.iv_units, args.source)
    elif args.command == "validate-options":
        validate_options_file(args.path, args.iv_units, args.source)
    elif args.command == "iv-surface":
        make_iv_surface(args.quotes, args.spot_forward_approximation, as_of=args.as_of)
    elif args.command == "forecast-iv":
        make_iv_surface(args.quotes, args.spot_forward_approximation, forecast=True,
                        n_components=args.n_components)
    elif args.command == "evaluate-iv":
        evaluate_iv_surface(args.quotes, args.spot_forward_approximation,
                            args.n_components, args.min_training_dates)
    elif args.command == "evaluate-iv-dl":
        evaluate_iv_surface_dl(args.quotes, args.spot_forward_approximation,
            args.min_training_dates, args.sequence_length, args.max_epochs, args.seed)
    elif args.command == "forecast-iv-dl":
        forecast_iv_surface_dl(args.quotes, args.spot_forward_approximation,
            args.sequence_length, args.max_epochs, args.seed)
    elif args.command == "project-iv":
        project_iv(args.quotes, args.spot_forward_approximation)
    elif args.command == "iv-factors":
        export_iv_factors(args.quotes, args.spot_forward_approximation, args.n_components)
    elif args.command == "backtest":
        horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
        if not horizons or any(h < 1 for h in horizons):
            parser.error("--horizons must be comma-separated positive session counts")
        backtest(args.quotes, args.forecasts, args.prices, horizons, args.min_vol_edge,
                 args.fee_per_contract, args.contract_multiplier, args.hedge_mode,
                 args.option_slippage_bps, args.hedge_slippage_bps,
                 args.hedge_fee_per_share, args.max_relative_spread,
                 args.min_option_volume, args.min_open_interest,
                 args.hedge_risk_free_rate, args.hedge_dividend_yield,
                 args.contracts, args.max_volume_participation,
                 args.max_open_interest_participation, args.max_cumulative_loss_krw,
                 args.short_margin_per_contract_krw, args.capital_base_krw,
                 args.annual_financing_rate, args.annual_stock_borrow_rate,
                 args.max_cumulative_drawdown_krw, args.exercise_fee_per_contract,
                 args.margin_schedule)


if __name__ == "__main__":
    main()
