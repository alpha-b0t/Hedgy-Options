"""Command-line entry point for the initial research pipeline."""
import argparse
import json
from pathlib import Path

import pandas as pd

from skhynix_vol.data.prices import download_prices, save_price_data
from skhynix_vol.data.options import load_option_file, save_option_quotes
from skhynix_vol.evaluation.walk_forward import forecast_latest_rv, walk_forward_rv
from skhynix_vol.features.implied import build_iv_surface, forecast_next_iv_surface
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
    out = Path("reports"); out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out / "rv_walk_forward_predictions.csv", index=False)
    metrics.to_csv(out / "rv_walk_forward_metrics.csv", index=False)
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
                       "ewma_lambda": 0.94, "metrics": ["mae", "rmse", "qlike_variance_scale", "r2", "directional_accuracy"]},
    }
    (out / "rv_walk_forward_run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if metrics.empty:
        print("No walk-forward folds available; check the price history length.")
    else:
        summary = metrics.groupby(["horizon", "model"])[["mae", "rmse", "qlike"]].mean()
        print(summary.to_string())
        print("Saved annual fold predictions and metrics under reports/.")


def forecast_rv() -> None:
    """Write latest-close 5/20/60 session RV forecasts."""
    path = Path("data/processed/daily_prices.parquet")
    if not path.exists():
        raise FileNotFoundError("Run `skhynix-vol download` before `skhynix-vol forecast-rv`")
    prices = pd.read_parquet(path)
    dataset = build_realized_dataset(prices)
    forecasts = forecast_latest_rv(dataset)
    out = Path("reports"); out.mkdir(parents=True, exist_ok=True)
    target = out / "rv_forecasts_latest.csv"
    forecasts.to_csv(target, index=False)
    print(f"As-of {forecasts.as_of_date.iloc[0].date()}:\n{forecasts[['horizon_sessions','har_rv_forecast','rolling_rv_forecast','ewma_vol_forecast']].to_string(index=False)}")
    print(f"Saved forecasts to {target}")


def import_options(path: str, iv_units: str) -> None:
    raw, normalized = load_option_file(path, iv_units=iv_units)
    raw_path, processed_path = save_option_quotes(raw, normalized)
    print(f"Imported {len(normalized):,} option quotes from {path}")
    print(f"Observation dates: {normalized.date.min().date()} to {normalized.date.max().date()}")
    print(f"Crossed-market rows flagged: {(normalized.quote_quality_flag != 'ok').sum()}")
    print(f"Saved raw: {raw_path}\nSaved normalized: {processed_path}")


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
    surface = build_iv_surface(quotes, spot_forward_approximation=spot_forward_approximation)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    surface.to_parquet("data/processed/iv_surface.parquet", index=False)
    out = Path("reports"); out.mkdir(parents=True, exist_ok=True)
    if as_of is not None:
        as_date = pd.Timestamp(as_of).normalize()
        surface = surface[surface.date.eq(as_date)]
        if surface.empty:
            raise ValueError(f"No supported IV surface grid for {as_date.date()}")
    if forecast:
        prediction = forecast_next_iv_surface(surface, n_components=n_components)
        prediction_path = out / "iv_surface_forecast.csv"
        prediction.to_csv(prediction_path, index=False)
        image_path = plot_iv_surface(prediction, out / "iv_surface_forecast.png",
            value_column="implied_volatility_forecast", title="SK Hynix next-session IV surface forecast")
        print(f"Saved {len(prediction)} forecast grid cells to {prediction_path}")
        print(f"Saved forecast image: {image_path}")
    else:
        image_path = plot_iv_surface(surface, out / f"iv_surface_{surface.date.max():%Y%m%d}.png")
        print(f"Saved supported surface points: data/processed/iv_surface.parquet")
        print(f"Saved image: {image_path}")


def backtest(quotes_path: str, predictions_path: str, prices_path: str,
             horizons: list[int], min_vol_edge: float, fee_per_contract: float,
             contract_multiplier: float | None) -> None:
    def read_table(path: str) -> pd.DataFrame:
        return pd.read_parquet(path) if Path(path).suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)
    quotes, predictions, prices = map(read_table, [quotes_path, predictions_path, prices_path])
    all_trades, all_summaries = [], []
    for horizon in horizons:
        trades, summary = backtest_straddles(quotes, predictions, prices, horizon=horizon,
            min_vol_edge=min_vol_edge, fee_per_contract=fee_per_contract,
            contract_multiplier=contract_multiplier)
        all_trades.append(trades); all_summaries.append(summary)
    out = Path("reports"); out.mkdir(parents=True, exist_ok=True)
    trade_frame = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    summary_frame = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    trade_frame.to_csv(out / "straddle_backtest_trades.csv", index=False)
    summary_frame.to_csv(out / "straddle_backtest_summary.csv", index=False)
    summary_frame.to_json(out / "straddle_backtest_summary.json", orient="records", indent=2, date_format="iso")
    if summary_frame.trade_count.sum() == 0:
        print("No eligible straddle trades. Confirm quote coverage, contract multipliers, and forecast dates.")
    else:
        print(summary_frame.to_string(index=False))
    print("Saved trade ledger and summary under reports/.")


def main() -> None:
    parser = argparse.ArgumentParser(description="SK Hynix price and realized-volatility research pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download prices and construct RV data")
    download.add_argument("--start", default="2015-01-01", help="First requested date (inclusive)")
    download.add_argument("--end", default=None, help="Last requested date (exclusive; default today)")
    commands.add_parser("evaluate", help="Run expanding-window RV baseline evaluation")
    commands.add_parser("forecast-rv", help="Fit RV baselines and export current forward forecasts")
    imp = commands.add_parser("import-options", help="Import a provider-supplied options CSV or Parquet")
    imp.add_argument("path")
    imp.add_argument("--iv-units", choices=["decimal", "percent"], default="decimal")
    iv = commands.add_parser("iv-surface", help="Build and plot coverage-supported daily IV surfaces")
    iv.add_argument("--quotes", default="data/processed/option_quotes.parquet")
    iv.add_argument("--as-of", default=None, help="Observation date to plot")
    iv.add_argument("--spot-forward-approximation", action="store_true", help="Use F=S*exp((r-q)T), with missing carry rates set to zero")
    ivf = commands.add_parser("forecast-iv", help="Forecast and export the next-session PCA IV surface")
    ivf.add_argument("--quotes", default="data/processed/option_quotes.parquet")
    ivf.add_argument("--spot-forward-approximation", action="store_true", help="Use documented spot/carry approximation when forward is absent")
    ivf.add_argument("--n-components", type=int, default=3)
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
    args = parser.parse_args()
    if args.command == "download":
        run(start=args.start, end=args.end)
    elif args.command == "evaluate":
        evaluate()
    elif args.command == "forecast-rv":
        forecast_rv()
    elif args.command == "import-options":
        import_options(args.path, args.iv_units)
    elif args.command == "iv-surface":
        make_iv_surface(args.quotes, args.spot_forward_approximation, as_of=args.as_of)
    elif args.command == "forecast-iv":
        make_iv_surface(args.quotes, args.spot_forward_approximation, forecast=True,
                        n_components=args.n_components)
    elif args.command == "backtest":
        horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
        if not horizons or any(h < 1 for h in horizons):
            parser.error("--horizons must be comma-separated positive session counts")
        backtest(args.quotes, args.forecasts, args.prices, horizons, args.min_vol_edge,
                 args.fee_per_contract, args.contract_multiplier)


if __name__ == "__main__":
    main()
