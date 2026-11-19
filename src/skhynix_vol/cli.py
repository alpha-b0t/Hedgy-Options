"""Command-line entry point for the initial research pipeline."""
import argparse
import json
from pathlib import Path

import pandas as pd

from skhynix_vol.data.prices import download_prices, save_price_data
from skhynix_vol.evaluation.walk_forward import walk_forward_rv
from skhynix_vol.features.realized import build_realized_dataset
from skhynix_vol.visualization.charts import create_exploratory_charts


def run(start: str = "2015-01-01", end: str | None = None) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="SK Hynix price and realized-volatility research pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download prices and construct RV data")
    download.add_argument("--start", default="2015-01-01", help="First requested date (inclusive)")
    download.add_argument("--end", default=None, help="Last requested date (exclusive; default today)")
    commands.add_parser("evaluate", help="Run expanding-window RV baseline evaluation")
    args = parser.parse_args()
    if args.command == "download":
        run(start=args.start, end=args.end)
    else:
        evaluate()


if __name__ == "__main__":
    main()
