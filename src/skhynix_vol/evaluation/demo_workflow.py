"""Offline end-to-end research workflow using explicitly synthetic data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from skhynix_vol.data.demo_options import generate_synthetic_option_quotes, generate_synthetic_prices
from skhynix_vol.data.option_quality import option_chain_quality
from skhynix_vol.data.options import normalize_option_quotes
from skhynix_vol.evaluation.iv_walk_forward import walk_forward_iv
from skhynix_vol.evaluation.walk_forward import walk_forward_rv
from skhynix_vol.features.arbitrage import static_arbitrage_diagnostics
from skhynix_vol.features.arbitrage_surface import project_iv_surface
from skhynix_vol.features.implied import build_iv_surface
from skhynix_vol.features.option_pricing import validate_option_iv
from skhynix_vol.features.realized import build_realized_dataset
from skhynix_vol.strategies.backtest import backtest_straddles


def run_synthetic_research_demo(output_dir: str | Path = "data/demo_synthetic",
                                reports_dir: str | Path = "reports",
                                sessions: int = 3024, quote_days: int = 126,
                                seed: int = 7) -> dict[str, str | int | bool]:
    """Run RV, quote audit, IV, projection, and strategy stages offline.

    Every data artifact is written to a demo-only directory or receives a
    `_synthetic_demo` report suffix. It never replaces processed data.
    """
    if quote_days < 20:
        raise ValueError("quote_days must be at least 20 for the IV demo workflow")
    data_dir, report_dir = Path(output_dir), Path(reports_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    prices = generate_synthetic_prices(sessions=sessions, seed=seed)
    prices_path = data_dir / "daily_prices_synthetic_demo.csv"
    prices.to_csv(prices_path, index=False)
    features = build_realized_dataset(prices)
    rv_predictions, rv_metrics = walk_forward_rv(features)
    rv_predictions_path = report_dir / "rv_walk_forward_predictions_synthetic_demo.csv"
    rv_metrics_path = report_dir / "rv_walk_forward_metrics_synthetic_demo.csv"
    rv_predictions.to_csv(rv_predictions_path, index=False)
    rv_metrics.to_csv(rv_metrics_path, index=False)

    raw_quotes = generate_synthetic_option_quotes(prices, days=quote_days)
    quotes_path = data_dir / "options_quotes_synthetic_demo.csv"
    raw_quotes.to_csv(quotes_path, index=False)
    digest_builder = hashlib.sha256()
    with quotes_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest_builder.update(block)
    quotes = normalize_option_quotes(raw_quotes, source=f"synthetic_demo_seed_{seed}")
    quotes["source_file"] = quotes_path.name
    quotes["source_sha256"] = digest_builder.hexdigest()

    quote_validation = validate_option_iv(quotes)
    arbitrage = static_arbitrage_diagnostics(quote_validation)
    chain_quality = option_chain_quality(quote_validation)
    quote_validation.to_csv(report_dir / "option_quote_iv_validation_synthetic_demo.csv", index=False)
    arbitrage.to_csv(report_dir / "option_static_arbitrage_diagnostics_synthetic_demo.csv", index=False)
    chain_quality.to_csv(report_dir / "option_chain_quality_synthetic_demo.csv", index=False)

    surface = build_iv_surface(quotes)
    surface.to_csv(data_dir / "iv_surface_synthetic_demo.csv", index=False)
    projected = project_iv_surface(surface)
    projected.to_csv(report_dir / "iv_surface_arbitrage_projection_synthetic_demo.csv", index=False)
    iv_predictions, iv_metrics = walk_forward_iv(surface, min_training_dates=8)
    iv_predictions.to_csv(report_dir / "iv_surface_walk_forward_predictions_synthetic_demo.csv", index=False)
    iv_metrics.to_csv(report_dir / "iv_surface_walk_forward_metrics_synthetic_demo.csv", index=False)

    all_trades, all_summaries = [], []
    synthetic_margin_schedule = quotes[["date", "expiry", "strike"]].drop_duplicates().copy()
    synthetic_margin_schedule["contracts"] = 2
    synthetic_margin_schedule["short_straddle_margin_krw"] = 2_000_000.0
    synthetic_margin_schedule["source"] = "SYNTHETIC_DEMO_NOT_KRX_MARGIN"
    for horizon in (5, 20, 60):
        trades, summary = backtest_straddles(
            quotes, rv_predictions, prices, horizon=horizon,
            min_vol_edge=0.02, fee_per_contract=10.0,
            hedge_mode="daily_delta", option_slippage_bps=5.0,
            hedge_slippage_bps=3.0, hedge_fee_per_share=0.01,
            max_relative_spread=0.25, contracts=2,
            max_volume_participation=0.10,
            max_open_interest_participation=0.05,
            capital_base_krw=20_000_000.0,
            annual_financing_rate=0.04,
            annual_stock_borrow_rate=0.03,
            max_cumulative_drawdown_krw=2_000_000.0,
            exercise_fee_per_contract=1.0,
            margin_schedule=synthetic_margin_schedule,
        )
        all_trades.append(trades)
        all_summaries.append(summary)
    trade_frame = pd.concat(all_trades, ignore_index=True)
    summary_frame = pd.concat(all_summaries, ignore_index=True)
    trade_frame.to_csv(report_dir / "straddle_backtest_trades_synthetic_demo.csv", index=False)
    summary_frame.to_csv(report_dir / "straddle_backtest_summary_synthetic_demo.csv", index=False)

    manifest = {
        "data_origin": "SYNTHETIC_DEMO_DO_NOT_USE_FOR_RESEARCH",
        "synthetic_data_flag": True,
        "seed": seed,
        "price_sessions": int(len(prices)),
        "price_start": str(pd.Timestamp(prices.date.min()).date()),
        "price_end": str(pd.Timestamp(prices.date.max()).date()),
        "option_quote_days": int(prices.tail(quote_days).date.nunique()),
        "option_quote_rows": int(len(quotes)),
        "option_source_sha256": digest_builder.hexdigest(),
        "rv_walk_forward_rows": int(len(rv_predictions)),
        "iv_surface_rows": int(len(surface)),
        "iv_walk_forward_rows": int(len(iv_predictions)),
        "projected_surface_rows": int(len(projected)),
        "backtest_trade_rows": int(len(trade_frame)),
        "backtest_fee_per_leg_krw": 10.0,
        "backtest_hedge_mode": "daily_delta",
        "backtest_option_slippage_bps": 5.0,
        "backtest_hedge_slippage_bps": 3.0,
        "backtest_contracts_per_trade": 2,
        "backtest_max_cumulative_drawdown_krw": 2_000_000.0,
        "backtest_margin_schedule_source": "SYNTHETIC_DEMO_NOT_KRX_MARGIN",
        "backtest_exercise_fee_per_contract_krw": 1.0,
        "backtest_max_volume_participation": 0.10,
        "backtest_max_open_interest_participation": 0.05,
        "backtest_synthetic_short_straddle_margin_krw": 2_000_000.0,
        "backtest_capital_base_krw": 20_000_000.0,
        "backtest_annual_financing_rate": 0.04,
        "backtest_annual_stock_borrow_rate": 0.03,
        "outputs": {
            "synthetic_prices": str(prices_path),
            "synthetic_quotes": str(quotes_path),
            "rv_metrics": str(rv_metrics_path),
            "iv_metrics": str(report_dir / "iv_surface_walk_forward_metrics_synthetic_demo.csv"),
            "backtest_summary": str(report_dir / "straddle_backtest_summary_synthetic_demo.csv"),
        },
    }
    manifest_path = data_dir / "research_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"manifest": str(manifest_path), "quotes": int(len(quotes)),
            "rv_predictions": int(len(rv_predictions)), "surface_cells": int(len(surface)),
            "backtest_trades": int(len(trade_frame)), "synthetic_data": True}
