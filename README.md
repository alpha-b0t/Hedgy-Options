# SK Hynix Option Quantitative Volatility Research

A reproducible Python research foundation for SK Hynix (KRX: 000660; Yahoo Finance: `000660.KS`). The initial pipeline downloads daily prices, validates and stores raw and normalized observations, and builds trailing realized-volatility features and forward 5-, 20-, and 60-session realized-volatility labels.

See [ROADMAP.md](ROADMAP.md) for stage status, exit criteria, and the remaining licensed-data work.

This repository is research software, not financial advice. Forecast accuracy does not establish trading profitability. No live trading is implemented. Historical options data is not fabricated or assumed to be available from Yahoo Finance.

## Setup

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Alternatively install from `requirements.txt`.

## Run the pipeline

```bash
skhynix-vol download --start 2015-01-01
```

Or run without installing the console entry point:

```bash
python -m skhynix_vol.cli download --start 2015-01-01
```

After downloading, run the first chronological forecasting comparison:

```bash
skhynix-vol evaluate
```

This runs annual expanding-window folds for rolling-volatility, EWMA, HAR-RV, GARCH(1,1), EGARCH(1,1,1), and GJR-GARCH(1,1) forecasts at 5-, 20-, and 60-session horizons. Five initial calendar years are reserved for the first training window. HAR training labels whose forward return windows touch the test period are purged. GARCH parameters are estimated on prior returns only; conditional variance then updates sequentially with observed test-period returns. EGARCH multi-step forecasts use seeded simulation. Baseline and GARCH predictions and fold metrics are written separately under `reports/`.

Fit the RV baselines on all available labeled history and export forecasts from the latest close:

```bash
skhynix-vol forecast-rv
```

This writes `reports/rv_forecasts_latest.csv` with 5-, 20-, and 60-session HAR-RV forecasts plus rolling and EWMA benchmarks, and `reports/rv_garch_forecasts_latest.csv` with the three GARCH-family estimates. The as-of date is the last observed close; future labels are never used as features.

## Phase 3: machine-learning RV forecasts

Run expanding-window, annual out-of-sample model comparisons and export predictions and fold metrics:

```bash
skhynix-vol evaluate-ml
```

The evaluator fits Random Forest, histogram gradient boosting, RBF-SVR, and a compact feed-forward neural network (MLP) on each training fold. XGBoost joins automatically when installed with `python -m pip install -e '.[ml-boosting]'`; LightGBM is available with `python -m pip install -e '.[ml-lightgbm]'`. All preprocessing is fit inside each fold, features are limited to information available at the forecast close, and training rows are purged by the target horizon. Fixed model settings and a fixed random seed are used; the reported test years are not used for tuning. Outputs are `reports/rv_ml_walk_forward_predictions.csv` and `reports/rv_ml_walk_forward_metrics.csv`.

Fit each installed model on all available labeled history and export the latest-close forecasts:

```bash
skhynix-vol forecast-ml
```

The output is `reports/rv_ml_forecasts_latest.csv`. A model is a forecast benchmark, not evidence of trading profitability; compare all models on the same historical folds and inspect the per-year metrics.

### Phase 5: neural RV benchmark

The Phase 3 walk-forward suite includes a fold-local MLP neural network. For an optional recurrent model, install PyTorch and run the same `evaluate-ml` / `forecast-ml` commands:

```bash
python -m pip install -e '.[ml-deep]'
```

When PyTorch is available, an LSTM joins both commands automatically. It consumes 20-session sequences of the same eight close-known features; feature scaling and chronological early stopping are fit within each training fold. Training labels are purged by the forecast horizon, while earlier unlabeled feature rows may provide sequence context. The optional dependency uses PyTorch 2.3.1+ where wheels are available; Intel macOS uses the final official x86 wheel, PyTorch 2.2.2. Without the optional extra, the suite continues with its standard models and MLP.

The downloader uses Yahoo Finance via `yfinance`, requests unadjusted OHLC, adjusted close, dividends, and splits, and fails clearly if the provider returns no data. The date range, actual observation coverage, and generated paths are printed after success. Yahoo Finance is an initial research source, not an exchange-grade licensed feed; verify important observations against KRX or a suitable vendor before using them in consequential research.

## Outputs

- `data/raw/`: timestamped provider response in Parquet, retained separately from transformations.
- `data/processed/daily_prices.parquet`: normalized daily prices and source metadata.
- `data/processed/rv_features.parquet`: returns and trailing volatility features.
- `data/processed/rv_targets.parquet`: forward 5-, 20-, and 60-session realized-volatility labels.
- `reports/sk_hynix_price_history.png`: adjusted price history.
- `reports/sk_hynix_realized_volatility.png`: trailing volatility history.
- `reports/rv_walk_forward_predictions.csv` and `reports/rv_walk_forward_metrics.csv`: annual out-of-sample forecasts and MAE, RMSE, variance-scale QLIKE, R-squared, and directional accuracy.
- `reports/rv_walk_forward_run.json`: data coverage, model settings, target definition, and evaluation configuration; each prediction row also carries its fold's train window and HAR coefficients.
- `reports/rv_forecasts_latest.csv`: current forward forecasts from the latest observed close.

Large downloaded data and generated charts are excluded from Git. No missing price rows are filled. Calendar gaps include weekends and KRX holidays and are not treated as missing sessions.

### `daily_prices` schema

`date`, `ticker`, `open`, `high`, `low`, `close`, `adj_close`, `volume`, `dividends`, `stock_splits`, `ohlc_quality_flag`, `source`, `retrieval_timestamp`. Prices are in KRW per share; volume and corporate actions retain the values returned by the provider. Raw source columns are retained in the raw Parquet file. Internally inconsistent OHLC bars are flagged and preserved verbatim; return research can use adjusted closes, while OHLC range estimators must exclude or resolve flagged rows.

### `rv_features` and `rv_targets` schemas and definitions

- `return_1d = ln(adj_close_t / adj_close_(t-1))`; `abs_return_1d` is its absolute value.
- `rv_w = sqrt(mean(r^2 over the latest w observed returns) * 252)`, for `w` in 1, 5, 20, and 60.
- `ewma_vol_20` is annualized EWMA volatility with lambda 0.94 and recursive daily variance `v_t = lambda*v_(t-1) + (1-lambda)*r_t^2`; initialization uses the first available squared return.
- `volume_change_1d` is the daily percentage change in reported volume; undefined changes after zero or unavailable prior volume are missing, not infinite. `drawdown_20` is adjusted close divided by its trailing 20-session high minus one.
- `target_rv_hd = sqrt((252/h) * sum(r_(t+i)^2, i=1..h))`, for `h` in 5, 20, and 60. Targets use future returns, are not features, and are missing where a complete forward horizon is unavailable.

Trailing windows count observed daily rows (sessions), not calendar days. Features are known at the row date; each target begins with the next session's return. This is a forward-label convention for forecasting after the row's close. Adjusted closes are used for returns; unadjusted close is retained for price analysis.

## Development

```bash
pytest
```

The current implementation includes latest-date RV forecasts, chronological RV evaluation with GARCH-family models, a machine-learning and neural RV comparison, a provider-file IV-surface workflow with source labels and import manifests, quote-level static-arbitrage diagnostics and surface projection, and a quote-based straddle simulation engine. The next stages are provider-specific KRX ingestion, validation on genuine option chains, then economic evaluation using verified historical quotes. The synthetic fixture is only for interface verification.

## Phase 4: KRX options and implied volatility

The target instrument is KRX single-stock options on SK hynix (000660). KRX specifies a 10-share contract, European exercise, cash settlement, and monthly contracts. SK hynix is among the single-stock option underlyings; the planned weekly options for SK hynix and three other stocks were postponed in June 2026. Reconfirm the live KRX contract list and current listing status when acquiring a new dataset. See [KRX single-stock option specifications](https://global.krx.co.kr/contents/GLB/02/0201/0201040402/GLB0201040402.jsp) and [Yonhap's report on the weekly-options postponement](https://en.yna.co.kr/view/AEN20260625004000320).

The code applies the verified 10-share multiplier and checks imported multiplier conflicts. It imports historical observations from a provider CSV or Parquet file; it does not scrape KRX or bypass vendor licensing. Yahoo Finance is not an options source. The quote file must identify the underlying, expiry, side, strike, bid, ask, spot, and source IV. Add a forward (preferred) or carry fields (rate and dividend yield) for independent IV recalculation. Record the vendor or dataset label during import so it remains visible in normalized quotes and diagnostics:

```bash
skhynix-vol import-options provider_quotes.csv --iv-units percent --source "Licensed Vendor daily options"
skhynix-vol validate-options provider_quotes.csv --iv-units percent --source "Licensed Vendor daily options"
```

For interface development only, generate a removable synthetic quote file from local price dates:

```bash
skhynix-vol generate-demo-options
```

To run the complete offline RV and options workflow on generated underlying prices and synthetic option quotes, use `skhynix-vol demo-research`. This writes only to `data/demo_synthetic/` and creates `_synthetic_demo` reports. See [ROADMAP.md](ROADMAP.md) for the included stages and the limits of synthetic results.

The file is written to `data/demo_synthetic/options_quotes.csv`, outside the processed market-data directory. It carries `synthetic_data_flag=true` and `data_origin=SYNTHETIC_DEMO_DO_NOT_USE_FOR_RESEARCH`; its Black-76 prices, IVs, and weekly expiries are simulated. You can pass it directly to `iv-surface`, `evaluate-iv`, and `validate-options` to exercise those commands. Do not import it as historical option data or use it for strategy-performance claims. Delete `data/demo_synthetic/` to remove the fixture.

Example chronological IV workflow:

```bash
skhynix-vol iv-surface --quotes data/demo_synthetic/options_quotes.csv
skhynix-vol evaluate-iv --quotes data/demo_synthetic/options_quotes.csv
skhynix-vol validate-options data/demo_synthetic/options_quotes.csv
skhynix-vol project-iv --quotes data/demo_synthetic/options_quotes.csv
skhynix-vol backtest --quotes data/demo_synthetic/options_quotes.csv --horizons 5
```

Synthetic runs write separate `*_synthetic_demo` report files and keep the surface under `data/demo_synthetic/`; they do not replace standard processed surfaces or reports.

`project-iv` converts each supported IV grid to forward-normalized Black-76 call prices, then minimizes price adjustments subject to strike monotonicity/convexity, intrinsic and upper bounds, and nondecreasing tenor prices at shared moneyness. It converts projected prices back to IV and exports per-cell adjustments. This is a static European-option projection and does not cover discrete dividends, American exercise, or quote uncertainty.

## Options, IV surfaces, and research backtests

Yahoo Finance is not used as an options source. Import a historical quote file from a licensed or otherwise verified provider. CSV or Parquet input must contain `date`, `underlying`, `expiry`, `option_type`, `strike`, `bid`, `ask`, `underlying_close`, and `implied_volatility`. Optional columns include `forward_price`, `risk_free_rate`, `dividend_yield`, `volume`, `open_interest`, `contract_multiplier`, and `quote_timestamp`. Rates, yields, and IV are decimals internally. Explicitly state source IV units when importing:

```bash
skhynix-vol import-options provider_quotes.csv --iv-units percent
```

Then construct and plot daily surfaces, retaining only grid cells bracketed by observed strikes and maturities:

```bash
skhynix-vol iv-surface
```

Without a source `forward_price`, surface construction requires an explicit approximation:

```bash
skhynix-vol iv-surface --spot-forward-approximation
```

That approximation uses `F=S*exp((r-q)T)` and assumes missing rate/dividend yields are zero. The assumption is recorded with the surface values. The default grid uses log-forward-moneyness from -0.30 to +0.30 and tenors from 7 to 365 calendar days. Unsupported cells are not extrapolated. The surface values are saved in `data/processed/iv_surface.parquet` and the chart in `reports/iv_surface_YYYYMMDD.png`.

With at least eight dates and enough repeated grid coverage, export historical PCA scores/loadings and forecast the next-session surface by forecasting PCA factor changes with an AR(1) benchmark:

```bash
skhynix-vol forecast-iv --spot-forward-approximation
```

Run `skhynix-vol iv-factors --spot-forward-approximation` to export historical factor scores and loadings. These are saved as `reports/iv_surface_pca_scores.csv` and `reports/iv_surface_pca_loadings.csv`. The forecast command also writes `reports/iv_surface_factor_change_forecast.csv`.

The forecast grid is exported to `reports/iv_surface_forecast.csv` and rendered to `reports/iv_surface_forecast.png`. Forecast grids are not constrained to be arbitrage-free; `project-iv` applies static European call-price constraints to a constructed grid. Use the chronological evaluator before economic interpretation. PCA components are sign-indeterminate, so interpret them from exported loadings rather than assigning level/skew/curvature labels automatically.

Each imported row retains the source label, input filename, and SHA-256 digest of the original file. `reports/option_import_manifest.json` records the digest, IV input units, quote count, and date coverage. Import also exports `reports/option_chain_quality.csv`, grouped by observation date and expiry, with call/put coverage, matched strikes, duplicates, crossed and zero-bid quotes, invalid prices, relative spreads, and carry/timestamp coverage. `validate-options` writes the same chain report next to its quote and arbitrage diagnostics. These summaries expose data quality; they do not certify a chain for trading.

Evaluate one-step-ahead IV surfaces on expanding chronological folds against persistence and historical-mean baselines:

```bash
skhynix-vol evaluate-iv --spot-forward-approximation
```

Each target-date surface is forecast using earlier dates only. Outputs are `reports/iv_surface_walk_forward_predictions.csv` and `reports/iv_surface_walk_forward_metrics.csv`; metrics include surface MAE/RMSE, ATM error, and downside/upside skew error where those grid points are supported. The PCA model forecasts changes in factor scores; persistence and historical mean remain comparison baselines.

An optional PyTorch sequence model forecasts the next IV grid directly from recent surface snapshots. It selects cells and imputes missing values using the training window only. Use at least 30 supported dates for chronological evaluation (the default sequence length is five sessions):

```bash
skhynix-vol evaluate-iv-dl --quotes data/processed/option_quotes.parquet
skhynix-vol forecast-iv-dl --quotes data/processed/option_quotes.parquet
```

Walk-forward predictions and metrics are saved to `reports/iv_surface_lstm_walk_forward_predictions.csv` and `reports/iv_surface_lstm_walk_forward_metrics.csv`; the latest forecast grid and image are `reports/iv_surface_lstm_forecast.csv` and `.png`. This is a learned benchmark, not proof of predictive or trading value. Compare it with persistence, historical mean, and PCA on identical dates.

Validate imported quotes and independently reprice provider IV from the bid/ask midpoint. The calculation needs a source forward and risk-free rate, or spot, risk-free rate, and dividend yield:

```bash
skhynix-vol validate-options provider_quotes.csv --iv-units percent
```

The report `reports/option_quote_iv_validation.csv` preserves provider IV, adds calculated IV and the difference, and flags crossed quotes, quote bounds, missing carry inputs, and discrepancies over five volatility points. The companion `reports/option_static_arbitrage_diagnostics.csv` checks quoted vertical-spread monotonicity, call-price convexity, and put-call parity by date/expiry. Missing carry inputs are reported rather than filled. These are diagnostics, not an arbitrage-free surface projection; surface construction continues to use provider IV, so review flags before research use.

The quote-based historical benchmark is an ATM straddle strategy driven by out-of-sample HAR-RV predictions. It selects a same-strike call/put pair with expiry near the forecast horizon, buys when forecast RV exceeds IV by the edge threshold, and sells when it is lower. It executes entries/exits at bid/ask and charges the supplied per-leg fee. Use KRX's 10-share multiplier for SK hynix single-stock options; the override remains available for other products:

```bash
skhynix-vol backtest --contract-multiplier 10 --fee-per-contract 10
```

This simulator requires historical options quotes aligned to both entry and exit dates; without them it cannot produce actual backtest results. It does not delta-hedge, model margin, slippage beyond quoted spreads, financing, or exchange-specific fees. Short straddles have potentially very large losses. Results are hypothetical research output, not a profitability claim. Trade ledger and summary are exported as `reports/straddle_backtest_trades.csv` and `reports/straddle_backtest_summary.csv`.

### Phase 6 and later stages

Phase 6 establishes a provider-neutral import and audit boundary with source identity, raw-file integrity, normalized quote data, and chain-level coverage diagnostics. The next stages are provider-specific KRX ingestion, validation on genuine option chains, and economic evaluation using verified historical quotes. Those stages require a licensed provider's field definitions, an authorized access method, and real history. After adding that adapter, run quote validation, static-arbitrage checks, surface projection, chronological IV scoring, and the strategy simulation on the same real dataset. Synthetic data is for interface checks only. No provider-specific network connector is enabled until a vendor and authorized access method are selected.
