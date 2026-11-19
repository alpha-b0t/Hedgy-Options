# SK Hynix Quantitative Volatility Research

A reproducible Python research foundation for SK Hynix (KRX: 000660; Yahoo Finance: `000660.KS`). The initial pipeline downloads daily prices, validates and stores raw and normalized observations, and builds trailing realized-volatility features and forward 5-, 20-, and 60-session realized-volatility labels.

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

This runs annual expanding-window folds for rolling-volatility, EWMA, and HAR-RV forecasts at 5-, 20-, and 60-session horizons. Five initial calendar years are reserved for the first training window. Training labels whose forward return windows touch the test period are purged. Predictions and fold metrics are written under `reports/`.

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

The current implementation covers Phase 1 and the initial Phase 2 baseline evaluation. GARCH variants, richer statistical diagnostics, machine-learning models, and a separate investigation of KRX options data and contract specifications remain future work. No options observations, implied-volatility surface, or strategy backtest is included yet.
