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

The evaluator fits Random Forest, histogram gradient boosting, and RBF-SVR on each training fold. XGBoost joins automatically when installed with `python -m pip install -e '.[ml-boosting]'`; LightGBM is available with `python -m pip install -e '.[ml-lightgbm]'`. All preprocessing is fit inside each fold, features are limited to information available at the forecast close, and training rows are purged by the target horizon. Fixed model settings and a fixed random seed are used; the reported test years are not used for tuning. Outputs are `reports/rv_ml_walk_forward_predictions.csv` and `reports/rv_ml_walk_forward_metrics.csv`.

Fit each installed model on all available labeled history and export the latest-close forecasts:

```bash
skhynix-vol forecast-ml
```

The output is `reports/rv_ml_forecasts_latest.csv`. A model is a forecast benchmark, not evidence of trading profitability; compare all models on the same historical folds and inspect the per-year metrics.

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

The current implementation includes latest-date RV forecasts, chronological RV evaluation with GARCH-family models, a machine-learning RV comparison, a provider-file IV-surface workflow, and a quote-based straddle simulation engine. Arbitrage-aware surface constraints and automatic discovery of a reliable KRX options data feed remain future work.

## Phase 4: KRX options and implied volatility

The target instrument is KRX single-stock options on SK hynix (000660). KRX specifies a 10-share contract, European exercise, cash settlement, and monthly contracts. SK hynix is among the single-stock option underlyings; the planned weekly options for SK hynix and three other stocks were postponed in June 2026. Reconfirm the live KRX contract list and current listing status when acquiring a new dataset. See [KRX single-stock option specifications](https://global.krx.co.kr/contents/GLB/02/0201/0201040402/GLB0201040402.jsp) and [Yonhap's report on the weekly-options postponement](https://en.yna.co.kr/view/AEN20260625004000320).

The code applies the verified 10-share multiplier and checks imported multiplier conflicts. It imports historical observations from a provider CSV or Parquet file; it does not scrape KRX or bypass vendor licensing. Yahoo Finance is not an options source. The quote file must identify the underlying, expiry, side, strike, bid, ask, spot, and source IV. Add a forward (preferred) or carry fields (rate and dividend yield) for independent IV recalculation.

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

The forecast grid is exported to `reports/iv_surface_forecast.csv` and rendered to `reports/iv_surface_forecast.png`. This interpolated surface is not constrained to be arbitrage-free; use the chronological evaluator before economic interpretation. PCA components are sign-indeterminate, so interpret them from exported loadings rather than assigning level/skew/curvature labels automatically.

Evaluate one-step-ahead IV surfaces on expanding chronological folds against persistence and historical-mean baselines:

```bash
skhynix-vol evaluate-iv --spot-forward-approximation
```

Each target-date surface is forecast using earlier dates only. Outputs are `reports/iv_surface_walk_forward_predictions.csv` and `reports/iv_surface_walk_forward_metrics.csv`; metrics include surface MAE/RMSE, ATM error, and downside/upside skew error where those grid points are supported. The PCA model forecasts changes in factor scores; persistence and historical mean remain comparison baselines.

Validate imported quotes and independently reprice provider IV from the bid/ask midpoint. The calculation needs a source forward and risk-free rate, or spot, risk-free rate, and dividend yield:

```bash
skhynix-vol validate-options provider_quotes.csv --iv-units percent
```

The report `reports/option_quote_iv_validation.csv` preserves provider IV, adds calculated IV and the difference, and flags crossed quotes, no-arbitrage violations, missing carry inputs, and discrepancies over five volatility points. Missing carry inputs are reported rather than filled. Surface construction continues to use the provider IV; review validation flags and exclude suspect records as appropriate before research use.

The quote-based historical benchmark is an ATM straddle strategy driven by out-of-sample HAR-RV predictions. It selects a same-strike call/put pair with expiry near the forecast horizon, buys when forecast RV exceeds IV by the edge threshold, and sells when it is lower. It executes entries/exits at bid/ask and charges the supplied per-leg fee. Use KRX's 10-share multiplier for SK hynix single-stock options; the override remains available for other products:

```bash
skhynix-vol backtest --contract-multiplier 10 --fee-per-contract 10
```

This simulator requires historical options quotes aligned to both entry and exit dates; without them it cannot produce actual backtest results. It does not delta-hedge, model margin, slippage beyond quoted spreads, financing, or exchange-specific fees. Short straddles have potentially very large losses. Results are hypothetical research output, not a profitability claim. Trade ledger and summary are exported as `reports/straddle_backtest_trades.csv` and `reports/straddle_backtest_summary.csv`.
