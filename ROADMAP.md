# Project roadmap

## Current project state

The repository is a reproducible research prototype. Forecasting, quote diagnostics, surface construction/projection, walk-forward scoring, and a quote-based strategy simulator are implemented. The provider-file workflows can now run end to end on generated fixtures. The project is **not empirically validated for SK hynix options** because no genuine option history is present.

## Stages

| Stage | Scope | Status | Exit criteria |
| --- | --- | --- | --- |
| 0. Research foundation | Package, CLI, settings, storage conventions, documentation | Complete | Installable project and repeatable commands |
| 1. Underlying prices | Daily adjusted/unadjusted prices, validation, raw/processed storage, RV features and labels | Implemented | Provider coverage and price adjustments independently reconciled |
| 2. Statistical RV | Rolling, EWMA, HAR-RV, GARCH family, walk-forward evaluation and current forecasts | Implemented | Results reproduced on archived, verified prices |
| 3. ML RV | Fold-local RF, histogram boosting, SVR, MLP, optional XGBoost/LightGBM/LSTM | Implemented | Stable gains confirmed on untouched out-of-sample periods |
| 4. Option quote interface | Standard quote schema, IV normalization, KRX contract facts, quote repricing, static checks | Implemented for files | Map and reconcile the selected licensed provider's actual fields |
| 5. IV research | Coverage-supported surfaces, PCA forecast, optional direct multi-output LSTM, chronological scoring, static projection | Implemented | Validate with genuine snapshots, including edge cases and projection residuals |
| 6. Reproducible quote ingestion | Source labels, file hashes, import manifest, chain coverage report | Implemented | Provider identity and units documented for every imported dataset |
| 7. Offline integration | Generated prices and chains run through RV evaluation, quote audit, IV surface, projection, and walk-forward IV scoring | Implemented | Workflow executes and all artifacts remain labeled synthetic |
| 8. Strategy research | Bid/ask straddle simulator, optional daily delta hedge, explicit slippage/fees, contract sizing, volume/OI participation caps, cumulative-loss halt, trade ledger and tail-risk summaries | Implemented as a simplified simulator | Validate hedge/fill assumptions against real quotes; add margin, financing, and contract-life controls before interpretation |
| 9. Licensed data and empirical validation | Provider connector, real-chain validation, sensitivity analysis, performance review | Blocked on external data/access | Genuine KRX history, reconciled contract terms, realistic costs, and robust out-of-sample findings |

## Offline end-to-end workflow

Run without network access or market data:

```bash
skhynix-vol demo-research
```

The command generates a deterministic regime-switching underlying path and simulated Black-76 option quotes, then runs the implemented RV, quote-quality, IV-surface, surface-projection, IV walk-forward, and straddle simulation stages. Data is written under `data/demo_synthetic/`; reports use `_synthetic_demo` suffixes. Adjust `--seed`, `--sessions`, and `--quote-days` to regenerate the fixture. These outputs verify interfaces and artifact flow only; they must not be used to claim forecast quality or trading performance.

## Work remaining after the demo

1. Select a licensed vendor and obtain its authorized access method, field definitions, units, timestamp meaning, and contract metadata.
2. Add a vendor adapter that maps source records into the normalized quote schema without discarding raw fields.
3. Reconcile genuine quotes, underlying timestamps, forwards/carry, expiry conventions, multiplier, and settlement terms.
4. Examine real-chain quote quality, IV repricing, static-arbitrage violations, projection changes, and supported surface coverage.
5. Re-run chronological evaluations on genuine data and report uncertainty, turnover, fees, spread sensitivity, drawdown, and benchmark comparisons.
6. Extend the simplified strategy model for the research question (e.g. contract rolls, exercise/settlement cash flows, margin and financing) before making economic claims.

No provider-specific connector, licensed dataset, or trading-ready strategy is included in the current completion status.
