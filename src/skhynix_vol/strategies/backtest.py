"""Quote-based hypothetical ATM straddle backtest with bid/ask execution."""
from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_straddles(quotes: pd.DataFrame, rv_predictions: pd.DataFrame,
                       underlying_prices: pd.DataFrame, horizon: int = 20,
                       min_vol_edge: float = 0.02, fee_per_contract: float = 0.0,
                       contract_multiplier: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate non-overlapping ATM straddles from historical forecast/quote data.

    A forecast above matched call/put IV by `min_vol_edge` buys a straddle;
    below by that amount sells one. Enter at ask for buys/bid for sells, exit
    exactly `horizon` underlying sessions later at bid/ask respectively. Rows
    without exact entry/exit quote coverage are skipped. Fee is charged on all
    four option legs. No delta hedge or margin model is applied; results are
    grossly limited to quoted straddle P&L less explicit fees and spread costs.
    """
    required_q = {"date", "expiry", "option_type", "strike", "bid", "ask", "implied_volatility", "underlying_close"}
    if not required_q.issubset(quotes.columns):
        raise ValueError(f"Quotes missing: {', '.join(sorted(required_q - set(quotes.columns)))}")
    required_p = {"date", "horizon", "har_rv"}
    if not required_p.issubset(rv_predictions.columns):
        raise ValueError(f"RV predictions missing: {', '.join(sorted(required_p - set(rv_predictions.columns)))}")
    if "date" not in underlying_prices:
        raise ValueError("underlying_prices must contain date")
    if horizon < 1 or min_vol_edge < 0 or fee_per_contract < 0:
        raise ValueError("horizon must be positive; edge and fees cannot be negative")
    q = quotes.copy()
    q["date"] = pd.to_datetime(q.date).dt.normalize()
    q["expiry"] = pd.to_datetime(q.expiry).dt.normalize()
    q["option_type"] = q.option_type.astype(str).str.lower().replace({"c": "call", "p": "put"})
    if "quote_quality_flag" in q:
        q = q[q.quote_quality_flag.eq("ok")]
    contract_keys = ["date", "expiry", "option_type", "strike"]
    if q.duplicated(contract_keys).any():
        if "quote_timestamp" not in q:
            raise ValueError("Duplicate same-day contract quotes need quote_timestamp to select the closing snapshot")
        q["quote_timestamp"] = pd.to_datetime(q.quote_timestamp, utc=True, errors="coerce")
        q = q.sort_values("quote_timestamp").drop_duplicates(contract_keys, keep="last")
    prices = underlying_prices.copy()
    prices["date"] = pd.to_datetime(prices.date).dt.normalize()
    calendar = pd.Index(prices.date.drop_duplicates().sort_values())
    index_by_date = {date: i for i, date in enumerate(calendar)}
    preds = rv_predictions[rv_predictions.horizon.eq(horizon)].copy()
    preds["date"] = pd.to_datetime(preds.date).dt.normalize()
    preds = preds.sort_values("date")
    trades = []
    last_exit_index = -1
    for forecast in preds.itertuples(index=False):
        entry_date = forecast.date
        entry_index = index_by_date.get(entry_date)
        if entry_index is None or entry_index <= last_exit_index or entry_index + horizon >= len(calendar):
            continue
        exit_date = calendar[entry_index + horizon]
        on_entry = q[q.date.eq(entry_date) & q.option_type.isin(["call", "put"])].copy()
        on_exit_date = q[q.date.eq(exit_date)].copy()
        if on_entry.empty or on_exit_date.empty:
            continue
        calls = on_entry[on_entry.option_type.eq("call")]
        puts = on_entry[on_entry.option_type.eq("put")]
        pairs = calls.merge(puts, on=["date", "expiry", "strike"], suffixes=("_call", "_put"))
        pairs = pairs[pairs.expiry > exit_date].copy()
        if pairs.empty:
            continue
        spot = float(pairs.underlying_close_call.iloc[0])
        pairs["abs_log_moneyness"] = np.abs(np.log(pairs.strike / spot))
        target_tenor_days = horizon * 365.0 / 252.0
        pairs["expiry_distance"] = np.abs((pairs.expiry - entry_date).dt.days - target_tenor_days)
        pair = pairs.sort_values(["abs_log_moneyness", "expiry_distance"]).iloc[0]
        exit_call = on_exit_date[(on_exit_date.option_type.eq("call")) &
            on_exit_date.expiry.eq(pair.expiry) & np.isclose(on_exit_date.strike, pair.strike)]
        exit_put = on_exit_date[(on_exit_date.option_type.eq("put")) &
            on_exit_date.expiry.eq(pair.expiry) & np.isclose(on_exit_date.strike, pair.strike)]
        if exit_call.empty or exit_put.empty:
            continue
        forecast_rv = float(forecast.har_rv)
        atm_iv = float((pair.implied_volatility_call + pair.implied_volatility_put) / 2.0)
        edge = forecast_rv - atm_iv
        if edge > min_vol_edge:
            side = "long"
            entry_cash = float(pair.ask_call + pair.ask_put)
            exit_cash = float(exit_call.bid.iloc[0] + exit_put.bid.iloc[0])
            multiplier = _multiplier(pair, contract_multiplier)
            gross = (exit_cash - entry_cash) * multiplier
        elif edge < -min_vol_edge:
            side = "short"
            entry_cash = float(pair.bid_call + pair.bid_put)
            exit_cash = float(exit_call.ask.iloc[0] + exit_put.ask.iloc[0])
            multiplier = _multiplier(pair, contract_multiplier)
            gross = (entry_cash - exit_cash) * multiplier
        else:
            continue
        fees = 4.0 * fee_per_contract
        trades.append({"entry_date": entry_date, "exit_date": exit_date, "horizon_sessions": horizon,
            "side": side, "strike": float(pair.strike), "expiry": pair.expiry,
            "entry_dte": int((pair.expiry - entry_date).days), "target_tenor_days": target_tenor_days,
            "forecast_rv": forecast_rv,
            "matched_atm_iv": atm_iv, "vol_edge": edge, "contract_multiplier": multiplier,
            "entry_premium_per_share": entry_cash, "exit_premium_per_share": exit_cash,
            "gross_pnl_krw": gross, "fees_krw": fees, "net_pnl_krw": gross - fees,
            "entry_underlying": spot})
        last_exit_index = entry_index + horizon
    trade_frame = pd.DataFrame(trades)
    if trade_frame.empty:
        columns = ["entry_date", "exit_date", "horizon_sessions", "side", "strike", "expiry", "entry_dte",
                   "target_tenor_days", "forecast_rv", "matched_atm_iv", "vol_edge", "contract_multiplier",
                   "entry_premium_per_share", "exit_premium_per_share", "gross_pnl_krw", "fees_krw",
                   "net_pnl_krw", "entry_underlying"]
        return pd.DataFrame(columns=columns), pd.DataFrame([_empty_summary(horizon)])
    pnl = trade_frame.net_pnl_krw.to_numpy(float)
    equity = np.r_[0.0, np.cumsum(pnl)]
    drawdowns = np.maximum.accumulate(equity) - equity
    positives = pnl[pnl > 0].sum()
    negatives = -pnl[pnl < 0].sum()
    summary = {"horizon_sessions": horizon, "trade_count": len(trade_frame),
        "long_trades": int(trade_frame.side.eq("long").sum()), "short_trades": int(trade_frame.side.eq("short").sum()),
        "total_gross_pnl_krw": float(trade_frame.gross_pnl_krw.sum()),
        "total_fees_krw": float(trade_frame.fees_krw.sum()), "total_net_pnl_krw": float(pnl.sum()),
        "win_rate": float(np.mean(pnl > 0)),
        "profit_factor": float(positives / negatives) if negatives else (float("inf") if positives else np.nan),
        "max_realized_drawdown_krw": float(drawdowns.max()),
        "first_entry": trade_frame.entry_date.min(), "last_exit": trade_frame.exit_date.max(),
        "fee_per_leg_per_contract_krw": fee_per_contract,
        "delta_hedging": False, "margin_model": False}
    return trade_frame, pd.DataFrame([summary])


def _multiplier(pair: pd.Series, override: float | None) -> float:
    if override is not None:
        value = float(override)
    else:
        call = pair.get("contract_multiplier_call", np.nan)
        put = pair.get("contract_multiplier_put", np.nan)
        if pd.isna(call) or pd.isna(put) or not np.isclose(call, put):
            raise ValueError("Contract multiplier is missing or inconsistent; supply an explicit override")
        value = float(call)
    if value <= 0:
        raise ValueError("Contract multiplier must be positive")
    return value


def _empty_summary(horizon: int) -> dict:
    return {"horizon_sessions": horizon, "trade_count": 0, "long_trades": 0, "short_trades": 0,
            "total_gross_pnl_krw": 0.0, "total_fees_krw": 0.0, "total_net_pnl_krw": 0.0,
            "win_rate": np.nan, "profit_factor": np.nan, "max_realized_drawdown_krw": 0.0,
            "first_entry": pd.NaT, "last_exit": pd.NaT,
            "delta_hedging": False, "margin_model": False}
