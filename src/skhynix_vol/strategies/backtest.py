"""Quote-based hypothetical ATM straddle backtest with bid/ask execution."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


def backtest_straddles(quotes: pd.DataFrame, rv_predictions: pd.DataFrame,
                       underlying_prices: pd.DataFrame, horizon: int = 20,
                       min_vol_edge: float = 0.02, fee_per_contract: float = 0.0,
                       contract_multiplier: float | None = None,
                       hedge_mode: str = "none", option_slippage_bps: float = 0.0,
                       hedge_slippage_bps: float = 0.0, hedge_fee_per_share: float = 0.0,
                       max_relative_spread: float | None = None,
                       min_option_volume: float = 0.0, min_open_interest: float = 0.0,
                       hedge_risk_free_rate: float | None = None,
                       hedge_dividend_yield: float | None = None,
                       contracts: int = 1, max_volume_participation: float | None = None,
                       max_open_interest_participation: float | None = None,
                       max_cumulative_loss_krw: float | None = None,
                       short_margin_per_contract_krw: float | None = None,
                       capital_base_krw: float | None = None,
                       annual_financing_rate: float = 0.0,
                       annual_stock_borrow_rate: float = 0.0,
                       max_cumulative_drawdown_krw: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate non-overlapping ATM straddles from historical forecast/quote data.

    A forecast above matched call/put IV by `min_vol_edge` buys a straddle;
    below by that amount sells one. Enter at ask for buys/bid for sells, exit
    exactly `horizon` underlying sessions later at bid/ask respectively. The
    optional daily delta hedge uses European Black-Scholes deltas from the
    same-date option quotes and close prices. Missing daily quotes/carry inputs
    cause a hedged trade to be skipped instead of silently interpolated.
    """
    required_q = {"date", "expiry", "option_type", "strike", "bid", "ask", "implied_volatility", "underlying_close"}
    if not required_q.issubset(quotes.columns):
        raise ValueError(f"Quotes missing: {', '.join(sorted(required_q - set(quotes.columns)))}")
    required_p = {"date", "horizon", "har_rv"}
    if not required_p.issubset(rv_predictions.columns):
        raise ValueError(f"RV predictions missing: {', '.join(sorted(required_p - set(rv_predictions.columns)))}")
    if "date" not in underlying_prices:
        raise ValueError("underlying_prices must contain date")
    if not np.isfinite([min_vol_edge, fee_per_contract, option_slippage_bps,
                        hedge_slippage_bps, hedge_fee_per_share,
                        min_option_volume, min_open_interest,
                        annual_financing_rate, annual_stock_borrow_rate]).all():
        raise ValueError("Edges, costs, and liquidity thresholds must be finite")
    if annual_financing_rate < 0 or annual_stock_borrow_rate < 0:
        raise ValueError("Financing and stock-borrow rates cannot be negative")
    for name, value in (("short_margin_per_contract_krw", short_margin_per_contract_krw),
                        ("capital_base_krw", capital_base_krw)):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    if not isinstance(contracts, (int, np.integer)) or contracts < 1:
        raise ValueError("contracts must be a positive integer")
    for name, value in (("max_volume_participation", max_volume_participation),
                        ("max_open_interest_participation", max_open_interest_participation)):
        if value is not None and (not np.isfinite(value) or not 0 < value <= 1):
            raise ValueError(f"{name} must be in (0, 1]")
    if max_cumulative_loss_krw is not None and (not np.isfinite(max_cumulative_loss_krw) or max_cumulative_loss_krw <= 0):
        raise ValueError("max_cumulative_loss_krw must be a finite positive loss budget")
    if max_cumulative_drawdown_krw is not None and (not np.isfinite(max_cumulative_drawdown_krw) or max_cumulative_drawdown_krw <= 0):
        raise ValueError("max_cumulative_drawdown_krw must be a finite positive drawdown limit")
    if (horizon < 1 or min_vol_edge < 0 or fee_per_contract < 0 or
        option_slippage_bps < 0 or hedge_slippage_bps < 0 or hedge_fee_per_share < 0 or
        min_option_volume < 0 or min_open_interest < 0):
        raise ValueError("horizon must be positive; edges, costs, and liquidity thresholds cannot be negative")
    if hedge_mode not in {"none", "daily_delta"}:
        raise ValueError("hedge_mode must be 'none' or 'daily_delta'")
    if max_relative_spread is not None and (not np.isfinite(max_relative_spread) or not 0 <= max_relative_spread <= 2):
        raise ValueError("max_relative_spread must be between 0 and 2")
    if hedge_mode == "daily_delta" and "close" not in underlying_prices:
        raise ValueError("Daily delta hedging requires unadjusted close prices")
    q = quotes.copy()
    q["date"] = pd.to_datetime(q.date).dt.normalize()
    q["expiry"] = pd.to_datetime(q.expiry).dt.normalize()
    q["option_type"] = q.option_type.astype(str).str.lower().replace({"c": "call", "p": "put"})
    q["bid"] = pd.to_numeric(q.bid, errors="coerce")
    q["ask"] = pd.to_numeric(q.ask, errors="coerce")
    if "quote_quality_flag" in q:
        q = q[q.quote_quality_flag.eq("ok")]
    else:
        q = q[q.bid.ge(0) & q.ask.ge(q.bid)]
    data_origins = q.data_origin.dropna().astype(str).unique() if "data_origin" in q else []
    data_origin = data_origins[0] if len(data_origins) == 1 else "mixed" if len(data_origins) > 1 else None
    is_synthetic = (q.synthetic_data_flag.fillna(False).astype(str).str.lower()
                    .isin({"true", "1", "yes"}).any()) if "synthetic_data_flag" in q else False
    contract_keys = ["date", "expiry", "option_type", "strike"]
    if q.duplicated(contract_keys).any():
        if "quote_timestamp" not in q:
            raise ValueError("Duplicate same-day contract quotes need quote_timestamp to select the closing snapshot")
        q["quote_timestamp"] = pd.to_datetime(q.quote_timestamp, utc=True, errors="coerce")
        q = q.sort_values("quote_timestamp").drop_duplicates(contract_keys, keep="last")
    prices = underlying_prices.copy()
    prices["date"] = pd.to_datetime(prices.date).dt.normalize()
    prices = prices.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
    if "close" in prices:
        prices["close"] = pd.to_numeric(prices.close, errors="coerce")
    if "dividends" not in prices:
        prices["dividends"] = 0.0
    prices["dividends"] = pd.to_numeric(prices.dividends, errors="coerce").fillna(0.0)
    calendar = pd.Index(prices.index.sort_values())
    index_by_date = {date: i for i, date in enumerate(calendar)}
    quotes_by_date = {date: group for date, group in q.groupby("date", sort=False)}
    preds = rv_predictions[rv_predictions.horizon.eq(horizon)].copy()
    preds["date"] = pd.to_datetime(preds.date).dt.normalize()
    # Forecast folds can cover years before the option dataset begins. Ignore
    # those dates up front, and use per-date quote slices in the loop rather
    # than rescanning the complete chain file for every forecast row.
    preds = preds[preds.date.isin(quotes_by_date)]
    preds = preds.sort_values("date")
    trades = []
    skipped_liquidity = 0
    skipped_hedge_coverage = 0
    skipped_capital_limit = 0
    last_exit_index = -1
    realized_net_pnl = 0.0
    risk_halt = False
    drawdown_halt = False
    peak_realized_pnl = 0.0
    peak_realized_pnl = 0.0
    for forecast in preds.itertuples(index=False):
        if risk_halt:
            break
        entry_date = forecast.date
        entry_index = index_by_date.get(entry_date)
        if entry_index is None or entry_index <= last_exit_index or entry_index + horizon >= len(calendar):
            continue
        exit_date = calendar[entry_index + horizon]
        on_entry = quotes_by_date.get(entry_date, q.iloc[0:0])
        on_entry = on_entry[on_entry.option_type.isin(["call", "put"])].copy()
        on_exit_date = quotes_by_date.get(exit_date, q.iloc[0:0])
        if on_entry.empty or on_exit_date.empty:
            continue
        calls = on_entry[on_entry.option_type.eq("call")]
        puts = on_entry[on_entry.option_type.eq("put")]
        pairs = calls.merge(puts, on=["date", "expiry", "strike"], suffixes=("_call", "_put"))
        pairs = pairs[pairs.expiry > exit_date].copy()
        if pairs.empty:
            continue
        for side in ("call", "put"):
            bid, ask = pairs[f"bid_{side}"], pairs[f"ask_{side}"]
            pairs[f"relative_spread_{side}"] = np.where((ask + bid) > 0, (ask - bid) / ((ask + bid) / 2.0), np.inf)
        pairs["pair_relative_spread"] = pairs[["relative_spread_call", "relative_spread_put"]].max(axis=1)
        eligible = pd.Series(True, index=pairs.index)
        if max_relative_spread is not None:
            eligible &= pairs.pair_relative_spread.le(max_relative_spread)
        for side in ("call", "put"):
            if min_option_volume > 0:
                field = f"volume_{side}"
                eligible &= pd.to_numeric(pairs[field], errors="coerce").ge(min_option_volume) if field in pairs else False
            if min_open_interest > 0:
                field = f"open_interest_{side}"
                eligible &= pd.to_numeric(pairs[field], errors="coerce").ge(min_open_interest) if field in pairs else False
            if max_volume_participation is not None:
                field = f"volume_{side}"
                eligible &= (pd.to_numeric(pairs[field], errors="coerce") >= contracts / max_volume_participation) if field in pairs else False
            if max_open_interest_participation is not None:
                field = f"open_interest_{side}"
                eligible &= (pd.to_numeric(pairs[field], errors="coerce") >= contracts / max_open_interest_participation) if field in pairs else False
        if not eligible.any():
            skipped_liquidity += 1
            continue
        pairs = pairs.loc[eligible].copy()
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
        exit_quote_pair = pd.concat([exit_call.iloc[[0]], exit_put.iloc[[0]]], ignore_index=True)
        if not _passes_quote_liquidity(exit_quote_pair, max_relative_spread, min_option_volume,
                min_open_interest, contracts, max_volume_participation,
                max_open_interest_participation):
            skipped_liquidity += 1
            continue
        forecast_rv = float(forecast.har_rv)
        atm_iv = float((pair.implied_volatility_call + pair.implied_volatility_put) / 2.0)
        edge = forecast_rv - atm_iv
        if edge > min_vol_edge:
            side = "long"
            entry_cash = float(pair.ask_call + pair.ask_put)
            exit_cash = float(exit_call.bid.iloc[0] + exit_put.bid.iloc[0])
            multiplier = _multiplier(pair, contract_multiplier)
            gross = (exit_cash - entry_cash) * multiplier * contracts
        elif edge < -min_vol_edge:
            side = "short"
            entry_cash = float(pair.bid_call + pair.bid_put)
            exit_cash = float(exit_call.ask.iloc[0] + exit_put.ask.iloc[0])
            multiplier = _multiplier(pair, contract_multiplier)
            gross = (entry_cash - exit_cash) * multiplier * contracts
        else:
            continue
        option_slippage = (entry_cash + exit_cash) * multiplier * contracts * option_slippage_bps / 10_000.0
        option_turnover_krw = (entry_cash + exit_cash) * multiplier * contracts
        hedge_pnl = hedge_fees = hedge_slippage = hedge_turnover = 0.0
        hedge_max_abs_shares = 0.0
        hedge_peak_notional = 0.0
        hedge_financing_notional_days = hedge_borrow_notional_days = 0.0
        if hedge_mode == "daily_delta":
            contract_quotes = q[q.expiry.eq(pair.expiry) & np.isclose(q.strike, pair.strike)]
            try:
                (hedge_pnl, hedge_fees, hedge_slippage, hedge_turnover, hedge_max_abs_shares,
                 hedge_peak_notional, hedge_financing_notional_days,
                 hedge_borrow_notional_days) = _daily_delta_hedge(
                    contract_quotes, prices, calendar, entry_index, horizon, exit_date,
                    side, multiplier, hedge_slippage_bps, hedge_fee_per_share,
                    hedge_risk_free_rate, hedge_dividend_yield,
                    max_relative_spread, min_option_volume, min_open_interest,
                    contracts, max_volume_participation, max_open_interest_participation)
            except (ValueError, KeyError, IndexError):
                skipped_hedge_coverage += 1
                continue
        fees = 4.0 * fee_per_contract * contracts + hedge_fees
        gross_total = gross + hedge_pnl
        holding_days = max(0, (pd.Timestamp(exit_date) - pd.Timestamp(entry_date)).days)
        option_capital = entry_cash * multiplier * contracts if side == "long" else 0.0
        margin_requirement = (short_margin_per_contract_krw * contracts
                              if side == "short" and short_margin_per_contract_krw is not None else np.nan)
        hedge_capital = hedge_peak_notional
        capital_requirement = (option_capital + hedge_capital if side == "long" else
            margin_requirement + hedge_capital if np.isfinite(margin_requirement) else np.nan)
        if capital_base_krw is not None:
            if not np.isfinite(capital_requirement) or capital_requirement > capital_base_krw:
                skipped_capital_limit += 1
                continue
        financing_basis = option_capital if side == "long" else (margin_requirement if np.isfinite(margin_requirement) else 0.0)
        financing_cost = (financing_basis * annual_financing_rate * holding_days / 365.0 +
                          hedge_financing_notional_days * annual_financing_rate / 365.0)
        stock_borrow_cost = hedge_borrow_notional_days * annual_stock_borrow_rate / 365.0
        all_costs = fees + option_slippage + hedge_slippage + financing_cost + stock_borrow_cost
        trades.append({"entry_date": entry_date, "exit_date": exit_date, "horizon_sessions": horizon,
            "side": side, "strike": float(pair.strike), "expiry": pair.expiry,
            "entry_dte": int((pair.expiry - entry_date).days), "target_tenor_days": target_tenor_days,
            "forecast_rv": forecast_rv,
            "matched_atm_iv": atm_iv, "vol_edge": edge, "contract_multiplier": multiplier,
            "entry_premium_per_share": entry_cash, "exit_premium_per_share": exit_cash,
            "contracts": contracts, "option_pnl_krw": gross, "hedge_pnl_krw": hedge_pnl,
            "gross_pnl_krw": gross_total, "option_fees_krw": 4.0 * fee_per_contract * contracts,
            "hedge_fees_krw": hedge_fees, "fees_krw": fees,
            "option_slippage_krw": option_slippage, "hedge_slippage_krw": hedge_slippage,
            "option_turnover_krw": option_turnover_krw,
            "hedge_turnover_shares": hedge_turnover, "hedge_max_abs_shares": hedge_max_abs_shares,
            "hedge_peak_notional_krw": hedge_peak_notional,
            "financing_cost_krw": financing_cost, "stock_borrow_cost_krw": stock_borrow_cost,
            "capital_requirement_krw": capital_requirement,
            "capital_utilization_pct": (100.0 * capital_requirement / capital_base_krw
                if capital_base_krw is not None else np.nan),
            "net_pnl_krw": gross_total - all_costs, "entry_underlying": spot,
            "hedge_mode": hedge_mode, "option_slippage_bps": option_slippage_bps,
            "hedge_slippage_bps": hedge_slippage_bps,
            "max_relative_spread": max_relative_spread,
            "min_option_volume": min_option_volume, "min_open_interest": min_open_interest,
            "max_volume_participation": max_volume_participation,
            "max_open_interest_participation": max_open_interest_participation,
            "max_cumulative_loss_krw": max_cumulative_loss_krw,
            "short_margin_per_contract_krw": short_margin_per_contract_krw,
            "capital_base_krw": capital_base_krw,
            "annual_financing_rate": annual_financing_rate,
            "annual_stock_borrow_rate": annual_stock_borrow_rate,
            "holding_calendar_days": holding_days})
        realized_net_pnl += gross_total - all_costs
        if max_cumulative_loss_krw is not None and realized_net_pnl <= -max_cumulative_loss_krw:
            risk_halt = True
        peak_realized_pnl = max(peak_realized_pnl, realized_net_pnl)
        if (max_cumulative_drawdown_krw is not None and
                peak_realized_pnl - realized_net_pnl >= max_cumulative_drawdown_krw):
            drawdown_halt = True
        risk_halt = risk_halt or drawdown_halt
        last_exit_index = entry_index + horizon
    trade_frame = pd.DataFrame(trades)
    if trade_frame.empty:
        columns = ["entry_date", "exit_date", "horizon_sessions", "side", "strike", "expiry", "entry_dte",
                   "target_tenor_days", "forecast_rv", "matched_atm_iv", "vol_edge", "contract_multiplier",
                   "entry_premium_per_share", "exit_premium_per_share", "option_pnl_krw", "hedge_pnl_krw",
                   "gross_pnl_krw", "option_fees_krw", "hedge_fees_krw", "fees_krw",
                   "option_slippage_krw", "hedge_slippage_krw", "hedge_turnover_shares",
                   "option_turnover_krw",
                   "hedge_max_abs_shares", "hedge_peak_notional_krw", "financing_cost_krw", "stock_borrow_cost_krw",
                   "capital_requirement_krw", "capital_utilization_pct", "net_pnl_krw",
                   "entry_underlying", "hedge_mode", "contracts"]
        trades = pd.DataFrame(columns=columns)
        summary = _empty_summary(horizon)
        summary["delta_hedging"] = hedge_mode == "daily_delta"
        summary["hedge_mode"] = hedge_mode
        summary["option_slippage_bps"] = option_slippage_bps
        summary["hedge_slippage_bps"] = hedge_slippage_bps
        summary["hedge_fee_per_share_krw"] = hedge_fee_per_share
        summary["max_relative_spread"] = max_relative_spread
        summary["min_option_volume"] = min_option_volume
        summary["min_open_interest"] = min_open_interest
        summary["contracts_per_trade"] = contracts
        summary["max_volume_participation"] = max_volume_participation
        summary["max_open_interest_participation"] = max_open_interest_participation
        summary["max_cumulative_loss_krw"] = max_cumulative_loss_krw
        summary["max_cumulative_drawdown_krw"] = max_cumulative_drawdown_krw
        summary["risk_halt_triggered"] = risk_halt
        summary["drawdown_halt_triggered"] = drawdown_halt
        summary["capital_base_krw"] = capital_base_krw
        summary["short_margin_per_contract_krw"] = short_margin_per_contract_krw
        summary["annual_financing_rate"] = annual_financing_rate
        summary["annual_stock_borrow_rate"] = annual_stock_borrow_rate
        summary["return_on_capital_pct"] = np.nan
        summary["estimated_capital_model"] = "user_supplied_margin_plus_peak_hedge_notional"
        summary["skipped_capital_limit_candidates"] = skipped_capital_limit
        summary["liquidity_filters"] = (max_relative_spread is not None or min_option_volume > 0 or
            min_open_interest > 0 or max_volume_participation is not None or
            max_open_interest_participation is not None)
        summary["skipped_liquidity_candidates"] = skipped_liquidity
        summary["skipped_hedge_coverage_candidates"] = skipped_hedge_coverage
        if data_origin is not None:
            trades["data_origin"] = data_origin
            summary["data_origin"] = data_origin
        trades["synthetic_data_flag"] = is_synthetic
        summary["synthetic_data_flag"] = is_synthetic
        return trades, pd.DataFrame([summary])
    if data_origin is not None:
        trade_frame["data_origin"] = data_origin
    trade_frame["synthetic_data_flag"] = is_synthetic
    pnl = trade_frame.net_pnl_krw.to_numpy(float)
    equity = np.r_[0.0, np.cumsum(pnl)]
    drawdowns = np.maximum.accumulate(equity) - equity
    positives = pnl[pnl > 0].sum()
    negatives = -pnl[pnl < 0].sum()
    summary = {"horizon_sessions": horizon, "trade_count": len(trade_frame),
        "long_trades": int(trade_frame.side.eq("long").sum()), "short_trades": int(trade_frame.side.eq("short").sum()),
        "total_gross_pnl_krw": float(trade_frame.gross_pnl_krw.sum()),
        "total_fees_krw": float(trade_frame.fees_krw.sum()),
        "total_slippage_krw": float((trade_frame.option_slippage_krw + trade_frame.hedge_slippage_krw).sum()),
        "total_option_turnover_krw": float(trade_frame.option_turnover_krw.sum()),
        "break_even_additional_option_slippage_bps": (
            float(pnl.sum() / trade_frame.option_turnover_krw.sum() * 10_000.0)
            if trade_frame.option_turnover_krw.sum() > 0 else np.nan),
        "total_financing_cost_krw": float(trade_frame.financing_cost_krw.sum()),
        "total_stock_borrow_cost_krw": float(trade_frame.stock_borrow_cost_krw.sum()),
        "total_hedge_pnl_krw": float(trade_frame.hedge_pnl_krw.sum()),
        "total_hedge_turnover_shares": float(trade_frame.hedge_turnover_shares.sum()),
        "total_option_contracts": int(trade_frame.contracts.sum()),
        "contracts_per_trade": contracts,
        "peak_capital_requirement_krw": float(trade_frame.capital_requirement_krw.max()) if trade_frame.capital_requirement_krw.notna().any() else np.nan,
        "max_capital_utilization_pct": float(trade_frame.capital_utilization_pct.max()) if trade_frame.capital_utilization_pct.notna().any() else np.nan,
        "capital_base_krw": capital_base_krw,
        "return_on_capital_pct": (float(pnl.sum() / capital_base_krw * 100.0)
            if capital_base_krw is not None else np.nan),
        "total_net_pnl_krw": float(pnl.sum()), "mean_net_pnl_krw": float(np.mean(pnl)),
        "median_net_pnl_krw": float(np.median(pnl)),
        "pnl_std_krw": float(np.std(pnl, ddof=1)) if len(pnl) > 1 else np.nan,
        "var_95_loss_krw": float(max(0.0, -np.quantile(pnl, 0.05))),
        "expected_shortfall_95_loss_krw": float(max(0.0, -np.mean(pnl[pnl <= np.quantile(pnl, 0.05)]))),
        "win_rate": float(np.mean(pnl > 0)),
        "profit_factor": float(positives / negatives) if negatives else (float("inf") if positives else np.nan),
        "max_realized_drawdown_krw": float(drawdowns.max()),
        "first_entry": trade_frame.entry_date.min(), "last_exit": trade_frame.exit_date.max(),
        "fee_per_leg_per_contract_krw": fee_per_contract,
        "hedge_mode": hedge_mode, "option_slippage_bps": option_slippage_bps,
        "hedge_slippage_bps": hedge_slippage_bps,
        "hedge_fee_per_share_krw": hedge_fee_per_share,
        "max_relative_spread": max_relative_spread,
        "min_option_volume": min_option_volume, "min_open_interest": min_open_interest,
        "max_volume_participation": max_volume_participation,
        "max_open_interest_participation": max_open_interest_participation,
        "max_cumulative_loss_krw": max_cumulative_loss_krw,
        "max_cumulative_drawdown_krw": max_cumulative_drawdown_krw,
        "short_margin_per_contract_krw": short_margin_per_contract_krw,
        "annual_financing_rate": annual_financing_rate,
        "annual_stock_borrow_rate": annual_stock_borrow_rate,
        "estimated_capital_model": "user_supplied_margin_plus_peak_hedge_notional",
        "risk_halt_triggered": risk_halt,
        "drawdown_halt_triggered": drawdown_halt,
        "delta_hedging": hedge_mode == "daily_delta", "margin_model": False,
        "liquidity_filters": (max_relative_spread is not None or min_option_volume > 0 or
            min_open_interest > 0 or max_volume_participation is not None or
            max_open_interest_participation is not None),
        "skipped_liquidity_candidates": skipped_liquidity,
        "skipped_hedge_coverage_candidates": skipped_hedge_coverage,
        "skipped_capital_limit_candidates": skipped_capital_limit}
    if data_origin is not None:
        summary["data_origin"] = data_origin
    summary["synthetic_data_flag"] = is_synthetic
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
            "delta_hedging": False, "margin_model": False,
            "total_slippage_krw": 0.0, "total_hedge_pnl_krw": 0.0,
            "total_option_turnover_krw": 0.0,
            "break_even_additional_option_slippage_bps": np.nan,
            "total_hedge_turnover_shares": 0.0, "var_95_loss_krw": np.nan,
            "total_financing_cost_krw": 0.0, "total_stock_borrow_cost_krw": 0.0,
            "total_option_contracts": 0, "contracts_per_trade": 1,
            "peak_capital_requirement_krw": np.nan,
            "max_capital_utilization_pct": np.nan,
            "return_on_capital_pct": np.nan,
            "expected_shortfall_95_loss_krw": np.nan,
            "mean_net_pnl_krw": np.nan, "median_net_pnl_krw": np.nan,
            "pnl_std_krw": np.nan, "liquidity_filters": False,
            "skipped_liquidity_candidates": 0,
            "skipped_hedge_coverage_candidates": 0,
            "skipped_capital_limit_candidates": 0}


def _passes_quote_liquidity(quotes: pd.DataFrame, max_relative_spread: float | None,
                            min_volume: float, min_open_interest: float,
                            contracts: int = 1, max_volume_participation: float | None = None,
                            max_open_interest_participation: float | None = None) -> bool:
    if quotes.empty:
        return False
    bid = pd.to_numeric(quotes.bid, errors="coerce")
    ask = pd.to_numeric(quotes.ask, errors="coerce")
    if (bid.isna() | ask.isna() | bid.lt(0) | ask.lt(bid)).any():
        return False
    spread = (ask - bid) / ((ask + bid) / 2.0).replace(0, np.nan)
    if max_relative_spread is not None and (spread.isna() | spread.gt(max_relative_spread)).any():
        return False
    for column, minimum, max_participation in (
            ("volume", min_volume, max_volume_participation),
            ("open_interest", min_open_interest, max_open_interest_participation)):
        if minimum <= 0 and max_participation is None:
            continue
        values = pd.to_numeric(quotes[column], errors="coerce") if column in quotes else pd.Series(np.nan, index=quotes.index)
        if ((minimum > 0 and (values.isna() | values.lt(minimum)).any()) or
            (max_participation is not None and
             (values.isna() | values.lt(contracts / max_participation)).any())):
            return False
    return True


def _daily_delta_hedge(contract_quotes: pd.DataFrame, prices: pd.DataFrame,
                       calendar: pd.Index, entry_index: int, horizon: int,
                       exit_date: pd.Timestamp, option_side: str, multiplier: float,
                       slippage_bps: float, fee_per_share: float,
                       rate_override: float | None, dividend_override: float | None,
                       max_relative_spread: float | None, min_volume: float,
                       min_open_interest: float, contracts: int = 1,
                       max_volume_participation: float | None = None,
                       max_open_interest_participation: float | None = None) -> tuple[float, float, float, float, float, float, float, float]:
    """Close-to-close delta hedge with explicit hedge execution costs."""
    entry_date = calendar[entry_index]
    entry_quotes = contract_quotes[contract_quotes.date.eq(entry_date)]
    entry_spot = float(prices.loc[entry_date, "close"])
    entry_delta = _straddle_delta(entry_quotes, entry_date, entry_spot,
        float(entry_quotes.strike.iloc[0]), rate_override, dividend_override)
    position_sign = 1.0 if option_side == "long" else -1.0
    multiplier *= contracts
    hedge_shares = -position_sign * multiplier * entry_delta
    hedge_peak_notional = abs(hedge_shares) * entry_spot
    turnover = abs(hedge_shares)
    hedge_slippage = turnover * entry_spot * slippage_bps / 10_000.0
    hedge_fees = turnover * fee_per_share
    hedge_pnl = 0.0
    hedge_financing_notional_days = 0.0
    hedge_borrow_notional_days = 0.0
    previous_spot = entry_spot
    previous_date = entry_date
    max_abs_shares = abs(hedge_shares)
    strike = float(entry_quotes.strike.iloc[0])
    expiry = pd.Timestamp(entry_quotes.expiry.iloc[0])
    for step in range(1, horizon + 1):
        date = calendar[entry_index + step]
        daily = contract_quotes[contract_quotes.date.eq(date)]
        if not _passes_quote_liquidity(daily[daily.option_type.isin(["call", "put"])],
                max_relative_spread, min_volume, min_open_interest, contracts,
                max_volume_participation, max_open_interest_participation):
            raise ValueError("Missing or illiquid option quote during hedge window")
        daily = daily[daily.option_type.isin(["call", "put"])]
        if set(daily.option_type) != {"call", "put"}:
            raise ValueError("Missing call or put during hedge window")
        spot = float(prices.loc[date, "close"])
        dividends = float(prices.loc[date, "dividends"])
        interval_days = max(0, (pd.Timestamp(date) - pd.Timestamp(previous_date)).days)
        hedge_financing_notional_days += max(hedge_shares, 0.0) * previous_spot * interval_days
        hedge_borrow_notional_days += max(-hedge_shares, 0.0) * previous_spot * interval_days
        hedge_pnl += hedge_shares * (spot - previous_spot + dividends)
        if date == exit_date:
            traded = abs(hedge_shares)
            turnover += traded
            hedge_slippage += traded * spot * slippage_bps / 10_000.0
            hedge_fees += traded * fee_per_share
            hedge_shares = 0.0
            break
        delta = _straddle_delta(daily, date, spot, strike, rate_override, dividend_override)
        target = -position_sign * multiplier * delta
        traded = abs(target - hedge_shares)
        turnover += traded
        hedge_slippage += traded * spot * slippage_bps / 10_000.0
        hedge_fees += traded * fee_per_share
        hedge_shares = target
        hedge_peak_notional = max(hedge_peak_notional, abs(hedge_shares) * spot)
        max_abs_shares = max(max_abs_shares, abs(hedge_shares))
        previous_spot = spot
        previous_date = date
    return (hedge_pnl, hedge_fees, hedge_slippage, turnover, max_abs_shares, hedge_peak_notional,
            hedge_financing_notional_days, hedge_borrow_notional_days)


def _straddle_delta(quotes: pd.DataFrame, date: pd.Timestamp, spot: float,
                    strike: float, rate_override: float | None,
                    dividend_override: float | None) -> float:
    if quotes.empty or not np.isfinite(spot) or spot <= 0 or not np.isfinite(strike) or strike <= 0:
        raise ValueError("Cannot calculate option delta without current quotes and a positive spot")
    tenor = (pd.Timestamp(quotes.expiry.iloc[0]) - date).days / 365.0
    if tenor <= 0:
        raise ValueError("Option expired before hedge exit")
    deltas = []
    for kind in ("call", "put"):
        rows = quotes[quotes.option_type.eq(kind)]
        if rows.empty:
            raise ValueError("Both call and put quotes are required to hedge a straddle")
        row = rows.iloc[0]
        rate = _input_or_override(row.get("risk_free_rate"), rate_override)
        dividend = row.get("dividend_yield", np.nan)
        if pd.isna(dividend):
            forward = row.get("forward_price", np.nan)
            if pd.isna(forward):
                dividend = dividend_override
            else:
                dividend = rate - math.log(float(forward) / spot) / tenor
        dividend = _input_or_override(dividend, dividend_override)
        volatility = float(row.implied_volatility)
        if not np.isfinite(volatility) or volatility <= 0:
            raise ValueError("Positive quote IV is required for delta hedging")
        if not np.isfinite([rate, dividend]).all():
            raise ValueError("Finite risk-free and dividend yields are required for delta hedging")
        d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * volatility ** 2) * tenor) / (volatility * math.sqrt(tenor))
        n_d1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
        deltas.append(math.exp(-dividend * tenor) * (n_d1 if kind == "call" else n_d1 - 1.0))
    return float(sum(deltas))


def _input_or_override(value, override: float | None) -> float:
    if pd.notna(value):
        result = float(value)
    elif override is not None:
        result = float(override)
    else:
        raise ValueError("Delta hedge needs quote carry fields or explicit rate/dividend overrides")
    if not np.isfinite(result):
        raise ValueError("Delta hedge carry inputs must be finite")
    return result
