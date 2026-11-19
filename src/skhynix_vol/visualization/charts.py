"""Exploratory chart exports for price and volatility research."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def create_exploratory_charts(prices: pd.DataFrame, features: pd.DataFrame,
                              output_dir: str | Path = "reports") -> list[Path]:
    """Save adjusted-price and trailing-volatility history charts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = prices.set_index("date") if "date" in prices else prices.copy()
    f = features.set_index("date") if "date" in features else features.copy()
    paths: list[Path] = []
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pd.to_datetime(p.index), p["adj_close"], color="#136f63", linewidth=1.2)
    ax.set(title="SK Hynix adjusted close", ylabel="KRW", xlabel="Date")
    ax.grid(alpha=0.25); fig.tight_layout()
    path = out / "sk_hynix_price_history.png"; fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in ("rv_5", "rv_20", "rv_60", "ewma_vol_20"):
        if col in f: ax.plot(pd.to_datetime(f.index), f[col], label=col, linewidth=1.0)
    ax.set(title="SK Hynix trailing annualized realized volatility", ylabel="Annualized volatility", xlabel="Date")
    ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
    path = out / "sk_hynix_realized_volatility.png"; fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)
    return paths
