"""Coverage-aware implied-volatility surface images."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_iv_surface(surface: pd.DataFrame, output_path: str | Path,
                    value_column: str = "implied_volatility",
                    title: str = "SK Hynix implied-volatility surface") -> Path:
    """Render a 3D surface with unsupported grid cells left blank."""
    required = {"log_forward_moneyness", "tenor_days", value_column}
    if not required.issubset(surface.columns):
        raise ValueError(f"Surface image is missing: {', '.join(sorted(required - set(surface.columns)))}")
    latest_date = pd.to_datetime(surface.date).max() if "date" in surface else None
    data = surface.loc[surface.date.eq(latest_date)] if latest_date is not None else surface
    grid = data.pivot_table(index="tenor_days", columns="log_forward_moneyness", values=value_column, aggfunc="mean")
    if grid.empty or grid.notna().sum().sum() < 3:
        raise ValueError("At least three supported IV grid points are required to plot")
    k, dte = np.meshgrid(grid.columns.to_numpy(float), grid.index.to_numpy(float))
    iv = grid.to_numpy(float)
    out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(k, dte, iv, cmap="viridis", edgecolor="none", antialiased=True)
    ax.set_title(title if latest_date is None else f"{title} — {latest_date:%Y-%m-%d}")
    ax.set_xlabel("Log forward moneyness  ln(K/F)")
    ax.set_ylabel("Tenor (calendar days)")
    ax.set_zlabel("Implied volatility (decimal)")
    fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.12, label="IV (decimal)")
    fig.tight_layout(); fig.savefig(out, dpi=170); plt.close(fig)
    return out
