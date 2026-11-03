"""Matplotlib figures saved as PNG (headless-safe)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, GREY, RED, GREEN = "#2a6f97", "#e07a1f", "#8a8f98", "#c0392b", "#2e8b57"


def _finish(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_cashflows(cf: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(cf["Year"], cf["Equity_FCF"], color=BLUE, label="Equity free cash flow")
    ax.set_xlabel("Year")
    ax.set_ylabel("INR crore")
    ax2 = ax.twinx()
    ax2.plot(cf["Year"], cf["Cum_Equity_FCF"], color=ORANGE, lw=2, label="Cumulative")
    ax2.axhline(0, color=GREY, lw=0.8)
    ax2.set_ylabel("Cumulative, INR crore")
    ax.set_title("Equity cash flows and payback")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)
    return _finish(fig, path)


def plot_dscr(cf: pd.DataFrame, covenant: float, path: Path) -> Path:
    d = cf[cf["DSCR"].notna()]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(d["Year"], d["DSCR"], marker="o", color=BLUE, label="DSCR")
    ax.axhline(covenant, color=ORANGE, ls="--", label=f"Covenant {covenant:.2f}x")
    ax.axhline(1.0, color=RED, ls=":", label="Default threshold 1.00x")
    ax.set_xlabel("Year")
    ax.set_ylabel("DSCR (x)")
    ax.set_title("Debt Service Coverage Ratio over the loan tenor")
    ax.legend(frameon=False)
    return _finish(fig, path)


def plot_heatmap(grid: pd.DataFrame, title: str, path: Path, pct: bool = True) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    im = ax.imshow(grid.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(grid.shape[1]), labels=grid.columns)
    ax.set_yticks(range(grid.shape[0]), labels=grid.index)
    ax.set_xlabel(grid.columns.name)
    ax.set_ylabel(grid.index.name)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.values[i, j]
            ax.text(j, i, f"{v:.1%}" if pct else f"{v:.2f}x", ha="center", va="center",
                    fontsize=8, color="black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.85)
    return _finish(fig, path)


def plot_tornado(tor: pd.DataFrame, path: Path, metric_label: str = "Equity IRR") -> Path:
    t = tor.iloc[::-1]
    base = t["Base"].iloc[0]
    lo_col, hi_col = t.columns[1], t.columns[2]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(t))
    ax.barh(y, t[lo_col] - base, left=base, color=ORANGE, label=lo_col)
    ax.barh(y, t[hi_col] - base, left=base, color=BLUE, label=hi_col)
    ax.axvline(base, color="black", lw=0.8)
    ax.set_yticks(y, labels=t["Input"])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.1%}")
    ax.set_xlabel(metric_label)
    ax.set_title(f"Tornado: {metric_label} sensitivity to one-at-a-time input shocks")
    ax.legend(frameon=False, title="Input change")
    return _finish(fig, path)


def plot_mc_hist(samples: pd.DataFrame, cfg_cost_of_equity: float, covenant: float,
                 path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.hist(samples["equity_irr"].dropna() * 100, bins=40, color=BLUE, edgecolor="white")
    ax.axvline(cfg_cost_of_equity * 100, color=RED, ls="--", label="Cost of equity")
    ax.axvline(samples["equity_irr"].median() * 100, color=ORANGE, label="Median")
    ax.set_xlabel("Equity IRR (%)")
    ax.set_ylabel("Simulations")
    ax.legend(frameon=False)
    ax.set_title("Equity IRR distribution")
    ax = axes[1]
    ax.hist(samples["min_dscr"].dropna(), bins=40, color=GREEN, edgecolor="white")
    ax.axvline(covenant, color=ORANGE, ls="--", label=f"Covenant {covenant:.2f}x")
    ax.axvline(1.0, color=RED, ls=":", label="1.00x")
    ax.set_xlabel("Minimum DSCR (x)")
    ax.legend(frameon=False)
    ax.set_title("Minimum DSCR distribution")
    return _finish(fig, path)
