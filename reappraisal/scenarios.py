"""Deterministic sensitivity analysis: PLF x tariff grids, tornado, break-even PLF."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .model import evaluate


def sensitivity_grid(cfg: ProjectConfig, metric: str = "equity_irr",
                     plf_steps=(-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15),
                     tariff_steps=(-0.20, -0.10, 0.0, 0.10, 0.20)) -> pd.DataFrame:
    """Metric for every PLF x tariff combination (steps are relative changes)."""
    plfs = [cfg.plf * (1 + s) for s in plf_steps]
    tariffs = [cfg.tariff * (1 + s) for s in tariff_steps]
    grid = np.empty((len(plfs), len(tariffs)))
    for i, p in enumerate(plfs):
        for j, t in enumerate(tariffs):
            grid[i, j] = evaluate(replace(cfg, plf=min(p, 1.0), tariff=t))[metric]
    return pd.DataFrame(
        grid,
        index=pd.Index([f"{p:.1%}" for p in plfs], name="PLF"),
        columns=pd.Index([f"{t:.2f}" for t in tariffs], name="Tariff (INR/kWh)"),
    )


_TORNADO_PARAMS = {
    "plf": "PLF",
    "tariff": "Tariff",
    "capex_cr_per_mw": "Capex / MW",
    "om_lakh_per_mw": "O&M cost",
    "interest_rate": "Interest rate",
    "degradation": "Degradation",
}


def tornado(cfg: ProjectConfig, shock: float = 0.10, metric: str = "equity_irr") -> pd.DataFrame:
    """Impact on `metric` of a +/- `shock` relative change in each input, one at a time."""
    base = evaluate(cfg)[metric]
    rows = []
    for attr, label in _TORNADO_PARAMS.items():
        v = getattr(cfg, attr)
        lo = evaluate(replace(cfg, **{attr: v * (1 - shock)}))[metric]
        hi = evaluate(replace(cfg, **{attr: v * (1 + shock)}))[metric]
        rows.append({"Input": label, f"-{shock:.0%}": lo, f"+{shock:.0%}": hi,
                     "Swing": abs(hi - lo), "Base": base})
    return pd.DataFrame(rows).sort_values("Swing", ascending=False).reset_index(drop=True)


def breakeven_plf(cfg: ProjectConfig, target_dscr: float = 1.0, tol: float = 1e-6) -> float:
    """Lowest PLF at which min DSCR still meets `target_dscr` (bisection).

    Returns NaN if the project has no debt or the target is unreachable below the base PLF.
    """
    if cfg.debt_share == 0:
        return float("nan")
    lo, hi = 0.01, cfg.plf
    if evaluate(replace(cfg, plf=hi))["min_dscr"] < target_dscr:
        return float("nan")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if evaluate(replace(cfg, plf=mid))["min_dscr"] >= target_dscr:
            hi = mid
        else:
            lo = mid
    return hi
