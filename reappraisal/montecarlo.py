"""Monte Carlo risk analysis on PLF, tariff realisation, capex and interest rate."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .config import MonteCarloConfig, ProjectConfig
from .model import evaluate

OUTPUTS = ("equity_irr", "project_irr", "project_npv_cr", "min_dscr", "lcoe_inr_per_kwh")


def run_monte_carlo(cfg: ProjectConfig, mc: MonteCarloConfig | None = None) -> pd.DataFrame:
    """Return one row per simulation with sampled inputs and resulting metrics.

    Each draw is a single "state of the world" held for the whole project life
    (e.g. a resource-assessment error or a capex overrun), not year-by-year noise.
    """
    mc = mc or MonteCarloConfig()
    rng = np.random.default_rng(mc.seed)
    n = mc.n_sims

    plf = rng.normal(cfg.plf, cfg.plf * mc.plf_sd_rel, n)
    plf = np.clip(plf, cfg.plf * 0.5, min(1.0, cfg.plf * 1.5))
    tariff_rel = rng.triangular(*mc.tariff_rel, n)
    capex_rel = rng.triangular(*mc.capex_rel, n)
    interest_abs = rng.triangular(*mc.interest_abs, n)

    rows = []
    for i in range(n):
        sim_cfg = replace(
            cfg,
            plf=float(plf[i]),
            tariff=cfg.tariff * (1 + float(tariff_rel[i])),
            capex_cr_per_mw=cfg.capex_cr_per_mw * (1 + float(capex_rel[i])),
            interest_rate=max(0.0, cfg.interest_rate + float(interest_abs[i])),
        )
        m = evaluate(sim_cfg)
        rows.append({
            "plf": sim_cfg.plf, "tariff": sim_cfg.tariff,
            "capex_cr_per_mw": sim_cfg.capex_cr_per_mw, "interest_rate": sim_cfg.interest_rate,
            **{k: m[k] for k in OUTPUTS},
        })
    return pd.DataFrame(rows)


def summarise(samples: pd.DataFrame, cfg: ProjectConfig) -> dict:
    """Percentiles and risk probabilities from the simulation output."""
    out = {"n_sims": int(len(samples)), "percentiles": {}, "risk": {}}
    for col in OUTPUTS:
        s = samples[col].dropna()
        out["percentiles"][col] = {
            "mean": float(s.mean()), "std": float(s.std()),
            "P10": float(s.quantile(0.10)), "P50": float(s.quantile(0.50)),
            "P90": float(s.quantile(0.90)),
        }
    out["risk"] = {
        "prob_min_dscr_below_covenant": float((samples["min_dscr"] < cfg.min_dscr_covenant).mean()),
        "prob_min_dscr_below_1.0": float((samples["min_dscr"] < 1.0).mean()),
        "prob_project_npv_negative": float((samples["project_npv_cr"] < 0).mean()),
        "prob_equity_irr_below_cost_of_equity": float(
            (samples["equity_irr"] < cfg.cost_of_equity).mean()),
        "covenant_used": cfg.min_dscr_covenant,
    }
    return out
