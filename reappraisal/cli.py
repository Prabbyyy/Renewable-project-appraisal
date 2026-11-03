"""Command-line entry point:  python -m reappraisal --config examples/solar_50mw.json"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import load_config
from .model import run_model
from .montecarlo import run_monte_carlo, summarise
from .plots import (plot_cashflows, plot_dscr, plot_heatmap, plot_mc_hist, plot_tornado)
from .report import export_excel
from .scenarios import breakeven_plf, sensitivity_grid, tornado


def run_pipeline(config_path: str, out_dir: str, n_sims: int | None = None,
                 seed: int | None = None) -> dict:
    cfg, mc = load_config(config_path)
    if n_sims is not None:
        mc = replace(mc, n_sims=n_sims)
    if seed is not None:
        mc = replace(mc, seed=seed)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    result = run_model(cfg)
    sens_irr = sensitivity_grid(cfg, "equity_irr")
    sens_dscr = sensitivity_grid(cfg, "min_dscr")
    tor = tornado(cfg)
    be_plf = breakeven_plf(cfg, target_dscr=1.0)
    be_cov = breakeven_plf(cfg, target_dscr=cfg.min_dscr_covenant)

    samples = run_monte_carlo(cfg, mc)
    mc_summary = summarise(samples, cfg)

    # --- files ---
    result.cashflow.to_csv(out / "cashflow.csv", index=False)
    samples.to_csv(out / "mc_samples.csv", index=False)
    export_excel(result, sens_irr, sens_dscr, tor, samples, mc_summary, out / "appraisal.xlsx")
    plot_cashflows(result.cashflow, out / "cashflows.png")
    plot_dscr(result.cashflow, cfg.min_dscr_covenant, out / "dscr.png")
    plot_heatmap(sens_irr, "Equity IRR: PLF vs tariff", out / "heatmap_equity_irr.png")
    plot_heatmap(sens_dscr, "Minimum DSCR: PLF vs tariff", out / "heatmap_min_dscr.png",
                 pct=False)
    plot_tornado(tor, out / "tornado.png")
    plot_mc_hist(samples, cfg.cost_of_equity, cfg.min_dscr_covenant, out / "mc_distributions.png")

    summary = {
        "project": cfg.name,
        "metrics": result.metrics,
        "breakeven_plf_dscr_1.0": be_plf,
        f"breakeven_plf_dscr_{cfg.min_dscr_covenant:.2f}": be_cov,
        "monte_carlo": mc_summary,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _print(summary: dict) -> None:
    m = summary["metrics"]
    print(f"\n=== {summary['project']} ===")
    print(f"Project cost      : INR {m['project_cost_cr']:.1f} cr "
          f"(debt {m['debt_cr']:.1f} / equity {m['equity_cr']:.1f})")
    print(f"Year-1 generation : {m['year1_generation_gwh']:.1f} GWh")
    print(f"Project IRR       : {m['project_irr']:.2%}")
    print(f"Equity IRR        : {m['equity_irr']:.2%}")
    print(f"WACC              : {m['wacc']:.2%}")
    print(f"Project NPV       : INR {m['project_npv_cr']:.1f} cr")
    print(f"Payback (project) : {m['payback_years']:.1f} yrs")
    print(f"Min / Avg DSCR    : {m['min_dscr']:.2f}x / {m['avg_dscr']:.2f}x")
    print(f"LCOE              : INR {m['lcoe_inr_per_kwh']:.2f}/kWh")
    be = summary["breakeven_plf_dscr_1.0"]
    print(f"Break-even PLF    : {be:.2%} (min DSCR = 1.0x)")
    r = summary["monte_carlo"]["risk"]
    p = summary["monte_carlo"]["percentiles"]["equity_irr"]
    print(f"\nMonte Carlo ({summary['monte_carlo']['n_sims']} runs)")
    print(f"Equity IRR P10/P50/P90 : {p['P10']:.2%} / {p['P50']:.2%} / {p['P90']:.2%}")
    print(f"P(min DSCR < covenant) : {r['prob_min_dscr_below_covenant']:.1%}")
    print(f"P(min DSCR < 1.0x)     : {r['prob_min_dscr_below_1.0']:.1%}")
    print(f"P(project NPV < 0)     : {r['prob_project_npv_negative']:.1%}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Renewable energy project appraisal model")
    ap.add_argument("--config", required=True, help="Path to JSON config (see examples/)")
    ap.add_argument("--out", default="outputs", help="Output directory")
    ap.add_argument("--sims", type=int, default=None, help="Override Monte Carlo runs")
    ap.add_argument("--seed", type=int, default=None, help="Override random seed")
    args = ap.parse_args(argv)
    summary = run_pipeline(args.config, args.out, args.sims, args.seed)
    _print(summary)
    print(f"\nOutputs written to: {Path(args.out).resolve()}")
