"""Core financial model: generation -> revenue -> EBITDA -> debt -> tax -> returns."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .metrics import NAN, discounted_payback_period, irr, npv, payback_period


@dataclass
class ModelResult:
    config: ProjectConfig
    cashflow: pd.DataFrame   # index = year 0..N, all money in INR crore
    metrics: dict


def _tax(profit_before_tax: np.ndarray, rate: float) -> np.ndarray:
    """Tax with simple loss carry-forward (unabsorbed depreciation / losses)."""
    tax = np.zeros_like(profit_before_tax)
    loss_pool = 0.0
    for i, p in enumerate(profit_before_tax):
        if p <= 0:
            loss_pool += -p
        else:
            used = min(loss_pool, p)
            loss_pool -= used
            tax[i] = rate * (p - used)
    return tax


def _debt_schedule(debt: float, rate: float, tenor: int, n_years: int, kind: str):
    interest = np.zeros(n_years)
    principal = np.zeros(n_years)
    balance = debt
    if debt <= 0:
        return interest, principal
    if kind == "annuity":
        pmt = debt / tenor if rate == 0 else debt * rate / (1 - (1 + rate) ** -tenor)
    for i in range(tenor):
        interest[i] = balance * rate
        principal[i] = (pmt - interest[i]) if kind == "annuity" else debt / tenor
        balance -= principal[i]
    return interest, principal


def simulate(cfg: ProjectConfig) -> dict:
    """Run the model and return raw arrays (index 0 = construction year, 1..N = operations)."""
    n = cfg.project_life
    yrs = np.arange(1, n + 1)

    # --- Generation & revenue (INR crore) ---
    gen_mwh = cfg.capacity_mw * cfg.hours_per_year * cfg.plf * (1 - cfg.degradation) ** (yrs - 1)
    tariff_t = cfg.tariff * (1 + cfg.tariff_escalation) ** (yrs - 1)
    revenue = gen_mwh * tariff_t / 1e4          # MWh * INR/kWh * 1000 / 1e7

    # --- Operating costs ---
    om = cfg.capacity_mw * cfg.om_lakh_per_mw / 100 * (1 + cfg.om_escalation) ** (yrs - 1)
    insurance = np.full(n, cfg.project_cost_cr * cfg.insurance_pct_capex)
    ebitda = revenue - om - insurance

    # --- Financing ---
    capex = cfg.project_cost_cr
    debt = capex * cfg.debt_share
    equity = capex - debt
    interest, principal = _debt_schedule(debt, cfg.interest_rate, cfg.loan_tenor, n, cfg.repayment)
    debt_service = interest + principal
    depreciation = np.full(n, capex / n)

    # --- Tax (levered for equity/DSCR, unlevered for project cash flows) ---
    tax_lev = _tax(ebitda - depreciation - interest, cfg.tax_rate)
    tax_unlev = _tax(ebitda - depreciation, cfg.tax_rate)

    cfads = ebitda - tax_lev
    with np.errstate(divide="ignore", invalid="ignore"):
        dscr = np.where(debt_service > 0, cfads / debt_service, np.nan)

    project_fcf = np.concatenate(([-capex], ebitda - tax_unlev))
    equity_fcf = np.concatenate(([-equity], cfads - debt_service))

    # --- Metrics ---
    wacc = cfg.debt_share * cfg.interest_rate * (1 - cfg.tax_rate) \
        + (1 - cfg.debt_share) * cfg.cost_of_equity
    disc = (1 + wacc) ** yrs
    pv_cost_cr = capex + np.sum((om + insurance) / disc)
    pv_gen_mwh = np.sum(gen_mwh / disc)
    has_debt = not np.all(np.isnan(dscr))

    metrics = {
        "project_cost_cr": capex,
        "debt_cr": debt,
        "equity_cr": equity,
        "wacc": wacc,
        "project_irr": irr(project_fcf),
        "equity_irr": irr(equity_fcf),
        "project_npv_cr": npv(wacc, project_fcf),
        "equity_npv_cr": npv(cfg.cost_of_equity, equity_fcf),
        "payback_years": payback_period(project_fcf),
        "discounted_payback_years": discounted_payback_period(wacc, project_fcf),
        "equity_payback_years": payback_period(equity_fcf),
        "min_dscr": float(np.nanmin(dscr)) if has_debt else NAN,
        "avg_dscr": float(np.nanmean(dscr)) if has_debt else NAN,
        "lcoe_inr_per_kwh": pv_cost_cr * 1e4 / pv_gen_mwh,
        "year1_generation_gwh": gen_mwh[0] / 1e3,
        "year1_revenue_cr": revenue[0],
        "year1_ebitda_cr": ebitda[0],
    }
    return {
        "yrs": yrs, "gen_mwh": gen_mwh, "tariff": tariff_t, "revenue": revenue, "om": om,
        "insurance": insurance, "ebitda": ebitda, "depreciation": depreciation,
        "interest": interest, "principal": principal, "tax_lev": tax_lev, "cfads": cfads,
        "debt_service": debt_service, "dscr": dscr, "project_fcf": project_fcf,
        "equity_fcf": equity_fcf, "metrics": metrics,
    }


def run_model(cfg: ProjectConfig) -> ModelResult:
    """Run the model and return a tidy year-by-year table plus summary metrics."""
    s = simulate(cfg)
    pad = lambda a: np.concatenate(([0.0], a))  # noqa: E731  (year-0 row = construction)
    df = pd.DataFrame({
        "Year": np.arange(0, cfg.project_life + 1),
        "Generation_MWh": pad(s["gen_mwh"]),
        "Tariff_INR_per_kWh": pad(s["tariff"]),
        "Revenue": pad(s["revenue"]),
        "O&M": pad(s["om"]),
        "Insurance": pad(s["insurance"]),
        "EBITDA": pad(s["ebitda"]),
        "Depreciation": pad(s["depreciation"]),
        "Interest": pad(s["interest"]),
        "Principal": pad(s["principal"]),
        "Tax": pad(s["tax_lev"]),
        "CFADS": pad(s["cfads"]),
        "Debt_Service": pad(s["debt_service"]),
        "DSCR": np.concatenate(([np.nan], s["dscr"])),
        "Project_FCF": s["project_fcf"],
        "Equity_FCF": s["equity_fcf"],
    })
    df["Cum_Project_FCF"] = df["Project_FCF"].cumsum()
    df["Cum_Equity_FCF"] = df["Equity_FCF"].cumsum()
    return ModelResult(config=cfg, cashflow=df, metrics=s["metrics"])


def evaluate(cfg: ProjectConfig) -> dict:
    """Fast path for sweeps / Monte Carlo: metrics only, no DataFrame."""
    return simulate(cfg)["metrics"]
