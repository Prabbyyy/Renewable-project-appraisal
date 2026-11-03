from dataclasses import replace

import numpy as np
import pytest

from reappraisal import ProjectConfig, irr, npv, payback_period, run_model, evaluate
from reappraisal.model import _debt_schedule, _tax
from reappraisal.montecarlo import run_monte_carlo, summarise
from reappraisal.config import MonteCarloConfig
from reappraisal.scenarios import breakeven_plf, sensitivity_grid, tornado

CFG = ProjectConfig()


# ---------- metrics ----------
def test_irr_known_value():
    assert irr([-100, 110]) == pytest.approx(0.10, abs=1e-8)


def test_irr_makes_npv_zero():
    cf = [-1000, 300, 300, 300, 300, 300]
    assert npv(irr(cf), cf) == pytest.approx(0.0, abs=1e-6)


def test_irr_nan_without_sign_change():
    assert np.isnan(irr([100, 200, 300]))


def test_npv_simple():
    assert npv(0.10, [-100, 110]) == pytest.approx(0.0, abs=1e-9)


def test_payback_interpolates():
    assert payback_period([-100, 50, 50]) == pytest.approx(2.0)
    assert payback_period([-100, 60, 60]) == pytest.approx(1.6667, abs=1e-3)


# ---------- debt & tax ----------
@pytest.mark.parametrize("kind", ["annuity", "equal_principal"])
def test_debt_fully_repaid(kind):
    interest, principal = _debt_schedule(100.0, 0.095, 15, 25, kind)
    assert principal.sum() == pytest.approx(100.0)
    assert principal[15:].sum() == 0 and interest[15:].sum() == 0


def test_annuity_equal_instalments():
    interest, principal = _debt_schedule(100.0, 0.10, 10, 10, "annuity")
    service = interest + principal
    assert np.allclose(service, service[0])


def test_tax_loss_carry_forward():
    tax = _tax(np.array([-10.0, 4.0, 10.0]), 0.25)
    assert tax[0] == 0 and tax[1] == 0
    assert tax[2] == pytest.approx(0.25 * (10 - 6))


# ---------- model ----------
def test_cashflow_shape_and_year0():
    res = run_model(CFG)
    cf = res.cashflow
    assert len(cf) == CFG.project_life + 1
    assert cf.loc[0, "Project_FCF"] == pytest.approx(-CFG.project_cost_cr)
    assert cf.loc[0, "Equity_FCF"] == pytest.approx(-CFG.project_cost_cr * (1 - CFG.debt_share))


def test_revenue_arithmetic():
    # 50 MW * 8760 h * 27% = 118,260 MWh; * 2.65 INR/kWh = INR 31.3389 cr
    assert run_model(CFG).cashflow.loc[1, "Revenue"] == pytest.approx(31.3389, abs=1e-6)


def test_debt_service_matches_schedule():
    cf = run_model(CFG).cashflow
    assert cf["Principal"].sum() == pytest.approx(CFG.project_cost_cr * CFG.debt_share)


def test_higher_plf_improves_returns():
    low = evaluate(replace(CFG, plf=0.24))
    high = evaluate(replace(CFG, plf=0.30))
    assert high["equity_irr"] > low["equity_irr"]
    assert high["min_dscr"] > low["min_dscr"]


def test_higher_tariff_raises_irr_but_not_lcoe():
    base = evaluate(CFG)
    up = evaluate(replace(CFG, tariff=CFG.tariff * 1.1))
    assert up["equity_irr"] > base["equity_irr"]
    assert up["lcoe_inr_per_kwh"] == pytest.approx(base["lcoe_inr_per_kwh"])  # cost-side only


def test_all_equity_has_no_dscr():
    m = evaluate(replace(CFG, debt_share=0.0))
    assert np.isnan(m["min_dscr"])
    assert m["equity_irr"] == pytest.approx(m["project_irr"], abs=1e-6)


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        ProjectConfig(plf=1.5)
    with pytest.raises(ValueError):
        ProjectConfig(loan_tenor=30)


# ---------- scenarios & Monte Carlo ----------
def test_sensitivity_grid_monotonic_in_plf():
    g = sensitivity_grid(CFG, "equity_irr")
    assert (g.diff().iloc[1:] > 0).all().all()


def test_tornado_sorted_by_swing():
    t = tornado(CFG)
    assert t["Swing"].is_monotonic_decreasing


def test_breakeven_plf_hits_target():
    be = breakeven_plf(CFG, target_dscr=1.0)
    assert evaluate(replace(CFG, plf=be))["min_dscr"] == pytest.approx(1.0, abs=1e-3)
    assert be < CFG.plf


def test_monte_carlo_reproducible_and_sane():
    mc = MonteCarloConfig(n_sims=200, seed=7)
    a = run_monte_carlo(CFG, mc)
    b = run_monte_carlo(CFG, mc)
    assert a.equals(b)
    s = summarise(a, CFG)
    p = s["percentiles"]["equity_irr"]
    assert p["P10"] <= p["P50"] <= p["P90"]
    assert 0.0 <= s["risk"]["prob_min_dscr_below_covenant"] <= 1.0
