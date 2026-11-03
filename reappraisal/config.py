"""Input assumptions for a utility-scale renewable project.

All money is in INR crore (1 crore = 10 million), energy in MWh, tariff in INR/kWh.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    # --- Project ---
    name: str = "Solar 50 MW"
    technology: str = "solar"          # label only: "solar" | "wind"
    capacity_mw: float = 50.0
    capex_cr_per_mw: float = 3.6       # INR crore per MW
    project_life: int = 25             # years of operation (PPA / useful life)

    # --- Generation ---
    plf: float = 0.27                  # Plant Load Factor (net, year-1)
    degradation: float = 0.005         # annual output degradation (0.5% = 0.005)
    hours_per_year: int = 8760

    # --- Revenue & opex ---
    tariff: float = 2.65               # INR/kWh, year-1
    tariff_escalation: float = 0.0     # annual escalation (0 for a flat PPA tariff)
    om_lakh_per_mw: float = 7.0        # O&M, INR lakh per MW per year (year-1)
    om_escalation: float = 0.05        # annual O&M escalation
    insurance_pct_capex: float = 0.0035  # annual insurance as % of project cost

    # --- Financing ---
    debt_share: float = 0.75           # debt / project cost
    interest_rate: float = 0.0925      # annual interest on term loan
    loan_tenor: int = 18               # years of repayment (starts year 1)
    repayment: str = "equal_principal"  # "equal_principal" | "annuity" (equal instalments)
    cost_of_equity: float = 0.14       # required return on equity (discount rate for equity NPV)
    min_dscr_covenant: float = 1.20    # lender covenant, used for risk flags

    # --- Tax ---
    tax_rate: float = 0.2517           # 22% + 10% surcharge + 4% cess
    # Straight-line depreciation of full project cost over project_life.
    # No salvage value is assumed (conservative).

    def __post_init__(self) -> None:
        if self.capacity_mw <= 0 or self.capex_cr_per_mw <= 0:
            raise ValueError("capacity_mw and capex_cr_per_mw must be positive")
        if not 0 < self.plf <= 1:
            raise ValueError("plf must be in (0, 1]")
        if not 0 <= self.debt_share < 1:
            raise ValueError("debt_share must be in [0, 1)")
        if self.project_life < 1:
            raise ValueError("project_life must be >= 1")
        if not 1 <= self.loan_tenor <= self.project_life:
            raise ValueError("loan_tenor must be between 1 and project_life")
        if self.repayment not in ("annuity", "equal_principal"):
            raise ValueError("repayment must be 'annuity' or 'equal_principal'")
        if self.interest_rate < 0 or self.tariff <= 0:
            raise ValueError("interest_rate must be >= 0 and tariff > 0")

    @property
    def project_cost_cr(self) -> float:
        return self.capacity_mw * self.capex_cr_per_mw


@dataclass(frozen=True)
class MonteCarloConfig:
    """Uncertainty ranges. Each draw is held constant over the project life."""
    n_sims: int = 5000
    seed: int = 42
    plf_sd_rel: float = 0.06           # std-dev of PLF as a fraction of the mean (normal)
    tariff_rel: tuple = (-0.10, 0.0, 0.05)    # triangular (low, mode, high), relative change
    capex_rel: tuple = (-0.05, 0.0, 0.15)     # triangular, relative change (cost overrun skew)
    interest_abs: tuple = (-0.01, 0.0, 0.02)  # triangular, absolute change in interest rate


def _filter(cls, data: dict) -> dict:
    allowed = {f.name for f in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return data


def load_config(path: str | Path) -> tuple[ProjectConfig, MonteCarloConfig]:
    """Load a JSON file with a 'project' block and an optional 'monte_carlo' block."""
    raw = json.loads(Path(path).read_text())
    project = ProjectConfig(**_filter(ProjectConfig, raw.get("project", {})))
    mc_raw = {k: tuple(v) if isinstance(v, list) else v
              for k, v in raw.get("monte_carlo", {}).items()}
    mc = MonteCarloConfig(**_filter(MonteCarloConfig, mc_raw))
    return project, mc
