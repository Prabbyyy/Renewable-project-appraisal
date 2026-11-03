"""Standalone financial metrics (no external dependency beyond numpy)."""
from __future__ import annotations

import numpy as np

NAN = float("nan")


def npv(rate: float, cashflows) -> float:
    """NPV with the first cash flow at t=0 (not discounted)."""
    cf = np.asarray(cashflows, dtype=float)
    t = np.arange(len(cf))
    return float(np.sum(cf / (1.0 + rate) ** t))


def irr(cashflows, low: float = -0.9, high: float = 10.0,
        tol: float = 1e-10, max_iter: int = 300) -> float:
    """IRR by bisection. Returns NaN if the flows have no sign change / no root."""
    cf = np.asarray(cashflows, dtype=float)
    if not (np.any(cf > 0) and np.any(cf < 0)):
        return NAN
    f_low, f_high = npv(low, cf), npv(high, cf)
    if f_low * f_high > 0:
        return NAN
    mid = 0.5 * (low + high)
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        f_mid = npv(mid, cf)
        if abs(f_mid) < tol or (high - low) < tol:
            return mid
        if f_low * f_mid < 0:
            high = mid
        else:
            low, f_low = mid, f_mid
    return mid


def payback_period(cashflows) -> float:
    """Years until cumulative cash flow turns positive (linear interpolation within the year)."""
    cf = np.asarray(cashflows, dtype=float)
    cum = np.cumsum(cf)
    for t in range(1, len(cum)):
        if cum[t] >= 0 > cum[t - 1]:
            return (t - 1) + (-cum[t - 1]) / cf[t]
    return NAN


def discounted_payback_period(rate: float, cashflows) -> float:
    cf = np.asarray(cashflows, dtype=float)
    disc = cf / (1.0 + rate) ** np.arange(len(cf))
    return payback_period(disc)
