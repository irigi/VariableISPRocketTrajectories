"""Dimensionless scaling helpers for trajectory atlas construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CanonicalUnits:
    """Canonical DU/TU units for a departure radius and central gravity parameter."""

    du_m: float
    tu_s: float



def compute_rho(r_target: float, r_departure: float) -> float:
    """Compute radius ratio rho = R_target / R_departure."""
    if r_departure <= 0:
        raise ValueError("r_departure must be positive")
    if r_target <= 0:
        raise ValueError("r_target must be positive")
    return float(r_target / r_departure)



def compute_canonical_units(r0: float, mu: float) -> CanonicalUnits:
    """Compute canonical distance/time units (DU, TU) for non-dimensionalization."""
    if r0 <= 0:
        raise ValueError("r0 must be positive")
    if mu <= 0:
        raise ValueError("mu must be positive")
    return CanonicalUnits(du_m=float(r0), tu_s=float(np.sqrt((r0**3) / mu)))



def compute_kappa_tilde(power: float, m_dry: float, m0: float, r0: float, mu: float) -> float:
    """
    Compute the dimensionless capability parameter:

        kappa_tilde = [2P (1/m_dry - 1/m0)] * r0^(2.5) / mu^(1.5)

    All inputs are expected in SI units.
    """
    if power <= 0:
        raise ValueError("power must be positive")
    if m_dry <= 0 or m0 <= 0:
        raise ValueError("m_dry and m0 must be positive")
    if m0 <= m_dry:
        raise ValueError("m0 must be greater than m_dry")
    if r0 <= 0:
        raise ValueError("r0 must be positive")
    if mu <= 0:
        raise ValueError("mu must be positive")

    fuel_effort = 2.0 * power * ((1.0 / m_dry) - (1.0 / m0))
    scale = (r0**2.5) / (mu**1.5)
    return float(fuel_effort * scale)



def solve_power_for_kappa(target_kappa: float, m_dry: float, m0: float, r0: float, mu: float) -> float:
    """Solve the required power to achieve a desired kappa_tilde."""
    if target_kappa <= 0:
        raise ValueError("target_kappa must be positive")
    if m_dry <= 0 or m0 <= 0:
        raise ValueError("m_dry and m0 must be positive")
    if m0 <= m_dry:
        raise ValueError("m0 must be greater than m_dry")
    if r0 <= 0:
        raise ValueError("r0 must be positive")
    if mu <= 0:
        raise ValueError("mu must be positive")

    denom = 2.0 * ((1.0 / m_dry) - (1.0 / m0)) * (r0**2.5) / (mu**1.5)
    if denom <= 0:
        raise ValueError("invalid parameters produce non-positive denominator")
    return float(target_kappa / denom)



def estimate_kappa_bounds(kappa_values: np.ndarray, pad_decades: float = 0.25) -> tuple[float, float]:
    """Estimate log-space bounds for atlas generation from sampled kappa values."""
    arr = np.asarray(kappa_values, dtype=float)
    if arr.size == 0:
        raise ValueError("kappa_values cannot be empty")
    if np.any(arr <= 0):
        raise ValueError("all kappa_values must be positive")
    if pad_decades < 0:
        raise ValueError("pad_decades must be non-negative")

    lo = np.min(arr)
    hi = np.max(arr)
    lo_log = np.log10(lo) - pad_decades
    hi_log = np.log10(hi) + pad_decades
    return float(10**lo_log), float(10**hi_log)
