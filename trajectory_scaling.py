"""Dimensionless scaling helpers for trajectory atlas construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ATLAS_VECTOR_SIZE = 6
SOLVER_PARAM_SIZE = 5


@dataclass(frozen=True)
class CanonicalUnits:
    """Canonical DU/TU units for a departure radius and central gravity parameter."""

    du_m: float
    tu_s: float


@dataclass(frozen=True)
class AtlasNormalization:
    """Affine normalization contract for solver vectors stored in the atlas.

    The atlas vector is expected in the order:
    [lam_r0, lam_vr0, lam_vtheta0, C_m, C_theta, t_f_days].
    """

    scale: tuple[float, float, float, float, float, float]
    offset: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    version: str = "atlas-normalization-v1"

    def scale_array(self) -> np.ndarray:
        arr = np.asarray(self.scale, dtype=float)
        if arr.shape != (ATLAS_VECTOR_SIZE,):
            raise ValueError("scale must have 6 elements")
        if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
            raise ValueError("scale values must be finite and positive")
        return arr

    def offset_array(self) -> np.ndarray:
        arr = np.asarray(self.offset, dtype=float)
        if arr.shape != (ATLAS_VECTOR_SIZE,):
            raise ValueError("offset must have 6 elements")
        if np.any(~np.isfinite(arr)):
            raise ValueError("offset values must be finite")
        return arr


def default_atlas_normalization(seed_params: np.ndarray | list[float], t_scale_days: float = 365.0) -> AtlasNormalization:
    """Build default deterministic normalization from a representative seed vector."""
    seed = np.asarray(seed_params, dtype=float).reshape(-1)
    if seed.shape[0] != SOLVER_PARAM_SIZE:
        raise ValueError("seed_params must have 5 elements")
    if t_scale_days <= 0:
        raise ValueError("t_scale_days must be positive")

    # keep contract stable and non-singular with floors for near-zero components
    param_scale = np.maximum(np.abs(seed), np.array([1.0e-8, 1.0, 1.0, 1.0e-8, 1.0]))
    scale = tuple(param_scale.tolist() + [float(t_scale_days)])
    return AtlasNormalization(scale=scale)


def normalize_atlas_vector(vector: np.ndarray | list[float], contract: AtlasNormalization) -> np.ndarray:
    """Convert physical solver vector to normalized atlas vector."""
    vec = np.asarray(vector, dtype=float).reshape(-1)
    if vec.shape[0] != ATLAS_VECTOR_SIZE:
        raise ValueError("vector must have 6 elements")
    if np.any(~np.isfinite(vec)):
        raise ValueError("vector must be finite")

    scale = contract.scale_array()
    offset = contract.offset_array()
    return (vec - offset) / scale


def denormalize_atlas_vector(vector_norm: np.ndarray | list[float], contract: AtlasNormalization) -> np.ndarray:
    """Convert normalized atlas vector back to physical units used by the solver."""
    vec = np.asarray(vector_norm, dtype=float).reshape(-1)
    if vec.shape[0] != ATLAS_VECTOR_SIZE:
        raise ValueError("vector_norm must have 6 elements")
    if np.any(~np.isfinite(vec)):
        raise ValueError("vector_norm must be finite")

    scale = contract.scale_array()
    offset = contract.offset_array()
    return vec * scale + offset


def normalize_solver_seed(seed_params: np.ndarray | list[float], t_days: float, contract: AtlasNormalization) -> np.ndarray:
    """Convenience wrapper to normalize solver params + transfer time."""
    params = np.asarray(seed_params, dtype=float).reshape(-1)
    if params.shape[0] != SOLVER_PARAM_SIZE:
        raise ValueError("seed_params must have 5 elements")
    vec = np.concatenate([params, [float(t_days)]])
    return normalize_atlas_vector(vec, contract)


def denormalize_solver_seed(vector_norm: np.ndarray | list[float], contract: AtlasNormalization) -> tuple[np.ndarray, float]:
    """Return denormalized (params[5], t_days) from normalized atlas vector."""
    vec = denormalize_atlas_vector(vector_norm, contract)
    return np.asarray(vec[:SOLVER_PARAM_SIZE], dtype=float), float(vec[SOLVER_PARAM_SIZE])


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
