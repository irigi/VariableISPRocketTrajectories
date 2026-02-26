"""
atlas_utils.py
Shared utilities for the Time-Optimal Trajectory Atlas.
Handles non-dimensionalization and grid definitions.
"""
import numpy as np

# ----------------------------- #
#  Physical Constants (SI)      #
# ----------------------------- #
AU = 1.495978707e11  # m
MU_SI = 1.32712440018e20  # m^3 s⁻² (GM_sun)

# ----------------------------- #
#  Grid Definitions             #
# ----------------------------- #
# 1. Radius Ratio (rho = r_target / r_start)
# Range: 0.05 (inward) to 100.0 (outward)
# Log-space because orbital dynamics scale geometrically.
RHO_MIN, RHO_MAX = 0.05, 100.0
N_RHO = 60  # Density of radius grid

# 2. Capability Parameter (kappa)
# Represents dimensionless "energy capacity" relative to local gravity depth.
# Range: Determined by verification script below.
KAPPA_MIN, KAPPA_MAX = 0.1, 5000.0
N_KAPPA = 40  # Density of ship capability grid

# 3. Angle (theta)
# Range: 0 to 6 revolutions (4*pi is usually enough, but 12*pi covers low-thrust spirals)
THETA_MAX_REV = 6.0
N_THETA = 120  # Density of angle grid (approx every 18 degrees)


def get_rho_grid():
    """Returns the logarithmic grid for radius ratios."""
    return np.logspace(np.log10(RHO_MIN), np.log10(RHO_MAX), N_RHO)


def get_kappa_grid():
    """Returns the logarithmic grid for capability parameters."""
    return np.logspace(np.log10(KAPPA_MIN), np.log10(KAPPA_MAX), N_KAPPA)


def get_theta_grid():
    """Returns the linear grid for transfer angles (radians)."""
    return np.linspace(0, THETA_MAX_REV * 2 * np.pi, N_THETA)


# ----------------------------- #
#  Dimensional Analysis         #
# ----------------------------- #

def physical_to_dimensionless(P, m0, m_dry, r0, r_target, mu=MU_SI):
    """
    Convert physical mission parameters to Atlas coordinates (rho, kappa).

    Args:
        P (float): Power (Watts)
        m0 (float): Initial mass (kg)
        m_dry (float): Dry mass (kg)
        r0 (float): Departure radius (m)
        r_target (float): Target radius (m)
        mu (float): Gravitational parameter (m^3/s^2)

    Returns:
        (rho, kappa): The coordinates in the Atlas.
    """
    # 1. Radius Ratio
    rho = r_target / r0

    # 2. Capability Parameter (Kappa)
    # Based on the energy integral constraint: J = 2P(1/m_dry - 1/m0)
    # Normalized by local gravitational energy depth.

    # Inverse mass difference (propellant fraction metric)
    delta_inv_m = (1.0 / m_dry) - (1.0 / m0)

    # Integrated acceleration capacity (J = integral(a^2 dt)) in SI units
    J_capacity_si = 2.0 * P * delta_inv_m

    # Scaling factor for J to Canonical Units (CU)
    # J_dim = J_si * (TU^3 / DU^2)
    # DU = r0, TU = sqrt(r0^3 / mu)
    # TU^3 / DU^2 = (r0^4.5 / mu^1.5) / r0^2 = r0^2.5 / mu^1.5
    scale_factor = (r0 ** 2.5) / (mu ** 1.5)

    kappa = J_capacity_si * scale_factor

    return rho, kappa


def dimensionless_to_physical(rho, kappa, r0, mu=MU_SI):
    """
    Recover physical scales from Atlas coordinates (for checking bounds).
    Returns the required 'J_capacity' (integral a^2 dt) in SI units.
    """
    scale_factor = (r0 ** 2.5) / (mu ** 1.5)
    J_capacity_si = kappa / scale_factor
    r_target = rho * r0
    return r_target, J_capacity_si
