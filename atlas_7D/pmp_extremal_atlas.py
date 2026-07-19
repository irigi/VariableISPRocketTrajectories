"""Planar Kepler PMP extremal atlas for a constant-power variable-Isp rocket.

The atlas is a compact catalogue of *forward* normal PMP extremals.  It stores
only launch parameters, endpoint descriptors, and diagnostics in an ``.npz``
file.  Full trajectories are reconstructed on demand by integrating the exact
extremal equations and applying a local shooting correction.

Model
-----
Dimensionless units are based on the query's initial radius r0:

    R0 = r0
    T0 = sqrt(r0**3 / mu)
    V0 = sqrt(mu / r0)
    A0 = mu / r0**2

The costate-free extremal system is

    R' = V
    V' = -R/|R|^3 + A
    A' = J
    J' = -(I - 3 Rhat Rhat^T) A / |R|^3
    s' = |A|^2
    theta' = cross(R, V) / |R|^2

where s is the dimensionless inverse-mass expenditure.  A boundary query has
resource

    kappa = 2 P (1/md - 1/mi) r0**(5/2) / mu**(3/2).

The offline atlas samples the seven-dimensional forward map

    (u0, w0, Ar0, At0, Jr0, L, tau_f)
        -> (u0, w0, rho, Theta, ur_f, ut_f, kappa),

where L is the conserved rotational quantity and the missing initial jerk is

    Jt0 = u0*At0 - w0*Ar0 - L.

The inverse map is multivalued.  A query therefore tries several nearby atlas
seeds, performs exact least-squares shooting for each, deduplicates converged
branches, and returns the shortest branch found.

Dependencies: numpy, scipy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.stats import qmc

FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Basic data objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CartesianState:
    """Planar dimensional Cartesian state."""

    position: tuple[float, float]
    velocity: tuple[float, float]

    def arrays(self) -> tuple[FloatArray, FloatArray]:
        r = np.asarray(self.position, dtype=float)
        v = np.asarray(self.velocity, dtype=float)
        if r.shape != (2,) or v.shape != (2,):
            raise ValueError("position and velocity must each have two components")
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(v)):
            raise ValueError("state contains non-finite values")
        if np.linalg.norm(r) <= 0.0:
            raise ValueError("position norm must be positive")
        return r, v


@dataclass(frozen=True)
class RocketCapability:
    """Dimensional rocket and central-body parameters."""

    useful_power: float
    initial_mass: float
    dry_mass: float
    mu: float

    def validate(self) -> None:
        if self.useful_power <= 0.0:
            raise ValueError("useful_power must be positive")
        if self.initial_mass <= self.dry_mass or self.dry_mass <= 0.0:
            raise ValueError("require initial_mass > dry_mass > 0")
        if self.mu <= 0.0:
            raise ValueError("mu must be positive")




@dataclass
class CoverageConfig:
    """Adaptive coverage and verified pruning settings.

    Coverage is measured on independent *feasible* holdout extremals generated
    from the same launch domain.  A holdout is covered only when the normal
    query-time shooting correction converges without using the holdout itself.
    """

    enabled: bool = True

    # Fraction of launch attempts used to create the initial seed cloud.  The
    # rest are consumed in independent validation/adaptation batches.
    initial_fraction: float = 0.25
    batch_launches: int = 4096
    validation_rows: int = 512
    validation_cell_size: float = 0.20
    target_success: float = 0.98
    patience: int = 2
    max_rounds: int = 32
    minimum_validation_rows: int = 128

    # Fast correction used during coverage testing.
    neighbours: int = 24
    direct_seeds: int = 4
    regression_neighbours: int = 16
    max_nfev: int = 35

    # Initial cloud thinning and adaptive insertion. Distances are measured in
    # endpoint coordinates divided by feature_scale.
    initial_cell_size: float = 0.30
    insertion_cell_size: float = 0.12
    insert_failures_per_round: int = 512
    max_atlas_rows: int = 250_000

    # Verified redundancy pruning. A candidate is removed only if its exact
    # endpoint can be recovered from other active seeds.
    prune_enabled: bool = True
    prune_distance: float = 0.08
    prune_max_checks: int = 4000
    prune_max_nfev: int = 25

    # Deterministic validation-row selection.
    validation_seed: int = 918273

    def validate(self) -> None:
        if not (0.0 < self.initial_fraction < 1.0):
            raise ValueError("coverage.initial_fraction must lie in (0,1)")
        if self.batch_launches <= 0 or self.validation_rows <= 0:
            raise ValueError("coverage batch sizes must be positive")
        if self.validation_cell_size <= 0.0:
            raise ValueError("coverage.validation_cell_size must be positive")
        if not (0.0 < self.target_success <= 1.0):
            raise ValueError("coverage.target_success must lie in (0,1]")
        if self.patience <= 0 or self.max_rounds <= 0:
            raise ValueError("coverage patience and max_rounds must be positive")
        if self.minimum_validation_rows <= 0:
            raise ValueError("coverage.minimum_validation_rows must be positive")
        if self.neighbours <= 0 or self.direct_seeds <= 0:
            raise ValueError("coverage neighbour counts must be positive")
        if self.regression_neighbours <= 0 or self.max_nfev <= 0:
            raise ValueError("coverage correction settings must be positive")
        if self.initial_cell_size <= 0.0 or self.insertion_cell_size <= 0.0:
            raise ValueError("coverage cell sizes must be positive")
        if self.insert_failures_per_round <= 0 or self.max_atlas_rows <= 0:
            raise ValueError("coverage insertion limits must be positive")
        if self.prune_distance <= 0.0 or self.prune_max_checks < 0:
            raise ValueError("invalid coverage pruning settings")


@dataclass
class AtlasConfig:
    """Bounds and numerical settings for offline atlas generation.

    Bounds are dimensionless and deliberately explicit.  There is no universal
    finite atlas for arbitrary radii, speeds, resources, and winding counts;
    choose ranges that cover the intended mission domain.
    """

    n_samples: int = 50_000
    seed: int = 12345
    workers: int = max(1, (os.cpu_count() or 2) - 1)

    # Initial radial/tangential velocity in local Kepler-speed units.
    u0_bounds: tuple[float, float] = (-1.5, 1.5)
    w0_bounds: tuple[float, float] = (-0.5, 2.0)

    # Initial acceleration and radial jerk in canonical dimensionless units.
    ar0_bounds: tuple[float, float] = (-8.0, 8.0)
    at0_bounds: tuple[float, float] = (-8.0, 8.0)
    jr0_bounds: tuple[float, float] = (-20.0, 20.0)

    # Conserved rotational quantity.
    ell_bounds: tuple[float, float] = (-20.0, 20.0)

    # Flight time is sampled log-uniformly.
    tau_bounds: tuple[float, float] = (0.03, 30.0)

    # Keep only endpoints in this useful domain.
    rho_bounds: tuple[float, float] = (0.05, 30.0)
    kappa_bounds: tuple[float, float] = (1.0e-5, 1.0e4)
    theta_bounds: tuple[float, float] = (-12.0 * math.pi, 12.0 * math.pi)
    final_speed_max: float = 8.0

    # Integration guards.
    r_collision: float = 0.015
    r_escape: float = 60.0
    max_acceleration: float = 100.0
    rtol: float = 2.0e-9
    atol: float = 2.0e-11
    max_step: float = 0.04
    diagnostic_points: int = 96

    # Each integrated extremal contributes this many canonically rescaled
    # subarcs.  One is the full arc; the rest are stratified interior arcs.
    subarcs_per_trajectory: int = 6
    subarc_min_fraction: float = 0.08

    # Feature scaling for nearest-neighbour search. If None, robust scales are
    # estimated from the generated endpoint cloud.
    feature_scale: Optional[tuple[float, ...]] = None

    # Adaptive coverage validation and verified pruning.
    coverage: CoverageConfig = field(default_factory=CoverageConfig)

    def validate(self) -> None:
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if self.workers <= 0:
            raise ValueError("workers must be positive")
        for name, bounds in (
            ("u0_bounds", self.u0_bounds),
            ("w0_bounds", self.w0_bounds),
            ("ar0_bounds", self.ar0_bounds),
            ("at0_bounds", self.at0_bounds),
            ("jr0_bounds", self.jr0_bounds),
            ("ell_bounds", self.ell_bounds),
            ("tau_bounds", self.tau_bounds),
            ("rho_bounds", self.rho_bounds),
            ("kappa_bounds", self.kappa_bounds),
            ("theta_bounds", self.theta_bounds),
        ):
            if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
                raise ValueError(f"invalid {name}: {bounds}")
        if self.tau_bounds[0] <= 0.0 or self.rho_bounds[0] <= 0.0 or self.kappa_bounds[0] <= 0.0:
            raise ValueError("tau, rho, and kappa lower bounds must be positive")
        if self.r_collision <= 0.0 or self.r_escape <= self.r_collision:
            raise ValueError("invalid radial integration guards")
        if self.diagnostic_points < 8:
            raise ValueError("diagnostic_points must be at least 8")
        if self.subarcs_per_trajectory < 1:
            raise ValueError("subarcs_per_trajectory must be positive")
        if not (0.0 < self.subarc_min_fraction < 1.0):
            raise ValueError("subarc_min_fraction must lie in (0,1)")
        self.coverage.validate()


@dataclass
class QueryConfig:
    """Settings for atlas retrieval and exact shooting correction."""

    neighbours: int = 48
    direct_seeds: int = 8
    regression_neighbours: int = 24
    max_nfev: int = 90
    rtol: float = 3.0e-10
    atol: float = 3.0e-12
    max_step: float = 0.025
    trajectory_points: int = 1200
    r_collision: float = 0.01
    r_escape: float = 100.0
    max_acceleration: float = 300.0

    # Residual scales: log-radius, angle, radial velocity, tangential velocity,
    # log-resource.  These affect conditioning, not the final acceptance test.
    residual_scale: tuple[float, float, float, float, float] = (
        0.03,
        0.08,
        0.12,
        0.12,
        0.08,
    )
    acceptance: tuple[float, float, float, float, float] = (
        2.0e-6,
        3.0e-6,
        3.0e-6,
        3.0e-6,
        3.0e-6,
    )
    # Allow modest extrapolation beyond sampled launch bounds.
    launch_bound_margin: float = 0.25


@dataclass
class CanonicalBoundary:
    r0: float
    t0: float
    v0_scale: float
    a0_scale: float
    rotation: FloatArray  # canonical -> dimensional orientation
    target: FloatArray  # [u0, w0, log(rho), theta, uf, wf, log(kappa)]
    kappa: float
    delta_inverse_mass: float


@dataclass
class ShootingSolution:
    launch: FloatArray  # [u0,w0,Ar0,At0,Jr0,ell,tau]
    residual: FloatArray
    residual_norm: float
    normal_constant: float
    minimum_radius: float
    maximum_acceleration: float
    radial_turns: int
    trajectory: dict[str, FloatArray]


# ---------------------------------------------------------------------------
# Canonical geometry and dynamics
# ---------------------------------------------------------------------------


def _cross2(a: FloatArray, b: FloatArray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _rot90(v: FloatArray) -> FloatArray:
    return np.array([-v[1], v[0]], dtype=float)


def _canonicalize_boundary(
    initial: CartesianState,
    final: CartesianState,
    capability: RocketCapability,
    revolutions: int,
) -> CanonicalBoundary:
    capability.validate()
    r_i, v_i = initial.arrays()
    r_f, v_f = final.arrays()

    r0 = float(np.linalg.norm(r_i))
    e_r0 = r_i / r0
    e_t0 = _rot90(e_r0)
    # Columns map canonical vectors back into the dimensional inertial frame.
    rotation = np.column_stack((e_r0, e_t0))

    mu = capability.mu
    t0 = math.sqrt(r0**3 / mu)
    vscale = math.sqrt(mu / r0)
    ascale = mu / r0**2

    u0 = float(np.dot(v_i, e_r0) / vscale)
    w0 = float(np.dot(v_i, e_t0) / vscale)

    rf_can = rotation.T @ r_f / r0
    vf_can = rotation.T @ v_f / vscale
    rho = float(np.linalg.norm(rf_can))
    if rho <= 0.0:
        raise ValueError("final position norm must be positive")

    phi = math.atan2(float(rf_can[1]), float(rf_can[0]))
    theta = phi + 2.0 * math.pi * int(revolutions)
    e_rf = rf_can / rho
    e_tf = _rot90(e_rf)
    uf = float(np.dot(vf_can, e_rf))
    wf = float(np.dot(vf_can, e_tf))

    delta = 1.0 / capability.dry_mass - 1.0 / capability.initial_mass
    kappa = (
        2.0
        * capability.useful_power
        * delta
        * r0 ** 2.5
        / mu ** 1.5
    )
    if kappa <= 0.0 or not math.isfinite(kappa):
        raise ValueError("computed dimensionless capability kappa is invalid")

    target = np.array(
        [u0, w0, math.log(rho), theta, uf, wf, math.log(kappa)],
        dtype=float,
    )
    return CanonicalBoundary(
        r0=r0,
        t0=t0,
        v0_scale=vscale,
        a0_scale=ascale,
        rotation=rotation,
        target=target,
        kappa=kappa,
        delta_inverse_mass=delta,
    )


def _initial_vector(launch: FloatArray) -> tuple[FloatArray, float]:
    """Build the 11-state initial vector and normal Hamiltonian constant."""
    u0, w0, ar0, at0, jr0, ell, _tau = map(float, launch)
    jt0 = u0 * at0 - w0 * ar0 - ell

    # state = Rx,Ry,Vx,Vy,Ax,Ay,Jx,Jy,s,theta,min_r_marker(not dynamic omitted)
    y0 = np.array(
        [1.0, 0.0, u0, w0, ar0, at0, jr0, jt0, 0.0, 0.0],
        dtype=float,
    )
    # g0 = (-1,0), K = -J.V + A.g + |A|^2/2
    normal_constant = (
        -(jr0 * u0 + jt0 * w0)
        - ar0
        + 0.5 * (ar0 * ar0 + at0 * at0)
    )
    return y0, float(normal_constant)


def _rhs(_t: float, y: FloatArray) -> FloatArray:
    r = y[0:2]
    v = y[2:4]
    a = y[4:6]
    j = y[6:8]
    rr = float(np.dot(r, r))
    radius = math.sqrt(rr)
    inv_r3 = 1.0 / (rr * radius)
    g = -r * inv_r3
    # Dg @ a = -a/r^3 + 3 r (r.a)/r^5
    jdot = -a * inv_r3 + 3.0 * r * float(np.dot(r, a)) / (rr * rr * radius)
    theta_dot = _cross2(r, v) / rr
    return np.array(
        [
            v[0],
            v[1],
            g[0] + a[0],
            g[1] + a[1],
            j[0],
            j[1],
            jdot[0],
            jdot[1],
            float(np.dot(a, a)),
            theta_dot,
        ],
        dtype=float,
    )



def _rhs_jacobian(y: FloatArray) -> FloatArray:
    """Analytic Jacobian of the 10-state extremal RHS."""
    r = y[0:2]
    v = y[2:4]
    a = y[4:6]
    rr = float(np.dot(r, r))
    radius = math.sqrt(rr)
    inv_r3 = 1.0 / (rr * radius)
    inv_r5 = inv_r3 / rr
    inv_r7 = inv_r5 / rr
    eye2 = np.eye(2)
    dg = -eye2 * inv_r3 + 3.0 * np.outer(r, r) * inv_r5

    jac = np.zeros((10, 10), dtype=float)
    jac[0:2, 2:4] = eye2
    jac[2:4, 0:2] = dg
    jac[2:4, 4:6] = eye2
    jac[4:6, 6:8] = eye2

    ra = float(np.dot(r, a))
    # d[(Dg)A]_i / dr_k
    h = (
        3.0
        * inv_r5
        * (np.outer(a, r) + ra * eye2 + np.outer(r, a))
        - 15.0 * inv_r7 * ra * np.outer(r, r)
    )
    jac[6:8, 0:2] = h
    jac[6:8, 4:6] = dg
    jac[8, 4:6] = 2.0 * a

    cross_rv = _cross2(r, v)
    jac[9, 0:2] = np.array([v[1], -v[0]]) / rr - 2.0 * cross_rv * r / (rr * rr)
    jac[9, 2:4] = np.array([-r[1], r[0]]) / rr
    return jac


def _rhs_with_sens(_t: float, aug: FloatArray) -> FloatArray:
    y = aug[:10]
    sens = aug[10:].reshape(10, 4)
    dy = _rhs(_t, y)
    dsens = _rhs_jacobian(y) @ sens
    return np.concatenate((dy, dsens.ravel()))


def _initial_sensitivity(u0: float, w0: float) -> FloatArray:
    """dy0/d(Ar0, At0, Jr0, ell), shape (10,4)."""
    sens = np.zeros((10, 4), dtype=float)
    sens[4, 0] = 1.0
    sens[5, 1] = 1.0
    sens[6, 2] = 1.0
    sens[7, 0] = -w0
    sens[7, 1] = u0
    sens[7, 3] = -1.0
    return sens


def _endpoint_output_jacobian(yf: FloatArray) -> FloatArray:
    """d(log R, theta, ur, ut, log s) / dy_final."""
    r = yf[0:2]
    v = yf[2:4]
    radius = float(np.linalg.norm(r))
    er = r / radius
    et = _rot90(er)
    ur = float(np.dot(v, er))
    ut = float(np.dot(v, et))
    resource = float(yf[8])
    if resource <= 0.0:
        raise ValueError("non-positive endpoint resource")

    g = np.zeros((5, 10), dtype=float)
    g[0, 0:2] = er / radius
    g[1, 9] = 1.0
    g[2, 0:2] = ut * et / radius
    g[2, 2:4] = er
    g[3, 0:2] = -ur * et / radius
    g[3, 2:4] = et
    g[4, 8] = 1.0 / resource
    return g


def _integrate_launch_with_sens(
    launch: FloatArray,
    *,
    rtol: float,
    atol: float,
    max_step: float,
    r_collision: float,
    r_escape: float,
    max_acceleration: float,
):
    y0, normal_constant = _initial_vector(launch)
    tau = float(launch[6])
    if tau <= 0.0 or normal_constant <= 0.0 or not np.isfinite(normal_constant):
        return None, None, normal_constant, "non-normal"
    sens0 = _initial_sensitivity(float(launch[0]), float(launch[1]))
    aug0 = np.concatenate((y0, sens0.ravel()))

    # Events inspect only the first ten augmented components.
    base_events = _make_events(r_collision, r_escape, max_acceleration)
    events = []
    for base in base_events:
        def event(t, aug, base=base):
            return base(t, aug[:10])
        event.terminal = True  # type: ignore[attr-defined]
        event.direction = -1.0  # type: ignore[attr-defined]
        events.append(event)

    try:
        sol = solve_ivp(
            _rhs_with_sens,
            (0.0, tau),
            aug0,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            events=tuple(events),
        )
    except (FloatingPointError, OverflowError, ValueError):
        return None, None, normal_constant, "integration-exception"
    if not sol.success:
        return None, None, normal_constant, "integration-failed"
    if sol.t[-1] < tau * (1.0 - 1.0e-9):
        return None, None, normal_constant, "guard-event"
    final = sol.y[:10, -1]
    sens = sol.y[10:, -1].reshape(10, 4)
    if not np.all(np.isfinite(final)) or not np.all(np.isfinite(sens)):
        return None, None, normal_constant, "non-finite"
    return final, sens, normal_constant, "ok"


def _make_events(r_collision: float, r_escape: float, max_acceleration: float):
    def collision(_t: float, y: FloatArray) -> float:
        return float(np.linalg.norm(y[0:2]) - r_collision)

    def escape(_t: float, y: FloatArray) -> float:
        return float(r_escape - np.linalg.norm(y[0:2]))

    def acceleration_guard(_t: float, y: FloatArray) -> float:
        return float(max_acceleration - np.linalg.norm(y[4:6]))

    for event in (collision, escape, acceleration_guard):
        event.terminal = True  # type: ignore[attr-defined]
        event.direction = -1.0  # type: ignore[attr-defined]
    return collision, escape, acceleration_guard


def _integrate_launch(
    launch: FloatArray,
    *,
    rtol: float,
    atol: float,
    max_step: float,
    r_collision: float,
    r_escape: float,
    max_acceleration: float,
    dense_output: bool = False,
    t_eval: Optional[FloatArray] = None,
) -> tuple[Optional[object], float, str]:
    y0, normal_constant = _initial_vector(launch)
    tau = float(launch[6])
    if tau <= 0.0 or normal_constant <= 0.0 or not np.isfinite(normal_constant):
        return None, normal_constant, "non-normal"

    events = _make_events(r_collision, r_escape, max_acceleration)
    try:
        sol = solve_ivp(
            _rhs,
            (0.0, tau),
            y0,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            dense_output=dense_output,
            t_eval=t_eval,
            events=events,
        )
    except (FloatingPointError, OverflowError, ValueError):
        return None, normal_constant, "integration-exception"

    if not sol.success:
        return None, normal_constant, "integration-failed"
    if sol.t[-1] < tau * (1.0 - 1.0e-9):
        return None, normal_constant, "guard-event"
    if not np.all(np.isfinite(sol.y[:, -1])):
        return None, normal_constant, "non-finite"
    return sol, normal_constant, "ok"


def _endpoint_from_state(yf: FloatArray) -> tuple[float, float, float, float, float]:
    r = yf[0:2]
    v = yf[2:4]
    radius = float(np.linalg.norm(r))
    er = r / radius
    et = _rot90(er)
    theta = float(yf[9])
    ur = float(np.dot(v, er))
    ut = float(np.dot(v, et))
    kappa = float(yf[8])
    return radius, theta, ur, ut, kappa


def _count_radial_turns(states: FloatArray) -> int:
    r = states[0:2, :].T
    v = states[2:4, :].T
    rv = np.einsum("ij,ij->i", r, v)
    # Ignore tiny values to reduce false counts at tangencies.
    threshold = 1.0e-8 * np.maximum(1.0, np.linalg.norm(r, axis=1) * np.linalg.norm(v, axis=1))
    signs = np.sign(np.where(np.abs(rv) > threshold, rv, 0.0))
    nz = signs[signs != 0.0]
    if len(nz) < 2:
        return 0
    return int(np.count_nonzero(nz[1:] != nz[:-1]))


# ---------------------------------------------------------------------------
# Offline atlas generation
# ---------------------------------------------------------------------------


def _atlas_config_from_dict(payload: dict) -> AtlasConfig:
    data = dict(payload)
    coverage_payload = data.get("coverage")
    if isinstance(coverage_payload, dict):
        data["coverage"] = CoverageConfig(**coverage_payload)
    return AtlasConfig(**data)


def _sample_launches(config: AtlasConfig) -> FloatArray:
    config.validate()
    # Sobol works best at powers of two. random(n) is still supported for any n.
    sampler = qmc.Sobol(d=7, scramble=True, seed=config.seed)
    unit = sampler.random(config.n_samples)

    linear_bounds = np.array(
        [
            config.u0_bounds,
            config.w0_bounds,
            config.ar0_bounds,
            config.at0_bounds,
            config.jr0_bounds,
            config.ell_bounds,
        ],
        dtype=float,
    )
    launch = np.empty((config.n_samples, 7), dtype=float)
    launch[:, :6] = qmc.scale(unit[:, :6], linear_bounds[:, 0], linear_bounds[:, 1])
    log_tau_lo, log_tau_hi = np.log(config.tau_bounds)
    launch[:, 6] = np.exp(log_tau_lo + unit[:, 6] * (log_tau_hi - log_tau_lo))
    return launch


def _subarc_pairs(n_points: int, count: int, min_fraction: float, seed_value: int) -> list[tuple[int, int]]:
    """Select deterministic, stratified subintervals from a sampled trajectory."""
    pairs: list[tuple[int, int]] = [(0, n_points - 1)]
    if count <= 1:
        return pairs
    rng = np.random.default_rng(seed_value)
    min_span = max(2, int(math.ceil(min_fraction * (n_points - 1))))
    attempts = 0
    while len(pairs) < count and attempts < 50 * count:
        attempts += 1
        # Bias span logarithmically so both short and long subarcs are harvested.
        span = int(round(math.exp(rng.uniform(math.log(min_span), math.log(n_points - 1)))))
        span = min(max(span, min_span), n_points - 1)
        start = int(rng.integers(0, n_points - span))
        pair = (start, start + span)
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def _canonical_subarc_row(
    launch: FloatArray,
    sol,
    ia: int,
    ib: int,
    cfg: AtlasConfig,
) -> Optional[tuple[FloatArray, FloatArray, FloatArray]]:
    ya = sol.y[:, ia]
    yb = sol.y[:, ib]
    ta = float(sol.t[ia])
    tb = float(sol.t[ib])
    if tb <= ta:
        return None

    ra_vec = ya[0:2]
    va = ya[2:4]
    aa = ya[4:6]
    ja = ya[6:8]
    rb_vec = yb[0:2]
    vb = yb[2:4]
    ra = float(np.linalg.norm(ra_vec))
    rb = float(np.linalg.norm(rb_vec))
    if ra <= 0.0 or rb <= 0.0:
        return None
    era = ra_vec / ra
    eta = _rot90(era)
    erb = rb_vec / rb
    etb = _rot90(erb)

    sqrt_ra = math.sqrt(ra)
    u0 = sqrt_ra * float(np.dot(va, era))
    w0 = sqrt_ra * float(np.dot(va, eta))
    ar0 = ra**2 * float(np.dot(aa, era))
    at0 = ra**2 * float(np.dot(aa, eta))
    jerk_scale = ra**3.5
    jr0 = jerk_scale * float(np.dot(ja, era))
    jt0 = jerk_scale * float(np.dot(ja, eta))
    ell = -jt0 + u0 * at0 - w0 * ar0
    tau = (tb - ta) / ra**1.5

    rho = rb / ra
    theta = float(yb[9] - ya[9])
    ur = sqrt_ra * float(np.dot(vb, erb))
    ut = sqrt_ra * float(np.dot(vb, etb))
    kappa = ra**2.5 * float(yb[8] - ya[8])
    if kappa <= 0.0 or tau <= 0.0:
        return None

    if not (cfg.rho_bounds[0] <= rho <= cfg.rho_bounds[1]):
        return None
    if not (cfg.theta_bounds[0] <= theta <= cfg.theta_bounds[1]):
        return None
    if not (cfg.kappa_bounds[0] <= kappa <= cfg.kappa_bounds[1]):
        return None
    if math.hypot(ur, ut) > cfg.final_speed_max:
        return None

    sub_launch = np.array([u0, w0, ar0, at0, jr0, ell, tau], dtype=float)
    _y0, normal_constant = _initial_vector(sub_launch)
    if normal_constant <= 0.0 or not np.isfinite(normal_constant):
        return None
    endpoint = np.array(
        [u0, w0, math.log(rho), theta, ur, ut, math.log(kappa)], dtype=float
    )

    segment = sol.y[:, ia : ib + 1]
    radii = np.linalg.norm(segment[0:2, :], axis=0) / ra
    accelerations = np.linalg.norm(segment[4:6, :], axis=0) * ra**2
    diagnostics = np.array(
        [
            normal_constant,
            float(np.min(radii)),
            float(np.max(radii)),
            float(np.max(accelerations)),
            float(_count_radial_turns(segment)),
            float(round(theta / (2.0 * math.pi))),
        ],
        dtype=float,
    )
    return sub_launch, endpoint, diagnostics


def _worker_generate(args):
    launch, cfg_dict = args
    cfg = _atlas_config_from_dict(cfg_dict)
    t_eval = np.linspace(0.0, float(launch[6]), cfg.diagnostic_points)
    sol, normal_constant, status = _integrate_launch(
        np.asarray(launch, dtype=float),
        rtol=cfg.rtol,
        atol=cfg.atol,
        max_step=cfg.max_step,
        r_collision=cfg.r_collision,
        r_escape=cfg.r_escape,
        max_acceleration=cfg.max_acceleration,
        t_eval=t_eval,
    )
    if sol is None:
        return []

    # Stable per-launch seed: deterministic across process scheduling.
    rounded = np.round(np.asarray(launch, dtype=float), 10)
    weights = np.array([73856093, 19349663, 83492791, 2654435761, 97531, 421, 17], dtype=np.float64)
    seed_value = int(abs(float(np.dot(rounded, weights))) * 1.0e6) % (2**32 - 1)
    rows = []
    for ia, ib in _subarc_pairs(
        len(sol.t), cfg.subarcs_per_trajectory, cfg.subarc_min_fraction, seed_value
    ):
        row = _canonical_subarc_row(np.asarray(launch), sol, ia, ib, cfg)
        if row is not None:
            rows.append(row)
    return rows


def _worker_generate_batch(args):
    launch_batch, cfg_dict = args
    results = []
    for launch in launch_batch:
        results.extend(_worker_generate((launch, cfg_dict)))
    return results


def _robust_feature_scale(endpoint: FloatArray) -> FloatArray:
    # Median absolute deviation, with sensible floors.  Coordinates are already
    # log-transformed where appropriate.
    med = np.median(endpoint, axis=0)
    mad = 1.4826 * np.median(np.abs(endpoint - med), axis=0)
    floors = np.array([0.25, 0.25, 0.25, 0.35, 0.25, 0.25, 0.35], dtype=float)
    scale = np.maximum(mad, floors)
    return scale



def _generate_rows_from_launches(
    launches: FloatArray, config: AtlasConfig
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Integrate launch attempts and return accepted forward/subarc rows."""
    if len(launches) == 0:
        return (
            np.empty((0, 7), dtype=float),
            np.empty((0, 7), dtype=float),
            np.empty((0, 6), dtype=float),
        )
    cfg_dict = asdict(config)
    accepted_launch: list[FloatArray] = []
    accepted_endpoint: list[FloatArray] = []
    accepted_diag: list[FloatArray] = []

    if config.workers == 1:
        iterator = (_worker_generate((row, cfg_dict)) for row in launches)
        for rows in iterator:
            for launch, endpoint, diag in rows:
                accepted_launch.append(launch)
                accepted_endpoint.append(endpoint)
                accepted_diag.append(diag)
    else:
        batch_size = max(16, min(512, len(launches) // (config.workers * 8) or 16))
        batches = [launches[i : i + batch_size] for i in range(0, len(launches), batch_size)]
        with ProcessPoolExecutor(max_workers=config.workers) as pool:
            futures = [pool.submit(_worker_generate_batch, (batch, cfg_dict)) for batch in batches]
            for future in as_completed(futures):
                for launch, endpoint, diag in future.result():
                    accepted_launch.append(launch)
                    accepted_endpoint.append(endpoint)
                    accepted_diag.append(diag)

    if not accepted_launch:
        return (
            np.empty((0, 7), dtype=float),
            np.empty((0, 7), dtype=float),
            np.empty((0, 6), dtype=float),
        )
    return (
        np.vstack(accepted_launch),
        np.vstack(accepted_endpoint),
        np.vstack(accepted_diag),
    )


def _branch_labels(diagnostics: FloatArray) -> NDArray[np.int64]:
    """Discrete labels used only to avoid merging obvious branch families."""
    if len(diagnostics) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.rint(diagnostics[:, 4:6]).astype(np.int64)


def _thin_endpoint_rows(
    launch: FloatArray,
    endpoint: FloatArray,
    diagnostics: FloatArray,
    feature_scale: FloatArray,
    cell_size: float,
    max_rows: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Cheap branch-aware grid thinning in normalized endpoint coordinates.

    This is only a first-pass reduction. Final deletion is verified by an
    actual shooting correction in :func:`_prune_redundant_rows`.
    """
    if len(launch) <= 1:
        return launch, endpoint, diagnostics
    features = endpoint / feature_scale
    labels = _branch_labels(diagnostics)
    # Keep shorter arcs first inside a coarse endpoint/branch cell.
    order = np.argsort(launch[:, 6], kind="stable")
    occupied: set[tuple[int, ...]] = set()
    keep: list[int] = []
    inv = 1.0 / cell_size
    for idx in order:
        cell = tuple(np.floor(features[idx] * inv).astype(np.int64)) + tuple(labels[idx])
        if cell in occupied:
            continue
        occupied.add(cell)
        keep.append(int(idx))
        if len(keep) >= max_rows:
            break
    keep_array = np.asarray(keep, dtype=int)
    return launch[keep_array], endpoint[keep_array], diagnostics[keep_array]


def _local_regression_seed_arrays(
    target: FloatArray,
    indices: FloatArray,
    launch: FloatArray,
    endpoint: FloatArray,
    feature_scale: FloatArray,
) -> Optional[FloatArray]:
    if len(indices) < 8:
        return None
    x = endpoint[indices]
    q = np.column_stack((launch[indices, 2:6], np.log(launch[indices, 6])))
    delta = (x - target) / feature_scale
    dist2 = np.sum(delta * delta, axis=1)
    positive = dist2[dist2 > 0.0]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    bandwidth = max(bandwidth, 1.0e-6)
    weights = np.exp(-0.5 * dist2 / bandwidth)
    design = np.column_stack((np.ones(len(indices)), delta))
    sw = np.sqrt(np.maximum(weights, 1.0e-12))
    aw = design * sw[:, None]
    bw = q * sw[:, None]
    lhs = aw.T @ aw
    lhs[1:, 1:] += 1.0e-7 * np.eye(7)
    rhs = aw.T @ bw
    try:
        coefficient = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return None
    prediction = coefficient[0]
    return prediction if np.all(np.isfinite(prediction)) else None


def _launch_correction_bounds(
    launch: FloatArray, margin: float
) -> tuple[FloatArray, FloatArray]:
    lo = np.min(launch[:, 2:7], axis=0)
    hi = np.max(launch[:, 2:7], axis=0)
    span = np.maximum(hi - lo, 1.0e-6)
    lo = lo - margin * span
    hi = hi + margin * span
    lo[-1] = math.log(max(lo[-1], 1.0e-5))
    hi[-1] = math.log(max(hi[-1], math.exp(lo[-1]) * 1.001))
    return lo, hi


def _fast_correct_target(
    target: FloatArray,
    launch: FloatArray,
    endpoint: FloatArray,
    feature_scale: FloatArray,
    config: QueryConfig,
    *,
    tree: Optional[cKDTree] = None,
    excluded: Optional[set[int]] = None,
    bounds: Optional[tuple[FloatArray, FloatArray]] = None,
) -> tuple[Optional[FloatArray], float, int]:
    """Correct one exact endpoint using only other atlas seeds.

    Returns ``(corrected_launch, nearest_distance, nfev)``. The corrected launch
    is ``None`` when no supplied seed converges.
    """
    if len(launch) == 0:
        return None, math.inf, 0
    if tree is None:
        tree = cKDTree(endpoint / feature_scale)
    excluded = excluded or set()
    requested = min(len(launch), max(config.neighbours + len(excluded) + 4, 1))
    distances, indices = tree.query(target / feature_scale, k=requested)
    distances = np.atleast_1d(distances).astype(float)
    indices = np.atleast_1d(indices).astype(int)
    retained = [(d, i) for d, i in zip(distances, indices) if int(i) not in excluded]
    if not retained:
        return None, math.inf, 0
    nearest_distance = float(retained[0][0])
    nearest = np.array([i for _d, i in retained[: config.neighbours]], dtype=int)

    seeds: list[FloatArray] = []
    regression_indices = nearest[: min(config.regression_neighbours, len(nearest))]
    regression = _local_regression_seed_arrays(
        target, regression_indices, launch, endpoint, feature_scale
    )
    if regression is not None:
        seeds.append(regression)
    for index in nearest[: min(config.direct_seeds, len(nearest))]:
        seeds.append(np.concatenate((launch[index, 2:6], [math.log(launch[index, 6])])))

    lo, hi = bounds if bounds is not None else _launch_correction_bounds(
        launch, config.launch_bound_margin
    )
    target5 = target[2:7]
    total_nfev = 0
    best: Optional[FloatArray] = None
    best_norm = math.inf

    for seed in seeds:
        seed = np.clip(np.asarray(seed, dtype=float), lo + 1.0e-10, hi - 1.0e-10)
        cache_x: Optional[FloatArray] = None
        cache_value = None

        def evaluate(x):
            nonlocal cache_x, cache_value
            x = np.asarray(x, dtype=float)
            if cache_x is None or not np.array_equal(x, cache_x):
                cache_x = x.copy()
                cache_value = ExtremalAtlas._shooting_residual_and_jacobian(
                    x, float(target[0]), float(target[1]), target5, config
                )
            return cache_value

        result = least_squares(
            lambda x: evaluate(x)[0],
            seed,
            bounds=(lo, hi),
            method="trf",
            jac=lambda x: evaluate(x)[1],
            x_scale="jac",
            max_nfev=config.max_nfev,
            ftol=1.0e-11,
            xtol=1.0e-11,
            gtol=1.0e-11,
        )
        total_nfev += int(result.nfev)
        _scaled, _jac, yf, normal_constant, status, corrected_launch = evaluate(result.x)
        if yf is None or status != "ok" or normal_constant <= 0.0:
            continue
        radius, theta, ur, ut, kappa = _endpoint_from_state(yf)
        if radius <= 0.0 or kappa <= 0.0:
            continue
        raw = np.array(
            [
                math.log(radius) - target5[0],
                theta - target5[1],
                ur - target5[2],
                ut - target5[3],
                math.log(kappa) - target5[4],
            ],
            dtype=float,
        )
        if np.any(np.abs(raw) > np.asarray(config.acceptance, dtype=float)):
            continue
        norm = float(np.linalg.norm(raw / np.asarray(config.residual_scale, dtype=float)))
        if norm < best_norm:
            best_norm = norm
            best = corrected_launch.copy()
    return best, nearest_distance, total_nfev


def _coverage_query_config(config: AtlasConfig, *, pruning: bool = False) -> QueryConfig:
    coverage = config.coverage
    return QueryConfig(
        neighbours=coverage.neighbours,
        direct_seeds=coverage.direct_seeds,
        regression_neighbours=coverage.regression_neighbours,
        max_nfev=coverage.prune_max_nfev if pruning else coverage.max_nfev,
        rtol=max(config.rtol, 5.0e-10),
        atol=max(config.atol, 5.0e-12),
        max_step=config.max_step,
        trajectory_points=max(64, config.diagnostic_points),
        r_collision=config.r_collision,
        r_escape=config.r_escape,
        max_acceleration=config.max_acceleration,
        residual_scale=(0.03, 0.08, 0.12, 0.12, 0.08),
        acceptance=(8.0e-6, 1.2e-5, 1.2e-5, 1.2e-5, 1.2e-5),
        launch_bound_margin=0.25,
    )


def _select_validation_indices(
    endpoint: FloatArray,
    diagnostics: FloatArray,
    feature_scale: FloatArray,
    requested: int,
    rng: np.random.Generator,
    cell_size: float,
) -> FloatArray:
    """Select spatially diverse holdouts instead of density-weighted rows."""
    count = len(endpoint)
    if count <= requested:
        return np.arange(count, dtype=int)
    labels = _branch_labels(diagnostics)
    inv = 1.0 / cell_size
    order = rng.permutation(count)
    representative: dict[tuple[int, ...], int] = {}
    for idx in order:
        cell = tuple(np.floor(endpoint[idx] / feature_scale * inv).astype(np.int64)) + tuple(labels[idx])
        representative.setdefault(cell, int(idx))
    diverse = np.fromiter(representative.values(), dtype=int)
    if len(diverse) >= requested:
        selected = rng.choice(diverse, size=requested, replace=False)
        return np.sort(selected).astype(int)
    selected_set = set(map(int, diverse))
    remaining = np.array([i for i in range(count) if i not in selected_set], dtype=int)
    need = requested - len(diverse)
    extra = rng.choice(remaining, size=need, replace=False)
    return np.sort(np.concatenate((diverse, extra))).astype(int)


def _failure_insertion_indices(
    failed_indices: FloatArray,
    distances: FloatArray,
    endpoint: FloatArray,
    diagnostics: FloatArray,
    feature_scale: FloatArray,
    cell_size: float,
    limit: int,
) -> FloatArray:
    if len(failed_indices) == 0:
        return np.empty(0, dtype=int)
    labels = _branch_labels(diagnostics)
    order = failed_indices[np.argsort(distances[failed_indices])[::-1]]
    occupied: set[tuple[int, ...]] = set()
    keep: list[int] = []
    inv = 1.0 / cell_size
    for idx in order:
        cell = tuple(np.floor(endpoint[idx] / feature_scale * inv).astype(np.int64)) + tuple(labels[idx])
        if cell in occupied:
            continue
        occupied.add(cell)
        keep.append(int(idx))
        if len(keep) >= limit:
            break
    return np.asarray(keep, dtype=int)


def _prune_redundant_rows(
    launch: FloatArray,
    endpoint: FloatArray,
    diagnostics: FloatArray,
    feature_scale: FloatArray,
    config: AtlasConfig,
) -> tuple[FloatArray, FloatArray, FloatArray, int]:
    coverage = config.coverage
    if not coverage.prune_enabled or coverage.prune_max_checks == 0 or len(launch) < 3:
        return launch, endpoint, diagnostics, 0

    features = endpoint / feature_scale
    tree = cKDTree(features)
    distances, neighbours = tree.query(features, k=2)
    labels = _branch_labels(diagnostics)
    candidates = np.argsort(distances[:, 1])
    active = np.ones(len(launch), dtype=bool)
    protected: set[int] = set()
    removed = 0
    checks = 0
    query_config = _coverage_query_config(config, pruning=True)
    correction_bounds = _launch_correction_bounds(
        launch, query_config.launch_bound_margin
    )

    for index in candidates:
        index = int(index)
        neighbour = int(neighbours[index, 1])
        if distances[index, 1] > coverage.prune_distance:
            break
        if checks >= coverage.prune_max_checks:
            break
        if not active[index] or not active[neighbour] or index in protected:
            continue
        if not np.array_equal(labels[index], labels[neighbour]):
            continue
        checks += 1
        excluded = {index}
        corrected, _distance, _nfev = _fast_correct_target(
            endpoint[index], launch, endpoint, feature_scale, query_config,
            tree=tree, excluded=excluded, bounds=correction_bounds,
        )
        if corrected is None:
            continue
        active[index] = False
        protected.add(neighbour)
        removed += 1

    return launch[active], endpoint[active], diagnostics[active], removed


def _estimate_coverage_cells(
    atlas_endpoint: FloatArray,
    feature_scale: FloatArray,
    probe_endpoint: FloatArray,
    probe_success: NDArray[np.bool_],
) -> tuple[FloatArray, NDArray[np.int32], NDArray[np.int32]]:
    radius = np.zeros(len(atlas_endpoint), dtype=float)
    success_count = np.zeros(len(atlas_endpoint), dtype=np.int32)
    failure_count = np.zeros(len(atlas_endpoint), dtype=np.int32)
    if len(atlas_endpoint) == 0 or len(probe_endpoint) == 0:
        return radius, success_count, failure_count
    tree = cKDTree(atlas_endpoint / feature_scale)
    distances, indices = tree.query(probe_endpoint / feature_scale, k=1)
    nearest_failure = np.full(len(atlas_endpoint), np.inf, dtype=float)
    for distance, index, success in zip(distances, indices, probe_success):
        index = int(index)
        distance = float(distance)
        if success:
            success_count[index] += 1
            radius[index] = max(radius[index], distance)
        else:
            failure_count[index] += 1
            nearest_failure[index] = min(nearest_failure[index], distance)
    constrained = np.isfinite(nearest_failure) & (radius > 0.0)
    radius[constrained] = np.minimum(radius[constrained], 0.9 * nearest_failure[constrained])
    return radius, success_count, failure_count


def generate_atlas(path: str | Path, config: AtlasConfig) -> Path:
    """Generate an adaptive, coverage-validated forward extremal atlas.

    When ``config.coverage.enabled`` is false, generation falls back to the
    original fixed-count forward cloud.  Adaptive coverage is tested on
    independent feasible holdout extremals from the configured launch domain.
    """
    config.validate()
    if not config.coverage.enabled:
        return _generate_atlas_fixed(path, config)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    coverage = config.coverage
    launches = _sample_launches(config)
    initial_count = int(round(config.n_samples * coverage.initial_fraction))
    initial_count = min(max(initial_count, 1), max(config.n_samples - 1, 1))

    print(f"Integrating initial coverage seed launches: {initial_count:,}")
    seed_launch, seed_endpoint, seed_diag = _generate_rows_from_launches(
        launches[:initial_count], config
    )
    if len(seed_launch) == 0:
        raise RuntimeError("No initial atlas samples survived integration")

    if config.feature_scale is None:
        feature_scale = _robust_feature_scale(seed_endpoint)
    else:
        feature_scale = np.asarray(config.feature_scale, dtype=float)
        if feature_scale.shape != (7,) or np.any(feature_scale <= 0.0):
            raise ValueError("feature_scale must contain seven positive values")

    atlas_launch, atlas_endpoint, atlas_diag = _thin_endpoint_rows(
        seed_launch,
        seed_endpoint,
        seed_diag,
        feature_scale,
        coverage.initial_cell_size,
        coverage.max_atlas_rows,
    )
    print(
        f"Initial accepted rows: {len(seed_launch):,}; "
        f"after coarse branch-aware thinning: {len(atlas_launch):,}"
    )

    rng = np.random.default_rng(coverage.validation_seed)
    query_config = _coverage_query_config(config)
    cursor = initial_count
    successful_rounds = 0
    round_number = 0
    coverage_history: list[dict] = []
    last_probe_endpoint = np.empty((0, 7), dtype=float)
    last_probe_success = np.empty(0, dtype=bool)

    while (
        cursor < config.n_samples
        and round_number < coverage.max_rounds
        and len(atlas_launch) < coverage.max_atlas_rows
    ):
        round_number += 1
        stop = min(cursor + coverage.batch_launches, config.n_samples)
        batch_launches = launches[cursor:stop]
        cursor = stop
        candidate_launch, candidate_endpoint, candidate_diag = _generate_rows_from_launches(
            batch_launches, config
        )
        if len(candidate_launch) == 0:
            coverage_history.append(
                {"round": round_number, "launches": int(len(batch_launches)), "rows": 0}
            )
            continue

        validation_indices = _select_validation_indices(
            candidate_endpoint,
            candidate_diag,
            feature_scale,
            coverage.validation_rows,
            rng,
            coverage.validation_cell_size,
        )
        tree = cKDTree(atlas_endpoint / feature_scale)
        correction_bounds = _launch_correction_bounds(
            atlas_launch, query_config.launch_bound_margin
        )
        success = np.zeros(len(validation_indices), dtype=bool)
        nearest_distance = np.full(len(validation_indices), np.inf, dtype=float)
        nfev_total = 0

        for local, candidate_index in enumerate(validation_indices):
            corrected, distance, nfev = _fast_correct_target(
                candidate_endpoint[candidate_index],
                atlas_launch,
                atlas_endpoint,
                feature_scale,
                query_config,
                tree=tree,
                bounds=correction_bounds,
            )
            success[local] = corrected is not None
            nearest_distance[local] = distance
            nfev_total += nfev

        tested = len(validation_indices)
        success_rate = float(np.mean(success)) if tested else 0.0
        failed_local = np.flatnonzero(~success)
        failed_candidate_indices = validation_indices[failed_local]

        # Distances indexed in candidate-row space for insertion ranking.
        candidate_distances = np.full(len(candidate_launch), -np.inf, dtype=float)
        candidate_distances[validation_indices] = nearest_distance
        remaining_capacity = max(0, coverage.max_atlas_rows - len(atlas_launch))
        insertion_limit = min(coverage.insert_failures_per_round, remaining_capacity)
        insert_indices = _failure_insertion_indices(
            failed_candidate_indices,
            candidate_distances,
            candidate_endpoint,
            candidate_diag,
            feature_scale,
            coverage.insertion_cell_size,
            insertion_limit,
        )
        if len(insert_indices):
            atlas_launch = np.vstack((atlas_launch, candidate_launch[insert_indices]))
            atlas_endpoint = np.vstack((atlas_endpoint, candidate_endpoint[insert_indices]))
            atlas_diag = np.vstack((atlas_diag, candidate_diag[insert_indices]))

        enough = tested >= coverage.minimum_validation_rows
        if enough and success_rate >= coverage.target_success:
            successful_rounds += 1
        else:
            successful_rounds = 0

        record = {
            "round": round_number,
            "launch_attempts": int(len(batch_launches)),
            "candidate_rows": int(len(candidate_launch)),
            "validation_rows": int(tested),
            "successes": int(np.count_nonzero(success)),
            "success_rate": success_rate,
            "inserted_failures": int(len(insert_indices)),
            "atlas_rows": int(len(atlas_launch)),
            "median_nearest_distance": float(np.median(nearest_distance)) if tested else math.nan,
            "p95_nearest_distance": float(np.quantile(nearest_distance, 0.95)) if tested else math.nan,
            "correction_nfev": int(nfev_total),
        }
        coverage_history.append(record)
        last_probe_endpoint = candidate_endpoint[validation_indices].copy()
        last_probe_success = success.copy()
        print(
            f"Coverage round {round_number}: success {success_rate:.1%} "
            f"({np.count_nonzero(success)}/{tested}), inserted {len(insert_indices)}, "
            f"atlas rows {len(atlas_launch):,}"
        )

        if successful_rounds >= coverage.patience:
            print(
                f"Coverage target {coverage.target_success:.1%} reached for "
                f"{coverage.patience} consecutive rounds."
            )
            break

    before_prune = len(atlas_launch)
    atlas_launch, atlas_endpoint, atlas_diag, pruned = _prune_redundant_rows(
        atlas_launch, atlas_endpoint, atlas_diag, feature_scale, config
    )
    if pruned:
        print(f"Verified redundant seeds removed: {pruned:,}")

    coverage_radius, coverage_success_count, coverage_failure_count = _estimate_coverage_cells(
        atlas_endpoint, feature_scale, last_probe_endpoint, last_probe_success
    )

    final_success_rate = (
        float(np.mean(last_probe_success)) if len(last_probe_success) else math.nan
    )
    converged = bool(
        len(last_probe_success) >= coverage.minimum_validation_rows
        and final_success_rate >= coverage.target_success
        and successful_rounds >= coverage.patience
    )
    cfg_dict = asdict(config)
    metadata = {
        "format_version": 2,
        "model": "planar-kepler-costate-free-normal-pmp",
        "atlas_strategy": "adaptive-feasible-holdout-coverage",
        "launch_columns": ["u0", "w0", "Ar0", "At0", "Jr0", "ell", "tau"],
        "endpoint_columns": [
            "u0", "w0", "log_rho", "theta_unwrapped", "ur_final", "ut_final", "log_kappa"
        ],
        "diagnostic_columns": [
            "normal_constant", "minimum_radius", "maximum_radius", "maximum_acceleration",
            "radial_turns", "rounded_winding"
        ],
        "config": cfg_dict,
        "accepted_atlas_rows": int(len(atlas_launch)),
        "initial_rows_before_thinning": int(len(seed_launch)),
        "rows_before_verified_pruning": int(before_prune),
        "verified_pruned_rows": int(pruned),
        "integrated_launch_attempts": int(cursor),
        "launch_attempt_budget": int(config.n_samples),
        "subarcs_per_trajectory": int(config.subarcs_per_trajectory),
        "coverage_converged": converged,
        "coverage_final_success_rate": final_success_rate,
        "coverage_successful_rounds": int(successful_rounds),
        "coverage_history": coverage_history,
        "coverage_definition": (
            "Independent feasible holdout endpoint is covered when exact shooting correction "
            "converges from atlas seeds without using the holdout launch."
        ),
    }
    np.savez_compressed(
        path,
        launch=atlas_launch,
        endpoint=atlas_endpoint,
        diagnostics=atlas_diag,
        feature_scale=feature_scale,
        coverage_radius=coverage_radius,
        coverage_success_count=coverage_success_count,
        coverage_failure_count=coverage_failure_count,
        metadata_json=np.array(json.dumps(metadata)),
    )
    if not converged:
        print(
            "WARNING: launch budget ended before the configured coverage criterion was met. "
            "The atlas was saved with coverage_converged=false."
        )
    return path


def _generate_atlas_fixed(path: str | Path, config: AtlasConfig) -> Path:
    """Generate and save a forward-extremal atlas.

    The saved NPZ contains no pickled Python objects and is safe to load with
    ``allow_pickle=False``.
    """
    config.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    launches = _sample_launches(config)
    cfg_dict = asdict(config)
    accepted_launch: list[FloatArray] = []
    accepted_endpoint: list[FloatArray] = []
    accepted_diag: list[FloatArray] = []

    if config.workers == 1:
        iterator = (_worker_generate((row, cfg_dict)) for row in launches)
        for rows in iterator:
            for l, e, d in rows:
                accepted_launch.append(l)
                accepted_endpoint.append(e)
                accepted_diag.append(d)
    else:
        batch_size = max(16, min(512, config.n_samples // (config.workers * 8) or 16))
        batches = [launches[i : i + batch_size] for i in range(0, len(launches), batch_size)]
        with ProcessPoolExecutor(max_workers=config.workers) as pool:
            futures = [pool.submit(_worker_generate_batch, (batch, cfg_dict)) for batch in batches]
            for future in as_completed(futures):
                for l, e, d in future.result():
                    accepted_launch.append(l)
                    accepted_endpoint.append(e)
                    accepted_diag.append(d)

    if not accepted_launch:
        raise RuntimeError(
            "No atlas samples survived. Broaden the endpoint domain or reduce "
            "the sampled launch bounds."
        )

    launch_array = np.vstack(accepted_launch)
    endpoint_array = np.vstack(accepted_endpoint)
    diagnostics_array = np.vstack(accepted_diag)
    if config.feature_scale is None:
        feature_scale = _robust_feature_scale(endpoint_array)
    else:
        feature_scale = np.asarray(config.feature_scale, dtype=float)
        if feature_scale.shape != (7,) or np.any(feature_scale <= 0.0):
            raise ValueError("feature_scale must contain seven positive values")

    metadata = {
        "format_version": 1,
        "model": "planar-kepler-costate-free-normal-pmp",
        "launch_columns": ["u0", "w0", "Ar0", "At0", "Jr0", "ell", "tau"],
        "endpoint_columns": [
            "u0",
            "w0",
            "log_rho",
            "theta_unwrapped",
            "ur_final",
            "ut_final",
            "log_kappa",
        ],
        "diagnostic_columns": [
            "normal_constant",
            "minimum_radius",
            "maximum_radius",
            "maximum_acceleration",
            "radial_turns",
            "rounded_winding",
        ],
        "config": cfg_dict,
        "accepted_atlas_rows": int(len(launch_array)),
        "integrated_launch_attempts": int(config.n_samples),
        "subarcs_per_trajectory": int(config.subarcs_per_trajectory),
    }
    np.savez_compressed(
        path,
        launch=launch_array,
        endpoint=endpoint_array,
        diagnostics=diagnostics_array,
        feature_scale=feature_scale,
        metadata_json=np.array(json.dumps(metadata)),
    )
    return path


# ---------------------------------------------------------------------------
# Atlas query and exact correction
# ---------------------------------------------------------------------------


class ExtremalAtlas:
    """Load, query, and enrich a forward-extremal NPZ atlas."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as data:
            self.launch = np.asarray(data["launch"], dtype=float)
            self.endpoint = np.asarray(data["endpoint"], dtype=float)
            self.diagnostics = np.asarray(data["diagnostics"], dtype=float)
            self.feature_scale = np.asarray(data["feature_scale"], dtype=float)
            self.coverage_radius = (
                np.asarray(data["coverage_radius"], dtype=float)
                if "coverage_radius" in data.files
                else np.zeros(len(self.launch), dtype=float)
            )
            self.coverage_success_count = (
                np.asarray(data["coverage_success_count"], dtype=np.int32)
                if "coverage_success_count" in data.files
                else np.zeros(len(self.launch), dtype=np.int32)
            )
            self.coverage_failure_count = (
                np.asarray(data["coverage_failure_count"], dtype=np.int32)
                if "coverage_failure_count" in data.files
                else np.zeros(len(self.launch), dtype=np.int32)
            )
            self.metadata = json.loads(str(data["metadata_json"].item()))
        if self.launch.ndim != 2 or self.launch.shape[1] != 7:
            raise ValueError("invalid atlas launch array")
        if self.endpoint.shape != self.launch.shape:
            raise ValueError("invalid atlas endpoint array")
        if self.feature_scale.shape != (7,) or np.any(self.feature_scale <= 0.0):
            raise ValueError("invalid atlas feature scales")
        if self.coverage_radius.shape != (len(self.launch),):
            raise ValueError("invalid atlas coverage_radius array")
        self._features = self.endpoint / self.feature_scale
        self._tree = cKDTree(self._features)
        self._launch_min = np.min(self.launch[:, 2:7], axis=0)
        self._launch_max = np.max(self.launch[:, 2:7], axis=0)

    def coverage_certificate(
        self,
        initial: CartesianState,
        final: CartesianState,
        capability: RocketCapability,
        revolutions: int = 0,
    ) -> dict:
        """Return the nearest seed and its empirically validated cell radius."""
        boundary = _canonicalize_boundary(initial, final, capability, revolutions)
        distance, index = self._tree.query(boundary.target / self.feature_scale, k=1)
        index = int(index)
        radius = float(self.coverage_radius[index])
        return {
            "seed_index": index,
            "distance": float(distance),
            "validated_radius": radius,
            "inside_validated_cell": bool(radius > 0.0 and distance <= radius),
            "success_count": int(self.coverage_success_count[index]),
            "failure_count": int(self.coverage_failure_count[index]),
        }

    def coverage_distance(
        self,
        initial: CartesianState,
        final: CartesianState,
        capability: RocketCapability,
        revolutions: int = 0,
    ) -> float:
        return float(
            self.coverage_certificate(initial, final, capability, revolutions)["distance"]
        )

    def _nearest_indices(self, target: FloatArray, k: int) -> FloatArray:
        k = min(max(1, k), len(self.launch))
        _dist, idx = self._tree.query(target / self.feature_scale, k=k)
        return np.atleast_1d(idx).astype(int)

    def _local_regression_seed(self, target: FloatArray, indices: FloatArray) -> Optional[FloatArray]:
        """Weighted local affine inverse chart endpoint -> launch variables."""
        if len(indices) < 8:
            return None
        x = self.endpoint[indices]
        # Regress q = [Ar,At,Jr,ell,log(tau)] rather than raw tau.
        q = np.column_stack((self.launch[indices, 2:6], np.log(self.launch[indices, 6])))
        delta = (x - target) / self.feature_scale
        dist2 = np.sum(delta * delta, axis=1)
        positive = dist2[dist2 > 0.0]
        bandwidth = float(np.median(positive)) if len(positive) else 1.0
        bandwidth = max(bandwidth, 1.0e-6)
        weights = np.exp(-0.5 * dist2 / bandwidth)
        design = np.column_stack((np.ones(len(indices)), delta))
        sw = np.sqrt(np.maximum(weights, 1.0e-12))
        aw = design * sw[:, None]
        bw = q * sw[:, None]
        # Mild ridge regularization, excluding intercept.
        ridge = 1.0e-7
        lhs = aw.T @ aw
        lhs[1:, 1:] += ridge * np.eye(7)
        rhs = aw.T @ bw
        try:
            coef = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            return None
        prediction = coef[0]
        if not np.all(np.isfinite(prediction)):
            return None
        return prediction

    @staticmethod
    def _shooting_residual_and_jacobian(
        x: FloatArray,
        u0: float,
        w0: float,
        target5: FloatArray,
        config: QueryConfig,
    ) -> tuple[FloatArray, FloatArray, Optional[FloatArray], float, str, FloatArray]:
        ar0, at0, jr0, ell, log_tau = map(float, x)
        tau = math.exp(log_tau)
        launch = np.array([u0, w0, ar0, at0, jr0, ell, tau], dtype=float)
        yf, sens, normal_constant, status = _integrate_launch_with_sens(
            launch,
            rtol=config.rtol,
            atol=config.atol,
            max_step=config.max_step,
            r_collision=config.r_collision,
            r_escape=config.r_escape,
            max_acceleration=config.max_acceleration,
        )
        if yf is None or sens is None:
            penalty = np.full(5, 1.0e3, dtype=float)
            return penalty, np.eye(5), None, normal_constant, status, launch

        radius, theta, ur, ut, kappa = _endpoint_from_state(yf)
        if radius <= 0.0 or kappa <= 0.0:
            penalty = np.full(5, 1.0e3, dtype=float)
            return penalty, np.eye(5), None, normal_constant, "invalid-endpoint", launch
        raw = np.array(
            [
                math.log(radius) - target5[0],
                theta - target5[1],
                ur - target5[2],
                ut - target5[3],
                math.log(kappa) - target5[4],
            ],
            dtype=float,
        )
        output_jac = _endpoint_output_jacobian(yf)
        jac = np.empty((5, 5), dtype=float)
        jac[:, :4] = output_jac @ sens
        # d y(tau) / d log(tau) = f(yf) * tau.
        jac[:, 4] = output_jac @ (_rhs(tau, yf) * tau)
        scales = np.asarray(config.residual_scale, dtype=float)
        return raw / scales, jac / scales[:, None], yf, normal_constant, status, launch

    @staticmethod
    def _shooting_residual(
        x: FloatArray,
        u0: float,
        w0: float,
        target5: FloatArray,
        config: QueryConfig,
        return_details: bool = False,
    ):
        ar0, at0, jr0, ell, log_tau = map(float, x)
        tau = math.exp(log_tau)
        launch = np.array([u0, w0, ar0, at0, jr0, ell, tau], dtype=float)
        sol, normal_constant, status = _integrate_launch(
            launch,
            rtol=config.rtol,
            atol=config.atol,
            max_step=config.max_step,
            r_collision=config.r_collision,
            r_escape=config.r_escape,
            max_acceleration=config.max_acceleration,
        )
        if sol is None:
            penalty = np.full(5, 1.0e3, dtype=float)
            return (penalty, None, normal_constant, status, penalty.copy(), launch) if return_details else penalty

        radius, theta, ur, ut, kappa = _endpoint_from_state(sol.y[:, -1])
        if radius <= 0.0 or kappa <= 0.0:
            penalty = np.full(5, 1.0e3, dtype=float)
            return (penalty, None, normal_constant, "invalid-endpoint", penalty.copy(), launch) if return_details else penalty
        raw = np.array(
            [
                math.log(radius) - target5[0],
                theta - target5[1],
                ur - target5[2],
                ut - target5[3],
                math.log(kappa) - target5[4],
            ],
            dtype=float,
        )
        scaled = raw / np.asarray(config.residual_scale, dtype=float)
        return (scaled, sol, normal_constant, status, raw, launch) if return_details else scaled

    def _bounds_for_correction(self, config: QueryConfig) -> tuple[FloatArray, FloatArray]:
        lo = self._launch_min.copy()
        hi = self._launch_max.copy()
        span = np.maximum(hi - lo, 1.0e-6)
        lo -= config.launch_bound_margin * span
        hi += config.launch_bound_margin * span
        # Convert the final tau coordinate to log(tau).
        lo[-1] = math.log(max(lo[-1], 1.0e-5))
        hi[-1] = math.log(max(hi[-1], math.exp(lo[-1]) * 1.001))
        return lo, hi

    def solve(
        self,
        initial: CartesianState,
        final: CartesianState,
        capability: RocketCapability,
        *,
        revolutions: int = 0,
        config: Optional[QueryConfig] = None,
        return_all: bool = False,
    ) -> ShootingSolution | list[ShootingSolution]:
        """Retrieve and exactly correct PMP extremals for a boundary query.

        ``revolutions`` selects the unwrapped final angle by adding 2*pi*N to
        the principal geometric angle.  Try several values to compare winding
        branches.
        """
        cfg = config or QueryConfig()
        boundary = _canonicalize_boundary(initial, final, capability, revolutions)
        target = boundary.target
        idx = self._nearest_indices(target, cfg.neighbours)

        seed_vectors: list[FloatArray] = []
        # Local inverse-chart prediction.
        reg_idx = idx[: min(cfg.regression_neighbours, len(idx))]
        reg = self._local_regression_seed(target, reg_idx)
        if reg is not None:
            seed_vectors.append(reg)

        # Several direct branch seeds. Convert tau to log(tau).
        for i in idx[: min(cfg.direct_seeds, len(idx))]:
            q = np.concatenate((self.launch[i, 2:6], [math.log(self.launch[i, 6])]))
            seed_vectors.append(q)

        lo, hi = self._bounds_for_correction(cfg)
        target5 = target[2:7]
        solutions: list[ShootingSolution] = []

        for seed in seed_vectors:
            seed = np.clip(np.asarray(seed, dtype=float), lo + 1.0e-10, hi - 1.0e-10)
            cache_x = None
            cache_value = None

            def evaluate(x):
                nonlocal cache_x, cache_value
                x = np.asarray(x, dtype=float)
                if cache_x is None or not np.array_equal(x, cache_x):
                    cache_x = x.copy()
                    cache_value = self._shooting_residual_and_jacobian(
                        x, target[0], target[1], target5, cfg
                    )
                return cache_value

            def fun(x):
                return evaluate(x)[0]

            def jac(x):
                return evaluate(x)[1]

            result = least_squares(
                fun,
                seed,
                bounds=(lo, hi),
                method="trf",
                jac=jac,
                x_scale="jac",
                max_nfev=cfg.max_nfev,
                ftol=1.0e-12,
                xtol=1.0e-12,
                gtol=1.0e-12,
            )
            scaled, _jac, yf, normal_constant, status, launch = evaluate(result.x)
            if yf is None or status != "ok" or normal_constant <= 0.0:
                continue
            radius, theta, ur, ut, kappa = _endpoint_from_state(yf)
            raw = np.array([
                math.log(radius) - target5[0],
                theta - target5[1],
                ur - target5[2],
                ut - target5[3],
                math.log(kappa) - target5[4],
            ], dtype=float)
            if np.any(np.abs(raw) > np.asarray(cfg.acceptance, dtype=float)):
                continue

            # Reintegrate densely for the returned trajectory and diagnostics.
            t_eval = np.linspace(0.0, launch[6], cfg.trajectory_points)
            dense_sol, normal_constant, status = _integrate_launch(
                launch,
                rtol=cfg.rtol,
                atol=cfg.atol,
                max_step=cfg.max_step,
                r_collision=cfg.r_collision,
                r_escape=cfg.r_escape,
                max_acceleration=cfg.max_acceleration,
                t_eval=t_eval,
            )
            if dense_sol is None or status != "ok":
                continue
            trajectory = _dimensional_trajectory(dense_sol.t, dense_sol.y, boundary, capability)
            radii = np.linalg.norm(dense_sol.y[0:2, :], axis=0)
            acc = np.linalg.norm(dense_sol.y[4:6, :], axis=0)
            solution = ShootingSolution(
                launch=launch,
                residual=raw,
                residual_norm=float(np.linalg.norm(scaled)),
                normal_constant=float(normal_constant),
                minimum_radius=float(np.min(radii)),
                maximum_acceleration=float(np.max(acc)),
                radial_turns=_count_radial_turns(dense_sol.y),
                trajectory=trajectory,
            )

            # Deduplicate in launch/time space.
            duplicate = False
            for old in solutions:
                scale = np.array([1, 1, 1, 1, 1, 1, max(1.0, old.launch[6])], dtype=float)
                if np.linalg.norm((solution.launch - old.launch) / scale) < 2.0e-5:
                    duplicate = True
                    if solution.residual_norm < old.residual_norm:
                        solutions.remove(old)
                        solutions.append(solution)
                    break
            if not duplicate:
                solutions.append(solution)

        if not solutions:
            certificate = self.coverage_certificate(
                initial, final, capability, revolutions
            )
            raise RuntimeError(
                "No branch converged from the atlas seeds. "
                f"Nearest normalized distance={certificate['distance']:.3g}; "
                f"validated cell radius={certificate['validated_radius']:.3g}; "
                f"inside validated cell={certificate['inside_validated_cell']}. "
                "Generate or adapt the atlas further, increase query seeds, or try a "
                "different revolution count."
            )

        solutions.sort(key=lambda s: (s.launch[6], s.residual_norm))
        return solutions if return_all else solutions[0]


# ---------------------------------------------------------------------------
# Dimensional reconstruction
# ---------------------------------------------------------------------------


def _dimensional_trajectory(
    tau: FloatArray,
    y: FloatArray,
    boundary: CanonicalBoundary,
    capability: RocketCapability,
) -> dict[str, FloatArray]:
    r_can = y[0:2, :].T
    v_can = y[2:4, :].T
    a_can = y[4:6, :].T
    j_can = y[6:8, :].T
    s = y[8, :]
    theta = y[9, :]

    rotation = boundary.rotation
    position = (r_can @ rotation.T) * boundary.r0
    velocity = (v_can @ rotation.T) * boundary.v0_scale
    acceleration = (a_can @ rotation.T) * boundary.a0_scale
    jerk_scale = boundary.a0_scale / boundary.t0
    jerk = (j_can @ rotation.T) * jerk_scale
    time = tau * boundary.t0

    frac = np.clip(s / boundary.kappa, 0.0, np.inf)
    inv_mass = 1.0 / capability.initial_mass + boundary.delta_inverse_mass * frac
    mass = 1.0 / inv_mass
    acceleration_magnitude = np.linalg.norm(acceleration, axis=1)
    thrust = mass * acceleration_magnitude
    with np.errstate(divide="ignore", invalid="ignore"):
        exhaust_speed = 2.0 * capability.useful_power / (mass * acceleration_magnitude)
    exhaust_speed[acceleration_magnitude == 0.0] = np.inf

    return {
        "time": np.asarray(time, dtype=float),
        "position": np.asarray(position, dtype=float),
        "velocity": np.asarray(velocity, dtype=float),
        "acceleration": np.asarray(acceleration, dtype=float),
        "jerk": np.asarray(jerk, dtype=float),
        "mass": np.asarray(mass, dtype=float),
        "thrust": np.asarray(thrust, dtype=float),
        "exhaust_speed": np.asarray(exhaust_speed, dtype=float),
        "theta_unwrapped": np.asarray(theta, dtype=float),
        "resource_fraction": np.asarray(frac, dtype=float),
    }


# ---------------------------------------------------------------------------
# Convenience example and command-line interface
# ---------------------------------------------------------------------------


def circular_state(radius: float, mu: float, angle: float = 0.0, prograde: bool = True) -> CartesianState:
    """Create a dimensional circular Kepler state."""
    if radius <= 0.0 or mu <= 0.0:
        raise ValueError("radius and mu must be positive")
    er = np.array([math.cos(angle), math.sin(angle)], dtype=float)
    et = _rot90(er)
    if not prograde:
        et = -et
    speed = math.sqrt(mu / radius)
    return CartesianState(tuple(radius * er), tuple(speed * et))



def validate_atlas_coverage(
    atlas_path: str | Path,
    *,
    launch_attempts: int = 4096,
    validation_rows: int = 512,
    seed: Optional[int] = None,
    workers: Optional[int] = None,
) -> dict:
    """Measure atlas coverage on a fresh independent feasible holdout set."""
    atlas = ExtremalAtlas(atlas_path)
    payload = atlas.metadata.get("config", {})
    config = _atlas_config_from_dict(payload) if payload else AtlasConfig()
    config.n_samples = int(launch_attempts)
    config.seed = int(seed if seed is not None else config.seed + 1_000_003)
    if workers is not None:
        config.workers = int(workers)
    config.coverage.enabled = False
    launches = _sample_launches(config)
    probe_launch, probe_endpoint, probe_diag = _generate_rows_from_launches(launches, config)
    if len(probe_launch) == 0:
        raise RuntimeError("No independent validation extremals survived integration")

    rng = np.random.default_rng(config.seed + 37)
    indices = _select_validation_indices(
        probe_endpoint,
        probe_diag,
        atlas.feature_scale,
        validation_rows,
        rng,
        config.coverage.validation_cell_size,
    )
    query_config = _coverage_query_config(config)
    tree = cKDTree(atlas.endpoint / atlas.feature_scale)
    bounds = _launch_correction_bounds(atlas.launch, query_config.launch_bound_margin)
    success = np.zeros(len(indices), dtype=bool)
    distances = np.full(len(indices), np.inf, dtype=float)
    total_nfev = 0
    for local, index in enumerate(indices):
        corrected, distance, nfev = _fast_correct_target(
            probe_endpoint[index],
            atlas.launch,
            atlas.endpoint,
            atlas.feature_scale,
            query_config,
            tree=tree,
            bounds=bounds,
        )
        success[local] = corrected is not None
        distances[local] = distance
        total_nfev += nfev

    return {
        "atlas": str(Path(atlas_path).resolve()),
        "launch_attempts": int(launch_attempts),
        "accepted_probe_rows": int(len(probe_launch)),
        "tested_probe_rows": int(len(indices)),
        "successes": int(np.count_nonzero(success)),
        "success_rate": float(np.mean(success)) if len(success) else math.nan,
        "median_nearest_distance": float(np.median(distances)) if len(distances) else math.nan,
        "p95_nearest_distance": float(np.quantile(distances, 0.95)) if len(distances) else math.nan,
        "maximum_nearest_distance": float(np.max(distances)) if len(distances) else math.nan,
        "correction_nfev": int(total_nfev),
        "seed": int(config.seed),
    }


def _parse_config(path: Optional[str]) -> AtlasConfig:
    if path is None:
        return AtlasConfig()
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return _atlas_config_from_dict(payload)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser("generate", help="generate an NPZ extremal atlas")
    make.add_argument("output", help="output .npz path")
    make.add_argument("--config", help="JSON file overriding AtlasConfig fields")
    make.add_argument("--samples", type=int, help="override number of Sobol samples")
    make.add_argument("--workers", type=int, help="override worker count")

    inspect = sub.add_parser("inspect", help="show atlas metadata")
    inspect.add_argument("atlas")

    validate = sub.add_parser("validate", help="test coverage on fresh feasible holdouts")
    validate.add_argument("atlas")
    validate.add_argument("--launches", type=int, default=4096)
    validate.add_argument("--rows", type=int, default=512)
    validate.add_argument("--seed", type=int)
    validate.add_argument("--workers", type=int)

    args = parser.parse_args(argv)
    if args.command == "generate":
        cfg = _parse_config(args.config)
        if args.samples is not None:
            cfg.n_samples = args.samples
        if args.workers is not None:
            cfg.workers = args.workers
        output = generate_atlas(args.output, cfg)
        print(output)
        return 0
    if args.command == "inspect":
        atlas = ExtremalAtlas(args.atlas)
        print(json.dumps(atlas.metadata, indent=2))
        return 0
    if args.command == "validate":
        report = validate_atlas_coverage(
            args.atlas,
            launch_attempts=args.launches,
            validation_rows=args.rows,
            seed=args.seed,
            workers=args.workers,
        )
        print(json.dumps(report, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
