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
import time
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
    """Adaptive endpoint-space coverage, targeted slices, and pruning.

    The generator uses three complementary checks:

    * a persistent, spatially diverse feasible holdout bank;
    * a fresh feasible audit batch on every round;
    * optional exact circular-to-circular probes sampled directly in
      ``(log(rho), theta, log(kappa))``.

    Coverage is accepted only when exact shooting correction succeeds and the
    nearest-seed distances are locally small.  Failed feasible holdouts are
    inserted as seeds; successful targeted circular corrections are also
    inserted so lower-dimensional mission families are represented explicitly.
    """

    enabled: bool = True

    # Launch budget allocation.
    initial_fraction: float = 0.25
    persistent_validation_launches: int = 16_384
    batch_launches: int = 8_192

    # Fresh audit size and cumulative feasible validation reservoir.  The
    # reservoir is persistent across rounds; fresh audits remain independent.
    validation_rows: int = 1_024
    validation_reservoir_rows: int = 8_192
    validation_retest_rows_per_round: int = 2_048
    validation_cell_size: float = 0.14
    validation_reservoir_cell_size: float = 0.10
    minimum_launch_fraction_before_stop: float = 0.50

    # Global and local stopping criteria.
    target_success: float = 0.99
    fresh_target_success: float = 0.90
    fresh_maximum_p95_distance: float = 3.0
    target_stratum_success: float = 0.90
    minimum_stratum_rows: int = 6
    minimum_strata_fraction: float = 0.95
    # Geometric distances remain diagnostic by default. The production
    # certificate is success under the bounded query solver; a global
    # Euclidean feature distance is not a reliable convergence radius.
    distance_criteria_enabled: bool = False
    maximum_p95_distance: float = 1.5
    maximum_p99_distance: float = 2.5
    patience: int = 3
    max_rounds: int = 32
    minimum_validation_rows: int = 512

    # Coarse output strata: log(rho), theta, log(kappa).  Branch labels
    # (radial turns and winding) are appended automatically.
    strata_bins: tuple[int, int, int] = (4, 8, 4)

    # Exact circular-to-circular validation family.  These probes are sampled
    # directly in the requested output domain, not from the random forward
    # distribution.  Continuation is attempted when a direct Newton correction
    # fails.  Set enabled=False when this mission family is irrelevant.
    circular_enabled: bool = True
    circular_validation_rows: int = 128
    circular_batch_rows: int = 16
    circular_bootstrap_per_round: int = 4
    circular_target_success: float = 0.90
    circular_distance_criteria_enabled: bool = False
    circular_maximum_p95_distance: float = 2.0
    circular_homotopy_steps: int = 8
    circular_nearest_seeds: int = 4
    circular_max_nfev: int = 20
    circular_insertions_per_round: int = 64
    circular_validation_seed: int = 719_231

    # Production query solver used identically by generation validation and the viewer.
    # ``fast_newton`` is deliberately bounded: if it cannot converge quickly,
    # the probe/pixel is marked failed instead of spending minutes in a robust solve.
    query_method: str = "fast_newton"
    neighbours: int = 24
    direct_seeds: int = 4
    regression_neighbours: int = 16
    max_nfev: int = 35  # retained for robust_least_squares compatibility
    query_max_iterations: int = 5
    query_max_seed_attempts: int = 3
    query_wall_seconds: float = 2.0
    query_line_search_steps: int = 3
    query_step_limit: float = 0.30
    query_regularization: float = 1.0e-8
    query_robust_fallback: bool = False
    query_allow_continuation: bool = False

    # Initial cloud thinning and acquisition-driven insertion.  Distances are
    # measured in endpoint coordinates divided by feature_scale unless a local
    # endpoint Jacobian is available.
    initial_cell_size: float = 0.24
    insertion_cell_size: float = 0.08
    max_atlas_rows: int = 350_000

    acquisition_enabled: bool = True
    acquisition_insertions_per_round: int = 1_536
    acquisition_min_separation: float = 0.07
    acquisition_candidate_limit: int = 30_000
    acquisition_distance_weight: float = 1.0
    acquisition_radius_weight: float = 1.5
    acquisition_failure_weight: float = 4.0
    acquisition_cell_deficit_weight: float = 2.0
    acquisition_branch_weight: float = 0.75
    acquisition_condition_weight: float = 0.35

    # Backward-compatible caps retained as sub-budgets in the acquisition pool.
    exploration_insertions_per_round: int = 512
    insert_failures_per_round: int = 1024

    # Lazy exact endpoint Jacobians.  Missing Jacobians are computed only for
    # seeds used by validation, acquisition, frontier expansion, or querying.
    jacobian_enabled: bool = True
    jacobian_initial_rows: int = 2_048
    jacobian_compute_per_round: int = 2_048
    jacobian_shortlist: int = 64
    jacobian_regularization: float = 1.0e-7
    jacobian_condition_limit: float = 1.0e10

    # Direct expansion from the boundary of empirical convergence cells.
    frontier_enabled: bool = True
    frontier_seeds_per_round: int = 64
    frontier_directions_per_seed: int = 2
    frontier_insertions_per_round: int = 128
    frontier_step_factor: float = 1.25
    frontier_min_step: float = 0.20
    frontier_max_step: float = 1.50
    frontier_max_nfev: int = 30

    # Adaptive endpoint partition used to emphasize poorly covered leaves.
    adaptive_partition_enabled: bool = True
    adaptive_max_depth: int = 5
    adaptive_min_points: int = 24
    adaptive_split_failure_rate: float = 0.12
    adaptive_split_p95_distance: float = 1.5

    # Verified redundancy pruning. A candidate is removed only if its exact
    # endpoint can be recovered from other active seeds.
    prune_enabled: bool = True
    prune_every_rounds: int = 4
    prune_distance: float = 0.06
    prune_max_checks: int = 5000
    prune_max_nfev: int = 30
    prune_probe_limit: int = 24
    prune_protect_condition: float = 1.0e8
    prune_min_branch_rows: int = 3

    # Deterministic feasible validation-row selection.
    validation_seed: int = 918_273

    # Atomic progress checkpoints.  Each checkpoint is a complete, queryable
    # atlas NPZ.  Generation state is included so a later --resume run can
    # continue from the next unused Sobol launch rather than restarting.
    checkpoint_enabled: bool = True
    checkpoint_every_rounds: int = 1
    checkpoint_bootstrap_launches: int = 2_048
    checkpoint_initial: bool = True
    checkpoint_on_interrupt: bool = True
    checkpoint_compressed: bool = True

    def validate(self) -> None:
        if not (0.0 < self.initial_fraction < 1.0):
            raise ValueError("coverage.initial_fraction must lie in (0,1)")
        if self.persistent_validation_launches <= 0 or self.batch_launches <= 0:
            raise ValueError("coverage launch batch sizes must be positive")
        if (
            self.validation_rows <= 0
            or self.validation_reservoir_rows <= 0
            or self.validation_retest_rows_per_round <= 0
            or self.validation_cell_size <= 0.0
            or self.validation_reservoir_cell_size <= 0.0
        ):
            raise ValueError("coverage validation settings must be positive")
        if not (0.0 <= self.minimum_launch_fraction_before_stop <= 1.0):
            raise ValueError("coverage.minimum_launch_fraction_before_stop must lie in [0,1]")
        for name, value in (
            ("target_success", self.target_success),
            ("fresh_target_success", self.fresh_target_success),
            ("target_stratum_success", self.target_stratum_success),
            ("minimum_strata_fraction", self.minimum_strata_fraction),
            ("circular_target_success", self.circular_target_success),
        ):
            if not (0.0 < value <= 1.0):
                raise ValueError(f"coverage.{name} must lie in (0,1]")
        if self.minimum_stratum_rows <= 0 or self.minimum_validation_rows <= 0:
            raise ValueError("coverage minimum sample counts must be positive")
        if (
            self.maximum_p95_distance <= 0.0
            or self.maximum_p99_distance <= 0.0
            or self.fresh_maximum_p95_distance <= 0.0
        ):
            raise ValueError("coverage distance thresholds must be positive")
        if self.maximum_p99_distance < self.maximum_p95_distance:
            raise ValueError("coverage.maximum_p99_distance must be >= maximum_p95_distance")
        if self.patience <= 0 or self.max_rounds <= 0:
            raise ValueError("coverage patience and max_rounds must be positive")
        if len(self.strata_bins) != 3 or any(int(v) <= 0 for v in self.strata_bins):
            raise ValueError("coverage.strata_bins must contain three positive integers")
        if self.circular_validation_rows <= 0 or self.circular_homotopy_steps <= 0:
            raise ValueError("coverage circular validation settings must be positive")
        if self.circular_batch_rows <= 0 or self.circular_bootstrap_per_round < 0:
            raise ValueError("coverage circular batch settings are invalid")
        if self.circular_nearest_seeds <= 0 or self.circular_insertions_per_round < 0:
            raise ValueError("coverage circular insertion settings are invalid")
        if self.circular_max_nfev <= 0:
            raise ValueError("coverage.circular_max_nfev must be positive")
        if self.circular_maximum_p95_distance <= 0.0:
            raise ValueError("coverage.circular_maximum_p95_distance must be positive")
        if self.neighbours <= 0 or self.direct_seeds <= 0:
            raise ValueError("coverage neighbour counts must be positive")
        if self.regression_neighbours <= 0 or self.max_nfev <= 0:
            raise ValueError("coverage correction settings must be positive")
        if self.query_method not in {"fast_newton", "robust_least_squares"}:
            raise ValueError("coverage.query_method must be fast_newton or robust_least_squares")
        if self.query_max_iterations <= 0 or self.query_max_seed_attempts <= 0:
            raise ValueError("coverage fast-query iteration and seed limits must be positive")
        if self.query_wall_seconds <= 0.0 or self.query_line_search_steps <= 0:
            raise ValueError("coverage fast-query time and line-search limits must be positive")
        if self.query_step_limit <= 0.0 or self.query_regularization < 0.0:
            raise ValueError("coverage fast-query step/regularization settings are invalid")
        if self.initial_cell_size <= 0.0 or self.insertion_cell_size <= 0.0:
            raise ValueError("coverage cell sizes must be positive")
        if self.acquisition_insertions_per_round <= 0 or self.acquisition_candidate_limit <= 0:
            raise ValueError("coverage acquisition limits must be positive")
        if self.acquisition_min_separation <= 0.0:
            raise ValueError("coverage.acquisition_min_separation must be positive")
        for name, value in (
            ("acquisition_distance_weight", self.acquisition_distance_weight),
            ("acquisition_radius_weight", self.acquisition_radius_weight),
            ("acquisition_failure_weight", self.acquisition_failure_weight),
            ("acquisition_cell_deficit_weight", self.acquisition_cell_deficit_weight),
            ("acquisition_branch_weight", self.acquisition_branch_weight),
            ("acquisition_condition_weight", self.acquisition_condition_weight),
        ):
            if value < 0.0:
                raise ValueError(f"coverage.{name} cannot be negative")
        if self.exploration_insertions_per_round < 0:
            raise ValueError("coverage.exploration_insertions_per_round cannot be negative")
        if self.insert_failures_per_round <= 0 or self.max_atlas_rows <= 0:
            raise ValueError("coverage insertion limits must be positive")
        if self.jacobian_initial_rows < 0 or self.jacobian_compute_per_round < 0:
            raise ValueError("coverage Jacobian budgets cannot be negative")
        if self.jacobian_shortlist <= 0 or self.jacobian_regularization <= 0.0:
            raise ValueError("coverage Jacobian settings must be positive")
        if self.jacobian_condition_limit <= 1.0:
            raise ValueError("coverage.jacobian_condition_limit must exceed one")
        if self.frontier_seeds_per_round < 0 or self.frontier_directions_per_seed <= 0:
            raise ValueError("coverage frontier counts are invalid")
        if self.frontier_insertions_per_round < 0 or self.frontier_max_nfev <= 0:
            raise ValueError("coverage frontier limits are invalid")
        if not (0.0 < self.frontier_min_step <= self.frontier_max_step):
            raise ValueError("coverage frontier step bounds are invalid")
        if self.frontier_step_factor <= 0.0:
            raise ValueError("coverage.frontier_step_factor must be positive")
        if self.adaptive_max_depth < 0 or self.adaptive_min_points <= 1:
            raise ValueError("coverage adaptive partition settings are invalid")
        if not (0.0 <= self.adaptive_split_failure_rate <= 1.0):
            raise ValueError("coverage.adaptive_split_failure_rate must lie in [0,1]")
        if self.adaptive_split_p95_distance <= 0.0:
            raise ValueError("coverage.adaptive_split_p95_distance must be positive")
        if self.prune_distance <= 0.0 or self.prune_max_checks < 0:
            raise ValueError("invalid coverage pruning settings")
        if self.prune_every_rounds <= 0 or self.prune_probe_limit <= 0:
            raise ValueError("coverage pruning cadence/probe limit must be positive")
        if self.prune_protect_condition <= 1.0 or self.prune_min_branch_rows <= 0:
            raise ValueError("coverage pruning protection settings are invalid")
        if self.checkpoint_every_rounds <= 0 or self.checkpoint_bootstrap_launches <= 0:
            raise ValueError("coverage checkpoint intervals must be positive")


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

    method: str = "fast_newton"
    neighbours: int = 32
    direct_seeds: int = 4
    regression_neighbours: int = 20
    max_nfev: int = 60  # robust_least_squares only
    max_iterations: int = 6
    max_seed_attempts: int = 3
    wall_time_seconds: float = 3.0
    line_search_steps: int = 3
    step_limit: float = 0.30
    regularization: float = 1.0e-8
    robust_fallback: bool = False
    allow_continuation: bool = False
    rtol: float = 8.0e-10
    atol: float = 8.0e-12
    max_step: float = 0.04
    trajectory_points: int = 500
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


class _IntegrationDeadline(RuntimeError):
    """Internal exception used to abort a query integration at its time budget."""


def _deadline_rhs(function, deadline: Optional[float]):
    if deadline is None:
        return function
    calls = 0

    def wrapped(t, y):
        nonlocal calls
        calls += 1
        if calls % 16 == 0 and time.monotonic() >= deadline:
            raise _IntegrationDeadline
        return function(t, y)

    return wrapped


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
    deadline: Optional[float] = None,
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
            _deadline_rhs(_rhs_with_sens, deadline),
            (0.0, tau),
            aug0,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            events=tuple(events),
        )
    except _IntegrationDeadline:
        return None, None, normal_constant, "timeout"
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
    deadline: Optional[float] = None,
) -> tuple[Optional[object], float, str]:
    y0, normal_constant = _initial_vector(launch)
    tau = float(launch[6])
    if tau <= 0.0 or normal_constant <= 0.0 or not np.isfinite(normal_constant):
        return None, normal_constant, "non-normal"

    events = _make_events(r_collision, r_escape, max_acceleration)
    try:
        sol = solve_ivp(
            _deadline_rhs(_rhs, deadline),
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
    except _IntegrationDeadline:
        return None, normal_constant, "timeout"
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


def _configured_correction_bounds(config: AtlasConfig) -> tuple[FloatArray, FloatArray]:
    """Exact configured bounds for (Ar0, At0, Jr0, ell, log(tau))."""
    lo = np.array(
        [
            config.ar0_bounds[0],
            config.at0_bounds[0],
            config.jr0_bounds[0],
            config.ell_bounds[0],
            math.log(config.tau_bounds[0]),
        ],
        dtype=float,
    )
    hi = np.array(
        [
            config.ar0_bounds[1],
            config.at0_bounds[1],
            config.jr0_bounds[1],
            config.ell_bounds[1],
            math.log(config.tau_bounds[1]),
        ],
        dtype=float,
    )
    return lo, hi



def _combined_correction_bounds(
    config: AtlasConfig, launch: FloatArray, margin: float = 0.25
) -> tuple[FloatArray, FloatArray]:
    """Union configured launch bounds with the actual atlas/subarc range."""
    configured_lo, configured_hi = _configured_correction_bounds(config)
    if len(launch) == 0:
        return configured_lo, configured_hi
    q = np.column_stack((launch[:, 2:6], np.log(np.maximum(launch[:, 6], 1.0e-12))))
    observed_lo = np.min(q, axis=0)
    observed_hi = np.max(q, axis=0)
    span = np.maximum(observed_hi - observed_lo, 1.0e-6)
    lo = np.minimum(configured_lo, observed_lo - margin * span)
    hi = np.maximum(configured_hi, observed_hi + margin * span)
    return lo, hi


def _launch_q_scale(config: AtlasConfig) -> FloatArray:
    """Characteristic scales for [Ar0, At0, Jr0, ell, log(tau)]."""
    spans = np.array(
        [
            config.ar0_bounds[1] - config.ar0_bounds[0],
            config.at0_bounds[1] - config.at0_bounds[0],
            config.jr0_bounds[1] - config.jr0_bounds[0],
            config.ell_bounds[1] - config.ell_bounds[0],
            math.log(config.tau_bounds[1] / config.tau_bounds[0]),
        ],
        dtype=float,
    )
    return np.maximum(0.25 * spans, np.array([0.25, 0.25, 0.5, 0.5, 0.25]))


def _endpoint_jacobian_for_launch(
    launch: FloatArray, config: AtlasConfig
) -> tuple[FloatArray, float, float, bool]:
    """Return raw d(logR,theta,ur,ut,logkappa)/d(Ar,At,Jr,ell,logtau)."""
    yf, sens, normal_constant, status = _integrate_launch_with_sens(
        np.asarray(launch, dtype=float),
        rtol=config.rtol,
        atol=config.atol,
        max_step=config.max_step,
        r_collision=config.r_collision,
        r_escape=config.r_escape,
        max_acceleration=config.max_acceleration,
    )
    if yf is None or sens is None or status != "ok" or normal_constant <= 0.0:
        return np.full((5, 5), np.nan), math.inf, 0.0, False
    try:
        output_jac = _endpoint_output_jacobian(yf)
        jac = np.empty((5, 5), dtype=float)
        jac[:, :4] = output_jac @ sens
        jac[:, 4] = output_jac @ (_rhs(float(launch[6]), yf) * float(launch[6]))
        singular = np.linalg.svd(jac, compute_uv=False)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return np.full((5, 5), np.nan), math.inf, 0.0, False
    if not np.all(np.isfinite(jac)) or singular[-1] <= 0.0:
        return np.full((5, 5), np.nan), math.inf, 0.0, False
    condition = float(singular[0] / singular[-1])
    return jac, condition, float(singular[-1]), True


def _worker_jacobian_batch(args):
    indices, launch_batch, cfg_dict = args
    config = _atlas_config_from_dict(cfg_dict)
    output = []
    for index, launch in zip(indices, launch_batch):
        jac, condition, sigma_min, valid = _endpoint_jacobian_for_launch(launch, config)
        output.append((int(index), jac, condition, sigma_min, bool(valid)))
    return output


def _empty_jacobian_arrays(count: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    return (
        np.full((count, 5, 5), np.nan, dtype=np.float32),
        np.full(count, np.inf, dtype=np.float64),
        np.zeros(count, dtype=np.float64),
    )


def _ensure_seed_jacobians(
    launch: FloatArray,
    jacobian: FloatArray,
    condition: FloatArray,
    sigma_min: FloatArray,
    indices: ArrayLike,
    config: AtlasConfig,
) -> int:
    """Compute missing exact endpoint Jacobians for selected seed rows in place."""
    if not config.coverage.jacobian_enabled or len(launch) == 0:
        return 0
    selected = np.unique(np.asarray(indices, dtype=int))
    selected = selected[(selected >= 0) & (selected < len(launch))]
    if len(selected) == 0:
        return 0
    missing = selected[~np.all(np.isfinite(jacobian[selected]), axis=(1, 2))]
    if len(missing) == 0:
        return 0
    cfg_dict = asdict(config)
    computed = 0
    if config.workers == 1 or len(missing) < 16:
        batches = [(missing, launch[missing])]
        results = [_worker_jacobian_batch((idx, rows, cfg_dict)) for idx, rows in batches]
    else:
        batch_size = max(8, min(128, len(missing) // (config.workers * 4) or 8))
        chunks = [missing[i : i + batch_size] for i in range(0, len(missing), batch_size)]
        results = []
        with ProcessPoolExecutor(max_workers=config.workers) as pool:
            futures = [
                pool.submit(_worker_jacobian_batch, (chunk, launch[chunk], cfg_dict))
                for chunk in chunks
            ]
            for future in as_completed(futures):
                results.append(future.result())
    for batch in results:
        for index, jac, cond, smin, valid in batch:
            if valid:
                jacobian[index] = np.asarray(jac, dtype=np.float32)
                condition[index] = float(cond)
                sigma_min[index] = float(smin)
            else:
                # A finite all-zero marker prevents repeated failed integration.
                jacobian[index] = np.zeros((5, 5), dtype=np.float32)
                condition[index] = math.inf
                sigma_min[index] = 0.0
            computed += 1
    return computed


def _regularized_inverse(jac: FloatArray, regularization: float) -> Optional[FloatArray]:
    try:
        u, singular, vt = np.linalg.svd(np.asarray(jac, dtype=float), full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(singular)) or singular[0] <= 0.0:
        return None
    cutoff = regularization * singular[0]
    inverse_s = singular / (singular * singular + cutoff * cutoff)
    result = (vt.T * inverse_s) @ u.T
    return result if np.all(np.isfinite(result)) else None


def _jacobian_seed_distance(
    target: FloatArray,
    seed_index: int,
    endpoint: FloatArray,
    feature_scale: FloatArray,
    jacobian: Optional[FloatArray],
    condition: Optional[FloatArray],
    q_scale: Optional[FloatArray],
    coverage: Optional[CoverageConfig],
) -> tuple[float, Optional[FloatArray]]:
    """Estimate correction effort and optional q prediction from one chart."""
    euclidean = float(np.linalg.norm((target - endpoint[seed_index]) / feature_scale))
    if (
        jacobian is None
        or condition is None
        or q_scale is None
        or coverage is None
        or seed_index >= len(jacobian)
        or not np.all(np.isfinite(jacobian[seed_index]))
        or not np.isfinite(condition[seed_index])
        or condition[seed_index] > coverage.jacobian_condition_limit
    ):
        return euclidean, None
    inverse = _regularized_inverse(jacobian[seed_index], coverage.jacobian_regularization)
    if inverse is None:
        return euclidean, None
    delta_q = inverse @ (target[2:7] - endpoint[seed_index, 2:7])
    initial_velocity_distance = np.linalg.norm(
        (target[:2] - endpoint[seed_index, :2]) / feature_scale[:2]
    )
    effort = float(math.sqrt(initial_velocity_distance**2 + np.sum((delta_q / q_scale) ** 2)))
    # A local inverse chart is a predictor, not a proof.  Use it when it offers a
    # shorter correction estimate than the global normalized endpoint metric;
    # otherwise retain the robust Euclidean distance.
    return max(min(euclidean, effort), 1.0e-12), delta_q


def _rank_seed_indices(
    target: FloatArray,
    launch: FloatArray,
    endpoint: FloatArray,
    feature_scale: FloatArray,
    tree: cKDTree,
    count: int,
    *,
    jacobian: Optional[FloatArray] = None,
    condition: Optional[FloatArray] = None,
    q_scale: Optional[FloatArray] = None,
    coverage: Optional[CoverageConfig] = None,
    excluded: Optional[set[int]] = None,
) -> tuple[FloatArray, FloatArray, dict[int, Optional[FloatArray]]]:
    excluded = excluded or set()
    shortlist = min(
        len(launch),
        max(count + len(excluded) + 4, coverage.jacobian_shortlist if coverage else count),
    )
    _distance, indices = tree.query(target / feature_scale, k=max(shortlist, 1))
    indices = np.atleast_1d(indices).astype(int)
    ranked: list[tuple[float, int]] = []
    predictions: dict[int, Optional[FloatArray]] = {}
    for index in indices:
        index = int(index)
        if index in excluded:
            continue
        distance, delta_q = _jacobian_seed_distance(
            target, index, endpoint, feature_scale, jacobian, condition, q_scale, coverage
        )
        ranked.append((distance, index))
        predictions[index] = delta_q
    ranked.sort(key=lambda item: item[0])
    ranked = ranked[:count]
    return (
        np.asarray([i for _d, i in ranked], dtype=int),
        np.asarray([d for d, _i in ranked], dtype=float),
        predictions,
    )



def _accepted_correction_details(
    details,
    target5: FloatArray,
    config: QueryConfig,
) -> tuple[bool, float, Optional[FloatArray]]:
    """Check one shooting evaluation against the unscaled endpoint tolerances."""
    scaled, _jac, yf, normal_constant, status, launch = details
    if yf is None or status != "ok" or normal_constant <= 0.0:
        return False, math.inf, None
    radius, theta, ur, ut, kappa = _endpoint_from_state(yf)
    if radius <= 0.0 or kappa <= 0.0:
        return False, math.inf, None
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
    accepted = bool(np.all(np.abs(raw) <= np.asarray(config.acceptance, dtype=float)))
    return accepted, float(np.linalg.norm(scaled)), launch.copy()


def _bounded_fast_newton(
    seed: FloatArray,
    lo: FloatArray,
    hi: FloatArray,
    target5: FloatArray,
    config: QueryConfig,
    evaluate_with_jac,
    evaluate_residual,
    *,
    deadline: Optional[float] = None,
) -> tuple[Optional[FloatArray], int, str]:
    """Small-budget damped Newton corrector.

    The method uses one exact variational Jacobian per accepted Newton iterate
    and cheaper state-only integrations for line search. It is intentionally
    fail-fast: the atlas is useful only when a nearby chart converges within a
    small, predictable budget.
    """
    x = np.clip(np.asarray(seed, dtype=float), lo + 1.0e-10, hi - 1.0e-10)
    span = np.maximum(hi - lo, 1.0e-8)
    evaluations = 0
    best_x = x.copy()
    best_norm = math.inf

    for _iteration in range(config.max_iterations + 1):
        if deadline is not None and time.monotonic() >= deadline:
            return None, evaluations, "timeout"
        details = evaluate_with_jac(x)
        evaluations += 1
        accepted, norm, launch = _accepted_correction_details(details, target5, config)
        if norm < best_norm:
            best_norm = norm
            best_x = x.copy()
        if accepted and launch is not None:
            return launch, evaluations, "converged"

        scaled, jac, yf, normal_constant, status, _launch = details
        if yf is None or status != "ok" or normal_constant <= 0.0:
            return None, evaluations, status
        if not np.all(np.isfinite(jac)) or not np.all(np.isfinite(scaled)):
            return None, evaluations, "nonfinite"

        # Tikhonov-regularized SVD Newton step.
        try:
            u, singular, vt = np.linalg.svd(jac, full_matrices=False)
        except np.linalg.LinAlgError:
            return None, evaluations, "svd-failed"
        reg = max(float(config.regularization), 0.0)
        inverse = singular / (singular * singular + reg)
        step = -(vt.T * inverse) @ (u.T @ scaled)
        if not np.all(np.isfinite(step)):
            return None, evaluations, "nonfinite-step"

        normalized_length = float(np.linalg.norm(step / span))
        if normalized_length > config.step_limit:
            step *= config.step_limit / normalized_length

        current_norm = float(np.linalg.norm(scaled))
        improved = False
        for line in range(config.line_search_steps):
            if deadline is not None and time.monotonic() >= deadline:
                return None, evaluations, "timeout"
            alpha = 0.5 ** line
            trial = np.clip(x + alpha * step, lo + 1.0e-10, hi - 1.0e-10)
            residual_details = evaluate_residual(trial)
            evaluations += 1
            trial_scaled, trial_state, trial_normal, trial_status, trial_raw, trial_launch = residual_details
            if (
                trial_state is not None
                and trial_status == "ok"
                and trial_normal > 0.0
                and np.all(np.isfinite(trial_scaled))
            ):
                trial_norm = float(np.linalg.norm(trial_scaled))
                if np.all(np.abs(trial_raw) <= np.asarray(config.acceptance, dtype=float)):
                    return trial_launch.copy(), evaluations, "converged"
                if trial_norm < current_norm * (1.0 - 1.0e-4 * alpha):
                    x = trial
                    improved = True
                    break
        if not improved:
            return None, evaluations, "no-descent"

    return None, evaluations, "iteration-limit"


def _correct_one_seed(
    seed: FloatArray,
    lo: FloatArray,
    hi: FloatArray,
    u0: float,
    w0: float,
    target5: FloatArray,
    config: QueryConfig,
    *,
    deadline: Optional[float] = None,
) -> tuple[Optional[FloatArray], int, str]:
    """Run the configured production query solver for one launch seed."""
    seed = np.clip(np.asarray(seed, dtype=float), lo + 1.0e-10, hi - 1.0e-10)

    def with_jac(x):
        return ExtremalAtlas._shooting_residual_and_jacobian(
            np.asarray(x, dtype=float), float(u0), float(w0), target5, config,
            deadline=deadline,
        )

    def residual_only(x):
        return ExtremalAtlas._shooting_residual(
            np.asarray(x, dtype=float), float(u0), float(w0), target5, config,
            return_details=True, deadline=deadline,
        )

    if config.method == "fast_newton":
        launch, nfev, reason = _bounded_fast_newton(
            seed, lo, hi, target5, config, with_jac, residual_only, deadline=deadline
        )
        if launch is not None or not config.robust_fallback:
            return launch, nfev, reason
    elif config.method != "robust_least_squares":
        raise ValueError(f"unsupported query method {config.method!r}")

    if deadline is not None and time.monotonic() >= deadline:
        return None, 0, "timeout"
    cache_x: Optional[FloatArray] = None
    cache_value = None

    def evaluate(x):
        nonlocal cache_x, cache_value
        x = np.asarray(x, dtype=float)
        if cache_x is None or not np.array_equal(x, cache_x):
            cache_x = x.copy()
            cache_value = with_jac(x)
        return cache_value

    result = least_squares(
        lambda x: evaluate(x)[0],
        seed,
        bounds=(lo, hi),
        method="trf",
        jac=lambda x: evaluate(x)[1],
        x_scale="jac",
        max_nfev=config.max_nfev,
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    details = evaluate(result.x)
    accepted, _norm, launch = _accepted_correction_details(details, target5, config)
    return (launch if accepted else None), int(result.nfev), ("converged" if accepted else "robust-failed")

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
    seed_jacobian: Optional[FloatArray] = None,
    seed_condition: Optional[FloatArray] = None,
    q_scale: Optional[FloatArray] = None,
    coverage_config: Optional[CoverageConfig] = None,
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
    nearest, ranked_distances, predictions = _rank_seed_indices(
        target,
        launch,
        endpoint,
        feature_scale,
        tree,
        config.neighbours,
        jacobian=seed_jacobian,
        condition=seed_condition,
        q_scale=q_scale,
        coverage=coverage_config,
        excluded=excluded,
    )
    if len(nearest) == 0:
        return None, math.inf, 0
    nearest_distance = float(ranked_distances[0])

    seeds: list[FloatArray] = []
    # Local inverse-Jacobian predictions are the most physically meaningful
    # initial guesses when the chart is regular.
    for index in nearest[: min(config.direct_seeds, len(nearest))]:
        delta_q = predictions.get(int(index))
        if delta_q is None:
            continue
        base_q = np.concatenate((launch[index, 2:6], [math.log(launch[index, 6])]))
        seeds.append(base_q + delta_q)

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
    deadline = time.monotonic() + max(config.wall_time_seconds, 1.0e-3)

    # Keep the production cost predictable: try only a small number of the
    # highest-quality chart/regression/direct seeds.
    unique_seeds: list[FloatArray] = []
    for seed in seeds:
        seed = np.asarray(seed, dtype=float)
        if not any(np.linalg.norm(seed - old) < 1.0e-10 for old in unique_seeds):
            unique_seeds.append(seed)
        if len(unique_seeds) >= config.max_seed_attempts:
            break

    for seed in unique_seeds:
        if time.monotonic() >= deadline:
            break
        corrected, nfev, _reason = _correct_one_seed(
            seed, lo, hi, float(target[0]), float(target[1]), target5, config,
            deadline=deadline,
        )
        total_nfev += int(nfev)
        if corrected is not None:
            best = corrected.copy()
            break
    return best, nearest_distance, total_nfev


def _coverage_query_config(
    config: AtlasConfig, *, pruning: bool = False, circular: bool = False
) -> QueryConfig:
    """Use the same bounded production query policy for validation and viewing."""
    coverage = config.coverage
    max_nfev = (
        coverage.prune_max_nfev
        if pruning
        else (coverage.circular_max_nfev if circular else coverage.max_nfev)
    )
    return QueryConfig(
        method=coverage.query_method,
        neighbours=coverage.neighbours,
        direct_seeds=coverage.direct_seeds,
        regression_neighbours=coverage.regression_neighbours,
        max_nfev=max_nfev,
        max_iterations=coverage.query_max_iterations,
        max_seed_attempts=coverage.query_max_seed_attempts,
        wall_time_seconds=coverage.query_wall_seconds,
        line_search_steps=coverage.query_line_search_steps,
        step_limit=coverage.query_step_limit,
        regularization=coverage.query_regularization,
        robust_fallback=coverage.query_robust_fallback,
        allow_continuation=coverage.query_allow_continuation,
        rtol=max(config.rtol, 8.0e-10),
        atol=max(config.atol, 8.0e-12),
        max_step=max(config.max_step, 0.04),
        trajectory_points=max(64, config.diagnostic_points),
        r_collision=config.r_collision,
        r_escape=config.r_escape,
        max_acceleration=config.max_acceleration,
        residual_scale=(0.03, 0.08, 0.12, 0.12, 0.08),
        acceptance=(1.0e-5, 1.5e-5, 1.5e-5, 1.5e-5, 1.5e-5),
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



def _merge_validation_reservoir(
    reservoir_launch: FloatArray,
    reservoir_endpoint: FloatArray,
    reservoir_diag: FloatArray,
    reservoir_tested: NDArray[np.bool_],
    reservoir_success: NDArray[np.bool_],
    reservoir_distance: FloatArray,
    new_launch: FloatArray,
    new_endpoint: FloatArray,
    new_diag: FloatArray,
    new_tested: NDArray[np.bool_],
    new_success: NDArray[np.bool_],
    new_distance: FloatArray,
    feature_scale: FloatArray,
    config: AtlasConfig,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    NDArray[np.bool_],
    NDArray[np.bool_],
    FloatArray,
]:
    """Maintain a bounded, spatially diverse cumulative feasible reservoir."""
    if len(new_launch) == 0:
        return (
            reservoir_launch,
            reservoir_endpoint,
            reservoir_diag,
            reservoir_tested,
            reservoir_success,
            reservoir_distance,
        )
    launch = np.vstack((reservoir_launch, new_launch)) if len(reservoir_launch) else new_launch.copy()
    endpoint = (
        np.vstack((reservoir_endpoint, new_endpoint)) if len(reservoir_endpoint) else new_endpoint.copy()
    )
    diag = np.vstack((reservoir_diag, new_diag)) if len(reservoir_diag) else new_diag.copy()
    tested = (
        np.concatenate((reservoir_tested, new_tested)) if len(reservoir_tested) else new_tested.copy()
    )
    success = (
        np.concatenate((reservoir_success, new_success)) if len(reservoir_success) else new_success.copy()
    )
    distance = (
        np.concatenate((reservoir_distance, new_distance)) if len(reservoir_distance) else new_distance.copy()
    )
    limit = min(config.coverage.validation_reservoir_rows, len(launch))
    if len(launch) <= limit:
        return launch, endpoint, diag, tested, success, distance

    labels = _branch_labels(diag)
    cell_size = config.coverage.validation_reservoir_cell_size
    inv = 1.0 / cell_size
    # Failed and distant probes are retained preferentially.  One representative
    # per endpoint/branch cell is chosen first, then the remaining budget is
    # filled by the same priority score.
    priority = (
        (~tested).astype(float) * 2.0e6
        + (tested & ~success).astype(float) * 1.0e6
        + np.nan_to_num(distance, nan=1.0e3, posinf=1.0e3)
    )
    order = np.argsort(priority)[::-1]
    occupied: set[tuple[int, ...]] = set()
    keep: list[int] = []
    deferred: list[int] = []
    for index in order:
        cell = tuple(np.floor(endpoint[index] / feature_scale * inv).astype(np.int64)) + tuple(
            labels[index]
        )
        if cell in occupied:
            deferred.append(int(index))
            continue
        occupied.add(cell)
        keep.append(int(index))
        if len(keep) >= limit:
            break
    if len(keep) < limit:
        selected = set(keep)
        for index in deferred:
            if index in selected:
                continue
            keep.append(index)
            if len(keep) >= limit:
                break
    idx = np.asarray(keep[:limit], dtype=int)
    return launch[idx], endpoint[idx], diag[idx], tested[idx], success[idx], distance[idx]


@dataclass
class _AdaptiveCellNode:
    deficit: float
    count: int
    success_rate: float
    p95_distance: float
    split_dim: int = -1
    split_value: float = 0.0
    left: Optional["_AdaptiveCellNode"] = None
    right: Optional["_AdaptiveCellNode"] = None

    def score(self, point: FloatArray) -> float:
        if self.split_dim < 0 or self.left is None or self.right is None:
            return self.deficit
        child = self.left if point[self.split_dim] <= self.split_value else self.right
        return child.score(point)


def _build_adaptive_partition(
    endpoint: FloatArray,
    success: NDArray[np.bool_],
    distance: FloatArray,
    feature_scale: FloatArray,
    config: AtlasConfig,
) -> Optional[_AdaptiveCellNode]:
    if not config.coverage.adaptive_partition_enabled or len(endpoint) == 0:
        return None
    features = endpoint / feature_scale
    coverage = config.coverage

    def build(indices: FloatArray, depth: int) -> _AdaptiveCellNode:
        values = success[indices]
        distances = distance[indices]
        rate = float(np.mean(values)) if len(values) else 0.0
        p95 = float(np.quantile(distances, 0.95)) if len(distances) else math.inf
        deficit = (
            (1.0 - rate)
            + min(4.0, p95 / max(coverage.maximum_p95_distance, 1.0e-12))
            + 1.0 / math.sqrt(max(len(indices), 1))
        )
        node = _AdaptiveCellNode(deficit, int(len(indices)), rate, p95)
        should_split = (
            depth < coverage.adaptive_max_depth
            and len(indices) >= 2 * coverage.adaptive_min_points
            and (
                (1.0 - rate) >= coverage.adaptive_split_failure_rate
                or p95 >= coverage.adaptive_split_p95_distance
            )
        )
        if not should_split:
            return node
        local = features[indices]
        spreads = np.ptp(local, axis=0)
        dim = int(np.argmax(spreads))
        if spreads[dim] <= 1.0e-12:
            return node
        split = float(np.median(local[:, dim]))
        left_mask = local[:, dim] <= split
        if np.all(left_mask) or not np.any(left_mask):
            return node
        node.split_dim = dim
        node.split_value = split
        node.left = build(indices[left_mask], depth + 1)
        node.right = build(indices[~left_mask], depth + 1)
        return node

    return build(np.arange(len(endpoint), dtype=int), 0)


def _branch_frequency(diagnostics: FloatArray) -> dict[tuple[int, int], int]:
    frequency: dict[tuple[int, int], int] = {}
    for label in _branch_labels(diagnostics):
        key = (int(label[0]), int(label[1]))
        frequency[key] = frequency.get(key, 0) + 1
    return frequency


def _acquisition_select_indices(
    candidate_endpoint: FloatArray,
    candidate_diag: FloatArray,
    failed: NDArray[np.bool_],
    atlas_endpoint: FloatArray,
    atlas_diag: FloatArray,
    feature_scale: FloatArray,
    coverage_radius: FloatArray,
    jacobian_condition: FloatArray,
    adaptive_partition: Optional[_AdaptiveCellNode],
    config: AtlasConfig,
    limit: int,
) -> tuple[FloatArray, FloatArray]:
    """Score and greedily diversify candidate insertions."""
    count = len(candidate_endpoint)
    if count == 0 or limit <= 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)
    coverage = config.coverage
    tree = cKDTree(atlas_endpoint / feature_scale)
    distance, nearest = tree.query(candidate_endpoint / feature_scale, k=1)
    distance = np.asarray(distance, dtype=float)
    nearest = np.asarray(nearest, dtype=int)
    radius = np.maximum(coverage_radius[nearest], 0.25 * coverage.insertion_cell_size)
    radius_ratio = distance / radius

    branch_frequency = _branch_frequency(atlas_diag)
    labels = _branch_labels(candidate_diag)
    branch_score = np.array(
        [1.0 / math.sqrt(branch_frequency.get((int(a), int(b)), 0) + 1.0) for a, b in labels],
        dtype=float,
    )
    if adaptive_partition is None:
        cell_deficit = np.ones(count, dtype=float)
    else:
        cell_deficit = np.array(
            [adaptive_partition.score(row / feature_scale) for row in candidate_endpoint],
            dtype=float,
        )
    nearest_condition = jacobian_condition[nearest]
    condition_score = np.where(
        np.isfinite(nearest_condition),
        np.clip(np.log10(np.maximum(nearest_condition, 1.0)) / 10.0, 0.0, 2.0),
        1.0,
    )
    score = (
        coverage.acquisition_distance_weight * distance
        + coverage.acquisition_radius_weight * np.clip(radius_ratio, 0.0, 20.0)
        + coverage.acquisition_failure_weight * failed.astype(float)
        + coverage.acquisition_cell_deficit_weight * cell_deficit
        + coverage.acquisition_branch_weight * branch_score
        + coverage.acquisition_condition_weight * condition_score
    )

    # Bound sorting/memory cost while always retaining failures and the farthest
    # candidates.  This pool reduction itself is deterministic.
    if count > coverage.acquisition_candidate_limit:
        mandatory = np.flatnonzero(failed)
        remaining_budget = max(0, coverage.acquisition_candidate_limit - len(mandatory))
        non_failed = np.flatnonzero(~failed)
        high = non_failed[np.argsort(score[non_failed])[::-1][:remaining_budget]]
        pool = np.unique(np.concatenate((mandatory, high)))
    else:
        pool = np.arange(count, dtype=int)
    order = pool[np.argsort(score[pool])[::-1]]

    minimum = coverage.acquisition_min_separation
    inv = 1.0 / minimum
    selected: list[int] = []
    occupied: set[tuple[int, ...]] = set()
    for index in order:
        # Failed known-feasible probes may be inserted even when geometrically
        # close, but only once per branch-aware fine cell.
        cell = tuple(np.floor(candidate_endpoint[index] / feature_scale * inv).astype(np.int64)) + tuple(
            labels[index]
        )
        if cell in occupied:
            continue
        if not failed[index] and distance[index] < 0.35 * minimum:
            continue
        occupied.add(cell)
        selected.append(int(index))
        if len(selected) >= limit:
            break
    idx = np.asarray(selected, dtype=int)
    return idx, score[idx]


def _clip_endpoint_target(target: FloatArray, config: AtlasConfig) -> FloatArray:
    result = np.asarray(target, dtype=float).copy()
    result[0] = np.clip(result[0], *config.u0_bounds)
    result[1] = np.clip(result[1], *config.w0_bounds)
    result[2] = np.clip(result[2], math.log(config.rho_bounds[0]), math.log(config.rho_bounds[1]))
    result[3] = np.clip(result[3], *config.theta_bounds)
    result[4] = np.clip(result[4], -config.final_speed_max, config.final_speed_max)
    result[5] = np.clip(result[5], -config.final_speed_max, config.final_speed_max)
    result[6] = np.clip(
        result[6], math.log(config.kappa_bounds[0]), math.log(config.kappa_bounds[1])
    )
    return result


def _frontier_expand(
    atlas_launch: FloatArray,
    atlas_endpoint: FloatArray,
    atlas_diag: FloatArray,
    atlas_jacobian: FloatArray,
    atlas_condition: FloatArray,
    coverage_radius: FloatArray,
    failed_probe_endpoint: FloatArray,
    feature_scale: FloatArray,
    config: AtlasConfig,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray, FloatArray, int]:
    """Expand local inverse charts toward uncovered probes and free directions."""
    coverage = config.coverage
    if (
        not coverage.frontier_enabled
        or coverage.frontier_insertions_per_round <= 0
        or len(atlas_launch) == 0
    ):
        return (
            np.empty((0, 7), dtype=float),
            np.empty((0, 7), dtype=float),
            np.empty((0, 6), dtype=float),
            0,
        )
    tree = cKDTree(atlas_endpoint / feature_scale)
    seed_indices: list[int] = []
    seed_targets: dict[int, list[FloatArray]] = {}
    if len(failed_probe_endpoint):
        _dist, nearest = tree.query(failed_probe_endpoint / feature_scale, k=1)
        for target, seed in zip(failed_probe_endpoint, np.asarray(nearest, dtype=int)):
            seed = int(seed)
            seed_targets.setdefault(seed, []).append(target)
        # Prioritize seeds facing several uncovered probes and seeds with small cells.
        seed_indices.extend(
            sorted(
                seed_targets,
                key=lambda i: (-len(seed_targets[i]), coverage_radius[i]),
            )
        )
    if len(seed_indices) < coverage.frontier_seeds_per_round:
        finite = np.flatnonzero(
            np.all(np.isfinite(atlas_jacobian), axis=(1, 2))
            & np.isfinite(atlas_condition)
            & (atlas_condition <= coverage.jacobian_condition_limit)
        )
        if len(finite):
            order = finite[np.argsort(coverage_radius[finite])]
            seed_indices.extend(map(int, order))
    # Stable unique ordering.
    seed_indices = list(dict.fromkeys(seed_indices))[: coverage.frontier_seeds_per_round]
    bounds = _combined_correction_bounds(config, atlas_launch, 0.25)
    query = _coverage_query_config(config)
    query.max_nfev = coverage.frontier_max_nfev
    q_scale = _launch_q_scale(config)

    output_launch: list[FloatArray] = []
    output_endpoint: list[FloatArray] = []
    output_diag: list[FloatArray] = []
    attempts = 0
    for seed in seed_indices:
        jac = atlas_jacobian[seed]
        inverse = None
        if (
            np.all(np.isfinite(jac))
            and np.isfinite(atlas_condition[seed])
            and atlas_condition[seed] <= coverage.jacobian_condition_limit
        ):
            inverse = _regularized_inverse(jac, coverage.jacobian_regularization)
        source = atlas_endpoint[seed]
        directions: list[FloatArray] = []
        for target in seed_targets.get(seed, []):
            delta = (target - source) / feature_scale
            norm = float(np.linalg.norm(delta))
            if norm > 1.0e-10:
                directions.append(delta / norm)
        while len(directions) < coverage.frontier_directions_per_seed:
            direction = rng.normal(size=7)
            norm = float(np.linalg.norm(direction))
            if norm > 1.0e-12:
                directions.append(direction / norm)
        for direction in directions[: coverage.frontier_directions_per_seed]:
            cell_radius = max(float(coverage_radius[seed]), coverage.frontier_min_step)
            step = np.clip(
                coverage.frontier_step_factor * cell_radius,
                coverage.frontier_min_step,
                coverage.frontier_max_step,
            )
            target = _clip_endpoint_target(source + direction * step * feature_scale, config)
            base_q = np.concatenate((atlas_launch[seed, 2:6], [math.log(atlas_launch[seed, 6])]))
            if inverse is not None:
                base_q = base_q + inverse @ (target[2:7] - source[2:7])
            predicted = np.array(
                [target[0], target[1], base_q[0], base_q[1], base_q[2], base_q[3], math.exp(base_q[4])],
                dtype=float,
            )
            predicted[2:6] = np.clip(predicted[2:6], bounds[0][:4], bounds[1][:4])
            predicted[6] = float(np.clip(predicted[6], config.tau_bounds[0], config.tau_bounds[1]))
            corrected, _nfev = _correct_target_from_explicit_seed(target, predicted, query, bounds)
            attempts += 1
            if corrected is None:
                continue
            diag = _diagnostics_for_exact_launch(corrected, config)
            if diag is None:
                continue
            output_launch.append(corrected)
            output_endpoint.append(target)
            output_diag.append(diag)
            if len(output_launch) >= coverage.frontier_insertions_per_round:
                break
        if len(output_launch) >= coverage.frontier_insertions_per_round:
            break
    if not output_launch:
        return (
            np.empty((0, 7), dtype=float),
            np.empty((0, 7), dtype=float),
            np.empty((0, 6), dtype=float),
            attempts,
        )
    return np.vstack(output_launch), np.vstack(output_endpoint), np.vstack(output_diag), attempts


def _prune_redundant_rows(
    launch: FloatArray,
    endpoint: FloatArray,
    diagnostics: FloatArray,
    feature_scale: FloatArray,
    config: AtlasConfig,
    *,
    jacobian: Optional[FloatArray] = None,
    condition: Optional[FloatArray] = None,
    sigma_min: Optional[FloatArray] = None,
    probe_endpoint: Optional[FloatArray] = None,
    probe_success: Optional[NDArray[np.bool_]] = None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, int]:
    """Conservative branch-aware coverage-dominance pruning.

    A close seed is removed only when its own endpoint and every successful
    validation probe assigned to it remain exactly recoverable from the active
    retained set.  Rare branches and ill-conditioned/fold-adjacent charts are
    protected.
    """
    coverage = config.coverage
    if jacobian is None or condition is None or sigma_min is None:
        jacobian, condition, sigma_min = _empty_jacobian_arrays(len(launch))
    if not coverage.prune_enabled or coverage.prune_max_checks == 0 or len(launch) < 3:
        return launch, endpoint, diagnostics, jacobian, condition, sigma_min, 0

    features = endpoint / feature_scale
    tree = cKDTree(features)
    distances, neighbours = tree.query(features, k=2)
    labels = _branch_labels(diagnostics)
    branch_counts = _branch_frequency(diagnostics)
    candidates = np.argsort(distances[:, 1])
    active = np.ones(len(launch), dtype=bool)
    protected: set[int] = set()
    removed = 0
    checks = 0
    query_config = _coverage_query_config(config, pruning=True)
    correction_bounds = _combined_correction_bounds(config, launch, 0.25)
    q_scale = _launch_q_scale(config)

    assigned: dict[int, list[int]] = {}
    if probe_endpoint is not None and len(probe_endpoint):
        probe_mask = (
            np.asarray(probe_success, dtype=bool)
            if probe_success is not None
            else np.ones(len(probe_endpoint), dtype=bool)
        )
        _probe_distance, probe_seed = tree.query(probe_endpoint / feature_scale, k=1)
        for probe_index, (seed, valid) in enumerate(zip(probe_seed, probe_mask)):
            if valid:
                assigned.setdefault(int(seed), []).append(int(probe_index))

    for index in candidates:
        index = int(index)
        neighbour = int(neighbours[index, 1])
        if distances[index, 1] > coverage.prune_distance:
            break
        if checks >= coverage.prune_max_checks:
            break
        if not active[index] or not active[neighbour] or index in protected:
            continue
        label_key = (int(labels[index, 0]), int(labels[index, 1]))
        if branch_counts.get(label_key, 0) <= coverage.prune_min_branch_rows:
            continue
        if not np.array_equal(labels[index], labels[neighbour]):
            continue
        if np.isfinite(condition[index]) and condition[index] >= coverage.prune_protect_condition:
            continue
        checks += 1
        excluded = set(map(int, np.flatnonzero(~active)))
        excluded.add(index)

        targets = [endpoint[index]]
        if probe_endpoint is not None:
            probe_indices = assigned.get(index, [])
            if len(probe_indices) > coverage.prune_probe_limit:
                # Prefer the farthest assigned probes; they are the strongest
                # evidence that this chart owns a unique convergence region.
                local_distance = np.linalg.norm(
                    (probe_endpoint[probe_indices] - endpoint[index]) / feature_scale,
                    axis=1,
                )
                order = np.argsort(local_distance)[::-1][: coverage.prune_probe_limit]
                probe_indices = [probe_indices[i] for i in order]
            targets.extend(probe_endpoint[probe_indices])

        dominated = True
        for target in targets:
            corrected, _distance, _nfev = _fast_correct_target(
                target,
                launch,
                endpoint,
                feature_scale,
                query_config,
                tree=tree,
                excluded=excluded,
                bounds=correction_bounds,
                seed_jacobian=jacobian,
                seed_condition=condition,
                q_scale=q_scale,
                coverage_config=coverage,
            )
            if corrected is None:
                dominated = False
                break
        if not dominated:
            continue
        active[index] = False
        protected.add(neighbour)
        branch_counts[label_key] -= 1
        removed += 1

    return (
        launch[active],
        endpoint[active],
        diagnostics[active],
        jacobian[active],
        condition[active],
        sigma_min[active],
        removed,
    )


def _estimate_coverage_cells(
    atlas_endpoint: FloatArray,
    feature_scale: FloatArray,
    probe_endpoint: FloatArray,
    probe_success: NDArray[np.bool_],
    *,
    jacobian: Optional[FloatArray] = None,
    condition: Optional[FloatArray] = None,
    config: Optional[AtlasConfig] = None,
) -> tuple[FloatArray, NDArray[np.int32], NDArray[np.int32]]:
    radius = np.zeros(len(atlas_endpoint), dtype=float)
    success_count = np.zeros(len(atlas_endpoint), dtype=np.int32)
    failure_count = np.zeros(len(atlas_endpoint), dtype=np.int32)
    if len(atlas_endpoint) == 0 or len(probe_endpoint) == 0:
        return radius, success_count, failure_count
    tree = cKDTree(atlas_endpoint / feature_scale)
    distances, indices = tree.query(probe_endpoint / feature_scale, k=1)
    nearest_failure = np.full(len(atlas_endpoint), np.inf, dtype=float)
    q_scale = _launch_q_scale(config) if config is not None else None
    coverage = config.coverage if config is not None else None
    for target, distance, index, success in zip(probe_endpoint, distances, indices, probe_success):
        index = int(index)
        if config is not None:
            distance = _jacobian_seed_distance(
                target, index, atlas_endpoint, feature_scale, jacobian, condition, q_scale, coverage
            )[0]
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



def _evaluate_probe_set(
    probe_endpoint: FloatArray,
    atlas_launch: FloatArray,
    atlas_endpoint: FloatArray,
    feature_scale: FloatArray,
    query_config: QueryConfig,
    bounds: Optional[tuple[FloatArray, FloatArray]] = None,
    *,
    seed_jacobian: Optional[FloatArray] = None,
    seed_condition: Optional[FloatArray] = None,
    q_scale: Optional[FloatArray] = None,
    coverage_config: Optional[CoverageConfig] = None,
) -> tuple[NDArray[np.bool_], FloatArray, list[Optional[FloatArray]], int]:
    """Correct a persistent probe set against the current atlas."""
    count = len(probe_endpoint)
    success = np.zeros(count, dtype=bool)
    distances = np.full(count, np.inf, dtype=float)
    corrected: list[Optional[FloatArray]] = [None] * count
    if count == 0 or len(atlas_launch) == 0:
        return success, distances, corrected, 0
    tree = cKDTree(atlas_endpoint / feature_scale)
    correction_bounds = bounds or _launch_correction_bounds(
        atlas_launch, query_config.launch_bound_margin
    )
    nfev_total = 0
    for i, target in enumerate(probe_endpoint):
        launch, distance, nfev = _fast_correct_target(
            target,
            atlas_launch,
            atlas_endpoint,
            feature_scale,
            query_config,
            tree=tree,
            bounds=correction_bounds,
            seed_jacobian=seed_jacobian,
            seed_condition=seed_condition,
            q_scale=q_scale,
            coverage_config=coverage_config,
        )
        corrected[i] = launch
        success[i] = launch is not None
        distances[i] = distance
        nfev_total += nfev
    return success, distances, corrected, nfev_total


def _stratum_keys(
    endpoint: FloatArray,
    diagnostics: FloatArray,
    config: AtlasConfig,
) -> list[tuple[int, ...]]:
    """Coarse output/branch strata used by the stopping certificate."""
    if len(endpoint) == 0:
        return []
    bins = np.asarray(config.coverage.strata_bins, dtype=int)
    lo = np.array(
        [math.log(config.rho_bounds[0]), config.theta_bounds[0], math.log(config.kappa_bounds[0])],
        dtype=float,
    )
    hi = np.array(
        [math.log(config.rho_bounds[1]), config.theta_bounds[1], math.log(config.kappa_bounds[1])],
        dtype=float,
    )
    values = endpoint[:, [2, 3, 6]]
    unit = np.clip((values - lo) / np.maximum(hi - lo, 1.0e-12), 0.0, 1.0 - 1.0e-12)
    spatial = np.floor(unit * bins).astype(int)
    labels = _branch_labels(diagnostics)
    # Cap labels so isolated extreme branches do not create one-row strata.
    labels = np.column_stack((np.clip(labels[:, 0], 0, 5), np.clip(labels[:, 1], -4, 4)))
    return [tuple(map(int, row)) for row in np.column_stack((spatial, labels))]


def _stratum_coverage_metrics(
    success: NDArray[np.bool_],
    endpoint: FloatArray,
    diagnostics: FloatArray,
    config: AtlasConfig,
) -> dict:
    keys = _stratum_keys(endpoint, diagnostics, config)
    groups: dict[tuple[int, ...], list[bool]] = {}
    for key, value in zip(keys, success):
        groups.setdefault(key, []).append(bool(value))
    qualified = {
        key: values
        for key, values in groups.items()
        if len(values) >= config.coverage.minimum_stratum_rows
    }
    if not qualified:
        return {
            "qualified_strata": 0,
            "passing_strata": 0,
            "passing_fraction": 0.0,
            "minimum_success_rate": 0.0,
        }
    rates = np.array([np.mean(values) for values in qualified.values()], dtype=float)
    passing = rates >= config.coverage.target_stratum_success
    return {
        "qualified_strata": int(len(rates)),
        "passing_strata": int(np.count_nonzero(passing)),
        "passing_fraction": float(np.mean(passing)),
        "minimum_success_rate": float(np.min(rates)),
    }


def _sample_circular_targets(config: AtlasConfig) -> FloatArray:
    coverage = config.coverage
    if not coverage.circular_enabled or coverage.circular_validation_rows <= 0:
        return np.empty((0, 7), dtype=float)
    sampler = qmc.Sobol(d=3, scramble=True, seed=coverage.circular_validation_seed)
    unit = sampler.random(coverage.circular_validation_rows)
    log_rho = math.log(config.rho_bounds[0]) + unit[:, 0] * math.log(
        config.rho_bounds[1] / config.rho_bounds[0]
    )
    theta = config.theta_bounds[0] + unit[:, 1] * (
        config.theta_bounds[1] - config.theta_bounds[0]
    )
    log_kappa = math.log(config.kappa_bounds[0]) + unit[:, 2] * math.log(
        config.kappa_bounds[1] / config.kappa_bounds[0]
    )
    rho = np.exp(log_rho)
    return np.column_stack(
        (
            np.zeros(len(unit)),
            np.ones(len(unit)),
            log_rho,
            theta,
            np.zeros(len(unit)),
            rho ** -0.5,
            log_kappa,
        )
    ).astype(float)


def _correct_target_from_explicit_seed(
    target: FloatArray,
    seed_launch: FloatArray,
    query_config: QueryConfig,
    bounds: tuple[FloatArray, FloatArray],
) -> tuple[Optional[FloatArray], int]:
    seed = np.concatenate((seed_launch[2:6], [math.log(seed_launch[6])]))
    lo, hi = bounds
    corrected, nfev, _reason = _correct_one_seed(
        seed, lo, hi, float(target[0]), float(target[1]), target[2:7], query_config,
        deadline=time.monotonic() + max(query_config.wall_time_seconds, 1.0e-3),
    )
    if corrected is not None:
        corrected = corrected.copy()
        corrected[0:2] = target[0:2]
    return corrected, int(nfev)


def _continuation_correct_target(
    source_launch: FloatArray,
    source_endpoint: FloatArray,
    target: FloatArray,
    query_config: QueryConfig,
    bounds: tuple[FloatArray, FloatArray],
    steps: int,
) -> tuple[Optional[FloatArray], int]:
    """Homotopy in all seven endpoint coordinates from a known extremal."""
    current_launch = np.asarray(source_launch, dtype=float).copy()
    total_nfev = 0
    for fraction in np.linspace(1.0 / steps, 1.0, steps):
        intermediate = (1.0 - fraction) * source_endpoint + fraction * target
        corrected, nfev = _correct_target_from_explicit_seed(
            intermediate, current_launch, query_config, bounds
        )
        total_nfev += nfev
        if corrected is None:
            return None, total_nfev
        current_launch = corrected
    current_launch[0:2] = target[0:2]
    return current_launch, total_nfev


def _diagnostics_for_exact_launch(
    launch: FloatArray,
    config: AtlasConfig,
) -> Optional[FloatArray]:
    t_eval = np.linspace(0.0, float(launch[6]), config.diagnostic_points)
    sol, normal_constant, status = _integrate_launch(
        launch,
        rtol=config.rtol,
        atol=config.atol,
        max_step=config.max_step,
        r_collision=config.r_collision,
        r_escape=config.r_escape,
        max_acceleration=config.max_acceleration,
        t_eval=t_eval,
    )
    if sol is None or status != "ok" or normal_constant <= 0.0:
        return None
    radii = np.linalg.norm(sol.y[0:2, :], axis=0)
    acceleration = np.linalg.norm(sol.y[4:6, :], axis=0)
    theta = float(sol.y[9, -1])
    return np.array(
        [
            normal_constant,
            float(np.min(radii)),
            float(np.max(radii)),
            float(np.max(acceleration)),
            float(_count_radial_turns(sol.y)),
            float(round(theta / (2.0 * math.pi))),
        ],
        dtype=float,
    )


def _evaluate_circular_targets(
    targets: FloatArray,
    atlas_launch: FloatArray,
    atlas_endpoint: FloatArray,
    feature_scale: FloatArray,
    query_config: QueryConfig,
    config: AtlasConfig,
    *,
    continuation_limit: Optional[int] = None,
    seed_jacobian: Optional[FloatArray] = None,
    seed_condition: Optional[FloatArray] = None,
    q_scale: Optional[FloatArray] = None,
) -> tuple[NDArray[np.bool_], FloatArray, list[Optional[FloatArray]], int]:
    """Solve exact circular probes with bounded continuation work.

    Every target first receives the normal nearest-neighbour/Newton attempt.
    Homotopy continuation is then applied only to the closest failed targets,
    capped by ``continuation_limit``.  This keeps large persistent circular
    banks computationally tractable while allowing the solved frontier to grow
    from round to round.
    """
    success = np.zeros(len(targets), dtype=bool)
    distances = np.full(len(targets), np.inf, dtype=float)
    corrected: list[Optional[FloatArray]] = [None] * len(targets)
    if len(targets) == 0 or len(atlas_launch) == 0:
        return success, distances, corrected, 0

    tree = cKDTree(atlas_endpoint / feature_scale)
    bounds = _combined_correction_bounds(config, atlas_launch, 0.25)
    k_nearest = min(config.coverage.circular_nearest_seeds, len(atlas_launch))
    nearest_distance, nearest_index = tree.query(
        targets / feature_scale,
        k=k_nearest,
    )
    nearest_distance = np.asarray(nearest_distance, dtype=float)
    nearest_index = np.asarray(nearest_index, dtype=int)
    if nearest_distance.ndim == 1:
        if len(targets) == 1:
            nearest_distance = nearest_distance.reshape(1, -1)
            nearest_index = nearest_index.reshape(1, -1)
        elif k_nearest == 1:
            nearest_distance = nearest_distance[:, None]
            nearest_index = nearest_index[:, None]
    distances[:] = nearest_distance[:, 0]

    total_nfev = 0
    # Direct correction is much cheaper than continuation and is attempted for
    # the entire selected batch.
    for probe_index, target in enumerate(targets):
        launch, _distance, nfev = _fast_correct_target(
            target,
            atlas_launch,
            atlas_endpoint,
            feature_scale,
            query_config,
            tree=tree,
            bounds=bounds,
            seed_jacobian=seed_jacobian,
            seed_condition=seed_condition,
            q_scale=q_scale,
            coverage_config=config.coverage,
        )
        total_nfev += nfev
        corrected[probe_index] = launch
        success[probe_index] = launch is not None

    failed = np.flatnonzero(~success)
    if not query_config.allow_continuation:
        continuation_limit = 0
    if continuation_limit is None:
        continuation_limit = len(failed)
    continuation_limit = max(0, min(int(continuation_limit), len(failed)))
    # Grow outward from the currently best represented circular region.
    failed = failed[np.argsort(distances[failed])]
    for probe_index in failed[:continuation_limit]:
        target = targets[probe_index]
        for seed_index in np.atleast_1d(nearest_index[probe_index]):
            seed_index = int(seed_index)
            launch, nfev = _continuation_correct_target(
                atlas_launch[seed_index],
                atlas_endpoint[seed_index],
                target,
                query_config,
                bounds,
                config.coverage.circular_homotopy_steps,
            )
            total_nfev += nfev
            if launch is not None:
                corrected[probe_index] = launch
                success[probe_index] = True
                break
    return success, distances, corrected, total_nfev


def _append_rows_with_distance_filter(
    atlas_launch: FloatArray,
    atlas_endpoint: FloatArray,
    atlas_diag: FloatArray,
    new_launch: FloatArray,
    new_endpoint: FloatArray,
    new_diag: FloatArray,
    feature_scale: FloatArray,
    minimum_distance: float,
    capacity: int,
) -> tuple[FloatArray, FloatArray, FloatArray, int]:
    if capacity <= 0 or len(new_launch) == 0:
        return atlas_launch, atlas_endpoint, atlas_diag, 0
    tree = cKDTree(atlas_endpoint / feature_scale)
    distances, _ = tree.query(new_endpoint / feature_scale, k=1)
    labels = _branch_labels(new_diag)
    order = np.argsort(distances)[::-1]
    occupied: set[tuple[int, ...]] = set()
    keep: list[int] = []
    inv = 1.0 / max(minimum_distance, 1.0e-12)
    for index in order:
        if distances[index] < minimum_distance:
            continue
        cell = tuple(np.floor(new_endpoint[index] / feature_scale * inv).astype(np.int64)) + tuple(labels[index])
        if cell in occupied:
            continue
        occupied.add(cell)
        keep.append(int(index))
        if len(keep) >= capacity:
            break
    if not keep:
        return atlas_launch, atlas_endpoint, atlas_diag, 0
    idx = np.asarray(keep, dtype=int)
    return (
        np.vstack((atlas_launch, new_launch[idx])),
        np.vstack((atlas_endpoint, new_endpoint[idx])),
        np.vstack((atlas_diag, new_diag[idx])),
        int(len(idx)),
    )



def _append_atlas_rows(
    atlas_launch: FloatArray,
    atlas_endpoint: FloatArray,
    atlas_diag: FloatArray,
    atlas_jacobian: FloatArray,
    atlas_condition: FloatArray,
    atlas_sigma_min: FloatArray,
    new_launch: FloatArray,
    new_endpoint: FloatArray,
    new_diag: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    if len(new_launch) == 0:
        return (
            atlas_launch,
            atlas_endpoint,
            atlas_diag,
            atlas_jacobian,
            atlas_condition,
            atlas_sigma_min,
        )
    new_jac, new_condition, new_sigma = _empty_jacobian_arrays(len(new_launch))
    return (
        np.vstack((atlas_launch, new_launch)),
        np.vstack((atlas_endpoint, new_endpoint)),
        np.vstack((atlas_diag, new_diag)),
        np.concatenate((atlas_jacobian, new_jac), axis=0),
        np.concatenate((atlas_condition, new_condition)),
        np.concatenate((atlas_sigma_min, new_sigma)),
    )


def _jacobian_priority_indices(
    atlas_endpoint: FloatArray,
    probe_endpoint: FloatArray,
    atlas_jacobian: FloatArray,
    feature_scale: FloatArray,
    limit: int,
) -> FloatArray:
    if limit <= 0 or len(atlas_endpoint) == 0:
        return np.empty(0, dtype=int)
    missing = ~np.all(np.isfinite(atlas_jacobian), axis=(1, 2))
    if not np.any(missing):
        return np.empty(0, dtype=int)
    candidates: list[int] = []
    if len(probe_endpoint):
        tree = cKDTree(atlas_endpoint / feature_scale)
        k = min(4, len(atlas_endpoint))
        _dist, idx = tree.query(probe_endpoint / feature_scale, k=k)
        candidates.extend(map(int, np.atleast_1d(idx).ravel()))
    candidates.extend(map(int, np.flatnonzero(missing)))
    unique = []
    seen = set()
    for index in candidates:
        if index in seen or not missing[index]:
            continue
        seen.add(index)
        unique.append(index)
        if len(unique) >= limit:
            break
    return np.asarray(unique, dtype=int)


def _probe_metrics(success: NDArray[np.bool_], distance: FloatArray) -> dict:
    if len(success) == 0:
        return {
            "rows": 0,
            "successes": 0,
            "success_rate": 0.0,
            "median_distance": math.inf,
            "p95_distance": math.inf,
            "p99_distance": math.inf,
        }
    return {
        "rows": int(len(success)),
        "successes": int(np.count_nonzero(success)),
        "success_rate": float(np.mean(success)),
        "median_distance": float(np.median(distance)),
        "p95_distance": float(np.quantile(distance, 0.95)),
        "p99_distance": float(np.quantile(distance, 0.99)),
    }



def _coverage_distances(
    targets: FloatArray,
    atlas_endpoint: FloatArray,
    feature_scale: FloatArray,
    jacobian: Optional[FloatArray],
    condition: Optional[FloatArray],
    config: AtlasConfig,
) -> FloatArray:
    """Nearest chart distance, using the local inverse Jacobian when regular."""
    if len(targets) == 0 or len(atlas_endpoint) == 0:
        return np.full(len(targets), np.inf, dtype=float)
    tree = cKDTree(atlas_endpoint / feature_scale)
    _euclidean, nearest = tree.query(targets / feature_scale, k=1)
    nearest = np.asarray(nearest, dtype=int)
    q_scale = _launch_q_scale(config)
    output = np.empty(len(targets), dtype=float)
    for i, (target, seed) in enumerate(zip(targets, nearest)):
        output[i] = _jacobian_seed_distance(
            target,
            int(seed),
            atlas_endpoint,
            feature_scale,
            jacobian,
            condition,
            q_scale,
            config.coverage,
        )[0]
    return output


def _atomic_save_npz(
    path: str | Path,
    *,
    compressed: bool = True,
    **arrays,
) -> Path:
    """Atomically replace an NPZ file after a complete temporary write.

    A process interruption can therefore leave either the previous valid
    checkpoint or the new valid checkpoint, but not a half-written target.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp.npz")
    writer = np.savez_compressed if compressed else np.savez
    try:
        writer(temporary, **arrays)
        # Flush the completed temporary file before the atomic rename.
        fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temporary, target)
        try:
            dir_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            # Directory fsync is not supported on every platform/filesystem.
            pass
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass
    return target


def _checkpoint_config_matches(
    saved: dict, current: AtlasConfig, *, allow_policy_change: bool = False
) -> bool:
    """Compare resume configuration.

    With ``allow_policy_change`` only the adaptive coverage/query policy may
    differ; the physical launch distribution, integration guards, feature
    scaling, seed, and Sobol budget must remain identical. This is useful when
    upgrading the production query method without discarding an expensive
    forward cloud.
    """
    current_dict = asdict(current)
    saved_dict = dict(saved)
    if allow_policy_change:
        saved_dict.pop("coverage", None)
        current_dict.pop("coverage", None)
    left = json.dumps(saved_dict, sort_keys=True, separators=(",", ":"))
    right = json.dumps(current_dict, sort_keys=True, separators=(",", ":"))
    return left == right


def generate_atlas(
    path: str | Path, config: AtlasConfig, *, resume: bool = False,
    allow_policy_change: bool = False,
) -> Path:
    """Generate an adaptive convergence-cell atlas.

    Point selection is driven by a cumulative feasible validation reservoir,
    independent fresh audits, acquisition scoring, lazy exact endpoint
    Jacobians, local frontier expansion, and conservative coverage-dominance
    pruning.  A complete atomic NPZ checkpoint is written after initialization,
    every configured number of rounds, and on graceful interruption.
    """
    config.validate()
    if not config.coverage.enabled:
        return _generate_atlas_fixed(path, config)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    coverage = config.coverage
    launches = _sample_launches(config)
    q_scale = _launch_q_scale(config)

    initial_count = int(round(config.n_samples * coverage.initial_fraction))
    initial_count = min(max(initial_count, 1), max(config.n_samples - 2, 1))
    remaining_after_initial = max(0, config.n_samples - initial_count)
    reservoir_launch_count = min(
        coverage.persistent_validation_launches,
        max(1, remaining_after_initial // 3),
    )

    resumed = False
    phase = "initial"
    initial_rows_before_thinning = 0
    last_fresh_launch = np.empty((0, 7), dtype=float)
    last_fresh_endpoint = np.empty((0, 7), dtype=float)
    last_fresh_diag = np.empty((0, 6), dtype=float)
    last_fresh_success = np.empty(0, dtype=bool)
    last_fresh_distance = np.empty(0, dtype=float)

    atlas_launch = np.empty((0, 7), dtype=float)
    atlas_endpoint = np.empty((0, 7), dtype=float)
    atlas_diag = np.empty((0, 6), dtype=float)
    atlas_jacobian, atlas_condition, atlas_sigma_min = _empty_jacobian_arrays(0)
    reservoir_launch = np.empty((0, 7), dtype=float)
    reservoir_endpoint = np.empty((0, 7), dtype=float)
    reservoir_diag = np.empty((0, 6), dtype=float)
    reservoir_tested = np.empty(0, dtype=bool)
    reservoir_success = np.empty(0, dtype=bool)
    reservoir_distance = np.empty(0, dtype=float)
    circular_targets = np.empty((0, 7), dtype=float)
    circular_tested_state = np.empty(0, dtype=bool)
    circular_success_state = np.empty(0, dtype=bool)
    circular_distance_state = np.empty(0, dtype=float)
    circular_corrected_state: list[Optional[FloatArray]] = []

    if config.feature_scale is None:
        feature_scale = np.ones(7, dtype=float)
        feature_scale_initialized = False
    else:
        feature_scale = np.asarray(config.feature_scale, dtype=float)
        if feature_scale.shape != (7,) or np.any(feature_scale <= 0.0):
            raise ValueError("feature_scale must contain seven positive values")
        feature_scale_initialized = True

    cursor = 0
    round_number = 0
    successful_rounds = 0
    coverage_history: list[dict] = []
    rng = np.random.default_rng(coverage.validation_seed)

    def save_bootstrap_checkpoint(reason: str, *, interrupted: bool = False) -> None:
        """Save initialization/reservoir progress before the main coverage loop exists."""
        if not coverage.checkpoint_enabled:
            return
        row_count = len(atlas_launch)
        checkpoint_metadata = {
            "format_version": 5,
            "model": "planar-kepler-costate-free-normal-pmp",
            "atlas_strategy": "cumulative-reservoir-jacobian-acquisition-frontier",
            "launch_columns": ["u0", "w0", "Ar0", "At0", "Jr0", "ell", "tau"],
            "endpoint_columns": [
                "u0", "w0", "log_rho", "theta_unwrapped", "ur_final", "ut_final", "log_kappa"
            ],
            "diagnostic_columns": [
                "normal_constant", "minimum_radius", "maximum_radius",
                "maximum_acceleration", "radial_turns", "rounded_winding"
            ],
            "jacobian_columns": [
                "log_rho", "theta_unwrapped", "ur_final", "ut_final", "log_kappa"
            ],
            "jacobian_parameters": ["Ar0", "At0", "Jr0", "ell", "log_tau"],
            "config": asdict(config),
            "accepted_atlas_rows": int(row_count),
            "initial_rows_before_thinning": int(initial_rows_before_thinning),
            "validation_reservoir_rows": int(len(reservoir_endpoint)),
            "validation_reservoir_tested": int(np.count_nonzero(reservoir_tested)),
            "circular_validation_rows": int(len(circular_targets)),
            "computed_jacobians": int(
                np.count_nonzero(np.all(np.isfinite(atlas_jacobian), axis=(1, 2)))
            ) if row_count else 0,
            "integrated_launch_attempts": int(cursor),
            "launch_attempt_budget": int(config.n_samples),
            "subarcs_per_trajectory": int(config.subarcs_per_trajectory),
            "coverage_converged": False,
            "generation_complete": False,
            "generation_interrupted": bool(interrupted),
            "checkpoint_reason": reason,
            "checkpoint_round": int(round_number),
            "coverage_successful_rounds": int(successful_rounds),
            "coverage_history": coverage_history,
            "generation_state": {
                "phase": phase,
                "cursor": int(cursor),
                "round_number": int(round_number),
                "successful_rounds": int(successful_rounds),
                "feature_scale_initialized": bool(feature_scale_initialized),
                "rng_state": rng.bit_generator.state,
            },
            "coverage_definition": (
                "Known-feasible cumulative reservoir and fresh audits are covered only "
                "when exact shooting correction converges. Local inverse-Jacobian effort "
                "defines chart distance when available."
            ),
        }
        _atomic_save_npz(
            path,
            compressed=coverage.checkpoint_compressed,
            launch=atlas_launch,
            endpoint=atlas_endpoint,
            diagnostics=atlas_diag,
            feature_scale=feature_scale,
            endpoint_jacobian=atlas_jacobian,
            jacobian_condition=atlas_condition,
            jacobian_sigma_min=atlas_sigma_min,
            coverage_radius=np.zeros(row_count, dtype=float),
            coverage_success_count=np.zeros(row_count, dtype=np.int32),
            coverage_failure_count=np.zeros(row_count, dtype=np.int32),
            persistent_probe_launch=reservoir_launch,
            persistent_probe_endpoint=reservoir_endpoint,
            persistent_probe_diagnostics=reservoir_diag,
            persistent_probe_tested=reservoir_tested,
            persistent_probe_success=reservoir_success,
            persistent_probe_distance=reservoir_distance,
            circular_probe_endpoint=circular_targets,
            circular_probe_tested=circular_tested_state,
            circular_probe_success=circular_success_state,
            circular_probe_distance=circular_distance_state,
            circular_probe_status=np.where(
                ~circular_tested_state,
                0,
                np.where(circular_success_state, 1, 2),
            ).astype(np.int8),
            last_fresh_launch=last_fresh_launch,
            last_fresh_endpoint=last_fresh_endpoint,
            last_fresh_diagnostics=last_fresh_diag,
            last_fresh_success=last_fresh_success,
            last_fresh_distance=last_fresh_distance,
            metadata_json=np.array(json.dumps(checkpoint_metadata)),
        )
        print(
            f"Bootstrap checkpoint saved after {reason}: {path} "
            f"(phase={phase}, rows={row_count:,}, cursor={cursor:,}/{config.n_samples:,})"
        )

    if resume and path.exists():
        with np.load(path, allow_pickle=False) as data:
            saved_metadata = json.loads(str(data["metadata_json"].item()))
            if not _checkpoint_config_matches(
                saved_metadata.get("config", {}), config,
                allow_policy_change=allow_policy_change,
            ):
                raise ValueError(
                    "Checkpoint configuration does not match the requested configuration. "
                    "Resume with the original JSON or choose a new output path."
                )
            if saved_metadata.get("generation_complete", False):
                print(f"Atlas generation is already complete: {path}")
                return path
            required = (
                "launch", "endpoint", "diagnostics", "feature_scale",
                "endpoint_jacobian", "jacobian_condition", "jacobian_sigma_min",
                "persistent_probe_launch", "persistent_probe_endpoint",
                "persistent_probe_diagnostics", "persistent_probe_tested",
                "persistent_probe_success", "persistent_probe_distance",
                "circular_probe_endpoint", "circular_probe_tested",
                "circular_probe_success", "circular_probe_distance",
            )
            missing = [name for name in required if name not in data.files]
            if missing:
                raise ValueError(
                    "The existing NPZ is not a resumable format-v5 checkpoint; missing arrays: "
                    + ", ".join(missing)
                )
            atlas_launch = np.asarray(data["launch"], dtype=float)
            atlas_endpoint = np.asarray(data["endpoint"], dtype=float)
            atlas_diag = np.asarray(data["diagnostics"], dtype=float)
            feature_scale = np.asarray(data["feature_scale"], dtype=float)
            atlas_jacobian = np.asarray(data["endpoint_jacobian"], dtype=np.float32)
            atlas_condition = np.asarray(data["jacobian_condition"], dtype=float)
            atlas_sigma_min = np.asarray(data["jacobian_sigma_min"], dtype=float)
            reservoir_launch = np.asarray(data["persistent_probe_launch"], dtype=float)
            reservoir_endpoint = np.asarray(data["persistent_probe_endpoint"], dtype=float)
            reservoir_diag = np.asarray(data["persistent_probe_diagnostics"], dtype=float)
            reservoir_tested = np.asarray(data["persistent_probe_tested"], dtype=bool)
            reservoir_success = np.asarray(data["persistent_probe_success"], dtype=bool)
            reservoir_distance = np.asarray(data["persistent_probe_distance"], dtype=float)
            circular_targets = np.asarray(data["circular_probe_endpoint"], dtype=float)
            circular_tested_state = np.asarray(data["circular_probe_tested"], dtype=bool)
            circular_success_state = np.asarray(data["circular_probe_success"], dtype=bool)
            circular_distance_state = np.asarray(data["circular_probe_distance"], dtype=float)
            if "last_fresh_launch" in data.files:
                last_fresh_launch = np.asarray(data["last_fresh_launch"], dtype=float)
                last_fresh_endpoint = np.asarray(data["last_fresh_endpoint"], dtype=float)
                last_fresh_diag = np.asarray(data["last_fresh_diagnostics"], dtype=float)
                last_fresh_success = np.asarray(data["last_fresh_success"], dtype=bool)
                last_fresh_distance = np.asarray(data["last_fresh_distance"], dtype=float)
        state = saved_metadata.get("generation_state", {})
        phase = str(state.get("phase", "coverage"))
        cursor = int(state.get("cursor", saved_metadata.get("integrated_launch_attempts", 0)))
        round_number = int(state.get("round_number", 0))
        successful_rounds = int(state.get("successful_rounds", 0))
        feature_scale_initialized = bool(state.get("feature_scale_initialized", True))
        coverage_history = list(saved_metadata.get("coverage_history", []))
        initial_rows_before_thinning = int(
            saved_metadata.get("initial_rows_before_thinning", len(atlas_launch))
        )
        if "rng_state" in state:
            rng.bit_generator.state = state["rng_state"]
        circular_corrected_state = [None] * len(circular_targets)
        resumed = True
        print(
            f"Resuming checkpoint {path}: phase={phase}, round {round_number}, "
            f"launch cursor {cursor:,}/{config.n_samples:,}, atlas rows {len(atlas_launch):,}"
        )
    elif resume:
        raise FileNotFoundError(f"Cannot resume; checkpoint does not exist: {path}")

    bootstrap_chunk = max(1, int(coverage.checkpoint_bootstrap_launches))
    reservoir_start = initial_count
    reservoir_stop = min(reservoir_start + reservoir_launch_count, config.n_samples)

    try:
        if not resumed and coverage.checkpoint_enabled and coverage.checkpoint_initial:
            save_bootstrap_checkpoint("empty initialization checkpoint")

        if phase == "initial":
            if cursor < initial_count:
                print(
                    f"Integrating initial coverage seed launches in chunks of "
                    f"{bootstrap_chunk:,}: {initial_count - cursor:,} remaining"
                )
            while cursor < initial_count:
                stop = min(cursor + bootstrap_chunk, initial_count)
                new_launch, new_endpoint, new_diag = _generate_rows_from_launches(
                    launches[cursor:stop], config
                )
                initial_rows_before_thinning += int(len(new_launch))
                if len(new_launch):
                    if not feature_scale_initialized:
                        feature_scale = _robust_feature_scale(new_endpoint)
                        feature_scale_initialized = True
                    combined_launch = (
                        np.vstack((atlas_launch, new_launch)) if len(atlas_launch) else new_launch
                    )
                    combined_endpoint = (
                        np.vstack((atlas_endpoint, new_endpoint)) if len(atlas_endpoint) else new_endpoint
                    )
                    combined_diag = (
                        np.vstack((atlas_diag, new_diag)) if len(atlas_diag) else new_diag
                    )
                    atlas_launch, atlas_endpoint, atlas_diag = _thin_endpoint_rows(
                        combined_launch,
                        combined_endpoint,
                        combined_diag,
                        feature_scale,
                        coverage.initial_cell_size,
                        coverage.max_atlas_rows,
                    )
                    atlas_jacobian, atlas_condition, atlas_sigma_min = _empty_jacobian_arrays(
                        len(atlas_launch)
                    )
                cursor = stop
                if coverage.checkpoint_enabled:
                    save_bootstrap_checkpoint("initial launch chunk")
            if len(atlas_launch) == 0:
                raise RuntimeError("No initial atlas samples survived integration")
            phase = "reservoir"
            print(
                f"Initial accepted rows before thinning: {initial_rows_before_thinning:,}; "
                f"retained atlas rows: {len(atlas_launch):,}"
            )
            if coverage.checkpoint_enabled:
                save_bootstrap_checkpoint("initial phase complete")

        # Initial exact Jacobians are computed once the seed cloud exists.  The
        # operation is idempotent, so it is safe to repeat after a bootstrap resume.
        if phase == "reservoir" and coverage.jacobian_initial_rows > 0:
            finite_j = np.all(np.isfinite(atlas_jacobian), axis=(1, 2))
            needed_j = max(0, min(coverage.jacobian_initial_rows, len(atlas_endpoint)) - int(np.count_nonzero(finite_j)))
            if needed_j:
                jac_idx = _select_validation_indices(
                    atlas_endpoint[~finite_j],
                    atlas_diag[~finite_j],
                    feature_scale,
                    min(needed_j, int(np.count_nonzero(~finite_j))),
                    rng,
                    coverage.validation_reservoir_cell_size,
                )
                missing_global = np.flatnonzero(~finite_j)[jac_idx]
                computed = _ensure_seed_jacobians(
                    atlas_launch,
                    atlas_jacobian,
                    atlas_condition,
                    atlas_sigma_min,
                    missing_global,
                    config,
                )
                print(f"Initial exact endpoint Jacobians computed: {computed:,}")
                if coverage.checkpoint_enabled:
                    save_bootstrap_checkpoint("initial Jacobian batch")

        if phase == "reservoir":
            if cursor < reservoir_start:
                cursor = reservoir_start
            if cursor < reservoir_stop:
                print(
                    f"Integrating cumulative feasible validation launches in chunks of "
                    f"{bootstrap_chunk:,}: {reservoir_stop - cursor:,} remaining"
                )
            while cursor < reservoir_stop:
                stop = min(cursor + bootstrap_chunk, reservoir_stop)
                new_launch, new_endpoint, new_diag = _generate_rows_from_launches(
                    launches[cursor:stop], config
                )
                if len(new_launch):
                    zeros = np.zeros(len(new_launch), dtype=bool)
                    infinities = np.full(len(new_launch), np.inf, dtype=float)
                    (
                        reservoir_launch,
                        reservoir_endpoint,
                        reservoir_diag,
                        reservoir_tested,
                        reservoir_success,
                        reservoir_distance,
                    ) = _merge_validation_reservoir(
                        reservoir_launch,
                        reservoir_endpoint,
                        reservoir_diag,
                        reservoir_tested,
                        reservoir_success,
                        reservoir_distance,
                        new_launch,
                        new_endpoint,
                        new_diag,
                        zeros,
                        zeros,
                        infinities,
                        feature_scale,
                        config,
                    )
                cursor = stop
                if coverage.checkpoint_enabled:
                    save_bootstrap_checkpoint("validation-reservoir launch chunk")
            if len(reservoir_launch) == 0:
                raise RuntimeError("No validation-reservoir extremals survived integration")
            print(f"Cumulative validation reservoir rows: {len(reservoir_endpoint):,}")
            if len(circular_targets) == 0:
                circular_targets = _sample_circular_targets(config)
                circular_tested_state = np.zeros(len(circular_targets), dtype=bool)
                circular_success_state = np.zeros(len(circular_targets), dtype=bool)
                circular_distance_state = np.full(len(circular_targets), np.inf, dtype=float)
                circular_corrected_state = [None] * len(circular_targets)
                if len(circular_targets):
                    print(f"Persistent exact circular targets: {len(circular_targets):,}")
            phase = "coverage"
            if coverage.checkpoint_enabled:
                save_bootstrap_checkpoint("bootstrap phases complete")
    except KeyboardInterrupt:
        if coverage.checkpoint_enabled and coverage.checkpoint_on_interrupt:
            try:
                save_bootstrap_checkpoint("keyboard interrupt during bootstrap", interrupted=True)
            except BaseException as exc:
                print(f"Bootstrap interrupt checkpoint could not be refreshed: {exc}")
        print(
            "Generation interrupted during bootstrap. Resume with the same command plus --resume."
        )
        return path

    if phase != "coverage":
        raise RuntimeError(f"Unexpected generation phase after bootstrap: {phase}")

    query_config = _coverage_query_config(config)
    circular_query_config = _coverage_query_config(config, circular=True)
    print(
        "Coverage query policy: "
        f"method={query_config.method}, seeds={query_config.max_seed_attempts}, "
        f"iterations={query_config.max_iterations}, "
        f"wall={query_config.wall_time_seconds:.3g}s, "
        f"robust_fallback={query_config.robust_fallback}, "
        f"continuation={query_config.allow_continuation}"
    )
    correction_bounds = _combined_correction_bounds(config, atlas_launch, 0.25)

    def coverage_cell_arrays() -> tuple[FloatArray, NDArray[np.int32], NDArray[np.int32]]:
        endpoints: list[FloatArray] = []
        successes: list[NDArray[np.bool_]] = []
        if len(reservoir_endpoint):
            endpoints.append(reservoir_endpoint[reservoir_tested])
            successes.append(reservoir_success[reservoir_tested])
        if len(circular_targets) and np.any(circular_tested_state):
            endpoints.append(circular_targets[circular_tested_state])
            successes.append(circular_success_state[circular_tested_state])
        if not endpoints:
            return (
                np.zeros(len(atlas_launch), dtype=float),
                np.zeros(len(atlas_launch), dtype=np.int32),
                np.zeros(len(atlas_launch), dtype=np.int32),
            )
        return _estimate_coverage_cells(
            atlas_endpoint,
            feature_scale,
            np.vstack(endpoints),
            np.concatenate(successes),
            jacobian=atlas_jacobian,
            condition=atlas_condition,
            config=config,
        )

    def save_checkpoint(
        reason: str, *, interrupted: bool = False, fast: bool = False
    ) -> None:
        if not coverage.checkpoint_enabled:
            return
        if fast:
            radius = np.zeros(len(atlas_launch), dtype=float)
            success_count = np.zeros(len(atlas_launch), dtype=np.int32)
            failure_count = np.zeros(len(atlas_launch), dtype=np.int32)
        else:
            radius, success_count, failure_count = coverage_cell_arrays()
        checkpoint_metadata = {
            "format_version": 5,
            "model": "planar-kepler-costate-free-normal-pmp",
            "atlas_strategy": "cumulative-reservoir-jacobian-acquisition-frontier",
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
            "jacobian_columns": [
                "log_rho",
                "theta_unwrapped",
                "ur_final",
                "ut_final",
                "log_kappa",
            ],
            "jacobian_parameters": ["Ar0", "At0", "Jr0", "ell", "log_tau"],
            "config": asdict(config),
            "accepted_atlas_rows": int(len(atlas_launch)),
            "initial_rows_before_thinning": int(initial_rows_before_thinning),
            "validation_reservoir_rows": int(len(reservoir_endpoint)),
            "validation_reservoir_tested": int(np.count_nonzero(reservoir_tested)),
            "circular_validation_rows": int(len(circular_targets)),
            "computed_jacobians": int(
                np.count_nonzero(np.all(np.isfinite(atlas_jacobian), axis=(1, 2)))
            ),
            "integrated_launch_attempts": int(cursor),
            "launch_attempt_budget": int(config.n_samples),
            "subarcs_per_trajectory": int(config.subarcs_per_trajectory),
            "coverage_converged": False,
            "generation_complete": False,
            "generation_interrupted": bool(interrupted),
            "checkpoint_reason": reason,
            "checkpoint_round": int(round_number),
            "coverage_successful_rounds": int(successful_rounds),
            "coverage_history": coverage_history,
            "generation_state": {
                "phase": phase,
                "cursor": int(cursor),
                "round_number": int(round_number),
                "successful_rounds": int(successful_rounds),
                "feature_scale_initialized": bool(feature_scale_initialized),
                "rng_state": rng.bit_generator.state,
            },
            "coverage_definition": (
                "Known-feasible cumulative reservoir and fresh audits are covered only "
                "when exact shooting correction converges.  Local inverse-Jacobian "
                "effort defines chart distance when available."
            ),
        }
        _atomic_save_npz(
            path,
            compressed=coverage.checkpoint_compressed,
            launch=atlas_launch,
            endpoint=atlas_endpoint,
            diagnostics=atlas_diag,
            feature_scale=feature_scale,
            endpoint_jacobian=atlas_jacobian,
            jacobian_condition=atlas_condition,
            jacobian_sigma_min=atlas_sigma_min,
            coverage_radius=radius,
            coverage_success_count=success_count,
            coverage_failure_count=failure_count,
            persistent_probe_launch=reservoir_launch,
            persistent_probe_endpoint=reservoir_endpoint,
            persistent_probe_diagnostics=reservoir_diag,
            persistent_probe_tested=reservoir_tested,
            persistent_probe_success=reservoir_success,
            persistent_probe_distance=reservoir_distance,
            circular_probe_endpoint=circular_targets,
            circular_probe_tested=circular_tested_state,
            circular_probe_success=circular_success_state,
            circular_probe_distance=circular_distance_state,
            circular_probe_status=np.where(
                ~circular_tested_state,
                0,
                np.where(circular_success_state, 1, 2),
            ).astype(np.int8),
            last_fresh_launch=last_fresh_launch,
            last_fresh_endpoint=last_fresh_endpoint,
            last_fresh_diagnostics=last_fresh_diag,
            last_fresh_success=last_fresh_success,
            last_fresh_distance=last_fresh_distance,
            metadata_json=np.array(json.dumps(checkpoint_metadata)),
        )
        print(
            f"Checkpoint saved atomically after {reason}: {path} "
            f"({len(atlas_launch):,} rows, cursor {cursor:,}/{config.n_samples:,})"
        )

    if not resumed and coverage.checkpoint_enabled and coverage.checkpoint_initial:
        save_checkpoint("initialization")

    try:
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
            # The canonical subarc transformation can create launch/time values
            # outside the original sampling box; refresh the safe correction
            # bounds as the atlas grows.
            correction_bounds = _combined_correction_bounds(config, atlas_launch, 0.25)

            if len(candidate_launch) == 0:
                coverage_history.append(
                    {
                        "round": round_number,
                        "launch_attempts": int(len(batch_launches)),
                        "candidate_rows": 0,
                    }
                )
                if coverage.checkpoint_enabled:
                    save_checkpoint(f"empty coverage round {round_number}")
                continue

            audit_indices = _select_validation_indices(
                candidate_endpoint,
                candidate_diag,
                feature_scale,
                min(coverage.validation_rows, len(candidate_endpoint)),
                rng,
                coverage.validation_cell_size,
            )
            audit_launch = candidate_launch[audit_indices]
            audit_endpoint = candidate_endpoint[audit_indices]
            audit_diag = candidate_diag[audit_indices]

            # Compute exact Jacobians where they provide the most value: near
            # unresolved reservoir probes, fresh audits, and circular targets.
            jacobian_probe_parts = [audit_endpoint]
            unresolved_for_jac = reservoir_endpoint[~reservoir_success]
            if len(unresolved_for_jac):
                jacobian_probe_parts.append(unresolved_for_jac)
            if len(circular_targets):
                jacobian_probe_parts.append(circular_targets[~circular_success_state])
            jacobian_probes = np.vstack(jacobian_probe_parts)
            jacobian_indices = _jacobian_priority_indices(
                atlas_endpoint,
                jacobian_probes,
                atlas_jacobian,
                feature_scale,
                coverage.jacobian_compute_per_round,
            )
            jacobians_computed = _ensure_seed_jacobians(
                atlas_launch,
                atlas_jacobian,
                atlas_condition,
                atlas_sigma_min,
                jacobian_indices,
                config,
            )

            # Independent fresh audit.
            audit_success, audit_distance, _audit_corrected, nfev_audit = _evaluate_probe_set(
                audit_endpoint,
                atlas_launch,
                atlas_endpoint,
                feature_scale,
                query_config,
                correction_bounds,
                seed_jacobian=atlas_jacobian,
                seed_condition=atlas_condition,
                q_scale=q_scale,
                coverage_config=coverage,
            )
            fresh_metrics = _probe_metrics(audit_success, audit_distance)
            last_fresh_launch = audit_launch.copy()
            last_fresh_endpoint = audit_endpoint.copy()
            last_fresh_diag = audit_diag.copy()
            last_fresh_success = audit_success.copy()
            last_fresh_distance = audit_distance.copy()

            # Add fresh feasible probes to the cumulative spatially diverse bank.
            (
                reservoir_launch,
                reservoir_endpoint,
                reservoir_diag,
                reservoir_tested,
                reservoir_success,
                reservoir_distance,
            ) = _merge_validation_reservoir(
                reservoir_launch,
                reservoir_endpoint,
                reservoir_diag,
                reservoir_tested,
                reservoir_success,
                reservoir_distance,
                audit_launch,
                audit_endpoint,
                audit_diag,
                np.ones(len(audit_endpoint), dtype=bool),
                audit_success,
                audit_distance,
                feature_scale,
                config,
            )

            # Retest a bounded priority subset: never-tested probes first, then
            # failed probes with the largest current chart distance.
            unresolved = np.flatnonzero(~reservoir_tested | ~reservoir_success)
            if len(unresolved):
                priority = (
                    (~reservoir_tested[unresolved]).astype(float) * 1.0e6
                    + np.nan_to_num(
                        reservoir_distance[unresolved], nan=1.0e3, posinf=1.0e3
                    )
                )
                unresolved = unresolved[np.argsort(priority)[::-1]]
                unresolved = unresolved[: coverage.validation_retest_rows_per_round]
            nfev_reservoir = 0
            if len(unresolved):
                (
                    retest_success,
                    retest_distance,
                    _retest_corrected,
                    nfev_reservoir,
                ) = _evaluate_probe_set(
                    reservoir_endpoint[unresolved],
                    atlas_launch,
                    atlas_endpoint,
                    feature_scale,
                    query_config,
                    correction_bounds,
                    seed_jacobian=atlas_jacobian,
                    seed_condition=atlas_condition,
                    q_scale=q_scale,
                    coverage_config=coverage,
                )
                reservoir_tested[unresolved] = True
                reservoir_success[unresolved] |= retest_success
                reservoir_distance[unresolved] = retest_distance
            # Distances are cheap enough to refresh for the whole reservoir and
            # use local inverse-Jacobian effort where available.
            reservoir_distance[:] = _coverage_distances(
                reservoir_endpoint,
                atlas_endpoint,
                feature_scale,
                atlas_jacobian,
                atlas_condition,
                config,
            )
            reservoir_metrics = _probe_metrics(reservoir_success, reservoir_distance)
            reservoir_tested_fraction = float(np.mean(reservoir_tested))
            strata = _stratum_coverage_metrics(
                reservoir_success, reservoir_endpoint, reservoir_diag, config
            )

            # Incremental exact circular family validation.
            circular_batch_indices = np.empty(0, dtype=int)
            circular_batch_success = np.empty(0, dtype=bool)
            nfev_circular = 0
            if len(circular_targets):
                untested = np.flatnonzero(~circular_tested_state)
                failed_tested = np.flatnonzero(circular_tested_state & ~circular_success_state)
                if len(failed_tested):
                    failed_tested = failed_tested[
                        np.argsort(circular_distance_state[failed_tested])
                    ]
                needed = coverage.circular_batch_rows
                circular_batch_indices = untested[:needed]
                if len(circular_batch_indices) < needed:
                    circular_batch_indices = np.concatenate(
                        (
                            circular_batch_indices,
                            failed_tested[: needed - len(circular_batch_indices)],
                        )
                    )
                if len(circular_batch_indices):
                    (
                        circular_batch_success,
                        circular_batch_distance,
                        circular_batch_corrected,
                        nfev_circular,
                    ) = _evaluate_circular_targets(
                        circular_targets[circular_batch_indices],
                        atlas_launch,
                        atlas_endpoint,
                        feature_scale,
                        circular_query_config,
                        config,
                        continuation_limit=coverage.circular_bootstrap_per_round,
                        seed_jacobian=atlas_jacobian,
                        seed_condition=atlas_condition,
                        q_scale=q_scale,
                    )
                    circular_tested_state[circular_batch_indices] = True
                    circular_success_state[circular_batch_indices] |= circular_batch_success
                    circular_distance_state[circular_batch_indices] = circular_batch_distance
                    for local, global_index in enumerate(circular_batch_indices):
                        if circular_batch_corrected[local] is not None:
                            circular_corrected_state[int(global_index)] = circular_batch_corrected[local]
                circular_distance_state[:] = _coverage_distances(
                    circular_targets,
                    atlas_endpoint,
                    feature_scale,
                    atlas_jacobian,
                    atlas_condition,
                    config,
                )
            circular_metrics = _probe_metrics(
                circular_success_state, circular_distance_state
            )
            circular_all_tested = bool(
                len(circular_targets) == 0 or np.all(circular_tested_state)
            )

            # Empirical cells and adaptive leaf deficits are computed before
            # insertion so acquisition targets the current holes.
            current_radius, _cell_success_count, _cell_failure_count = coverage_cell_arrays()
            adaptive_partition = _build_adaptive_partition(
                reservoir_endpoint,
                reservoir_success,
                reservoir_distance,
                feature_scale,
                config,
            )

            # Direct frontier expansion from local charts facing unresolved probes.
            frontier_launch = np.empty((0, 7), dtype=float)
            frontier_endpoint = np.empty((0, 7), dtype=float)
            frontier_diag = np.empty((0, 6), dtype=float)
            frontier_attempts = 0
            if coverage.frontier_enabled:
                frontier_launch, frontier_endpoint, frontier_diag, frontier_attempts = (
                    _frontier_expand(
                        atlas_launch,
                        atlas_endpoint,
                        atlas_diag,
                        atlas_jacobian,
                        atlas_condition,
                        current_radius,
                        reservoir_endpoint[~reservoir_success],
                        feature_scale,
                        config,
                        rng,
                    )
                )

            # Build one acquisition pool.  Known feasible failures and successful
            # circular/frontier continuations receive a strong priority bonus.
            pool_launch: list[FloatArray] = []
            pool_endpoint: list[FloatArray] = []
            pool_diag: list[FloatArray] = []
            pool_failed: list[NDArray[np.bool_]] = []

            failed_reservoir = ~reservoir_success
            if np.any(failed_reservoir):
                pool_launch.append(reservoir_launch[failed_reservoir])
                pool_endpoint.append(reservoir_endpoint[failed_reservoir])
                pool_diag.append(reservoir_diag[failed_reservoir])
                pool_failed.append(np.ones(np.count_nonzero(failed_reservoir), dtype=bool))

            candidate_failure = np.zeros(len(candidate_launch), dtype=bool)
            candidate_failure[audit_indices] = ~audit_success
            pool_launch.append(candidate_launch)
            pool_endpoint.append(candidate_endpoint)
            pool_diag.append(candidate_diag)
            pool_failed.append(candidate_failure)

            if len(frontier_launch):
                pool_launch.append(frontier_launch)
                pool_endpoint.append(frontier_endpoint)
                pool_diag.append(frontier_diag)
                pool_failed.append(np.ones(len(frontier_launch), dtype=bool))

            circular_rows_launch: list[FloatArray] = []
            circular_rows_endpoint: list[FloatArray] = []
            circular_rows_diag: list[FloatArray] = []
            for index in np.flatnonzero(circular_success_state):
                launch = circular_corrected_state[int(index)]
                if launch is None:
                    continue
                diag = _diagnostics_for_exact_launch(launch, config)
                if diag is None:
                    continue
                circular_rows_launch.append(launch)
                circular_rows_endpoint.append(circular_targets[int(index)])
                circular_rows_diag.append(diag)
            if circular_rows_launch:
                pool_launch.append(np.vstack(circular_rows_launch))
                pool_endpoint.append(np.vstack(circular_rows_endpoint))
                pool_diag.append(np.vstack(circular_rows_diag))
                pool_failed.append(np.ones(len(circular_rows_launch), dtype=bool))

            acquisition_launch = np.vstack(pool_launch)
            acquisition_endpoint = np.vstack(pool_endpoint)
            acquisition_diag = np.vstack(pool_diag)
            acquisition_failed = np.concatenate(pool_failed)
            capacity = max(0, coverage.max_atlas_rows - len(atlas_launch))
            insertion_limit = min(coverage.acquisition_insertions_per_round, capacity)
            if coverage.acquisition_enabled:
                acquisition_indices, acquisition_scores = _acquisition_select_indices(
                    acquisition_endpoint,
                    acquisition_diag,
                    acquisition_failed,
                    atlas_endpoint,
                    atlas_diag,
                    feature_scale,
                    current_radius,
                    atlas_condition,
                    adaptive_partition,
                    config,
                    insertion_limit,
                )
            else:
                tree_simple = cKDTree(atlas_endpoint / feature_scale)
                simple_distance, _ = tree_simple.query(
                    acquisition_endpoint / feature_scale, k=1
                )
                acquisition_indices = _failure_insertion_indices(
                    np.arange(len(acquisition_endpoint), dtype=int),
                    np.asarray(simple_distance, dtype=float),
                    acquisition_endpoint,
                    acquisition_diag,
                    feature_scale,
                    coverage.insertion_cell_size,
                    insertion_limit,
                )
                acquisition_scores = np.asarray(simple_distance)[acquisition_indices]
            inserted = int(len(acquisition_indices))
            if inserted:
                (
                    atlas_launch,
                    atlas_endpoint,
                    atlas_diag,
                    atlas_jacobian,
                    atlas_condition,
                    atlas_sigma_min,
                ) = _append_atlas_rows(
                    atlas_launch,
                    atlas_endpoint,
                    atlas_diag,
                    atlas_jacobian,
                    atlas_condition,
                    atlas_sigma_min,
                    acquisition_launch[acquisition_indices],
                    acquisition_endpoint[acquisition_indices],
                    acquisition_diag[acquisition_indices],
                )

            # Conservative periodic coverage-dominance pruning.
            periodic_pruned = 0
            if (
                coverage.prune_enabled
                and round_number % coverage.prune_every_rounds == 0
                and len(atlas_launch) > 2
            ):
                (
                    atlas_launch,
                    atlas_endpoint,
                    atlas_diag,
                    atlas_jacobian,
                    atlas_condition,
                    atlas_sigma_min,
                    periodic_pruned,
                ) = _prune_redundant_rows(
                    atlas_launch,
                    atlas_endpoint,
                    atlas_diag,
                    feature_scale,
                    config,
                    jacobian=atlas_jacobian,
                    condition=atlas_condition,
                    sigma_min=atlas_sigma_min,
                    probe_endpoint=reservoir_endpoint,
                    probe_success=reservoir_success,
                )

            launch_fraction = cursor / config.n_samples
            reservoir_ok = (
                len(reservoir_success) >= coverage.minimum_validation_rows
                and reservoir_tested_fraction >= 0.99
                and reservoir_metrics["success_rate"] >= coverage.target_success
                and (
                    not coverage.distance_criteria_enabled
                    or (
                        reservoir_metrics["p95_distance"] <= coverage.maximum_p95_distance
                        and reservoir_metrics["p99_distance"] <= coverage.maximum_p99_distance
                    )
                )
                and strata["passing_fraction"] >= coverage.minimum_strata_fraction
            )
            fresh_ok = (
                fresh_metrics["success_rate"] >= coverage.fresh_target_success
                and (
                    not coverage.distance_criteria_enabled
                    or fresh_metrics["p95_distance"] <= coverage.fresh_maximum_p95_distance
                )
            )
            circular_ok = (
                not coverage.circular_enabled
                or (
                    circular_all_tested
                    and circular_metrics["success_rate"] >= coverage.circular_target_success
                    and (
                        not coverage.circular_distance_criteria_enabled
                        or circular_metrics["p95_distance"]
                        <= coverage.circular_maximum_p95_distance
                    )
                )
            )
            budget_ok = launch_fraction >= coverage.minimum_launch_fraction_before_stop
            round_ok = reservoir_ok and fresh_ok and circular_ok and budget_ok
            successful_rounds = successful_rounds + 1 if round_ok else 0

            record = {
                "round": int(round_number),
                "launch_attempts": int(len(batch_launches)),
                "launch_fraction": float(launch_fraction),
                "candidate_rows": int(len(candidate_launch)),
                "reservoir": reservoir_metrics,
                "reservoir_tested_fraction": reservoir_tested_fraction,
                "fresh_audit": fresh_metrics,
                "strata": strata,
                "circular": circular_metrics,
                "circular_tested": int(np.count_nonzero(circular_tested_state)),
                "circular_all_tested": bool(circular_all_tested),
                "jacobians_computed": int(jacobians_computed),
                "frontier_attempts": int(frontier_attempts),
                "frontier_solutions": int(len(frontier_launch)),
                "acquisition_pool_rows": int(len(acquisition_launch)),
                "acquisition_inserted": int(inserted),
                "acquisition_score_median": (
                    float(np.median(acquisition_scores)) if len(acquisition_scores) else math.nan
                ),
                "periodic_pruned": int(periodic_pruned),
                "atlas_rows": int(len(atlas_launch)),
                "correction_nfev": int(nfev_audit + nfev_reservoir + nfev_circular),
                "reservoir_ok": bool(reservoir_ok),
                "fresh_ok": bool(fresh_ok),
                "circular_ok": bool(circular_ok),
                "budget_ok": bool(budget_ok),
                "round_ok": bool(round_ok),
            }
            coverage_history.append(record)

            print(
                f"Coverage round {round_number}: reservoir "
                f"{reservoir_metrics['success_rate']:.1%} "
                f"({reservoir_tested_fraction:.1%} tested), "
                f"p95={reservoir_metrics['p95_distance']:.3g}, "
                f"p99={reservoir_metrics['p99_distance']:.3g}; fresh "
                f"{fresh_metrics['success_rate']:.1%}, "
                f"p95={fresh_metrics['p95_distance']:.3g}; "
                f"strata={strata['passing_fraction']:.1%}; circular "
                f"{circular_metrics['success_rate']:.1%}, "
                f"p95={circular_metrics['p95_distance']:.3g}; "
                f"inserted={inserted}, frontier={len(frontier_launch)}, "
                f"pruned={periodic_pruned}, atlas={len(atlas_launch):,}"
            )

            if (
                coverage.checkpoint_enabled
                and round_number % coverage.checkpoint_every_rounds == 0
            ):
                save_checkpoint(f"coverage round {round_number}")

            if successful_rounds >= coverage.patience:
                print(
                    "Coverage criteria reached for "
                    f"{coverage.patience} consecutive cumulative/fresh rounds."
                )
                break

    except KeyboardInterrupt:
        if coverage.checkpoint_enabled and coverage.checkpoint_on_interrupt:
            try:
                save_checkpoint("keyboard interrupt", interrupted=True, fast=True)
            except BaseException as exc:
                print(f"Interrupt checkpoint could not be refreshed: {exc}")
        print(
            "Generation interrupted. The latest completed-round atlas checkpoint "
            f"is available at {path}. Resume with the same command plus --resume."
        )
        return path

    # Final conservative dominance pruning and exact re-audit.
    before_prune = len(atlas_launch)
    (
        atlas_launch,
        atlas_endpoint,
        atlas_diag,
        atlas_jacobian,
        atlas_condition,
        atlas_sigma_min,
        pruned,
    ) = _prune_redundant_rows(
        atlas_launch,
        atlas_endpoint,
        atlas_diag,
        feature_scale,
        config,
        jacobian=atlas_jacobian,
        condition=atlas_condition,
        sigma_min=atlas_sigma_min,
        probe_endpoint=reservoir_endpoint,
        probe_success=reservoir_success,
    )
    if pruned:
        print(f"Coverage-dominated seeds removed: {pruned:,}")

    correction_bounds = _combined_correction_bounds(config, atlas_launch, 0.25)
    final_jac_indices = _jacobian_priority_indices(
        atlas_endpoint,
        np.vstack((reservoir_endpoint, circular_targets))
        if len(circular_targets)
        else reservoir_endpoint,
        atlas_jacobian,
        feature_scale,
        coverage.jacobian_compute_per_round,
    )
    _ensure_seed_jacobians(
        atlas_launch,
        atlas_jacobian,
        atlas_condition,
        atlas_sigma_min,
        final_jac_indices,
        config,
    )

    final_success, final_distance, _corrected, final_nfev = _evaluate_probe_set(
        reservoir_endpoint,
        atlas_launch,
        atlas_endpoint,
        feature_scale,
        query_config,
        correction_bounds,
        seed_jacobian=atlas_jacobian,
        seed_condition=atlas_condition,
        q_scale=q_scale,
        coverage_config=coverage,
    )
    reservoir_tested[:] = True
    reservoir_success[:] = final_success
    reservoir_distance[:] = _coverage_distances(
        reservoir_endpoint,
        atlas_endpoint,
        feature_scale,
        atlas_jacobian,
        atlas_condition,
        config,
    )
    final_reservoir_metrics = _probe_metrics(reservoir_success, reservoir_distance)
    final_strata = _stratum_coverage_metrics(
        reservoir_success, reservoir_endpoint, reservoir_diag, config
    )

    final_fresh_nfev = 0
    if len(last_fresh_endpoint):
        (
            last_fresh_success,
            _last_fresh_distance_exact,
            _last_fresh_corrected,
            final_fresh_nfev,
        ) = _evaluate_probe_set(
            last_fresh_endpoint,
            atlas_launch,
            atlas_endpoint,
            feature_scale,
            query_config,
            correction_bounds,
            seed_jacobian=atlas_jacobian,
            seed_condition=atlas_condition,
            q_scale=q_scale,
            coverage_config=coverage,
        )
        last_fresh_distance = _coverage_distances(
            last_fresh_endpoint,
            atlas_endpoint,
            feature_scale,
            atlas_jacobian,
            atlas_condition,
            config,
        )
    final_fresh_metrics = _probe_metrics(last_fresh_success, last_fresh_distance)
    final_circular_nfev = 0
    if len(circular_targets):
        (
            circular_success_state,
            _circular_distance_exact,
            _circular_corrected,
            final_circular_nfev,
        ) = _evaluate_circular_targets(
            circular_targets,
            atlas_launch,
            atlas_endpoint,
            feature_scale,
            circular_query_config,
            config,
            continuation_limit=len(circular_targets),
            seed_jacobian=atlas_jacobian,
            seed_condition=atlas_condition,
            q_scale=q_scale,
        )
        circular_tested_state[:] = True
        circular_distance_state[:] = _coverage_distances(
            circular_targets,
            atlas_endpoint,
            feature_scale,
            atlas_jacobian,
            atlas_condition,
            config,
        )
    final_circular_metrics = _probe_metrics(
        circular_success_state, circular_distance_state
    )

    launch_fraction = cursor / config.n_samples
    final_reservoir_ok = (
        len(reservoir_success) >= coverage.minimum_validation_rows
        and final_reservoir_metrics["success_rate"] >= coverage.target_success
        and (
            not coverage.distance_criteria_enabled
            or (
                final_reservoir_metrics["p95_distance"] <= coverage.maximum_p95_distance
                and final_reservoir_metrics["p99_distance"] <= coverage.maximum_p99_distance
            )
        )
        and final_strata["passing_fraction"] >= coverage.minimum_strata_fraction
    )
    final_fresh_ok = (
        len(last_fresh_success) > 0
        and final_fresh_metrics["success_rate"] >= coverage.fresh_target_success
        and (
            not coverage.distance_criteria_enabled
            or final_fresh_metrics["p95_distance"] <= coverage.fresh_maximum_p95_distance
        )
    )
    final_circular_ok = (
        not coverage.circular_enabled
        or (
            len(circular_success_state) > 0
            and final_circular_metrics["success_rate"] >= coverage.circular_target_success
            and (
                not coverage.circular_distance_criteria_enabled
                or final_circular_metrics["p95_distance"]
                <= coverage.circular_maximum_p95_distance
            )
        )
    )
    final_budget_ok = launch_fraction >= coverage.minimum_launch_fraction_before_stop
    coverage_converged = bool(
        final_reservoir_ok and final_fresh_ok and final_circular_ok and final_budget_ok
    )

    radius, success_count, failure_count = coverage_cell_arrays()
    metadata = {
        "format_version": 5,
        "model": "planar-kepler-costate-free-normal-pmp",
        "atlas_strategy": "cumulative-reservoir-jacobian-acquisition-frontier",
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
        "jacobian_columns": [
            "log_rho",
            "theta_unwrapped",
            "ur_final",
            "ut_final",
            "log_kappa",
        ],
        "jacobian_parameters": ["Ar0", "At0", "Jr0", "ell", "log_tau"],
        "config": asdict(config),
        "accepted_atlas_rows": int(len(atlas_launch)),
        "initial_rows_before_thinning": int(initial_rows_before_thinning),
        "rows_before_final_pruning": int(before_prune),
        "coverage_dominated_rows_removed": int(pruned),
        "computed_jacobians": int(
            np.count_nonzero(np.all(np.isfinite(atlas_jacobian), axis=(1, 2)))
        ),
        "integrated_launch_attempts": int(cursor),
        "launch_attempt_budget": int(config.n_samples),
        "subarcs_per_trajectory": int(config.subarcs_per_trajectory),
        "generation_complete": True,
        "coverage_converged": coverage_converged,
        "coverage_final_reservoir": final_reservoir_metrics,
        "coverage_final_fresh_audit": final_fresh_metrics,
        "coverage_final_strata": final_strata,
        "coverage_final_circular": final_circular_metrics,
        "coverage_final_reservoir_ok": bool(final_reservoir_ok),
        "coverage_final_fresh_ok": bool(final_fresh_ok),
        "coverage_final_circular_ok": bool(final_circular_ok),
        "coverage_final_budget_ok": bool(final_budget_ok),
        "coverage_successful_rounds": int(successful_rounds),
        "coverage_history": coverage_history,
        "final_audit_nfev": int(final_nfev + final_fresh_nfev + final_circular_nfev),
        "coverage_definition": (
            "A cumulative known-feasible endpoint reservoir plus independent fresh "
            "audits must converge by exact shooting.  Seed selection uses empirical "
            "cell radii, adaptive-cell deficits, branch novelty, and local exact "
            "endpoint Jacobians."
        ),
    }
    _atomic_save_npz(
        path,
        compressed=coverage.checkpoint_compressed,
        launch=atlas_launch,
        endpoint=atlas_endpoint,
        diagnostics=atlas_diag,
        feature_scale=feature_scale,
        endpoint_jacobian=atlas_jacobian,
        jacobian_condition=atlas_condition,
        jacobian_sigma_min=atlas_sigma_min,
        coverage_radius=radius,
        coverage_success_count=success_count,
        coverage_failure_count=failure_count,
        persistent_probe_launch=reservoir_launch,
        persistent_probe_endpoint=reservoir_endpoint,
        persistent_probe_diagnostics=reservoir_diag,
        persistent_probe_tested=reservoir_tested,
        persistent_probe_success=reservoir_success,
        persistent_probe_distance=reservoir_distance,
        circular_probe_endpoint=circular_targets,
        circular_probe_tested=circular_tested_state,
        circular_probe_success=circular_success_state,
        circular_probe_distance=circular_distance_state,
        circular_probe_status=np.where(
            ~circular_tested_state,
            0,
            np.where(circular_success_state, 1, 2),
        ).astype(np.int8),
        last_fresh_launch=last_fresh_launch,
        last_fresh_endpoint=last_fresh_endpoint,
        last_fresh_diagnostics=last_fresh_diag,
        last_fresh_success=last_fresh_success,
        last_fresh_distance=last_fresh_distance,
        metadata_json=np.array(json.dumps(metadata)),
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
            self.endpoint_jacobian = (
                np.asarray(data["endpoint_jacobian"], dtype=np.float32)
                if "endpoint_jacobian" in data.files
                else np.full((len(self.launch), 5, 5), np.nan, dtype=np.float32)
            )
            self.jacobian_condition = (
                np.asarray(data["jacobian_condition"], dtype=float)
                if "jacobian_condition" in data.files
                else np.full(len(self.launch), np.inf, dtype=float)
            )
            self.jacobian_sigma_min = (
                np.asarray(data["jacobian_sigma_min"], dtype=float)
                if "jacobian_sigma_min" in data.files
                else np.zeros(len(self.launch), dtype=float)
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
        if self.endpoint_jacobian.shape != (len(self.launch), 5, 5):
            raise ValueError("invalid atlas endpoint_jacobian array")
        if self.jacobian_condition.shape != (len(self.launch),):
            raise ValueError("invalid atlas jacobian_condition array")
        try:
            self._generation_config = _atlas_config_from_dict(self.metadata.get("config", {}))
        except (TypeError, ValueError):
            self._generation_config = AtlasConfig()
        self._coverage_config = self._generation_config.coverage
        self._q_scale = _launch_q_scale(self._generation_config)
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
        indices, distances, _predictions = _rank_seed_indices(
            boundary.target,
            self.launch,
            self.endpoint,
            self.feature_scale,
            self._tree,
            1,
            jacobian=self.endpoint_jacobian,
            condition=self.jacobian_condition,
            q_scale=self._q_scale,
            coverage=self._coverage_config,
        )
        index = int(indices[0])
        distance = float(distances[0])
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
        idx, _distance, _predictions = _rank_seed_indices(
            target,
            self.launch,
            self.endpoint,
            self.feature_scale,
            self._tree,
            k,
            jacobian=self.endpoint_jacobian,
            condition=self.jacobian_condition,
            q_scale=self._q_scale,
            coverage=self._coverage_config,
        )
        return idx

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
        deadline: Optional[float] = None,
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
            deadline=deadline,
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
        deadline: Optional[float] = None,
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
            deadline=deadline,
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
        return _combined_correction_bounds(
            self._generation_config, self.launch, config.launch_bound_margin
        )

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
        idx, _ranked_distance, predictions = _rank_seed_indices(
            target,
            self.launch,
            self.endpoint,
            self.feature_scale,
            self._tree,
            min(cfg.neighbours, len(self.launch)),
            jacobian=self.endpoint_jacobian,
            condition=self.jacobian_condition,
            q_scale=self._q_scale,
            coverage=self._coverage_config,
        )

        seed_vectors: list[FloatArray] = []
        # Exact local inverse-Jacobian chart predictions.
        for i in idx[: min(cfg.direct_seeds, len(idx))]:
            delta_q = predictions.get(int(i))
            if delta_q is None:
                continue
            base = np.concatenate((self.launch[i, 2:6], [math.log(self.launch[i, 6])]))
            seed_vectors.append(base + delta_q)

        # Local inverse-chart regression prediction.
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

        deadline = time.monotonic() + max(cfg.wall_time_seconds, 1.0e-3)
        unique_seeds: list[FloatArray] = []
        for seed in seed_vectors:
            seed = np.asarray(seed, dtype=float)
            if not any(np.linalg.norm(seed - old) < 1.0e-10 for old in unique_seeds):
                unique_seeds.append(seed)
            if len(unique_seeds) >= cfg.max_seed_attempts:
                break

        for seed in unique_seeds:
            if time.monotonic() >= deadline:
                break
            corrected_launch, _nfev, _reason = _correct_one_seed(
                seed, lo, hi, float(target[0]), float(target[1]), target5, cfg,
                deadline=deadline,
            )
            if corrected_launch is None:
                continue

            # Reintegrate only after convergence. This is the first dense output
            # calculation, so failed pixels/probes remain cheap.
            t_eval = np.linspace(0.0, corrected_launch[6], cfg.trajectory_points)
            dense_sol, normal_constant, status = _integrate_launch(
                corrected_launch,
                rtol=cfg.rtol,
                atol=cfg.atol,
                max_step=cfg.max_step,
                r_collision=cfg.r_collision,
                r_escape=cfg.r_escape,
                max_acceleration=cfg.max_acceleration,
                t_eval=t_eval,
                deadline=deadline,
            )
            if dense_sol is None or status != "ok" or normal_constant <= 0.0:
                continue
            radius, theta, ur, ut, kappa = _endpoint_from_state(dense_sol.y[:, -1])
            raw = np.array([
                math.log(radius) - target5[0],
                theta - target5[1],
                ur - target5[2],
                ut - target5[3],
                math.log(kappa) - target5[4],
            ], dtype=float)
            if np.any(np.abs(raw) > np.asarray(cfg.acceptance, dtype=float)):
                continue
            scaled = raw / np.asarray(cfg.residual_scale, dtype=float)
            trajectory = _dimensional_trajectory(dense_sol.t, dense_sol.y, boundary, capability)
            radii = np.linalg.norm(dense_sol.y[0:2, :], axis=0)
            acc = np.linalg.norm(dense_sol.y[4:6, :], axis=0)
            solution = ShootingSolution(
                launch=corrected_launch,
                residual=raw,
                residual_norm=float(np.linalg.norm(scaled)),
                normal_constant=float(normal_constant),
                minimum_radius=float(np.min(radii)),
                maximum_acceleration=float(np.max(acc)),
                radial_turns=_count_radial_turns(dense_sol.y),
                trajectory=trajectory,
            )

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
            if not return_all:
                break

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
    query_method: Optional[str] = None,
    query_seconds: Optional[float] = None,
    query_iterations: Optional[int] = None,
    query_seeds: Optional[int] = None,
    robust_fallback: Optional[bool] = None,
    allow_continuation: Optional[bool] = None,
) -> dict:
    """Audit an atlas with fresh feasible probes and exact circular targets."""
    atlas = ExtremalAtlas(atlas_path)
    payload = atlas.metadata.get("config", {})
    config = _atlas_config_from_dict(payload) if payload else AtlasConfig()
    config.n_samples = int(launch_attempts)
    config.seed = int(seed if seed is not None else config.seed + 1_000_003)
    if workers is not None:
        config.workers = int(workers)
    if query_method is not None:
        config.coverage.query_method = query_method
    if query_seconds is not None:
        config.coverage.query_wall_seconds = float(query_seconds)
    if query_iterations is not None:
        config.coverage.query_max_iterations = int(query_iterations)
    if query_seeds is not None:
        config.coverage.query_max_seed_attempts = int(query_seeds)
    if robust_fallback is not None:
        config.coverage.query_robust_fallback = bool(robust_fallback)
    if allow_continuation is not None:
        config.coverage.query_allow_continuation = bool(allow_continuation)
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
        min(validation_rows, len(probe_endpoint)),
        rng,
        config.coverage.validation_cell_size,
    )
    query_config = _coverage_query_config(config)
    success, distances, _corrected, total_nfev = _evaluate_probe_set(
        probe_endpoint[indices],
        atlas.launch,
        atlas.endpoint,
        atlas.feature_scale,
        query_config,
        _combined_correction_bounds(config, atlas.launch, 0.25),
        seed_jacobian=atlas.endpoint_jacobian,
        seed_condition=atlas.jacobian_condition,
        q_scale=atlas._q_scale,
        coverage_config=atlas._coverage_config,
    )
    strata = _stratum_coverage_metrics(
        success, probe_endpoint[indices], probe_diag[indices], config
    )

    # Use a fresh circular Sobol sequence for the audit.
    original_circular_seed = config.coverage.circular_validation_seed
    config.coverage.circular_validation_seed = int(config.seed + 2_000_033)
    config.coverage.circular_validation_rows = min(
        max(64, validation_rows // 2),
        max(64, config.coverage.circular_validation_rows),
    )
    circular_targets = _sample_circular_targets(config)
    circular_query_config = _coverage_query_config(config, circular=True)
    circular_success = np.empty(0, dtype=bool)
    circular_distances = np.empty(0, dtype=float)
    circular_nfev = 0
    if len(circular_targets):
        circular_success, circular_distances, _launches, circular_nfev = (
            _evaluate_circular_targets(
                circular_targets,
                atlas.launch,
                atlas.endpoint,
                atlas.feature_scale,
                circular_query_config,
                config,
                continuation_limit=config.coverage.circular_bootstrap_per_round,
                seed_jacobian=atlas.endpoint_jacobian,
                seed_condition=atlas.jacobian_condition,
                q_scale=atlas._q_scale,
            )
        )
    config.coverage.circular_validation_seed = original_circular_seed

    p95 = float(np.quantile(distances, 0.95)) if len(distances) else math.nan
    p99 = float(np.quantile(distances, 0.99)) if len(distances) else math.nan
    circular_p95 = (
        float(np.quantile(circular_distances, 0.95))
        if len(circular_distances)
        else math.nan
    )
    general_rate = float(np.mean(success)) if len(success) else math.nan
    circular_rate = (
        float(np.mean(circular_success)) if len(circular_success) else math.nan
    )
    general_ok = bool(
        len(success) >= config.coverage.minimum_validation_rows
        and general_rate >= config.coverage.target_success
        and (
            not config.coverage.distance_criteria_enabled
            or (
                p95 <= config.coverage.maximum_p95_distance
                and p99 <= config.coverage.maximum_p99_distance
            )
        )
        and strata["passing_fraction"] >= config.coverage.minimum_strata_fraction
    )
    circular_ok = bool(
        not config.coverage.circular_enabled
        or (
            len(circular_success) > 0
            and circular_rate >= config.coverage.circular_target_success
            and (
                not config.coverage.circular_distance_criteria_enabled
                or circular_p95 <= config.coverage.circular_maximum_p95_distance
            )
        )
    )

    return {
        "atlas": str(Path(atlas_path).resolve()),
        "launch_attempts": int(launch_attempts),
        "accepted_probe_rows": int(len(probe_launch)),
        "tested_probe_rows": int(len(indices)),
        "successes": int(np.count_nonzero(success)),
        "success_rate": general_rate,
        "median_nearest_distance": float(np.median(distances)) if len(distances) else math.nan,
        "p95_nearest_distance": p95,
        "p99_nearest_distance": p99,
        "maximum_nearest_distance": float(np.max(distances)) if len(distances) else math.nan,
        "strata": strata,
        "circular_tested": int(len(circular_targets)),
        "circular_successes": int(np.count_nonzero(circular_success)),
        "circular_success_rate": circular_rate,
        "circular_p95_nearest_distance": circular_p95,
        "general_criteria_passed": general_ok,
        "circular_criteria_passed": circular_ok,
        "all_criteria_passed": bool(general_ok and circular_ok),
        "correction_nfev": int(total_nfev + circular_nfev),
        "seed": int(config.seed),
        "query_policy": {
            "method": query_config.method,
            "wall_time_seconds": query_config.wall_time_seconds,
            "max_iterations": query_config.max_iterations,
            "max_seed_attempts": query_config.max_seed_attempts,
            "robust_fallback": query_config.robust_fallback,
            "allow_continuation": query_config.allow_continuation,
        },
    }


def _parse_config(path: Optional[str]) -> AtlasConfig:
    if path is None:
        return AtlasConfig()
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return _atlas_config_from_dict(payload)



def _mark_checkpoint_interrupted(path: str | Path, reason: str) -> bool:
    """Atomically mark an existing incomplete checkpoint after an outer interrupt."""
    target = Path(path)
    if not target.exists():
        return False
    try:
        with np.load(target, allow_pickle=False) as data:
            if "metadata_json" not in data.files:
                return False
            metadata = json.loads(str(data["metadata_json"].item()))
            if metadata.get("generation_complete", False):
                return True
            arrays = {
                name: np.asarray(data[name])
                for name in data.files
                if name != "metadata_json"
            }
        metadata["generation_interrupted"] = True
        metadata["checkpoint_reason"] = reason
        compressed = bool(
            metadata.get("config", {})
            .get("coverage", {})
            .get("checkpoint_compressed", True)
        )
        arrays["metadata_json"] = np.array(json.dumps(metadata))
        _atomic_save_npz(target, compressed=compressed, **arrays)
        return True
    except BaseException as exc:
        print(f"Could not mark checkpoint as interrupted: {exc}")
        return False

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser("generate", help="generate an NPZ extremal atlas")
    make.add_argument("output", help="output .npz path")
    make.add_argument("--config", help="JSON file overriding AtlasConfig fields")
    make.add_argument("--samples", type=int, help="override number of Sobol samples")
    make.add_argument("--workers", type=int, help="override worker count")
    make.add_argument(
        "--resume",
        action="store_true",
        help="resume an incomplete atomic checkpoint at the output path",
    )
    make.add_argument(
        "--resume-policy-change",
        action="store_true",
        help=(
            "resume while allowing coverage/query-policy changes; physical launch "
            "and integration configuration must still match"
        ),
    )

    inspect = sub.add_parser("inspect", help="show atlas metadata")
    inspect.add_argument("atlas")

    validate = sub.add_parser("validate", help="test coverage on fresh feasible holdouts")
    validate.add_argument("atlas")
    validate.add_argument("--launches", type=int, default=4096)
    validate.add_argument("--rows", type=int, default=512)
    validate.add_argument("--seed", type=int)
    validate.add_argument("--workers", type=int)
    validate.add_argument(
        "--query-method", choices=("fast_newton", "robust_least_squares")
    )
    validate.add_argument("--query-seconds", type=float)
    validate.add_argument("--query-iterations", type=int)
    validate.add_argument("--query-seeds", type=int)
    validate.add_argument("--robust-fallback", action="store_true", default=None)
    validate.add_argument("--allow-continuation", action="store_true", default=None)

    args = parser.parse_args(argv)
    if args.command == "generate":
        cfg = _parse_config(args.config)
        if args.samples is not None:
            cfg.n_samples = args.samples
        if args.workers is not None:
            cfg.workers = args.workers
        try:
            output = generate_atlas(
                args.output, cfg, resume=args.resume,
                allow_policy_change=args.resume_policy_change,
            )
        except KeyboardInterrupt:
            marked = _mark_checkpoint_interrupted(
                args.output, "keyboard interrupt outside adaptive coverage loop"
            )
            if marked:
                print(
                    f"Generation interrupted. The latest atomic checkpoint remains at "
                    f"{args.output}; resume with the same command plus --resume."
                )
            else:
                print("Generation interrupted before a checkpoint could be created.")
            return 130
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
            query_method=args.query_method,
            query_seconds=args.query_seconds,
            query_iterations=args.query_iterations,
            query_seeds=args.query_seeds,
            robust_fallback=args.robust_fallback,
            allow_continuation=args.allow_continuation,
        )
        print(json.dumps(report, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
