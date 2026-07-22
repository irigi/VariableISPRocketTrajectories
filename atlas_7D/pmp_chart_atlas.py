#!/usr/bin/env python3
"""Validated local-inverse chart atlas for planar Kepler PMP extremals.

This module implements a chart atlas rather than a point cloud.  Each chart
stores a feasible extremal, a local normalized inverse map from an analytical
center-rotated Cartesian endpoint basis to the five nontrivial launch
variables, an optional low-rank quadratic correction, and an empirically
validated anisotropic trust ellipsoid.  Canonical polar endpoints remain the
external query interface and exact-correction target.

The existing ``pmp_extremal_atlas.py`` module is used only as the authoritative
dynamics and shooting backend.  Generation, chart fitting, coverage logic,
checkpointing, restart, and query routing are implemented here.

NPZ format ``pmp-local-inverse-chart-atlas`` version 5 is continuously and
atomically checkpointed.  Incomplete checkpoints are queryable and can be
resumed with ``generate ... --resume``.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import asdict, dataclass, field, replace
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Iterable, Optional, Sequence
import warnings
import zipfile

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from scipy.stats import qmc

FloatArray = NDArray[np.float64]

FORMAT_NAME = "pmp-local-inverse-chart-atlas"
FORMAT_VERSION = 5

ENDPOINT_COLUMNS = (
    "u0",
    "w0",
    "log_rho",
    "theta_unwrapped",
    "ur_final",
    "ut_final",
    "log_kappa",
)
LAUNCH_COLUMNS = ("u0", "w0", "Ar0", "At0", "Jr0", "ell", "tau")
Q_COLUMNS = ("Ar0", "At0", "Jr0", "ell", "log_tau")
DIAGNOSTIC_COLUMNS = (
    "normal_constant",
    "minimum_radius",
    "maximum_radius",
    "maximum_acceleration",
    "radial_turns",
    "rounded_winding",
)

TRANSFORM_RAW = 0
TRANSFORM_CUSTOM_ROTATED_CARTESIAN = 1
TRANSFORM_NAMES = {
    TRANSFORM_RAW: "raw",
    TRANSFORM_CUSTOM_ROTATED_CARTESIAN: "custom_rotated_cartesian",
}
TRANSFORMED_ENDPOINT_COLUMNS = (
    "u0",
    "w0",
    "x_rot",
    "y_rot",
    "vx_rot",
    "vy_rot",
    "log_kappa",
)
SEARCH_ENDPOINT_COLUMNS = (
    "u0",
    "w0",
    "x_final",
    "y_final",
    "vx_final",
    "vy_final",
    "log_kappa",
)


def _transform_mode_index(name: str) -> int:
    normalized = str(name).strip().lower()
    aliases = {
        "raw": TRANSFORM_RAW,
        "custom": TRANSFORM_CUSTOM_ROTATED_CARTESIAN,
        "cartesian_rotated": TRANSFORM_CUSTOM_ROTATED_CARTESIAN,
        "custom_rotated_cartesian": TRANSFORM_CUSTOM_ROTATED_CARTESIAN,
    }
    if normalized not in aliases:
        raise ValueError(f"unknown endpoint transform: {name}")
    return aliases[normalized]


def _global_search_transform(raw_endpoints: FloatArray) -> FloatArray:
    """Map canonical endpoints to a single inertial Cartesian search basis.

    The chart-local map below rotates by each chart's center angle.  A KD-tree
    cannot use a different rotation for every row, so routing uses the
    equivalent inertial Cartesian state.  Winding remains separate discrete
    branch metadata.
    """
    raw = np.atleast_2d(np.asarray(raw_endpoints, dtype=float))
    rho = np.exp(raw[:, 2])
    theta = raw[:, 3]
    ur = raw[:, 4]
    ut = raw[:, 5]
    c = np.cos(theta)
    ss = np.sin(theta)
    x = rho * c
    y = rho * ss
    vx = ur * c - ut * ss
    vy = ur * ss + ut * c
    return np.column_stack((raw[:, 0], raw[:, 1], x, y, vx, vy, raw[:, 6]))


def _local_endpoint_transform(
    raw_endpoints: FloatArray,
    center_raw_endpoint: FloatArray,
    transform_mode: int,
    asinh_scale: float = 0.0,
) -> FloatArray:
    """Apply the swarm-dashboard analytical map in a chart-local frame.

    For transformed charts this is exactly the default custom dashboard map:
    final Cartesian position/velocity in a frame rotated by the center final
    angle.  The raw canonical endpoint is retained only for exact correction,
    branch labels, and external APIs.
    """
    raw = np.atleast_2d(np.asarray(raw_endpoints, dtype=float))
    center = np.asarray(center_raw_endpoint, dtype=float)
    if int(transform_mode) == TRANSFORM_RAW:
        return raw.copy()
    rho = np.exp(raw[:, 2])
    delta = raw[:, 3] - float(center[3])
    ur = raw[:, 4]
    ut = raw[:, 5]
    c = np.cos(delta)
    ss = np.sin(delta)
    state = np.column_stack((
        rho * c,
        rho * ss,
        ur * c - ut * ss,
        ur * ss + ut * c,
    ))
    if float(asinh_scale) > 0.0:
        rho0 = math.exp(float(center[2]))
        state0 = np.array([rho0, 0.0, float(center[4]), float(center[5])], dtype=float)
        state = np.arcsinh((state - state0[None, :]) / float(asinh_scale))
    return np.column_stack((raw[:, 0], raw[:, 1], state, raw[:, 6]))


def _inverse_local_endpoint_transform(
    transformed_endpoints: FloatArray,
    center_raw_endpoint: FloatArray,
    transform_mode: int,
    asinh_scale: float = 0.0,
    winding: Optional[FloatArray] = None,
) -> FloatArray:
    """Inverse of the chart-local analytical map.

    This is primarily a diagnostic/migration utility.  Search starts from a
    canonical target and applies the forward map, while exact correction always
    uses the original canonical target.
    """
    t = np.atleast_2d(np.asarray(transformed_endpoints, dtype=float))
    center = np.asarray(center_raw_endpoint, dtype=float)
    if int(transform_mode) == TRANSFORM_RAW:
        return t.copy()
    state = t[:, 2:6].copy()
    rho0 = math.exp(float(center[2]))
    state0 = np.array([rho0, 0.0, float(center[4]), float(center[5])], dtype=float)
    if float(asinh_scale) > 0.0:
        state = state0[None, :] + float(asinh_scale) * np.sinh(state)
    x, y, vx, vy = state.T
    rho = np.hypot(x, y)
    delta = np.arctan2(y, x)
    c = np.cos(delta)
    ss = np.sin(delta)
    ur = vx * c + vy * ss
    ut = -vx * ss + vy * c
    theta = float(center[3]) + delta
    if winding is not None:
        wind = np.asarray(winding, dtype=float).reshape(-1)
        theta += 2.0 * math.pi * (wind - np.rint(theta / (2.0 * math.pi)))
    return np.column_stack((t[:, 0], t[:, 1], np.log(np.maximum(rho, np.finfo(float).tiny)), theta, ur, ut, t[:, 6]))


def _local_transform_scale(
    center_raw_endpoint: FloatArray,
    raw_feature_scale: FloatArray,
    transform_mode: int,
    *,
    dynamic: bool,
    fallback_scale: FloatArray,
    asinh_scale: float,
    minimum_scale: float,
) -> FloatArray:
    raw_scale = np.asarray(raw_feature_scale, dtype=float)
    if int(transform_mode) == TRANSFORM_RAW:
        return np.maximum(raw_scale.copy(), float(minimum_scale))
    if not dynamic:
        return np.maximum(np.asarray(fallback_scale, dtype=float).copy(), float(minimum_scale))
    center = np.asarray(center_raw_endpoint, dtype=float)
    rho = math.exp(float(center[2]))
    ur = float(center[4])
    ut = float(center[5])
    dtheta = float(raw_scale[3])
    scale = np.array([
        raw_scale[0],
        raw_scale[1],
        rho * raw_scale[2],
        rho * dtheta,
        math.hypot(raw_scale[4], ut * dtheta),
        math.hypot(raw_scale[5], ur * dtheta),
        raw_scale[6],
    ], dtype=float)
    if float(asinh_scale) > 0.0:
        scale[2:6] /= float(asinh_scale)
    return np.maximum(scale, float(minimum_scale))


def _local_output_jacobian5(
    center_raw_endpoint: FloatArray,
    transform_mode: int,
    asinh_scale: float = 0.0,
) -> FloatArray:
    """Jacobian of transformed output5 wrt raw (logrho,theta,ur,ut,logkappa)."""
    if int(transform_mode) == TRANSFORM_RAW:
        return np.eye(5, dtype=float)
    center = np.asarray(center_raw_endpoint, dtype=float)
    rho = math.exp(float(center[2]))
    ur = float(center[4])
    ut = float(center[5])
    jac = np.array([
        [rho, 0.0, 0.0, 0.0, 0.0],
        [0.0, rho, 0.0, 0.0, 0.0],
        [0.0, -ut, 1.0, 0.0, 0.0],
        [0.0, ur, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ], dtype=float)
    if float(asinh_scale) > 0.0:
        jac[:4] /= float(asinh_scale)
    return jac


def _normalized_chart_delta(
    targets: FloatArray,
    center_raw_endpoint: FloatArray,
    transform_mode: int,
    transform_scale: FloatArray,
    asinh_scale: float = 0.0,
) -> FloatArray:
    target_t = _local_endpoint_transform(targets, center_raw_endpoint, transform_mode, asinh_scale)
    center_t = _local_endpoint_transform(
        np.asarray(center_raw_endpoint, dtype=float)[None, :], center_raw_endpoint, transform_mode, asinh_scale
    )[0]
    return (target_t - center_t[None, :]) / np.asarray(transform_scale, dtype=float)[None, :]


def _normalized_chart_delta_paired(
    targets: FloatArray,
    centers: FloatArray,
    transform_modes: NDArray[np.integer],
    transform_scales: FloatArray,
    asinh_scale: float = 0.0,
) -> FloatArray:
    """Vectorized one-target/one-center local transformed differences."""
    targets = np.asarray(targets, dtype=float)
    centers = np.asarray(centers, dtype=float)
    modes = np.asarray(transform_modes, dtype=np.int8)
    scales = np.asarray(transform_scales, dtype=float)
    z = np.empty_like(targets, dtype=float)
    raw_mask = modes == TRANSFORM_RAW
    if np.any(raw_mask):
        z[raw_mask] = (targets[raw_mask] - centers[raw_mask]) / scales[raw_mask]
    custom = ~raw_mask
    if np.any(custom):
        t = targets[custom]
        c0 = centers[custom]
        rho = np.exp(t[:, 2])
        rho0 = np.exp(c0[:, 2])
        delta = t[:, 3] - c0[:, 3]
        cc = np.cos(delta)
        ss = np.sin(delta)
        state = np.column_stack((
            rho * cc,
            rho * ss,
            t[:, 4] * cc - t[:, 5] * ss,
            t[:, 4] * ss + t[:, 5] * cc,
        ))
        state0 = np.column_stack((rho0, np.zeros_like(rho0), c0[:, 4], c0[:, 5]))
        if float(asinh_scale) > 0.0:
            state = np.arcsinh((state - state0) / float(asinh_scale))
            state0 = np.zeros_like(state0)
        delta_t = np.column_stack((t[:, 0] - c0[:, 0], t[:, 1] - c0[:, 1], state - state0, t[:, 6] - c0[:, 6]))
        z[custom] = delta_t / scales[custom]
    return z


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ChartBuildConfig:
    """Local inverse chart construction and validation settings."""

    quadratic_rank: int = 5
    fit_neighbours: int = 160
    minimum_fit_neighbours: int = 48
    fit_max_normalized_distance: float = 4.0
    fit_ridge: float = 2.0e-4
    quadratic_ridge: float = 1.0e-3
    fit_trim_quantile: float = 0.88
    exact_validation_points: int = 8
    minimum_exact_successes: int = 3
    local_target_success: float = 0.90
    trust_safety_factor: float = 0.88
    minimum_trust_radius: float = 0.025
    maximum_trust_radius: float = 3.0
    trust_endpoint_weight: float = 0.04
    jacobian_regularization: float = 1.0e-8
    maximum_jacobian_condition: float = 1.0e12
    reject_if_no_local_coverage: bool = False
    coefficient_clip: float = 25.0
    prediction_error_limit: float = 0.10
    prediction_target_success: float = 0.90
    require_center_self_test: bool = True
    minimum_local_training_success: float = 0.50
    minimum_local_training_tests: int = 2

    # Controlled same-branch local patch.  These points are generated by
    # perturbing the chart center itself, rather than borrowing unrelated
    # neighbours from the global forward pool.
    controlled_patch_enabled: bool = True
    patch_axis_steps: tuple[float, ...] = (0.02, 0.05, 0.10)
    patch_mixed_directions: int = 8
    patch_random_directions: int = 8
    patch_max_points: int = 64
    patch_min_points: int = 24
    patch_validation_fraction: float = 0.38
    patch_strict_branch: bool = True
    patch_use_pool_fallback: bool = False

    # Per-chart bounded-corrector portfolio.  The builder chooses the profile
    # with the largest independently validated trust radius, then stores that
    # profile in the chart for production queries and the viewer.
    corrector_profiles: tuple[str, ...] = (
        "newton_balanced",
        "newton_aggressive",
        "newton_damped",
        "trf_fast",
    )
    profile_wall_seconds: float = 0.45
    profile_trf_max_nfev: int = 10

    # Adaptive local-patch cascade.  The builder starts with the largest
    # connected patch and shrinks only when a fold/branch boundary prevents a
    # certifiable chart.
    adaptive_patch_scale_factors: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625)

    # Anisotropic endpoint-space trust ellipsoid.  ``trust_radii`` are stored
    # in the local orthonormal basis of each chart.
    minimum_directional_radius: float = 0.004
    maximum_directional_radius: float = 3.0
    trust_mixed_validation_points: int = 16

    # Compact fallback charts near folds.  They retain only the affine inverse
    # and a small ellipsoid when a regional quadratic chart cannot be certified.
    microcharts_enabled: bool = True
    microchart_patch_scale: float = 0.0625
    microchart_min_points: int = 10
    microchart_validation_points: int = 6
    microchart_minimum_successes: int = 2
    microchart_max_directional_radius: float = 0.18

    # Small fixed witness set used to prevent destructive refinement.
    witness_capacity: int = 8

    # Analytical endpoint transformation shared with the swarm dashboard.
    # New charts are fitted in local center-rotated Cartesian position/velocity
    # coordinates.  The canonical endpoint remains the external query format.
    output_transform: str = "custom_rotated_cartesian"
    output_transform_asinh_scale: float = 0.0
    dynamic_transform_scale: bool = True
    search_feature_scale: tuple[float, ...] = (0.08, 0.08, 0.12, 0.18, 0.22, 0.16, 0.18)
    minimum_transform_scale: float = 1.0e-7

    def validate(self) -> None:
        if not (0 <= self.quadratic_rank <= 7):
            raise ValueError("quadratic_rank must be between 0 and 7")
        if self.fit_neighbours < 16 or self.minimum_fit_neighbours < 8:
            raise ValueError("chart neighbour counts are too small")
        if self.minimum_fit_neighbours > self.fit_neighbours:
            raise ValueError("minimum_fit_neighbours exceeds fit_neighbours")
        if self.fit_max_normalized_distance <= 0.0:
            raise ValueError("fit_max_normalized_distance must be positive")
        if not (0.5 <= self.fit_trim_quantile <= 1.0):
            raise ValueError("fit_trim_quantile must be in [0.5,1]")
        if self.exact_validation_points < 0 or self.minimum_exact_successes < 0:
            raise ValueError("invalid exact validation counts")
        if not (0.0 < self.local_target_success <= 1.0):
            raise ValueError("local_target_success must be in (0,1]")
        if not (0.0 < self.trust_safety_factor <= 1.0):
            raise ValueError("trust_safety_factor must be in (0,1]")
        if self.minimum_trust_radius <= 0.0 or self.maximum_trust_radius <= 0.0:
            raise ValueError("trust radii must be positive")
        if self.minimum_trust_radius > self.maximum_trust_radius:
            raise ValueError("minimum trust radius exceeds maximum")
        if self.prediction_error_limit <= 0.0:
            raise ValueError("prediction_error_limit must be positive")
        if not (0.0 < self.prediction_target_success <= 1.0):
            raise ValueError("prediction_target_success must be in (0,1]")
        if not (0.0 <= self.minimum_local_training_success <= 1.0):
            raise ValueError("minimum_local_training_success must be in [0,1]")
        if self.minimum_local_training_tests < 0:
            raise ValueError("minimum_local_training_tests cannot be negative")
        _transform_mode_index(self.output_transform)
        if self.output_transform_asinh_scale < 0.0:
            raise ValueError("output_transform_asinh_scale cannot be negative")
        search_scale = np.asarray(self.search_feature_scale, dtype=float)
        if search_scale.shape != (7,) or np.any(~np.isfinite(search_scale)) or np.any(search_scale <= 0.0):
            raise ValueError("search_feature_scale must contain seven positive values")
        if self.minimum_transform_scale <= 0.0:
            raise ValueError("minimum_transform_scale must be positive")
        if not self.patch_axis_steps or any(float(v) <= 0.0 for v in self.patch_axis_steps):
            raise ValueError("patch_axis_steps must contain positive values")
        if self.patch_mixed_directions < 0 or self.patch_random_directions < 0:
            raise ValueError("patch direction counts cannot be negative")
        if self.patch_max_points < 8 or self.patch_min_points < 8:
            raise ValueError("controlled patch point limits are too small")
        if self.patch_min_points > self.patch_max_points:
            raise ValueError("patch_min_points exceeds patch_max_points")
        if not (0.1 <= self.patch_validation_fraction <= 0.8):
            raise ValueError("patch_validation_fraction must be in [0.1,0.8]")
        allowed_profiles = {"newton_balanced", "newton_aggressive", "newton_damped", "trf_fast"}
        if not self.corrector_profiles or any(v not in allowed_profiles for v in self.corrector_profiles):
            raise ValueError("corrector_profiles contains an unsupported profile")
        if self.profile_wall_seconds <= 0.0 or self.profile_trf_max_nfev <= 0:
            raise ValueError("corrector profile budgets must be positive")
        if not self.adaptive_patch_scale_factors or any(float(v) <= 0.0 for v in self.adaptive_patch_scale_factors):
            raise ValueError("adaptive_patch_scale_factors must be positive")
        if self.minimum_directional_radius <= 0.0 or self.maximum_directional_radius <= 0.0:
            raise ValueError("directional trust radii must be positive")
        if self.minimum_directional_radius > self.maximum_directional_radius:
            raise ValueError("minimum directional radius exceeds maximum")
        if self.trust_mixed_validation_points < 0:
            raise ValueError("trust_mixed_validation_points cannot be negative")
        if self.microchart_patch_scale <= 0.0 or self.microchart_min_points < 6:
            raise ValueError("invalid microchart patch settings")
        if self.microchart_validation_points < 2 or self.microchart_minimum_successes < 1:
            raise ValueError("invalid microchart validation settings")
        if self.microchart_minimum_successes > self.microchart_validation_points:
            raise ValueError("microchart minimum successes exceed validation points")
        if self.microchart_max_directional_radius <= 0.0:
            raise ValueError("microchart_max_directional_radius must be positive")
        if self.witness_capacity < 1 or self.witness_capacity > 64:
            raise ValueError("witness_capacity must be between 1 and 64")


@dataclass
class GeneratorConfig:
    """Long-running adaptive chart discovery settings."""

    max_launch_attempts: int = 20_000_000
    launch_batch_size: int = 32_768
    max_rounds: int = 10_000
    max_charts: int = 350_000
    initial_charts_per_round: int = 1024
    charts_per_round: int = 1024
    chart_build_batch: int = 64
    chart_workers: int = max(1, (os.cpu_count() or 2) - 1)
    refine_charts_per_round: int = 128
    refine_after_round: int = 3
    maximum_refinements_per_chart: int = 3
    refinement_min_radius_gain: float = 1.15

    fit_pool_max_rows: int = 500_000
    fit_pool_cell_size: float = 0.10
    candidate_cell_size: float = 0.055
    candidate_min_separation: float = 0.12
    candidate_audit_rows: int = 256
    failed_probe_centers_per_round: int = 1024
    bootstrap_bulk_centers: int = 0
    bulk_fallback_centers_per_round: int = 64
    require_probe_repair: bool = True

    validation_reservoir_rows: int = 12_288
    validation_add_per_round: int = 768
    validation_test_per_round: int = 2048
    fresh_validation_rows: int = 512
    target_success: float = 0.97
    fresh_target_success: float = 0.94
    target_p95_seconds: float = 0.50
    patience: int = 5
    minimum_rounds_before_stop: int = 8
    diagnostic_ungated_rows: int = 32
    diagnostic_routing_neighbours: int = 256
    diagnostic_center_self_tests: int = 32
    diagnostic_local_training_tests: int = 32

    # Fixed repair epochs: grow the target population only after repeatedly
    # repairing the current feasible reservoir.
    repair_epoch_target_success: float = 0.85
    repair_epoch_max_passes: int = 8
    repair_epoch_min_new_charts: int = 4

    # Strictly independent validation populations.  Holdout probes are never
    # used as chart centers; fresh probes are generated after construction.
    holdout_reservoir_rows: int = 8192
    holdout_add_per_epoch: int = 256
    fresh_audit_attempt_factor: int = 4

    # Witness-preserving refinement defaults to append-only, avoiding loss of
    # previously demonstrated coverage.
    refinement_append_only: bool = True

    checkpoint_enabled: bool = True
    checkpoint_every_chart_batches: int = 1
    checkpoint_compressed: bool = True
    checkpoint_on_interrupt: bool = True

    seed: int = 20260720

    def validate(self) -> None:
        for name in (
            "max_launch_attempts",
            "launch_batch_size",
            "max_rounds",
            "max_charts",
            "initial_charts_per_round",
            "charts_per_round",
            "chart_build_batch",
            "chart_workers",
            "fit_pool_max_rows",
            "validation_reservoir_rows",
            "validation_test_per_round",
            "fresh_validation_rows",
            "patience",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.candidate_audit_rows < 0 or self.validation_add_per_round < 0:
            raise ValueError("audit and validation additions cannot be negative")
        for name in (
            "failed_probe_centers_per_round", "bootstrap_bulk_centers",
            "bulk_fallback_centers_per_round", "diagnostic_ungated_rows",
            "diagnostic_center_self_tests", "diagnostic_local_training_tests",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.diagnostic_routing_neighbours <= 0:
            raise ValueError("diagnostic_routing_neighbours must be positive")
        if not (0.0 < self.repair_epoch_target_success <= 1.0):
            raise ValueError("repair_epoch_target_success must be in (0,1]")
        if self.repair_epoch_max_passes <= 0 or self.repair_epoch_min_new_charts < 0:
            raise ValueError("invalid repair epoch settings")
        if self.holdout_reservoir_rows <= 0 or self.holdout_add_per_epoch < 0:
            raise ValueError("invalid holdout reservoir settings")
        if self.fresh_audit_attempt_factor <= 0:
            raise ValueError("fresh_audit_attempt_factor must be positive")
        if self.refine_charts_per_round < 0 or self.refine_after_round < 0:
            raise ValueError("chart refinement counts cannot be negative")
        if self.maximum_refinements_per_chart < 0 or self.refinement_min_radius_gain < 1.0:
            raise ValueError("invalid chart refinement settings")
        if self.fit_pool_cell_size <= 0.0 or self.candidate_cell_size <= 0.0:
            raise ValueError("cell sizes must be positive")
        if self.candidate_min_separation <= 0.0:
            raise ValueError("candidate_min_separation must be positive")
        if not (0.0 < self.target_success <= 1.0):
            raise ValueError("target_success must be in (0,1]")
        if not (0.0 < self.fresh_target_success <= 1.0):
            raise ValueError("fresh_target_success must be in (0,1]")
        if self.target_p95_seconds <= 0.0:
            raise ValueError("target_p95_seconds must be positive")


@dataclass
class ChartAtlasConfig:
    """Complete chart-atlas configuration."""

    backend_config: dict[str, Any] = field(default_factory=dict)
    query_config: dict[str, Any] = field(default_factory=dict)
    chart: ChartBuildConfig = field(default_factory=ChartBuildConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    feature_scale: tuple[float, ...] = (0.08, 0.08, 0.12, 0.18, 0.12, 0.12, 0.18)
    q_scale: tuple[float, ...] = (16.0, 16.0, 64.0, 64.0, 2.5)
    routing_neighbours: int = 48
    routing_max_ratio: float = 1.0
    routing_winding_window: int = 1

    def validate(self) -> None:
        self.chart.validate()
        self.generator.validate()
        fs = np.asarray(self.feature_scale, dtype=float)
        qs = np.asarray(self.q_scale, dtype=float)
        if fs.shape != (7,) or np.any(~np.isfinite(fs)) or np.any(fs <= 0.0):
            raise ValueError("feature_scale must contain seven positive values")
        if qs.shape != (5,) or np.any(~np.isfinite(qs)) or np.any(qs <= 0.0):
            raise ValueError("q_scale must contain five positive values")
        if self.routing_neighbours <= 0 or self.routing_max_ratio <= 0.0:
            raise ValueError("invalid routing settings")
        if self.routing_winding_window < 0:
            raise ValueError("routing_winding_window cannot be negative")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChartAtlasConfig":
        data = dict(payload)
        chart_data = dict(data.get("chart", {}))
        if "patch_axis_steps" in chart_data:
            chart_data["patch_axis_steps"] = tuple(chart_data["patch_axis_steps"])
        if "corrector_profiles" in chart_data:
            chart_data["corrector_profiles"] = tuple(chart_data["corrector_profiles"])
        if "adaptive_patch_scale_factors" in chart_data:
            chart_data["adaptive_patch_scale_factors"] = tuple(chart_data["adaptive_patch_scale_factors"])
        if "search_feature_scale" in chart_data:
            chart_data["search_feature_scale"] = tuple(chart_data["search_feature_scale"])
        data["chart"] = ChartBuildConfig(**chart_data)
        data["generator"] = GeneratorConfig(**data.get("generator", {}))
        if "feature_scale" in data:
            data["feature_scale"] = tuple(data["feature_scale"])
        if "q_scale" in data:
            data["q_scale"] = tuple(data["q_scale"])
        cfg = cls(**data)
        cfg.validate()
        return cfg


# ---------------------------------------------------------------------------
# Dynamic backend loading
# ---------------------------------------------------------------------------


def load_backend(path: str | Path):
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    name = f"pmp_chart_backend_{hashlib.sha1(str(path).encode()).hexdigest()[:12]}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import dynamics backend from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _query_config(backend, payload: dict[str, Any]):
    fields = {f.name for f in dataclasses.fields(backend.QueryConfig)}
    return backend.QueryConfig(**{k: v for k, v in payload.items() if k in fields})


def _backend_config(backend, payload: dict[str, Any]):
    return backend._atlas_config_from_dict(payload)


CORRECTOR_PROFILE_NAMES = (
    "newton_balanced",
    "newton_aggressive",
    "newton_damped",
    "trf_fast",
)


def _corrector_profile_config(base, profile: str, chart_cfg: ChartBuildConfig):
    """Return one bounded production-corrector variant.

    All profiles share the same physical integration tolerances and acceptance
    thresholds.  Only the nonlinear correction strategy/budget changes.
    """
    wall = min(float(base.wall_time_seconds), float(chart_cfg.profile_wall_seconds))
    if profile == "newton_balanced":
        return replace(
            base,
            method="fast_newton",
            max_iterations=max(3, int(base.max_iterations)),
            line_search_steps=max(2, int(base.line_search_steps)),
            step_limit=float(base.step_limit),
            regularization=max(float(base.regularization), 1.0e-10),
            wall_time_seconds=wall,
            robust_fallback=False,
        )
    if profile == "newton_aggressive":
        return replace(
            base,
            method="fast_newton",
            max_iterations=max(5, int(base.max_iterations) + 2),
            line_search_steps=max(3, int(base.line_search_steps) + 1),
            step_limit=max(0.45, float(base.step_limit)),
            regularization=min(max(float(base.regularization), 1.0e-12), 1.0e-8),
            wall_time_seconds=wall,
            robust_fallback=False,
        )
    if profile == "newton_damped":
        return replace(
            base,
            method="fast_newton",
            max_iterations=max(5, int(base.max_iterations) + 2),
            line_search_steps=max(4, int(base.line_search_steps) + 2),
            step_limit=min(0.18, float(base.step_limit)),
            regularization=max(1.0e-5, float(base.regularization)),
            wall_time_seconds=wall,
            robust_fallback=False,
        )
    if profile == "trf_fast":
        return replace(
            base,
            method="robust_least_squares",
            max_nfev=min(max(4, int(chart_cfg.profile_trf_max_nfev)), 20),
            wall_time_seconds=wall,
            robust_fallback=False,
        )
    raise ValueError(f"unsupported corrector profile {profile!r}")


def _corrector_profile_index(profile: str, chart_cfg: ChartBuildConfig) -> int:
    del chart_cfg  # stored IDs use the stable canonical profile ordering
    try:
        return CORRECTOR_PROFILE_NAMES.index(profile)
    except ValueError as exc:
        raise ValueError(f"unknown corrector profile {profile!r}") from exc


# ---------------------------------------------------------------------------
# Deterministic chunked Sobol launch sampling
# ---------------------------------------------------------------------------


def _signed_log_sample(unit: FloatArray, bounds: tuple[float, float]) -> FloatArray:
    lo, hi = map(float, bounds)
    centered = 2.0 * np.asarray(unit, dtype=float) - 1.0
    sign = np.sign(centered)
    magnitude_u = np.abs(centered)
    magnitude = np.exp(math.log(lo) + magnitude_u * (math.log(hi) - math.log(lo)))
    # Preserve a narrow near-zero region around the Sobol midpoint.
    magnitude[magnitude_u < 1.0e-6] = 0.0
    return sign * magnitude


def _default_magnitude_bounds(component_bounds: tuple[float, float], floor: float) -> tuple[float, float]:
    hi = max(abs(float(component_bounds[0])), abs(float(component_bounds[1])))
    return (min(max(floor, hi * 1.0e-6), hi * 0.1), max(hi, floor * 10.0))


def sample_launch_chunk(backend_cfg, cursor: int, count: int, seed: int) -> FloatArray:
    """Generate the exact Sobol subsequence ``[cursor,cursor+count)``."""
    if count <= 0:
        return np.empty((0, 7), dtype=float)
    shards = tuple(getattr(backend_cfg, "bulk_shards", ()))
    dimension = 8 if shards else 7
    sampler = qmc.Sobol(d=dimension, scramble=True, seed=int(seed))
    if cursor:
        sampler.fast_forward(int(cursor))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        unit = sampler.random(int(count))

    if not shards:
        bounds = np.asarray(
            [
                backend_cfg.u0_bounds,
                backend_cfg.w0_bounds,
                backend_cfg.ar0_bounds,
                backend_cfg.at0_bounds,
                backend_cfg.jr0_bounds,
                backend_cfg.ell_bounds,
            ],
            dtype=float,
        )
        launch = np.empty((count, 7), dtype=float)
        launch[:, :6] = qmc.scale(unit[:, :6], bounds[:, 0], bounds[:, 1])
        lo, hi = np.log(backend_cfg.tau_bounds)
        launch[:, 6] = np.exp(lo + unit[:, 6] * (hi - lo))
        return launch

    weights = np.asarray([s.weight for s in shards], dtype=float)
    cumulative = np.cumsum(weights / np.sum(weights))
    shard_index = np.searchsorted(cumulative, unit[:, 0], side="right")
    shard_index = np.minimum(shard_index, len(shards) - 1)
    launch = np.empty((count, 7), dtype=float)

    for index, shard in enumerate(shards):
        mask = shard_index == index
        if not np.any(mask):
            continue
        u = unit[mask, 1:]
        ub = backend_cfg.u0_bounds if shard.u0_bounds is None else shard.u0_bounds
        wb = backend_cfg.w0_bounds if shard.w0_bounds is None else shard.w0_bounds
        tb = backend_cfg.tau_bounds if shard.tau_bounds is None else shard.tau_bounds
        launch[mask, 0] = ub[0] + u[:, 0] * (ub[1] - ub[0])
        launch[mask, 1] = wb[0] + u[:, 1] * (wb[1] - wb[0])

        if shard.acceleration_sampling == "cartesian_uniform":
            launch[mask, 2] = backend_cfg.ar0_bounds[0] + u[:, 2] * (
                backend_cfg.ar0_bounds[1] - backend_cfg.ar0_bounds[0]
            )
            launch[mask, 3] = backend_cfg.at0_bounds[0] + u[:, 3] * (
                backend_cfg.at0_bounds[1] - backend_cfg.at0_bounds[0]
            )
        else:
            mag_bounds = shard.acceleration_magnitude_bounds or _default_magnitude_bounds(
                (
                    min(backend_cfg.ar0_bounds[0], backend_cfg.at0_bounds[0]),
                    max(backend_cfg.ar0_bounds[1], backend_cfg.at0_bounds[1]),
                ),
                1.0e-5,
            )
            magnitude = np.exp(
                math.log(mag_bounds[0])
                + u[:, 2] * (math.log(mag_bounds[1]) - math.log(mag_bounds[0]))
            )
            angle = 2.0 * math.pi * u[:, 3]
            launch[mask, 2] = np.clip(magnitude * np.cos(angle), *backend_cfg.ar0_bounds)
            launch[mask, 3] = np.clip(magnitude * np.sin(angle), *backend_cfg.at0_bounds)

        if shard.jr_sampling == "linear":
            launch[mask, 4] = backend_cfg.jr0_bounds[0] + u[:, 4] * (
                backend_cfg.jr0_bounds[1] - backend_cfg.jr0_bounds[0]
            )
        else:
            bounds = shard.jr_magnitude_bounds or _default_magnitude_bounds(
                backend_cfg.jr0_bounds, 1.0e-5
            )
            launch[mask, 4] = np.clip(_signed_log_sample(u[:, 4], bounds), *backend_cfg.jr0_bounds)

        if shard.ell_sampling == "linear":
            launch[mask, 5] = backend_cfg.ell_bounds[0] + u[:, 5] * (
                backend_cfg.ell_bounds[1] - backend_cfg.ell_bounds[0]
            )
        else:
            bounds = shard.ell_magnitude_bounds or _default_magnitude_bounds(
                backend_cfg.ell_bounds, 1.0e-5
            )
            launch[mask, 5] = np.clip(_signed_log_sample(u[:, 5], bounds), *backend_cfg.ell_bounds)

        lo, hi = np.log(tb)
        launch[mask, 6] = np.exp(lo + u[:, 6] * (hi - lo))
    return launch


# ---------------------------------------------------------------------------
# Chart representation and fitting utilities
# ---------------------------------------------------------------------------


def launch_to_q(launch: FloatArray) -> FloatArray:
    launch = np.asarray(launch, dtype=float)
    if launch.ndim == 1:
        return np.concatenate((launch[2:6], [math.log(float(launch[6]))]))
    return np.column_stack((launch[:, 2:6], np.log(launch[:, 6])))


def q_to_launch(
    u0: float,
    w0: float,
    q: FloatArray,
    bounds: Optional[tuple[FloatArray, FloatArray]] = None,
    *,
    clip: bool = False,
) -> FloatArray:
    """Convert the five shooting variables to a seven-column launch vector.

    Local quadratic charts are only valid inside their empirical trust regions.
    Outside those regions they can occasionally predict a non-finite value or an
    enormous ``log(tau)``.  Such a prediction must be rejected (or explicitly
    clipped), rather than allowing ``exp(log(tau))`` to raise ``OverflowError``
    and terminate a long atlas-generation run.
    """
    q = np.asarray(q, dtype=float).copy()
    if q.shape != (5,) or not np.all(np.isfinite(q)):
        raise ValueError("non-finite or malformed chart launch prediction")

    if bounds is not None:
        lo = np.asarray(bounds[0], dtype=float)
        hi = np.asarray(bounds[1], dtype=float)
        if lo.shape != (5,) or hi.shape != (5,):
            raise ValueError("shooting bounds must each contain five values")
        if clip:
            q = np.clip(q, lo, hi)
        elif np.any(q < lo) or np.any(q > hi):
            raise ValueError("chart launch prediction lies outside configured bounds")

    # The configured bounds normally make this redundant, but retain an
    # explicit floating-point guard for callers that do not supply bounds.
    log_tau = float(q[4])
    max_log = math.log(np.finfo(float).max)
    min_log = math.log(np.finfo(float).tiny)
    if not (min_log <= log_tau <= max_log):
        raise ValueError("chart prediction produced an unrepresentable flight time")
    tau = math.exp(log_tau)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("chart prediction produced an invalid flight time")

    return np.array([u0, w0, q[0], q[1], q[2], q[3], tau], dtype=float)


def branch_labels(diagnostics: FloatArray) -> NDArray[np.int16]:
    if len(diagnostics) == 0:
        return np.empty((0, 2), dtype=np.int16)
    return np.rint(np.asarray(diagnostics)[:, 4:6]).astype(np.int16)


def _regularized_inverse(jac: FloatArray, regularization: float) -> tuple[Optional[FloatArray], float, float]:
    try:
        u, s, vt = np.linalg.svd(np.asarray(jac, dtype=float), full_matrices=False)
    except np.linalg.LinAlgError:
        return None, math.inf, 0.0
    if len(s) != 5 or not np.all(np.isfinite(s)) or s[-1] <= 0.0:
        return None, math.inf, 0.0
    inv = (vt.T * (s / (s * s + max(regularization, 0.0)))) @ u.T
    return inv, float(s[0] / s[-1]), float(s[-1])


def _quadratic_pairs(dimension: int = 7) -> list[tuple[int, int]]:
    return [(i, j) for i in range(dimension) for j in range(i, dimension)]


_QUAD_PAIRS = _quadratic_pairs(7)


def _full_quadratic_features(z: FloatArray) -> FloatArray:
    z = np.asarray(z, dtype=float)
    output = np.empty((len(z), len(_QUAD_PAIRS)), dtype=float)
    for column, (i, j) in enumerate(_QUAD_PAIRS):
        output[:, column] = z[:, i] * z[:, j] * (2.0 if i != j else 1.0)
    return output


def _coefficient_to_hessians(coef: FloatArray) -> FloatArray:
    # coef shape (28,5), prediction residual = Phi @ coef
    h = np.zeros((5, 7, 7), dtype=float)
    for column, (i, j) in enumerate(_QUAD_PAIRS):
        h[:, i, j] = coef[column]
        h[:, j, i] = coef[column]
    return h


def _fit_low_rank_quadratic(
    z: FloatArray,
    residual: FloatArray,
    weights: FloatArray,
    rank: int,
    ridge: float,
    coefficient_clip: float,
) -> tuple[FloatArray, FloatArray]:
    if rank <= 0 or len(z) < 20:
        return np.zeros((0, 7), dtype=float), np.zeros((5, 0), dtype=float)
    phi = _full_quadratic_features(z)
    sw = np.sqrt(np.maximum(weights, 1.0e-12))
    aw = phi * sw[:, None]
    bw = residual * sw[:, None]
    lhs = aw.T @ aw + ridge * np.eye(aw.shape[1])
    try:
        coef = np.linalg.solve(lhs, aw.T @ bw)
    except np.linalg.LinAlgError:
        return np.zeros((0, 7), dtype=float), np.zeros((5, 0), dtype=float)
    hessians = _coefficient_to_hessians(coef)
    aggregate = np.zeros((7, 7), dtype=float)
    for h in hessians:
        aggregate += h.T @ h
    try:
        eigenvalue, eigenvector = np.linalg.eigh(aggregate)
    except np.linalg.LinAlgError:
        return np.zeros((0, 7), dtype=float), np.zeros((5, 0), dtype=float)
    order = np.argsort(eigenvalue)[::-1]
    directions = eigenvector[:, order[:rank]].T
    basis = (z @ directions.T) ** 2
    aw = basis * sw[:, None]
    lhs = aw.T @ aw + ridge * np.eye(len(directions))
    try:
        coeff = np.linalg.solve(lhs, aw.T @ bw).T
    except np.linalg.LinAlgError:
        return np.zeros((0, 7), dtype=float), np.zeros((5, 0), dtype=float)
    coeff = np.clip(coeff, -coefficient_clip, coefficient_clip)
    return directions, coeff


def chart_predict_q_normalized(
    target: FloatArray,
    center_endpoint: FloatArray,
    center_launch: FloatArray,
    affine: FloatArray,
    quad_dirs: FloatArray,
    quad_coeff: FloatArray,
    transform_mode: int,
    transform_scale: FloatArray,
    q_scale: FloatArray,
    trust_endpoint_weight: float = 0.04,
    transform_asinh_scale: float = 0.0,
) -> tuple[FloatArray, FloatArray, float]:
    z = _normalized_chart_delta(
        np.asarray(target, dtype=float)[None, :],
        np.asarray(center_endpoint, dtype=float),
        int(transform_mode),
        np.asarray(transform_scale, dtype=float),
        float(transform_asinh_scale),
    )[0]
    y = np.asarray(affine, dtype=float) @ z
    if len(quad_dirs):
        basis = (np.asarray(quad_dirs, dtype=float) @ z) ** 2
        y = y + np.asarray(quad_coeff, dtype=float) @ basis
    score = math.sqrt(
        float(np.dot(y, y)) + float(trust_endpoint_weight) * float(np.dot(z, z))
    )
    q = launch_to_q(center_launch) + np.asarray(q_scale, dtype=float) * y
    return q, z, score


def _wilson_lower(successes: int, total: int, z: float = 1.645) -> float:
    if total <= 0:
        return 0.0
    p = successes / total
    denom = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return (center - radius) / denom


def _validated_radius(
    scores: FloatArray,
    success: NDArray[np.bool_],
    cfg: ChartBuildConfig,
) -> float:
    if len(scores) == 0:
        return cfg.minimum_trust_radius
    order = np.argsort(scores)
    s = np.asarray(scores, dtype=float)[order]
    ok = np.asarray(success, dtype=bool)[order]
    cumulative = np.cumsum(ok)
    best = 0.0
    for i in range(len(s)):
        total = i + 1
        successes = int(cumulative[i])
        if successes < cfg.minimum_exact_successes:
            continue
        if _wilson_lower(successes, total) >= cfg.local_target_success:
            best = float(s[i])
    if best <= 0.0:
        # A conservative point chart remains useful for exact/very-near queries.
        positive_success = s[ok & (s > 0.0)]
        if len(positive_success):
            best = float(np.min(positive_success)) * 0.5
        else:
            best = cfg.minimum_trust_radius
    return float(np.clip(best * cfg.trust_safety_factor, cfg.minimum_trust_radius, cfg.maximum_trust_radius))


def _fit_trust_ellipsoid(
    z_points: FloatArray,
    success: NDArray[np.bool_],
    cfg: ChartBuildConfig,
    *,
    max_radius: Optional[float] = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Fit a conservative anisotropic trust ellipsoid in normalized endpoint space.

    Returns ``basis, radii, ratios`` where rows of ``basis`` are orthonormal
    local directions and ``ratios`` are the validation Mahalanobis radii.
    """
    z = np.asarray(z_points, dtype=float)
    ok = np.asarray(success, dtype=bool)
    if z.ndim != 2 or z.shape[1] != 7 or len(z) == 0:
        return np.eye(7), np.full(7, cfg.minimum_directional_radius), np.full(len(z), np.inf)
    finite = np.all(np.isfinite(z), axis=1)
    z = z[finite]
    ok = ok[finite]
    if len(z) == 0:
        return np.eye(7), np.full(7, cfg.minimum_directional_radius), np.empty(0)
    try:
        _u, _s, vt = np.linalg.svd(z, full_matrices=True)
        basis = np.asarray(vt, dtype=float)
    except np.linalg.LinAlgError:
        basis = np.eye(7)
    coords = z @ basis.T
    upper = float(max_radius if max_radius is not None else cfg.maximum_directional_radius)
    radii = np.full(7, cfg.minimum_directional_radius, dtype=float)
    for j in range(7):
        succ = np.abs(coords[ok, j])
        fail = np.abs(coords[~ok, j])
        succ = succ[np.isfinite(succ) & (succ > 1.0e-12)]
        fail = fail[np.isfinite(fail) & (fail > 1.0e-12)]
        if len(succ):
            r = float(np.quantile(succ, 0.90 if len(succ) >= 4 else 1.0))
        else:
            r = cfg.minimum_directional_radius
        if len(fail):
            outside = fail[fail >= max(0.5 * r, cfg.minimum_directional_radius)]
            if len(outside):
                r = min(r, 0.90 * float(np.min(outside)))
        radii[j] = np.clip(r, cfg.minimum_directional_radius, upper)
    raw_ratio = np.sqrt(np.sum((coords / np.maximum(radii, 1.0e-12)) ** 2, axis=1))
    order = np.argsort(raw_ratio)
    best = 0.0
    cumulative = 0
    for n, idx in enumerate(order, start=1):
        cumulative += int(ok[idx])
        if cumulative < cfg.minimum_exact_successes:
            continue
        if _wilson_lower(cumulative, n) >= cfg.local_target_success:
            best = float(raw_ratio[idx])
    if best <= 0.0:
        successful = raw_ratio[ok & np.isfinite(raw_ratio)]
        best = float(np.min(successful)) if len(successful) else 0.0
    scale = max(0.0, best * cfg.trust_safety_factor)
    radii = np.clip(radii * scale, cfg.minimum_directional_radius, upper)
    ratios = np.sqrt(np.sum((coords / np.maximum(radii, 1.0e-12)) ** 2, axis=1))
    return basis, radii, ratios


def _ellipsoid_ratio(
    target: FloatArray,
    center: FloatArray,
    transform_mode: int,
    transform_scale: FloatArray,
    basis: FloatArray,
    radii: FloatArray,
    asinh_scale: float = 0.0,
) -> float:
    z = _normalized_chart_delta(
        np.asarray(target, dtype=float)[None, :],
        np.asarray(center, dtype=float),
        int(transform_mode),
        np.asarray(transform_scale, dtype=float),
        float(asinh_scale),
    )[0]
    xi = np.asarray(basis, dtype=float) @ z
    return float(np.sqrt(np.sum((xi / np.maximum(np.asarray(radii, dtype=float), 1.0e-12)) ** 2)))


@dataclass
class BuiltChart:
    launch: FloatArray
    endpoint: FloatArray
    diagnostics: FloatArray
    affine: FloatArray
    quad_dirs: FloatArray
    quad_coeff: FloatArray
    trust_radius: float
    local_successes: int
    local_failures: int
    local_p95_seconds: float
    jacobian_condition: float
    jacobian_sigma_min: float
    solver_method: int = 0
    patch_points: int = 0
    trust_basis: FloatArray = field(default_factory=lambda: np.eye(7, dtype=float))
    trust_radii: FloatArray = field(default_factory=lambda: np.ones(7, dtype=float))
    chart_type: int = 0  # 0 quadratic, 1 affine, 2 micro
    patch_scale: float = 1.0
    transform_mode: int = TRANSFORM_CUSTOM_ROTATED_CARTESIAN
    transform_scale: FloatArray = field(default_factory=lambda: np.ones(7, dtype=float))
    witnesses: FloatArray = field(default_factory=lambda: np.empty((0, 7), dtype=float))


# Worker globals for parallel chart fitting.
_WORKER_BACKEND = None
_WORKER_BACKEND_CFG = None
_WORKER_QUERY_CFG = None
_WORKER_CHART_CFG = None
_WORKER_FEATURE_SCALE = None
_WORKER_Q_SCALE = None
_WORKER_BOUNDS = None


def _chart_worker_init(
    solver_path: str,
    backend_config_dict: dict[str, Any],
    query_config_dict: dict[str, Any],
    chart_config_dict: dict[str, Any],
    feature_scale: Sequence[float],
    q_scale: Sequence[float],
) -> None:
    global _WORKER_BACKEND, _WORKER_BACKEND_CFG, _WORKER_QUERY_CFG
    global _WORKER_CHART_CFG, _WORKER_FEATURE_SCALE, _WORKER_Q_SCALE, _WORKER_BOUNDS
    _WORKER_BACKEND = load_backend(solver_path)
    _WORKER_BACKEND_CFG = _backend_config(_WORKER_BACKEND, backend_config_dict)
    # Prevent nested generation process pools inside chart workers.
    _WORKER_BACKEND_CFG.workers = 1
    _WORKER_QUERY_CFG = _query_config(_WORKER_BACKEND, query_config_dict)
    _WORKER_CHART_CFG = ChartBuildConfig(**chart_config_dict)
    _WORKER_FEATURE_SCALE = np.asarray(feature_scale, dtype=float)
    _WORKER_Q_SCALE = np.asarray(q_scale, dtype=float)
    _WORKER_BOUNDS = _WORKER_BACKEND._configured_correction_bounds(_WORKER_BACKEND_CFG)


def _build_chart_worker(payload: tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, int]):
    center_launch, center_endpoint, center_diag, neighbour_launch, neighbour_endpoint, neighbour_diag, seed = payload
    return _build_chart(
        _WORKER_BACKEND,
        _WORKER_BACKEND_CFG,
        _WORKER_QUERY_CFG,
        _WORKER_CHART_CFG,
        _WORKER_FEATURE_SCALE,
        _WORKER_Q_SCALE,
        _WORKER_BOUNDS,
        center_launch,
        center_endpoint,
        center_diag,
        neighbour_launch,
        neighbour_endpoint,
        neighbour_diag,
        int(seed),
    )


def _exact_endpoint_jacobian(backend, launch: FloatArray, query_cfg) -> tuple[Optional[FloatArray], Optional[FloatArray], float, float]:
    """Accurate chart center and d(output5)/d(q5) at production tolerances."""
    yf, sens, normal_constant, status = backend._integrate_launch_with_sens(
        np.asarray(launch, dtype=float),
        rtol=query_cfg.rtol,
        atol=query_cfg.atol,
        max_step=query_cfg.max_step,
        r_collision=query_cfg.r_collision,
        r_escape=query_cfg.r_escape,
        max_acceleration=query_cfg.max_acceleration,
    )
    if yf is None or sens is None or status != "ok" or normal_constant <= 0.0:
        return None, None, math.inf, 0.0
    radius, theta, ur, ut, kappa = backend._endpoint_from_state(yf)
    if radius <= 0.0 or kappa <= 0.0:
        return None, None, math.inf, 0.0
    endpoint = np.array(
        [launch[0], launch[1], math.log(radius), theta, ur, ut, math.log(kappa)],
        dtype=float,
    )
    try:
        output_jac = backend._endpoint_output_jacobian(yf)
        jac = np.empty((5, 5), dtype=float)
        jac[:, :4] = output_jac @ sens
        jac[:, 4] = output_jac @ (backend._rhs(float(launch[6]), yf) * float(launch[6]))
        singular = np.linalg.svd(jac, compute_uv=False)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return None, None, math.inf, 0.0
    if not np.all(np.isfinite(jac)) or singular[-1] <= 0.0:
        return None, None, math.inf, 0.0
    return endpoint, jac, float(singular[0] / singular[-1]), float(singular[-1])


def _exact_endpoint(backend, launch: FloatArray, query_cfg) -> Optional[FloatArray]:
    sol, normal_constant, status = backend._integrate_launch(
        np.asarray(launch, dtype=float),
        rtol=query_cfg.rtol,
        atol=query_cfg.atol,
        max_step=query_cfg.max_step,
        r_collision=query_cfg.r_collision,
        r_escape=query_cfg.r_escape,
        max_acceleration=query_cfg.max_acceleration,
    )
    if sol is None or status != "ok" or normal_constant <= 0.0:
        return None
    radius, theta, ur, ut, kappa = backend._endpoint_from_state(sol.y[:, -1])
    if radius <= 0.0 or kappa <= 0.0:
        return None
    return np.array(
        [launch[0], launch[1], math.log(radius), theta, ur, ut, math.log(kappa)],
        dtype=float,
    )


def _exact_endpoint_and_diagnostics(
    backend,
    launch: FloatArray,
    query_cfg,
) -> tuple[Optional[FloatArray], Optional[FloatArray]]:
    """Accurate endpoint plus branch/guard diagnostics for a local patch point."""
    sol, normal_constant, status = backend._integrate_launch(
        np.asarray(launch, dtype=float),
        rtol=query_cfg.rtol,
        atol=query_cfg.atol,
        max_step=query_cfg.max_step,
        r_collision=query_cfg.r_collision,
        r_escape=query_cfg.r_escape,
        max_acceleration=query_cfg.max_acceleration,
    )
    if sol is None or status != "ok" or normal_constant <= 0.0:
        return None, None
    states = np.asarray(sol.y, dtype=float)
    yf = states[:, -1]
    radius, theta, ur, ut, kappa = backend._endpoint_from_state(yf)
    if radius <= 0.0 or kappa <= 0.0:
        return None, None
    endpoint = np.array(
        [launch[0], launch[1], math.log(radius), theta, ur, ut, math.log(kappa)],
        dtype=float,
    )
    radii = np.linalg.norm(states[0:2, :], axis=0)
    acceleration = np.linalg.norm(states[4:6, :], axis=0)
    diagnostics = np.array(
        [
            float(normal_constant),
            float(np.min(radii)),
            float(np.max(radii)),
            float(np.max(acceleration)),
            float(backend._count_radial_turns(states)),
            float(round(theta / (2.0 * math.pi))),
        ],
        dtype=float,
    )
    return endpoint, diagnostics


def _controlled_local_patch(
    backend,
    center_launch: FloatArray,
    center_endpoint: FloatArray,
    center_diag: FloatArray,
    jac: FloatArray,
    query_cfg,
    chart_cfg: ChartBuildConfig,
    feature_scale: FloatArray,
    q_scale: FloatArray,
    bounds: tuple[FloatArray, FloatArray],
    seed: int,
    patch_scale: float = 1.0,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Generate a connected, same-branch local experimental design.

    The first two normalized launch coordinates perturb ``u0`` and ``w0``.
    The remaining five coordinates perturb the shooting vector along right
    singular vectors of the normalized forward Jacobian.  Mixed and random
    directions identify important quadratic couplings without borrowing points
    from unrelated inverse branches.
    """
    if not chart_cfg.controlled_patch_enabled:
        return np.empty((0, 7)), np.empty((0, 7)), np.empty((0, 6))
    fs = np.asarray(feature_scale, dtype=float)
    qs = np.asarray(q_scale, dtype=float)
    q0 = launch_to_q(center_launch)
    lo, hi = (np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float))

    normalized_jac = (
        np.diag(1.0 / fs[2:7])
        @ np.asarray(jac, dtype=float)
        @ np.diag(qs)
    )
    try:
        _u, _singular, vt = np.linalg.svd(normalized_jac, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.empty((0, 7)), np.empty((0, 7)), np.empty((0, 6))

    directions: list[FloatArray] = []
    e0 = np.zeros(7); e0[0] = 1.0
    e1 = np.zeros(7); e1[1] = 1.0
    directions.extend((e0, e1))
    for row in vt:
        d = np.zeros(7, dtype=float)
        d[2:] = row / max(np.linalg.norm(row), 1.0e-12)
        directions.append(d)

    # Deterministic mixed singular-vector directions.
    mixed_candidates: list[FloatArray] = []
    for i in range(len(directions)):
        for j in range(i + 1, len(directions)):
            for sign in (1.0, -1.0):
                d = directions[i] + sign * directions[j]
                norm = np.linalg.norm(d)
                if norm > 0.0:
                    mixed_candidates.append(d / norm)
    rng = np.random.default_rng(int(seed))
    if mixed_candidates and chart_cfg.patch_mixed_directions:
        choose = rng.choice(
            len(mixed_candidates),
            size=min(chart_cfg.patch_mixed_directions, len(mixed_candidates)),
            replace=False,
        )
        directions.extend([mixed_candidates[int(i)] for i in choose])
    for _ in range(chart_cfg.patch_random_directions):
        d = rng.normal(size=7)
        # Keep random designs dominated by the nontrivial shooting variables,
        # while still sampling conditioning on u0/w0.
        d[:2] *= 0.6
        norm = np.linalg.norm(d)
        if norm > 0.0:
            directions.append(d / norm)

    candidates: list[FloatArray] = []
    # Small-to-large ordering ensures a useful connected patch even when the
    # configured point cap is reached.
    for step in sorted(float(patch_scale) * float(v) for v in chart_cfg.patch_axis_steps):
        for d in directions:
            for sign in (-1.0, 1.0):
                delta = sign * step * d
                u0 = float(center_launch[0] + fs[0] * delta[0])
                w0 = float(center_launch[1] + fs[1] * delta[1])
                q = q0 + qs * delta[2:]
                try:
                    launch = q_to_launch(u0, w0, q, bounds=bounds)
                except (ValueError, OverflowError, FloatingPointError):
                    continue
                # Respect explicit u0/w0 generation bounds when available via
                # the supplied center neighbourhood; q bounds are already checked.
                if not np.all(np.isfinite(launch)):
                    continue
                candidates.append(launch)
                if len(candidates) >= chart_cfg.patch_max_points:
                    break
            if len(candidates) >= chart_cfg.patch_max_points:
                break
        if len(candidates) >= chart_cfg.patch_max_points:
            break

    patch_launch: list[FloatArray] = []
    patch_endpoint: list[FloatArray] = []
    patch_diag: list[FloatArray] = []
    center_label = branch_labels(np.asarray(center_diag, dtype=float)[None, :])[0]
    for launch in candidates:
        endpoint, diag = _exact_endpoint_and_diagnostics(backend, launch, query_cfg)
        if endpoint is None or diag is None:
            continue
        if chart_cfg.patch_strict_branch:
            label = branch_labels(diag[None, :])[0]
            if not np.array_equal(label, center_label):
                continue
        patch_launch.append(launch)
        patch_endpoint.append(endpoint)
        patch_diag.append(diag)
    if not patch_launch:
        return np.empty((0, 7)), np.empty((0, 7)), np.empty((0, 6))
    return np.vstack(patch_launch), np.vstack(patch_endpoint), np.vstack(patch_diag)


def _exact_probe_rows(backend, backend_cfg, query_cfg, launches: FloatArray):
    if len(launches) == 0:
        return (np.empty((0, 7)), np.empty((0, 7)), np.empty((0, 6)))
    cfg = replace(
        backend_cfg,
        generation_backend="dop853",
        rtol=query_cfg.rtol,
        atol=query_cfg.atol,
        max_step=query_cfg.max_step,
        r_collision=query_cfg.r_collision,
        r_escape=query_cfg.r_escape,
        max_acceleration=query_cfg.max_acceleration,
        diagnostic_points=min(48, max(16, backend_cfg.diagnostic_points)),
        subarcs_per_trajectory=1,
    )
    return backend._generate_rows_from_launches(np.asarray(launches, dtype=float), cfg)


def _audit_candidate_failures(
    work: "WorkingSet",
    backend,
    query_cfg,
    bounds: tuple[FloatArray, FloatArray],
    candidate_launch: FloatArray,
    candidate_endpoint: FloatArray,
    ratios: FloatArray,
    best_index: NDArray[np.int64],
    rng: np.random.Generator,
    count: int,
    chart_cfg: ChartBuildConfig,
) -> NDArray[np.int64]:
    inside = np.flatnonzero(np.isfinite(ratios) & (ratios <= 1.5) & (best_index >= 0))
    if len(inside) == 0 or count <= 0:
        return np.empty(0, dtype=np.int64)
    # Prefer the trust boundary, where empirical charts are least certain.
    order = inside[np.argsort(ratios[inside])[::-1]]
    if len(order) > count:
        # Keep mostly boundary cases plus a small random interior component.
        boundary_count = max(1, int(0.8 * count))
        boundary = order[:boundary_count]
        remainder = order[boundary_count:]
        random_count = min(count - len(boundary), len(remainder))
        random_part = rng.choice(remainder, size=random_count, replace=False) if random_count else np.empty(0, dtype=int)
        order = np.concatenate((boundary, random_part))
    failed: list[int] = []
    for candidate_idx in order:
        exact_target = _exact_endpoint(backend, candidate_launch[candidate_idx], query_cfg)
        if exact_target is None:
            continue
        chart_idx = int(best_index[candidate_idx])
        try:
            seed_launch, score = work.bank.predict_launch(
                chart_idx, exact_target, bounds=bounds
            )
            method_index = int(work.bank.solver_method[chart_idx]) if len(work.bank.solver_method) else 0
            method_index = min(max(method_index, 0), len(CORRECTOR_PROFILE_NAMES) - 1)
            method_cfg = _corrector_profile_config(
                query_cfg, CORRECTOR_PROFILE_NAMES[method_index], chart_cfg
            )
            corrected, _nfev = backend._correct_target_from_explicit_seed(
                exact_target, seed_launch, method_cfg, bounds
            )
        except (ValueError, OverflowError, FloatingPointError):
            # Invalid polynomial extrapolation: classify it as a chart failure
            # and shrink the claimed trust region instead of aborting the run.
            _q_pred, score = work.bank.predict_q(chart_idx, exact_target)
            corrected = None
        exact_ratio = work.bank.trust_ratio(chart_idx, exact_target)
        if corrected is None:
            failed.append(int(candidate_idx))
            work.bank.failure_count[chart_idx] += 1
            if exact_ratio <= 1.0:
                work.bank.shrink_trust(chart_idx, max(0.25, 0.95 * exact_ratio))
        else:
            work.bank.success_count[chart_idx] += 1
    if failed:
        work.bank.invalidate_tree()
    return np.asarray(failed, dtype=np.int64)


def _build_chart_single(
    backend,
    backend_cfg,
    query_cfg,
    chart_cfg: ChartBuildConfig,
    feature_scale: FloatArray,
    q_scale: FloatArray,
    bounds: tuple[FloatArray, FloatArray],
    center_launch: FloatArray,
    center_endpoint: FloatArray,
    center_diag: FloatArray,
    neighbour_launch: FloatArray,
    neighbour_endpoint: FloatArray,
    neighbour_diag: FloatArray,
    seed: int,
    patch_scale: float = 1.0,
    model_kind: str = "quadratic",
    prepared_center: Optional[tuple[FloatArray, FloatArray, float, float, FloatArray]] = None,
    prepared_patch: Optional[tuple[FloatArray, FloatArray, FloatArray]] = None,
) -> Optional[BuiltChart]:
    """Build one connected local inverse chart around an exact extremal.

    Unlike the earlier implementation, the polynomial fit is based primarily
    on controlled perturbations of the chart's own launch.  This prevents
    nearby-but-disconnected inverse branches from contaminating the local map.
    Several small-budget correctors are benchmarked on held-out patch points;
    the chart stores the profile with the largest certified trust radius.
    """
    center_launch = np.asarray(center_launch, dtype=float)
    if prepared_center is None:
        exact_center_endpoint, jac, exact_condition, exact_sigma = _exact_endpoint_jacobian(
            backend, center_launch, query_cfg
        )
        if exact_center_endpoint is None or jac is None:
            return None
        center_endpoint = exact_center_endpoint
        _center_ep2, exact_center_diag = _exact_endpoint_and_diagnostics(
            backend, center_launch, query_cfg
        )
        if exact_center_diag is not None:
            center_diag = exact_center_diag
        else:
            center_diag = np.asarray(center_diag, dtype=float).copy()
            center_diag[5] = float(round(center_endpoint[3] / (2.0 * math.pi)))
    else:
        center_endpoint, jac, exact_condition, exact_sigma, center_diag = prepared_center
        center_endpoint = np.asarray(center_endpoint, dtype=float)
        jac = np.asarray(jac, dtype=float)
        center_diag = np.asarray(center_diag, dtype=float)

    transform_mode = _transform_mode_index(chart_cfg.output_transform)
    transform_scale = _local_transform_scale(
        center_endpoint,
        np.asarray(feature_scale, dtype=float),
        transform_mode,
        dynamic=bool(chart_cfg.dynamic_transform_scale),
        fallback_scale=np.asarray(chart_cfg.search_feature_scale, dtype=float),
        asinh_scale=float(chart_cfg.output_transform_asinh_scale),
        minimum_scale=float(chart_cfg.minimum_transform_scale),
    )
    jac = _local_output_jacobian5(
        center_endpoint, transform_mode, chart_cfg.output_transform_asinh_scale
    ) @ np.asarray(jac, dtype=float)

    if chart_cfg.require_center_self_test:
        center_ok = False
        for profile in chart_cfg.corrector_profiles:
            profile_cfg = _corrector_profile_config(query_cfg, profile, chart_cfg)
            try:
                corrected, _center_nfev = backend._correct_target_from_explicit_seed(
                    center_endpoint, center_launch, profile_cfg, bounds
                )
            except Exception:
                corrected = None
            if corrected is not None:
                center_ok = True
                break
        if not center_ok:
            return None

    inv_jac, condition, sigma_min = _regularized_inverse(
        jac, chart_cfg.jacobian_regularization
    )
    if inv_jac is None or condition > chart_cfg.maximum_jacobian_condition:
        return None

    if prepared_patch is None:
        patch_launch, patch_endpoint, patch_diag = _controlled_local_patch(
            backend,
            center_launch,
            center_endpoint,
            center_diag,
            jac,
            query_cfg,
            chart_cfg,
            transform_scale,
            np.asarray(q_scale, dtype=float),
            bounds,
            int(seed),
            float(patch_scale),
        )
    else:
        patch_launch, patch_endpoint, patch_diag = prepared_patch
        patch_launch = np.asarray(patch_launch, dtype=float)
        patch_endpoint = np.asarray(patch_endpoint, dtype=float)
        patch_diag = np.asarray(patch_diag, dtype=float)

    # Optional emergency fallback for early/sparse runs.  The recommended long
    # configuration disables this because unrelated neighbours can cross folds.
    if len(patch_launch) < chart_cfg.patch_min_points and chart_cfg.patch_use_pool_fallback:
        labels = branch_labels(np.asarray(neighbour_diag, dtype=float))
        center_label = branch_labels(np.asarray(center_diag, dtype=float)[None, :])[0]
        same = np.all(labels == center_label[None, :], axis=1)
        nl = np.asarray(neighbour_launch, dtype=float)[same]
        ne = np.asarray(neighbour_endpoint, dtype=float)[same]
        nd = np.asarray(neighbour_diag, dtype=float)[same]
        z = (
            _normalized_chart_delta(
                ne, center_endpoint, transform_mode, transform_scale,
                chart_cfg.output_transform_asinh_scale,
            )
            if len(ne) else np.empty((0, 7))
        )
        d = np.linalg.norm(z, axis=1) if len(z) else np.empty(0)
        keep = np.isfinite(d) & (d > 1.0e-12) & (d <= chart_cfg.fit_max_normalized_distance)
        if np.any(keep):
            patch_launch = np.vstack((patch_launch, nl[keep])) if len(patch_launch) else nl[keep]
            patch_endpoint = np.vstack((patch_endpoint, ne[keep])) if len(patch_endpoint) else ne[keep]
            patch_diag = np.vstack((patch_diag, nd[keep])) if len(patch_diag) else nd[keep]

    if len(patch_launch) < chart_cfg.patch_min_points:
        return None

    fs = np.asarray(transform_scale, dtype=float)
    qs = np.asarray(q_scale, dtype=float)
    z_all = _normalized_chart_delta(
        patch_endpoint, center_endpoint, transform_mode, fs,
        chart_cfg.output_transform_asinh_scale,
    )
    q0 = launch_to_q(center_launch)
    y_all = (launch_to_q(patch_launch) - q0) / qs
    finite = np.all(np.isfinite(z_all), axis=1) & np.all(np.isfinite(y_all), axis=1)
    z_all = z_all[finite]
    y_all = y_all[finite]
    patch_launch = patch_launch[finite]
    patch_endpoint = patch_endpoint[finite]
    if len(z_all) < chart_cfg.patch_min_points:
        return None

    rng = np.random.default_rng(int(seed) ^ 0x5EEDBEEF)
    order = rng.permutation(len(z_all))
    desired_validation = max(
        chart_cfg.minimum_local_training_tests,
        min(
            chart_cfg.exact_validation_points,
            max(1, int(math.ceil(chart_cfg.patch_validation_fraction * len(order)))),
        ),
    )
    minimum_fit_rows = 6 if model_kind == "micro" else 12
    desired_validation = min(desired_validation, max(1, len(order) - minimum_fit_rows))
    validation_indices = order[:desired_validation]
    fit_indices = order[desired_validation:]
    if len(fit_indices) < minimum_fit_rows or len(validation_indices) < chart_cfg.minimum_local_training_tests:
        return None

    z_fit = z_all[fit_indices]
    y_fit = y_all[fit_indices]
    fit_distance = np.linalg.norm(z_fit, axis=1)

    # Exact variational inverse for the five fixed-(u0,w0) output columns.
    affine = np.zeros((5, 7), dtype=float)
    affine[:, 2:7] = np.diag(1.0 / qs) @ inv_jac @ np.diag(fs[2:7])

    # Fit the two initial-velocity conditioning columns from the connected patch.
    residual_u = y_fit - z_fit[:, 2:7] @ affine[:, 2:7].T
    zu = z_fit[:, :2]
    bandwidth = max(float(np.median(fit_distance)), 1.0e-3)
    weights = np.exp(-0.5 * (fit_distance / bandwidth) ** 2)
    sw = np.sqrt(np.maximum(weights, 1.0e-12))
    lhs = (zu * sw[:, None]).T @ (zu * sw[:, None]) + chart_cfg.fit_ridge * np.eye(2)
    rhs = (zu * sw[:, None]).T @ (residual_u * sw[:, None])
    try:
        affine[:, :2] = np.linalg.solve(lhs, rhs).T
    except np.linalg.LinAlgError:
        affine[:, :2] = 0.0

    initial_prediction = z_fit @ affine.T
    residual = y_fit - initial_prediction
    residual_norm = np.linalg.norm(residual, axis=1)
    cutoff = float(np.quantile(residual_norm, chart_cfg.fit_trim_quantile))
    fit_keep = residual_norm <= max(cutoff, 1.0e-12)
    if np.count_nonzero(fit_keep) < minimum_fit_rows:
        fit_keep = np.argsort(residual_norm)[: min(len(residual_norm), max(minimum_fit_rows, len(residual_norm) // 2))]
    if model_kind == "quadratic" and chart_cfg.quadratic_rank > 0:
        directions, coefficients = _fit_low_rank_quadratic(
            z_fit[fit_keep],
            residual[fit_keep],
            weights[fit_keep],
            chart_cfg.quadratic_rank,
            chart_cfg.quadratic_ridge,
            chart_cfg.coefficient_clip,
        )
    else:
        directions = np.empty((0, 7), dtype=float)
        coefficients = np.empty((5, 0), dtype=float)

    # Known-launch interpolation error supplies a cheap guard against unstable
    # quadratic fits, but exact bounded correction determines certification.
    predicted_fit = z_fit @ affine.T
    if len(directions):
        predicted_fit += ((z_fit @ directions.T) ** 2) @ coefficients.T
    fit_error = np.linalg.norm(predicted_fit - y_fit, axis=1)
    fit_scores = np.sqrt(
        np.sum(predicted_fit * predicted_fit, axis=1)
        + chart_cfg.trust_endpoint_weight * np.sum(z_fit * z_fit, axis=1)
    )
    fit_good = fit_error <= chart_cfg.prediction_error_limit
    prediction_cfg = replace(
        chart_cfg,
        minimum_exact_successes=max(
            1,
            min(chart_cfg.minimum_exact_successes, int(np.count_nonzero(fit_good))),
        ),
        local_target_success=min(chart_cfg.prediction_target_success, 0.90),
    )
    prediction_radius = _validated_radius(fit_scores, fit_good, prediction_cfg)

    val_targets = patch_endpoint[validation_indices]
    val_z = z_all[validation_indices]
    val_q: list[Optional[FloatArray]] = []
    val_scores: list[float] = []
    for target in val_targets:
        q_pred, _z_pred, score = chart_predict_q_normalized(
            target,
            center_endpoint,
            center_launch,
            affine,
            directions,
            coefficients,
            transform_mode,
            fs,
            qs,
            chart_cfg.trust_endpoint_weight,
            chart_cfg.output_transform_asinh_scale,
        )
        val_q.append(q_pred if np.all(np.isfinite(q_pred)) else None)
        val_scores.append(float(score))
    score_array = np.asarray(val_scores, dtype=float)

    profiles = tuple(chart_cfg.corrector_profiles)
    profile_results: dict[str, dict[str, Any]] = {}
    pilot_count = min(6, len(val_targets))
    pilot_indices = np.arange(pilot_count, dtype=int)

    def test_profile(profile: str, indices: Sequence[int], existing: Optional[dict[str, Any]] = None):
        record = existing or {"success": [], "latency": [], "tested_indices": []}
        cfg = _corrector_profile_config(query_cfg, profile, chart_cfg)
        already = set(record["tested_indices"])
        for idx in indices:
            idx = int(idx)
            if idx in already:
                continue
            q_pred = val_q[idx]
            started = time.monotonic()
            corrected = None
            if q_pred is not None:
                try:
                    seed_launch = q_to_launch(
                        float(val_targets[idx, 0]),
                        float(val_targets[idx, 1]),
                        q_pred,
                        bounds=bounds,
                    )
                    corrected, _nfev = backend._correct_target_from_explicit_seed(
                        val_targets[idx], seed_launch, cfg, bounds
                    )
                except Exception:
                    corrected = None
            record["tested_indices"].append(idx)
            record["success"].append(corrected is not None)
            record["latency"].append(float(time.monotonic() - started))
        return record

    # Cheap pilot tournament, then complete only the two most promising profiles.
    for profile in profiles:
        profile_results[profile] = test_profile(profile, pilot_indices)
    ranked_profiles = sorted(
        profiles,
        key=lambda name: (
            sum(profile_results[name]["success"]),
            -np.median(profile_results[name]["latency"]) if profile_results[name]["latency"] else -math.inf,
        ),
        reverse=True,
    )
    finalists = ranked_profiles[: min(2, len(ranked_profiles))]
    all_indices = np.arange(len(val_targets), dtype=int)
    for profile in finalists:
        profile_results[profile] = test_profile(profile, all_indices, profile_results[profile])

    best_profile: Optional[str] = None
    best_radius = -math.inf
    best_successes = -1
    best_p95 = math.inf
    best_success_vector: Optional[NDArray[np.bool_]] = None
    for profile in finalists:
        record = profile_results[profile]
        tested_indices = np.asarray(record["tested_indices"], dtype=int)
        success = np.asarray(record["success"], dtype=bool)
        latency = np.asarray(record["latency"], dtype=float)
        if len(tested_indices) != len(val_targets):
            continue
        # Records are appended in pilot/all order; reorder to target order.
        order2 = np.argsort(tested_indices)
        success = success[order2]
        latency = latency[order2]
        successes = int(np.count_nonzero(success))
        if successes < chart_cfg.minimum_exact_successes:
            radius = chart_cfg.minimum_trust_radius
        else:
            radius = _validated_radius(score_array, success, chart_cfg)
        # Exact bounded correction is the certification criterion.  Use the
        # interpolation fit only as a soft cap when it has itself demonstrated
        # a nontrivial radius; otherwise do not force every chart back to the
        # minimum point-chart radius.
        if prediction_radius > chart_cfg.minimum_trust_radius * 1.01:
            radius = min(float(radius), 1.25 * float(prediction_radius))
        else:
            radius = float(radius)
        p95 = float(np.quantile(latency, 0.95)) if len(latency) else math.inf
        key = (radius, successes, -p95)
        if key > (best_radius, best_successes, -best_p95):
            best_profile = profile
            best_radius = radius
            best_successes = successes
            best_p95 = p95
            best_success_vector = success

    if best_profile is None or best_success_vector is None:
        return None
    if len(fit_good) and float(np.mean(fit_good)) < chart_cfg.minimum_local_training_success:
        return None
    failures = int(len(best_success_vector) - best_successes)

    if best_successes < chart_cfg.minimum_exact_successes:
        return None
    if len(best_success_vector) < chart_cfg.minimum_local_training_tests:
        return None
    if best_successes / max(len(best_success_vector), 1) < chart_cfg.minimum_local_training_success:
        return None

    ellipsoid_max = (
        chart_cfg.microchart_max_directional_radius
        if model_kind == "micro" else chart_cfg.maximum_directional_radius
    )
    trust_basis, trust_radii, validation_ratios = _fit_trust_ellipsoid(
        val_z, best_success_vector, chart_cfg, max_radius=ellipsoid_max
    )
    trust_radius = float(np.exp(np.mean(np.log(np.maximum(trust_radii, 1.0e-12)))))
    if chart_cfg.reject_if_no_local_coverage and float(np.max(trust_radii)) <= chart_cfg.minimum_directional_radius * 1.001:
        return None

    successful_targets = val_targets[np.asarray(best_success_vector, dtype=bool)]
    witness_rows = [center_endpoint.copy()]
    if len(successful_targets):
        # Prefer geometrically diverse successful witnesses.
        wratio = validation_ratios[np.asarray(best_success_vector, dtype=bool)]
        for wi in np.argsort(wratio)[::-1]:
            witness_rows.append(successful_targets[int(wi)].copy())
            if len(witness_rows) >= chart_cfg.witness_capacity:
                break
    witnesses = np.vstack(witness_rows[: chart_cfg.witness_capacity])
    chart_type = 0 if model_kind == "quadratic" else (1 if model_kind == "affine" else 2)

    return BuiltChart(
        launch=center_launch.copy(),
        endpoint=center_endpoint.copy(),
        diagnostics=np.asarray(center_diag, dtype=float).copy(),
        affine=affine,
        quad_dirs=directions,
        quad_coeff=coefficients,
        trust_radius=trust_radius,
        local_successes=best_successes,
        local_failures=failures,
        local_p95_seconds=best_p95,
        jacobian_condition=float(condition),
        jacobian_sigma_min=float(sigma_min),
        solver_method=_corrector_profile_index(best_profile, chart_cfg),
        patch_points=int(len(patch_launch)),
        trust_basis=trust_basis,
        trust_radii=trust_radii,
        chart_type=chart_type,
        patch_scale=float(patch_scale),
        transform_mode=transform_mode,
        transform_scale=transform_scale.copy(),
        witnesses=witnesses,
    )


def _build_chart(
    backend, backend_cfg, query_cfg, chart_cfg: ChartBuildConfig,
    feature_scale: FloatArray, q_scale: FloatArray,
    bounds: tuple[FloatArray, FloatArray],
    center_launch: FloatArray, center_endpoint: FloatArray, center_diag: FloatArray,
    neighbour_launch: FloatArray, neighbour_endpoint: FloatArray, neighbour_diag: FloatArray,
    seed: int,
) -> Optional[BuiltChart]:
    """Adaptive chart cascade reusing each expensive local integration patch."""
    del backend_cfg
    center_launch = np.asarray(center_launch, dtype=float)
    exact_endpoint, jac, exact_condition, exact_sigma = _exact_endpoint_jacobian(
        backend, center_launch, query_cfg
    )
    if exact_endpoint is None or jac is None:
        return None
    _ep, exact_diag = _exact_endpoint_and_diagnostics(backend, center_launch, query_cfg)
    if exact_diag is None:
        exact_diag = np.asarray(center_diag, dtype=float).copy()
        exact_diag[5] = float(round(exact_endpoint[3] / (2.0 * math.pi)))
    prepared_center = (
        np.asarray(exact_endpoint, dtype=float), np.asarray(jac, dtype=float),
        float(exact_condition), float(exact_sigma), np.asarray(exact_diag, dtype=float),
    )
    scales = tuple(sorted({float(v) for v in chart_cfg.adaptive_patch_scale_factors}, reverse=True))
    for scale_index, scale in enumerate(scales):
        patch = _controlled_local_patch(
            backend, center_launch, exact_endpoint, exact_diag, jac, query_cfg, chart_cfg,
            np.asarray(feature_scale, dtype=float), np.asarray(q_scale, dtype=float),
            bounds, int(seed) ^ (scale_index * 0x1F123BB5), scale,
        )
        if len(patch[0]) < min(chart_cfg.patch_min_points, chart_cfg.microchart_min_points):
            continue
        for model_index, model_kind in enumerate(("quadratic", "affine")):
            built = _build_chart_single(
                backend, None, query_cfg, chart_cfg, feature_scale, q_scale, bounds,
                center_launch, exact_endpoint, exact_diag, neighbour_launch, neighbour_endpoint,
                neighbour_diag, seed ^ (model_index * 0xA551), scale, model_kind,
                prepared_center=prepared_center, prepared_patch=patch,
            )
            if built is not None:
                return built
    if chart_cfg.microcharts_enabled:
        micro_cfg = replace(
            chart_cfg,
            patch_min_points=min(chart_cfg.patch_min_points, chart_cfg.microchart_min_points),
            exact_validation_points=min(chart_cfg.exact_validation_points, chart_cfg.microchart_validation_points),
            minimum_exact_successes=min(chart_cfg.minimum_exact_successes, chart_cfg.microchart_minimum_successes),
            minimum_local_training_tests=min(chart_cfg.minimum_local_training_tests, chart_cfg.microchart_validation_points),
            minimum_local_training_success=min(chart_cfg.minimum_local_training_success, 0.50),
            reject_if_no_local_coverage=False,
        )
        scale = float(chart_cfg.microchart_patch_scale)
        patch = _controlled_local_patch(
            backend, center_launch, exact_endpoint, exact_diag, jac, query_cfg, micro_cfg,
            np.asarray(feature_scale, dtype=float), np.asarray(q_scale, dtype=float),
            bounds, int(seed) ^ 0xC0FFEE, scale,
        )
        if len(patch[0]) >= micro_cfg.patch_min_points:
            return _build_chart_single(
                backend, None, query_cfg, micro_cfg, feature_scale, q_scale, bounds,
                center_launch, exact_endpoint, exact_diag, neighbour_launch, neighbour_endpoint,
                neighbour_diag, seed ^ 0xC0FFEE, scale, "micro",
                prepared_center=prepared_center, prepared_patch=patch,
            )
    return None


# ---------------------------------------------------------------------------
# In-memory chart bank
# ---------------------------------------------------------------------------


def _normalize_witness_storage(
    raw_endpoint: Optional[FloatArray],
    raw_count: Optional[NDArray[np.int8]],
    centers: FloatArray,
    capacity: int,
) -> tuple[FloatArray, NDArray[np.int8]]:
    n = len(centers)
    endpoint = np.full((n, int(capacity), 7), np.nan, dtype=np.float32)
    count = np.zeros(n, dtype=np.int8)
    if raw_endpoint is not None:
        raw = np.asarray(raw_endpoint, dtype=np.float32)
        if raw.ndim == 3 and raw.shape[0] == n and raw.shape[2] == 7:
            copy_n = min(int(capacity), raw.shape[1])
            endpoint[:, :copy_n] = raw[:, :copy_n]
            if raw_count is not None:
                count = np.clip(np.asarray(raw_count, dtype=np.int16), 0, copy_n).astype(np.int8)
            else:
                count[:] = copy_n
    missing = count == 0
    if np.any(missing):
        endpoint[missing, 0] = np.asarray(centers, dtype=np.float32)[missing]
        count[missing] = 1
    return endpoint, count


class ChartBank:
    def __init__(
        self, feature_scale: FloatArray, q_scale: FloatArray, quadratic_rank: int,
        trust_endpoint_weight: float = 0.04, witness_capacity: int = 8,
        search_feature_scale: Optional[FloatArray] = None,
        transform_asinh_scale: float = 0.0,
    ):
        # feature_scale remains the canonical/raw scale used for generation and
        # backward-compatible raw charts.  Every transformed chart stores its
        # own local transformed scale.
        self.feature_scale = np.asarray(feature_scale, dtype=float)
        self.search_feature_scale = np.asarray(
            search_feature_scale if search_feature_scale is not None else feature_scale,
            dtype=float,
        )
        self.q_scale = np.asarray(q_scale, dtype=float)
        self.rank = int(quadratic_rank)
        self.trust_endpoint_weight = float(trust_endpoint_weight)
        self.witness_capacity = int(witness_capacity)
        self.transform_asinh_scale = float(transform_asinh_scale)
        self.launch = np.empty((0, 7), dtype=float)
        self.endpoint = np.empty((0, 7), dtype=float)  # canonical/raw endpoint
        self.diagnostics = np.empty((0, 6), dtype=float)
        self.affine = np.empty((0, 5, 7), dtype=np.float32)
        self.quad_dirs = np.empty((0, self.rank, 7), dtype=np.float32)
        self.quad_coeff = np.empty((0, 5, self.rank), dtype=np.float32)
        self.trust_radius = np.empty(0, dtype=np.float32)
        self.success_count = np.empty(0, dtype=np.int32)
        self.failure_count = np.empty(0, dtype=np.int32)
        self.local_p95_seconds = np.empty(0, dtype=np.float32)
        self.jacobian_condition = np.empty(0, dtype=np.float32)
        self.jacobian_sigma_min = np.empty(0, dtype=np.float32)
        self.solver_method = np.empty(0, dtype=np.int8)
        self.patch_points = np.empty(0, dtype=np.int16)
        self.refinement_count = np.empty(0, dtype=np.int16)
        self.trust_basis = np.empty((0, 7, 7), dtype=np.float32)
        self.trust_radii = np.empty((0, 7), dtype=np.float32)
        self.chart_type = np.empty(0, dtype=np.int8)
        self.patch_scale = np.empty(0, dtype=np.float32)
        self.transform_mode = np.empty(0, dtype=np.int8)
        self.transform_scale = np.empty((0, 7), dtype=np.float32)
        self.witness_endpoint = np.full((0, self.witness_capacity, 7), np.nan, dtype=np.float32)
        self.witness_count = np.empty(0, dtype=np.int8)
        self._tree: Optional[cKDTree] = None

    def __len__(self) -> int:
        return len(self.launch)

    def invalidate_tree(self) -> None:
        self._tree = None

    def transform_name(self, index: int) -> str:
        return TRANSFORM_NAMES.get(int(self.transform_mode[int(index)]), "unknown")

    def normalized_delta(self, index: int, targets: FloatArray) -> FloatArray:
        i = int(index)
        return _normalized_chart_delta(
            targets,
            self.endpoint[i],
            int(self.transform_mode[i]),
            self.transform_scale[i],
            self.transform_asinh_scale,
        )

    def shrink_trust(self, index: int, factor: float) -> None:
        i = int(index)
        f = float(np.clip(factor, 0.05, 1.0))
        self.trust_radii[i] *= np.float32(f)
        self.trust_radii[i] = np.maximum(self.trust_radii[i], np.float32(1.0e-6))
        self.trust_radius[i] = np.float32(
            np.exp(np.mean(np.log(np.maximum(self.trust_radii[i].astype(float), 1.0e-12))))
        )

    def add_witness(self, index: int, target: FloatArray) -> None:
        i = int(index)
        target = np.asarray(target, dtype=float)
        count = int(self.witness_count[i])
        if count:
            existing = self.witness_endpoint[i, :count].astype(float)
            distance = np.linalg.norm(self.normalized_delta(i, existing) - self.normalized_delta(i, target[None, :]), axis=1)
            if np.any(distance < 1.0e-6):
                return
        if count < self.witness_capacity:
            self.witness_endpoint[i, count] = target.astype(np.float32)
            self.witness_count[i] = np.int8(count + 1)
            return
        if self.witness_capacity <= 1:
            return
        existing = self.witness_endpoint[i].astype(float)
        normed = self.normalized_delta(i, existing)
        pair_distance = np.full((self.witness_capacity, self.witness_capacity), np.inf)
        for a in range(1, self.witness_capacity):
            for b in range(a + 1, self.witness_capacity):
                pair_distance[a, b] = np.linalg.norm(normed[a] - normed[b])
        flat = int(np.argmin(pair_distance))
        a, b = np.unravel_index(flat, pair_distance.shape)
        replace_index = b if b > 0 else a
        candidate = self.normalized_delta(i, target[None, :])[0]
        candidate_distance = np.min(np.linalg.norm(normed - candidate, axis=1))
        if candidate_distance > pair_distance[a, b]:
            self.witness_endpoint[i, replace_index] = target.astype(np.float32)

    @property
    def tree(self) -> cKDTree:
        if self._tree is None:
            if len(self) == 0:
                raise RuntimeError("chart bank is empty")
            search = _global_search_transform(self.endpoint)
            self._tree = cKDTree(search / self.search_feature_scale)
        return self._tree

    def append(self, charts: Sequence[BuiltChart]) -> int:
        if not charts:
            return 0
        n = len(charts)
        launch = np.vstack([c.launch for c in charts])
        endpoint = np.vstack([c.endpoint for c in charts])
        diagnostics = np.vstack([c.diagnostics for c in charts])
        affine = np.stack([c.affine for c in charts]).astype(np.float32)
        dirs = np.zeros((n, self.rank, 7), dtype=np.float32)
        coeff = np.zeros((n, 5, self.rank), dtype=np.float32)
        for i, c in enumerate(charts):
            r = min(self.rank, len(c.quad_dirs))
            if r:
                dirs[i, :r] = c.quad_dirs[:r]
                coeff[i, :, :r] = c.quad_coeff[:, :r]
        self.launch = np.vstack((self.launch, launch))
        self.endpoint = np.vstack((self.endpoint, endpoint))
        self.diagnostics = np.vstack((self.diagnostics, diagnostics))
        self.affine = np.concatenate((self.affine, affine), axis=0)
        self.quad_dirs = np.concatenate((self.quad_dirs, dirs), axis=0)
        self.quad_coeff = np.concatenate((self.quad_coeff, coeff), axis=0)
        self.trust_radius = np.concatenate((self.trust_radius, np.asarray([c.trust_radius for c in charts], dtype=np.float32)))
        self.success_count = np.concatenate((self.success_count, np.asarray([c.local_successes for c in charts], dtype=np.int32)))
        self.failure_count = np.concatenate((self.failure_count, np.asarray([c.local_failures for c in charts], dtype=np.int32)))
        self.local_p95_seconds = np.concatenate((self.local_p95_seconds, np.asarray([c.local_p95_seconds for c in charts], dtype=np.float32)))
        self.jacobian_condition = np.concatenate((self.jacobian_condition, np.asarray([c.jacobian_condition for c in charts], dtype=np.float32)))
        self.jacobian_sigma_min = np.concatenate((self.jacobian_sigma_min, np.asarray([c.jacobian_sigma_min for c in charts], dtype=np.float32)))
        self.solver_method = np.concatenate((self.solver_method, np.asarray([c.solver_method for c in charts], dtype=np.int8)))
        self.patch_points = np.concatenate((self.patch_points, np.asarray([c.patch_points for c in charts], dtype=np.int16)))
        self.refinement_count = np.concatenate((self.refinement_count, np.zeros(n, dtype=np.int16)))
        self.trust_basis = np.concatenate((self.trust_basis, np.stack([c.trust_basis for c in charts]).astype(np.float32)), axis=0)
        self.trust_radii = np.concatenate((self.trust_radii, np.stack([c.trust_radii for c in charts]).astype(np.float32)), axis=0)
        self.chart_type = np.concatenate((self.chart_type, np.asarray([c.chart_type for c in charts], dtype=np.int8)))
        self.patch_scale = np.concatenate((self.patch_scale, np.asarray([c.patch_scale for c in charts], dtype=np.float32)))
        self.transform_mode = np.concatenate((self.transform_mode, np.asarray([c.transform_mode for c in charts], dtype=np.int8)))
        self.transform_scale = np.concatenate((self.transform_scale, np.stack([c.transform_scale for c in charts]).astype(np.float32)), axis=0)
        witness = np.full((n, self.witness_capacity, 7), np.nan, dtype=np.float32)
        witness_count = np.zeros(n, dtype=np.int8)
        for i, c in enumerate(charts):
            wc = min(self.witness_capacity, len(c.witnesses))
            if wc:
                witness[i, :wc] = np.asarray(c.witnesses[:wc], dtype=np.float32)
                witness_count[i] = wc
        self.witness_endpoint = np.concatenate((self.witness_endpoint, witness), axis=0)
        self.witness_count = np.concatenate((self.witness_count, witness_count))
        self.invalidate_tree()
        return n

    def replace_chart(self, index: int, chart: BuiltChart) -> None:
        i = int(index)
        self.launch[i] = chart.launch
        self.endpoint[i] = chart.endpoint
        self.diagnostics[i] = chart.diagnostics
        self.affine[i] = np.asarray(chart.affine, dtype=np.float32)
        self.quad_dirs[i] = 0.0
        self.quad_coeff[i] = 0.0
        r = min(self.rank, len(chart.quad_dirs))
        if r:
            self.quad_dirs[i, :r] = chart.quad_dirs[:r]
            self.quad_coeff[i, :, :r] = chart.quad_coeff[:, :r]
        self.trust_radius[i] = np.float32(chart.trust_radius)
        self.success_count[i] = np.int32(chart.local_successes)
        self.failure_count[i] = np.int32(chart.local_failures)
        self.local_p95_seconds[i] = np.float32(chart.local_p95_seconds)
        self.jacobian_condition[i] = np.float32(chart.jacobian_condition)
        self.jacobian_sigma_min[i] = np.float32(chart.jacobian_sigma_min)
        self.solver_method[i] = np.int8(chart.solver_method)
        self.patch_points[i] = np.int16(chart.patch_points)
        self.trust_basis[i] = np.asarray(chart.trust_basis, dtype=np.float32)
        self.trust_radii[i] = np.asarray(chart.trust_radii, dtype=np.float32)
        self.chart_type[i] = np.int8(chart.chart_type)
        self.patch_scale[i] = np.float32(chart.patch_scale)
        self.transform_mode[i] = np.int8(chart.transform_mode)
        self.transform_scale[i] = np.asarray(chart.transform_scale, dtype=np.float32)
        self.witness_endpoint[i] = np.nan
        wc = min(self.witness_capacity, len(chart.witnesses))
        if wc:
            self.witness_endpoint[i, :wc] = np.asarray(chart.witnesses[:wc], dtype=np.float32)
        self.witness_count[i] = np.int8(wc)
        self.refinement_count[i] = np.int16(min(32767, int(self.refinement_count[i]) + 1))
        self.invalidate_tree()

    def candidate_indices(self, targets: FloatArray, k: int) -> NDArray[np.int64]:
        targets = np.atleast_2d(np.asarray(targets, dtype=float))
        if len(self) == 0:
            return np.empty((len(targets), 0), dtype=np.int64)
        k = min(max(1, int(k)), len(self))
        search = _global_search_transform(targets)
        _distance, index = self.tree.query(search / self.search_feature_scale, k=k, workers=-1)
        index = np.asarray(index, dtype=np.int64)
        if index.ndim == 1:
            index = index[:, None]
        return index

    def coverage_scores(
        self,
        targets: FloatArray,
        k: int = 48,
        winding_window: int = 1,
    ) -> tuple[FloatArray, NDArray[np.int64]]:
        targets = np.atleast_2d(np.asarray(targets, dtype=float))
        if len(self) == 0:
            return np.full(len(targets), np.inf), np.full(len(targets), -1, dtype=np.int64)
        indices = self.candidate_indices(targets, k)
        best_ratio = np.full(len(targets), np.inf, dtype=float)
        best_index = np.full(len(targets), -1, dtype=np.int64)
        target_winding = np.rint(targets[:, 3] / (2.0 * math.pi)).astype(int)
        for column in range(indices.shape[1]):
            idx = indices[:, column]
            center = self.endpoint[idx]
            z = _normalized_chart_delta_paired(
                targets, center, self.transform_mode[idx], self.transform_scale[idx],
                self.transform_asinh_scale,
            )
            affine = self.affine[idx].astype(float)
            y = np.einsum("nij,nj->ni", affine, z)
            if self.rank:
                dirs = self.quad_dirs[idx].astype(float)
                coeff = self.quad_coeff[idx].astype(float)
                qbasis = np.einsum("nkj,nj->nk", dirs, z) ** 2
                y += np.einsum("nik,nk->ni", coeff, qbasis)
            basis = self.trust_basis[idx].astype(float)
            radii = np.maximum(self.trust_radii[idx].astype(float), 1.0e-12)
            xi = np.einsum("nij,nj->ni", basis, z)
            ratio = np.sqrt(np.sum((xi / radii) ** 2, axis=1))
            ratio[~np.all(np.isfinite(y), axis=1)] = np.inf
            winding = np.rint(self.diagnostics[idx, 5]).astype(int)
            ratio[np.abs(winding - target_winding) > winding_window] = np.inf
            improve = ratio < best_ratio
            best_ratio[improve] = ratio[improve]
            best_index[improve] = idx[improve]
        return best_ratio, best_index

    def trust_ratio(self, chart_index: int, target: FloatArray) -> float:
        i = int(chart_index)
        return _ellipsoid_ratio(
            target,
            self.endpoint[i],
            int(self.transform_mode[i]),
            self.transform_scale[i],
            self.trust_basis[i],
            self.trust_radii[i],
            self.transform_asinh_scale,
        )

    def predict_q(self, chart_index: int, target: FloatArray) -> tuple[FloatArray, float]:
        """Return the raw five-variable chart prediction and chart score."""
        i = int(chart_index)
        q, _z, score = chart_predict_q_normalized(
            target,
            self.endpoint[i],
            self.launch[i],
            self.affine[i],
            self.quad_dirs[i],
            self.quad_coeff[i],
            int(self.transform_mode[i]),
            self.transform_scale[i],
            self.q_scale,
            self.trust_endpoint_weight,
            self.transform_asinh_scale,
        )
        return q, score

    def predict_launch(
        self,
        chart_index: int,
        target: FloatArray,
        bounds: Optional[tuple[FloatArray, FloatArray]] = None,
        *,
        clip: bool = False,
    ) -> tuple[FloatArray, float]:
        q, score = self.predict_q(chart_index, target)
        return (
            q_to_launch(float(target[0]), float(target[1]), q, bounds=bounds, clip=clip),
            score,
        )


# ---------------------------------------------------------------------------
# Queryable atlas
# ---------------------------------------------------------------------------


class LocalInverseChartAtlas:
    """Load and query a continuously checkpointed local-inverse chart atlas."""

    def __init__(self, path: str | Path, solver_path: Optional[str | Path] = None):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as data:
            self.metadata = json.loads(str(data["metadata_json"].item()))
            if self.metadata.get("format") != FORMAT_NAME:
                raise ValueError("not a local inverse chart atlas")
            self.config = ChartAtlasConfig.from_dict(self.metadata["config"])
            self.feature_scale = np.asarray(data["feature_scale"], dtype=float)
            self.q_scale = np.asarray(data["q_scale"], dtype=float)
            self.bank = ChartBank(
                self.feature_scale,
                self.q_scale,
                self.config.chart.quadratic_rank,
                self.config.chart.trust_endpoint_weight,
                self.config.chart.witness_capacity,
                np.asarray(self.config.chart.search_feature_scale, dtype=float),
                self.config.chart.output_transform_asinh_scale,
            )
            self.bank.launch = np.asarray(data["chart_launch"], dtype=float)
            self.bank.endpoint = np.asarray(data["chart_endpoint"], dtype=float)
            self.bank.diagnostics = np.asarray(data["chart_diagnostics"], dtype=float)
            self.bank.affine = np.asarray(data["chart_affine"], dtype=np.float32)
            self.bank.quad_dirs = np.asarray(data["chart_quad_dirs"], dtype=np.float32)
            self.bank.quad_coeff = np.asarray(data["chart_quad_coeff"], dtype=np.float32)
            self.bank.trust_radius = np.asarray(data["chart_trust_radius"], dtype=np.float32)
            self.bank.success_count = np.asarray(data["chart_success_count"], dtype=np.int32)
            self.bank.failure_count = np.asarray(data["chart_failure_count"], dtype=np.int32)
            self.bank.local_p95_seconds = np.asarray(data["chart_local_p95_seconds"], dtype=np.float32)
            self.bank.jacobian_condition = np.asarray(data["chart_jacobian_condition"], dtype=np.float32)
            self.bank.jacobian_sigma_min = np.asarray(data["chart_jacobian_sigma_min"], dtype=np.float32)
            self.bank.solver_method = np.asarray(data["chart_solver_method"], dtype=np.int8) if "chart_solver_method" in data.files else np.zeros(len(self.bank.launch), dtype=np.int8)
            self.bank.patch_points = np.asarray(data["chart_patch_points"], dtype=np.int16) if "chart_patch_points" in data.files else np.zeros(len(self.bank.launch), dtype=np.int16)
            self.bank.refinement_count = np.asarray(data["chart_refinement_count"], dtype=np.int16) if "chart_refinement_count" in data.files else np.zeros(len(self.bank.launch), dtype=np.int16)
            nchart = len(self.bank.launch)
            self.bank.trust_basis = np.asarray(data["chart_trust_basis"], dtype=np.float32) if "chart_trust_basis" in data.files else np.repeat(np.eye(7, dtype=np.float32)[None, :, :], nchart, axis=0)
            self.bank.trust_radii = np.asarray(data["chart_trust_radii"], dtype=np.float32) if "chart_trust_radii" in data.files else np.repeat(np.maximum(self.bank.trust_radius[:, None], 1.0e-6), 7, axis=1).astype(np.float32)
            self.bank.chart_type = np.asarray(data["chart_type"], dtype=np.int8) if "chart_type" in data.files else np.zeros(nchart, dtype=np.int8)
            self.bank.patch_scale = np.asarray(data["chart_patch_scale"], dtype=np.float32) if "chart_patch_scale" in data.files else np.ones(nchart, dtype=np.float32)
            # Pre-v5 checkpoints contain coefficients in the raw canonical basis.
            # They remain queryable as raw charts while all newly generated charts
            # use the analytical center-rotated Cartesian basis.
            self.bank.transform_mode = np.asarray(data["chart_transform_mode"], dtype=np.int8) if "chart_transform_mode" in data.files else np.zeros(nchart, dtype=np.int8)
            self.bank.transform_scale = np.asarray(data["chart_transform_scale"], dtype=np.float32) if "chart_transform_scale" in data.files else np.repeat(self.feature_scale[None, :], nchart, axis=0).astype(np.float32)
            raw_witness = np.asarray(data["chart_witness_endpoint"], dtype=np.float32) if "chart_witness_endpoint" in data.files else None
            raw_witness_count = np.asarray(data["chart_witness_count"], dtype=np.int8) if "chart_witness_count" in data.files else None
            self.bank.witness_endpoint, self.bank.witness_count = _normalize_witness_storage(
                raw_witness, raw_witness_count, self.bank.endpoint, self.bank.witness_capacity
            )
        default_solver = Path(__file__).with_name("pmp_extremal_atlas.py")
        self.solver_path = Path(solver_path or self.metadata.get("solver_path") or default_solver).resolve()
        stored_solver_digest = self.metadata.get("solver_digest")
        if stored_solver_digest and stored_solver_digest != _file_digest(self.solver_path):
            warnings.warn(
                "the query dynamics solver differs from the solver used to build this chart atlas",
                RuntimeWarning,
            )
        self.backend = load_backend(self.solver_path)
        self.backend_cfg = _backend_config(self.backend, self.config.backend_config)
        self.query_cfg = _query_config(self.backend, self.config.query_config)
        self.corrector_profile_configs = {
            name: _corrector_profile_config(self.query_cfg, name, self.config.chart)
            for name in CORRECTOR_PROFILE_NAMES
        }
        self.bounds = self.backend._configured_correction_bounds(self.backend_cfg)

    def coverage_scores(self, targets: FloatArray) -> tuple[FloatArray, NDArray[np.int64]]:
        return self.bank.coverage_scores(
            targets,
            self.config.routing_neighbours,
            self.config.routing_winding_window,
        )

    def correct_target(
        self,
        target: FloatArray,
        *,
        max_ratio: Optional[float] = None,
        return_reason: bool = False,
        routing_neighbours: Optional[int] = None,
        max_seed_attempts: Optional[int] = None,
    ):
        """Correct one canonical target from the validated chart bank.

        The returned reason deliberately separates routing, trust-gate,
        polynomial, and corrector failures so coverage diagnostics can identify
        the actual bottleneck instead of reporting one undifferentiated miss.
        """
        target = np.asarray(target, dtype=float)
        if target.shape != (7,):
            raise ValueError("target must have seven canonical endpoint coordinates")
        if len(self.bank) == 0:
            result = (None, -1, math.inf, "empty-atlas", 0.0)
            return result if return_reason else None
        k = int(routing_neighbours or self.config.routing_neighbours)
        indices = self.bank.candidate_indices(target[None, :], k)[0]
        target_winding = int(round(float(target[3]) / (2.0 * math.pi)))
        candidates: list[tuple[float, int, FloatArray]] = []
        branch_compatible = 0
        invalid_predictions = 0
        for idx in indices:
            idx = int(idx)
            winding = int(round(float(self.bank.diagnostics[idx, 5])))
            if abs(winding - target_winding) > self.config.routing_winding_window:
                continue
            branch_compatible += 1
            q_pred, _score = self.bank.predict_q(idx, target)
            ratio = self.bank.trust_ratio(idx, target)
            if not math.isfinite(ratio):
                invalid_predictions += 1
                continue
            try:
                seed_launch = q_to_launch(
                    float(target[0]), float(target[1]), q_pred, bounds=self.bounds
                )
            except (ValueError, OverflowError, FloatingPointError):
                invalid_predictions += 1
                continue
            candidates.append((ratio, idx, seed_launch))
        candidates.sort(key=lambda item: item[0])
        limit = self.config.routing_max_ratio if max_ratio is None else float(max_ratio)
        if candidates and candidates[0][0] <= 1.0e-12:
            idx = candidates[0][1]
            result = (self.bank.launch[idx].copy(), idx, 0.0, "chart-center", 0.0)
            return result if return_reason else result[0]
        if branch_compatible == 0:
            result = (None, -1, math.inf, "no-branch-compatible-chart", 0.0)
            return result if return_reason else None
        if not candidates:
            result = (None, -1, math.inf, "invalid-chart-prediction", 0.0)
            return result if return_reason else None
        best_ratio = candidates[0][0]
        if best_ratio > limit:
            result = (None, candidates[0][1], best_ratio, "outside-trust-region", 0.0)
            return result if return_reason else None
        started = time.monotonic()
        attempts = 0
        attempt_limit = int(max_seed_attempts or self.query_cfg.max_seed_attempts)
        for ratio, idx, seed_launch in candidates:
            if ratio > limit:
                break
            method_index = int(self.bank.solver_method[idx]) if len(self.bank.solver_method) else 0
            method_index = min(max(method_index, 0), len(CORRECTOR_PROFILE_NAMES) - 1)
            method_name = CORRECTOR_PROFILE_NAMES[method_index]
            corrected, _nfev = self.backend._correct_target_from_explicit_seed(
                target, seed_launch, self.corrector_profile_configs[method_name], self.bounds
            )
            attempts += 1
            if corrected is not None:
                elapsed = time.monotonic() - started
                result = (corrected, idx, ratio, "converged", elapsed)
                return result if return_reason else corrected
            if attempts >= attempt_limit:
                break
        elapsed = time.monotonic() - started
        wall = float(getattr(self.query_cfg, "wall_time_seconds", math.inf))
        reason = "timeout-or-budget" if elapsed >= 0.90 * wall else "correction-failed"
        result = (None, candidates[0][1], best_ratio, reason, elapsed)
        return result if return_reason else None

    def solve(
        self,
        initial,
        final,
        capability,
        *,
        revolutions: int = 0,
        return_all: bool = False,
    ):
        boundary = self.backend._canonicalize_boundary(initial, final, capability, revolutions)
        corrected, chart_index, ratio, reason, _elapsed = self.correct_target(
            boundary.target, return_reason=True
        )
        if corrected is None:
            raise RuntimeError(
                f"No validated chart converged: reason={reason}, best trust ratio={ratio:.3g}"
            )
        t_eval = np.linspace(0.0, corrected[6], self.query_cfg.trajectory_points)
        sol, normal_constant, status = self.backend._integrate_launch(
            corrected,
            rtol=self.query_cfg.rtol,
            atol=self.query_cfg.atol,
            max_step=self.query_cfg.max_step,
            r_collision=self.query_cfg.r_collision,
            r_escape=self.query_cfg.r_escape,
            max_acceleration=self.query_cfg.max_acceleration,
            t_eval=t_eval,
        )
        if sol is None or status != "ok":
            raise RuntimeError("corrected launch failed final trajectory integration")
        radius, theta, ur, ut, kappa = self.backend._endpoint_from_state(sol.y[:, -1])
        raw = np.array(
            [
                math.log(radius) - boundary.target[2],
                theta - boundary.target[3],
                ur - boundary.target[4],
                ut - boundary.target[5],
                math.log(kappa) - boundary.target[6],
            ],
            dtype=float,
        )
        trajectory = self.backend._dimensional_trajectory(sol.t, sol.y, boundary, capability)
        radii = np.linalg.norm(sol.y[0:2, :], axis=0)
        acceleration = np.linalg.norm(sol.y[4:6, :], axis=0)
        solution = self.backend.ShootingSolution(
            launch=corrected,
            residual=raw,
            residual_norm=float(np.linalg.norm(raw / np.asarray(self.query_cfg.residual_scale))),
            normal_constant=float(normal_constant),
            minimum_radius=float(np.min(radii)),
            maximum_acceleration=float(np.max(acceleration)),
            radial_turns=int(self.backend._count_radial_turns(sol.y)),
            trajectory=trajectory,
        )
        return [solution] if return_all else solution


# ---------------------------------------------------------------------------
# Generator state and persistence
# ---------------------------------------------------------------------------


@dataclass
class GeneratorState:
    phase: str = "running"
    sobol_cursor: int = 0
    round_number: int = 0
    successful_rounds: int = 0
    generation_complete: bool = False
    coverage_converged: bool = False
    interrupted: bool = False
    interruption_reason: str = ""
    coverage_history: list[dict[str, Any]] = field(default_factory=list)
    repair_epoch: int = 0
    repair_pass: int = 0
    repair_epoch_active: bool = False
    audit_cursor: int = 0


@dataclass
class WorkingSet:
    bank: ChartBank
    fit_launch: FloatArray
    fit_endpoint: FloatArray
    fit_diagnostics: FloatArray
    probe_launch: FloatArray  # repair reservoir; known launches may become centers
    probe_endpoint: FloatArray
    probe_diagnostics: FloatArray
    holdout_launch: FloatArray  # never used as chart centers or fit observations
    holdout_endpoint: FloatArray
    holdout_diagnostics: FloatArray
    pending_launch: FloatArray
    pending_endpoint: FloatArray
    pending_diagnostics: FloatArray
    state: GeneratorState


def _empty_working_set(config: ChartAtlasConfig) -> WorkingSet:
    bank = ChartBank(
        np.asarray(config.feature_scale), np.asarray(config.q_scale),
        config.chart.quadratic_rank, config.chart.trust_endpoint_weight,
        config.chart.witness_capacity,
        np.asarray(config.chart.search_feature_scale, dtype=float),
        config.chart.output_transform_asinh_scale,
    )
    return WorkingSet(
        bank=bank,
        fit_launch=np.empty((0, 7), dtype=float),
        fit_endpoint=np.empty((0, 7), dtype=float),
        fit_diagnostics=np.empty((0, 6), dtype=float),
        probe_launch=np.empty((0, 7), dtype=float),
        probe_endpoint=np.empty((0, 7), dtype=float),
        probe_diagnostics=np.empty((0, 6), dtype=float),
        holdout_launch=np.empty((0, 7), dtype=float),
        holdout_endpoint=np.empty((0, 7), dtype=float),
        holdout_diagnostics=np.empty((0, 6), dtype=float),
        pending_launch=np.empty((0, 7), dtype=float),
        pending_endpoint=np.empty((0, 7), dtype=float),
        pending_diagnostics=np.empty((0, 6), dtype=float),
        state=GeneratorState(),
    )


def _config_digest(config: ChartAtlasConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_save(
    path: Path,
    work: WorkingSet,
    config: ChartAtlasConfig,
    solver_path: Path,
    *,
    compressed: bool,
) -> None:
    metadata = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "model": "planar-kepler-costate-free-normal-pmp",
        "representation": "validated-anisotropic-mixed-local-inverse-charts-in-analytically-transformed-output-basis",
        "solver_path": str(solver_path),
        "solver_digest": _file_digest(solver_path),
        "config": asdict(config),
        "config_digest": _config_digest(config),
        "state": asdict(work.state),
        "chart_count": len(work.bank),
        "fit_pool_rows": len(work.fit_launch),
        "probe_rows": len(work.probe_launch),
        "repair_probe_rows": len(work.probe_launch),
        "holdout_probe_rows": len(work.holdout_launch),
        "pending_chart_centers": len(work.pending_launch),
        "endpoint_columns": list(ENDPOINT_COLUMNS),
        "transformed_endpoint_columns": list(TRANSFORMED_ENDPOINT_COLUMNS),
        "search_endpoint_columns": list(SEARCH_ENDPOINT_COLUMNS),
        "launch_columns": list(LAUNCH_COLUMNS),
        "q_columns": list(Q_COLUMNS),
        "diagnostic_columns": list(DIAGNOSTIC_COLUMNS),
        "saved_utc_unix": time.time(),
    }
    arrays = {
        "chart_launch": work.bank.launch,
        "chart_endpoint": work.bank.endpoint,
        "chart_diagnostics": work.bank.diagnostics.astype(np.float32),
        "chart_affine": work.bank.affine,
        "chart_quad_dirs": work.bank.quad_dirs,
        "chart_quad_coeff": work.bank.quad_coeff,
        "chart_trust_radius": work.bank.trust_radius,
        "chart_success_count": work.bank.success_count,
        "chart_failure_count": work.bank.failure_count,
        "chart_local_p95_seconds": work.bank.local_p95_seconds,
        "chart_jacobian_condition": work.bank.jacobian_condition,
        "chart_jacobian_sigma_min": work.bank.jacobian_sigma_min,
        "chart_solver_method": work.bank.solver_method,
        "chart_patch_points": work.bank.patch_points,
        "chart_refinement_count": work.bank.refinement_count,
        "chart_trust_basis": work.bank.trust_basis,
        "chart_trust_radii": work.bank.trust_radii,
        "chart_type": work.bank.chart_type,
        "chart_patch_scale": work.bank.patch_scale,
        "chart_transform_mode": work.bank.transform_mode,
        "chart_transform_scale": work.bank.transform_scale,
        "chart_witness_endpoint": work.bank.witness_endpoint,
        "chart_witness_count": work.bank.witness_count,
        "feature_scale": work.bank.feature_scale,
        "q_scale": work.bank.q_scale,
        "fit_launch": work.fit_launch,
        "fit_endpoint": work.fit_endpoint,
        "fit_diagnostics": work.fit_diagnostics.astype(np.float32),
        "probe_launch": work.probe_launch,
        "probe_endpoint": work.probe_endpoint,
        "probe_diagnostics": work.probe_diagnostics.astype(np.float32),
        "holdout_launch": work.holdout_launch,
        "holdout_endpoint": work.holdout_endpoint,
        "holdout_diagnostics": work.holdout_diagnostics.astype(np.float32),
        "pending_launch": work.pending_launch,
        "pending_endpoint": work.pending_endpoint,
        "pending_diagnostics": work.pending_diagnostics.astype(np.float32),
        "metadata_json": np.array(json.dumps(metadata, separators=(",", ":"))),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    save = np.savez_compressed if compressed else np.savez
    with open(temp, "wb") as handle:
        save(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _load_working_set(
    path: Path,
    config: ChartAtlasConfig,
    solver_path: Path,
    *,
    allow_policy_change: bool = False,
) -> WorkingSet:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        if metadata.get("format") != FORMAT_NAME:
            raise ValueError("resume file is not a chart atlas")
        if metadata.get("config_digest") != _config_digest(config):
            if not allow_policy_change:
                raise ValueError(
                    "configuration does not match checkpoint; use the exact original JSON "
                    "or pass --resume-policy-change for compatible generator-policy changes"
                )
            previous = metadata.get("config", {})
            previous_chart = previous.get("chart", {})
            old_format = int(metadata.get("format_version", 0))
            transform_compatible = (
                old_format < 5
                or (
                    previous_chart.get("output_transform", "raw") == config.chart.output_transform
                    and float(previous_chart.get("output_transform_asinh_scale", 0.0))
                    == float(config.chart.output_transform_asinh_scale)
                )
            )
            critical_equal = (
                previous.get("backend_config") == asdict(config).get("backend_config")
                and previous.get("query_config") == asdict(config).get("query_config")
                and previous.get("feature_scale") == list(config.feature_scale)
                and previous.get("q_scale") == list(config.q_scale)
                and int(previous_chart.get("quadratic_rank", -1)) == int(config.chart.quadratic_rank)
                and transform_compatible
            )
            if not critical_equal:
                raise ValueError(
                    "checkpoint dynamics/query/scales/quadratic rank are incompatible with the new configuration"
                )
            warnings.warn(
                "resuming with changed chart/generator policy; existing chart coefficients are retained "
                "and will be audited under the new policy",
                RuntimeWarning,
            )
        stored_solver_digest = metadata.get("solver_digest")
        if stored_solver_digest and stored_solver_digest != _file_digest(solver_path):
            raise ValueError("dynamics solver does not match the checkpointed solver")
        bank = ChartBank(
            np.asarray(data["feature_scale"], dtype=float),
            np.asarray(data["q_scale"], dtype=float),
            config.chart.quadratic_rank,
            config.chart.trust_endpoint_weight,
            config.chart.witness_capacity,
            np.asarray(config.chart.search_feature_scale, dtype=float),
            config.chart.output_transform_asinh_scale,
        )
        bank.launch = np.asarray(data["chart_launch"], dtype=float)
        bank.endpoint = np.asarray(data["chart_endpoint"], dtype=float)
        bank.diagnostics = np.asarray(data["chart_diagnostics"], dtype=float)
        bank.affine = np.asarray(data["chart_affine"], dtype=np.float32)
        bank.quad_dirs = np.asarray(data["chart_quad_dirs"], dtype=np.float32)
        bank.quad_coeff = np.asarray(data["chart_quad_coeff"], dtype=np.float32)
        bank.trust_radius = np.asarray(data["chart_trust_radius"], dtype=np.float32)
        bank.success_count = np.asarray(data["chart_success_count"], dtype=np.int32)
        bank.failure_count = np.asarray(data["chart_failure_count"], dtype=np.int32)
        bank.local_p95_seconds = np.asarray(data["chart_local_p95_seconds"], dtype=np.float32)
        bank.jacobian_condition = np.asarray(data["chart_jacobian_condition"], dtype=np.float32)
        bank.jacobian_sigma_min = np.asarray(data["chart_jacobian_sigma_min"], dtype=np.float32)
        bank.solver_method = (
            np.asarray(data["chart_solver_method"], dtype=np.int8)
            if "chart_solver_method" in data.files
            else np.zeros(len(bank.launch), dtype=np.int8)
        )
        bank.patch_points = (
            np.asarray(data["chart_patch_points"], dtype=np.int16)
            if "chart_patch_points" in data.files
            else np.zeros(len(bank.launch), dtype=np.int16)
        )
        bank.refinement_count = (
            np.asarray(data["chart_refinement_count"], dtype=np.int16)
            if "chart_refinement_count" in data.files
            else np.zeros(len(bank.launch), dtype=np.int16)
        )
        nchart = len(bank.launch)
        bank.trust_basis = (
            np.asarray(data["chart_trust_basis"], dtype=np.float32)
            if "chart_trust_basis" in data.files
            else np.repeat(np.eye(7, dtype=np.float32)[None, :, :], nchart, axis=0)
        )
        bank.trust_radii = (
            np.asarray(data["chart_trust_radii"], dtype=np.float32)
            if "chart_trust_radii" in data.files
            else np.repeat(np.maximum(bank.trust_radius[:, None], 1.0e-6), 7, axis=1).astype(np.float32)
        )
        bank.chart_type = (
            np.asarray(data["chart_type"], dtype=np.int8)
            if "chart_type" in data.files else np.zeros(nchart, dtype=np.int8)
        )
        bank.patch_scale = (
            np.asarray(data["chart_patch_scale"], dtype=np.float32)
            if "chart_patch_scale" in data.files else np.ones(nchart, dtype=np.float32)
        )
        bank.transform_mode = (
            np.asarray(data["chart_transform_mode"], dtype=np.int8)
            if "chart_transform_mode" in data.files else np.zeros(nchart, dtype=np.int8)
        )
        bank.transform_scale = (
            np.asarray(data["chart_transform_scale"], dtype=np.float32)
            if "chart_transform_scale" in data.files
            else np.repeat(np.asarray(data["feature_scale"], dtype=np.float32)[None, :], nchart, axis=0)
        )
        raw_witness = (
            np.asarray(data["chart_witness_endpoint"], dtype=np.float32)
            if "chart_witness_endpoint" in data.files else None
        )
        raw_witness_count = (
            np.asarray(data["chart_witness_count"], dtype=np.int8)
            if "chart_witness_count" in data.files else None
        )
        bank.witness_endpoint, bank.witness_count = _normalize_witness_storage(
            raw_witness, raw_witness_count, bank.endpoint, bank.witness_capacity
        )
        state_payload = metadata.get("state", {})
        state_kwargs: dict[str, Any] = {}
        for f in dataclasses.fields(GeneratorState):
            if f.name in state_payload:
                state_kwargs[f.name] = state_payload[f.name]
            elif f.default is not dataclasses.MISSING:
                state_kwargs[f.name] = f.default
            elif f.default_factory is not dataclasses.MISSING:  # type: ignore[comparison-overlap]
                state_kwargs[f.name] = f.default_factory()  # type: ignore[misc]
        state = GeneratorState(**state_kwargs)
        state.generation_complete = False
        state.interrupted = False
        state.interruption_reason = ""
        return WorkingSet(
            bank=bank,
            fit_launch=np.asarray(data["fit_launch"], dtype=float),
            fit_endpoint=np.asarray(data["fit_endpoint"], dtype=float),
            fit_diagnostics=np.asarray(data["fit_diagnostics"], dtype=float),
            probe_launch=np.asarray(data["probe_launch"], dtype=float),
            probe_endpoint=np.asarray(data["probe_endpoint"], dtype=float),
            probe_diagnostics=np.asarray(data["probe_diagnostics"], dtype=float),
            holdout_launch=(np.asarray(data["holdout_launch"], dtype=float) if "holdout_launch" in data.files else np.empty((0, 7), dtype=float)),
            holdout_endpoint=(np.asarray(data["holdout_endpoint"], dtype=float) if "holdout_endpoint" in data.files else np.empty((0, 7), dtype=float)),
            holdout_diagnostics=(np.asarray(data["holdout_diagnostics"], dtype=float) if "holdout_diagnostics" in data.files else np.empty((0, 6), dtype=float)),
            pending_launch=np.asarray(data["pending_launch"], dtype=float),
            pending_endpoint=np.asarray(data["pending_endpoint"], dtype=float),
            pending_diagnostics=np.asarray(data["pending_diagnostics"], dtype=float),
            state=state,
        )


# ---------------------------------------------------------------------------
# Adaptive generation helpers
# ---------------------------------------------------------------------------


def _merge_thinned(
    backend,
    old_launch: FloatArray,
    old_endpoint: FloatArray,
    old_diag: FloatArray,
    new_launch: FloatArray,
    new_endpoint: FloatArray,
    new_diag: FloatArray,
    feature_scale: FloatArray,
    cell_size: float,
    max_rows: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    if len(old_launch) == 0:
        combined = (new_launch, new_endpoint, new_diag)
    elif len(new_launch) == 0:
        combined = (old_launch, old_endpoint, old_diag)
    else:
        combined = (
            np.vstack((old_launch, new_launch)),
            np.vstack((old_endpoint, new_endpoint)),
            np.vstack((old_diag, new_diag)),
        )
    return backend._thin_endpoint_rows(
        combined[0], combined[1], combined[2], feature_scale, cell_size, max_rows
    )


def _select_spatially_separated(
    endpoint: FloatArray,
    diagnostics: FloatArray,
    priority: FloatArray,
    feature_scale: FloatArray,
    count: int,
    minimum_separation: float,
) -> NDArray[np.int64]:
    if len(endpoint) == 0 or count <= 0:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(np.asarray(priority, dtype=float))[::-1]
    features = _global_search_transform(endpoint) / np.asarray(feature_scale, dtype=float)
    labels = branch_labels(diagnostics)
    selected: list[int] = []
    selected_by_branch: dict[tuple[int, int], list[FloatArray]] = {}
    for idx in order:
        label = tuple(int(v) for v in labels[idx])
        old = selected_by_branch.setdefault(label, [])
        point = features[idx]
        if old:
            distance = np.linalg.norm(np.asarray(old) - point, axis=1)
            if np.min(distance) < minimum_separation:
                continue
        old.append(point)
        selected.append(int(idx))
        if len(selected) >= count:
            break
    return np.asarray(selected, dtype=np.int64)


def _neighbour_payloads(
    work: WorkingSet,
    center_launch: FloatArray,
    center_endpoint: FloatArray,
    center_diag: FloatArray,
    fit_neighbours: int,
    seed_base: int,
) -> list[tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, int]]:
    if len(center_launch) == 0:
        return []
    pool_labels = branch_labels(work.fit_diagnostics)
    center_labels = branch_labels(center_diag)
    payloads = []
    branch_cache: dict[tuple[int, int], tuple[NDArray[np.int64], Optional[cKDTree]]] = {}
    for i in range(len(center_launch)):
        label = tuple(int(v) for v in center_labels[i])
        if label not in branch_cache:
            indices = np.flatnonzero(np.all(pool_labels == np.asarray(label)[None, :], axis=1))
            tree = cKDTree(
                _global_search_transform(work.fit_endpoint[indices]) / work.bank.search_feature_scale
            ) if len(indices) else None
            branch_cache[label] = (indices, tree)
        indices, tree = branch_cache[label]
        if tree is None or len(indices) == 0:
            selected = np.empty(0, dtype=int)
        else:
            k = min(fit_neighbours, len(indices))
            center_search = _global_search_transform(center_endpoint[i][None, :])[0]
            _dist, local = tree.query(center_search / work.bank.search_feature_scale, k=k)
            selected = indices[np.atleast_1d(local).astype(int)]
        payloads.append(
            (
                center_launch[i].copy(),
                center_endpoint[i].copy(),
                center_diag[i].copy(),
                work.fit_launch[selected].copy(),
                work.fit_endpoint[selected].copy(),
                work.fit_diagnostics[selected].copy(),
                int(seed_base + i),
            )
        )
    return payloads


def _build_pending_charts(
    work: WorkingSet,
    config: ChartAtlasConfig,
    solver_path: Path,
    output_path: Path,
) -> None:
    batch_size = config.generator.chart_build_batch
    batch_number = 0
    executor: Optional[ProcessPoolExecutor] = None
    if config.generator.chart_workers > 1:
        executor = ProcessPoolExecutor(
            max_workers=config.generator.chart_workers,
            initializer=_chart_worker_init,
            initargs=(
                str(solver_path),
                config.backend_config,
                config.query_config,
                asdict(config.chart),
                config.feature_scale,
                config.q_scale,
            ),
        )
    try:
        while len(work.pending_launch) and len(work.bank) < config.generator.max_charts:
            batch_number += 1
            take = min(batch_size, len(work.pending_launch))
            center_launch = work.pending_launch[:take]
            center_endpoint = work.pending_endpoint[:take]
            center_diag = work.pending_diagnostics[:take]
            payloads = _neighbour_payloads(
                work,
                center_launch,
                center_endpoint,
                center_diag,
                config.chart.fit_neighbours,
                config.generator.seed
                + 10_000_000 * work.state.round_number
                + batch_number * batch_size,
            )
            built: list[BuiltChart] = []
            if executor is None:
                backend = load_backend(solver_path)
                backend_cfg = _backend_config(backend, config.backend_config)
                query_cfg = _query_config(backend, config.query_config)
                bounds = backend._configured_correction_bounds(backend_cfg)
                for payload in payloads:
                    chart = _build_chart(
                        backend, backend_cfg, query_cfg, config.chart,
                        work.bank.feature_scale, work.bank.q_scale, bounds, *payload
                    )
                    if chart is not None:
                        built.append(chart)
            else:
                futures = [executor.submit(_build_chart_worker, payload) for payload in payloads]
                for future in as_completed(futures):
                    chart = future.result()
                    if chart is not None:
                        built.append(chart)
            capacity = config.generator.max_charts - len(work.bank)
            if len(built) > capacity:
                built = built[:capacity]
            added = work.bank.append(built)
            work.pending_launch = work.pending_launch[take:]
            work.pending_endpoint = work.pending_endpoint[take:]
            work.pending_diagnostics = work.pending_diagnostics[take:]
            print(
                f"    chart batch {batch_number}: built={added}, "
                f"charts={len(work.bank):,}, pending={len(work.pending_launch):,}"
            )
            if config.generator.checkpoint_enabled and (
                batch_number % config.generator.checkpoint_every_chart_batches == 0
            ):
                _atomic_save(
                    output_path, work, config, solver_path,
                    compressed=config.generator.checkpoint_compressed,
                )
                print(f"    checkpoint: {output_path}")
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)


def _built_chart_solves_targets(
    built: BuiltChart, targets: FloatArray, backend, query_cfg, chart_cfg: ChartBuildConfig,
    bounds: tuple[FloatArray, FloatArray], feature_scale: FloatArray, q_scale: FloatArray,
) -> bool:
    if len(targets) == 0:
        return True
    method_index = min(max(int(built.solver_method), 0), len(CORRECTOR_PROFILE_NAMES) - 1)
    method_cfg = _corrector_profile_config(query_cfg, CORRECTOR_PROFILE_NAMES[method_index], chart_cfg)
    for target in np.asarray(targets, dtype=float):
        q, _z, _score = chart_predict_q_normalized(
            target, built.endpoint, built.launch, built.affine, built.quad_dirs,
            built.quad_coeff, built.transform_mode, built.transform_scale, q_scale,
            chart_cfg.trust_endpoint_weight, chart_cfg.output_transform_asinh_scale,
        )
        try:
            seed_launch = q_to_launch(float(target[0]), float(target[1]), q, bounds=bounds)
            corrected, _nfev = backend._correct_target_from_explicit_seed(
                target, seed_launch, method_cfg, bounds
            )
        except Exception:
            corrected = None
        if corrected is None:
            return False
    return True


def _refine_existing_charts(
    work: WorkingSet,
    config: ChartAtlasConfig,
    solver_path: Path,
    output_path: Path,
) -> int:
    count = config.generator.refine_charts_per_round
    if (
        count <= 0
        or work.state.round_number < config.generator.refine_after_round
        or len(work.bank) == 0
        or len(work.fit_launch) == 0
    ):
        return 0
    eligible = np.flatnonzero(
        work.bank.refinement_count < config.generator.maximum_refinements_per_chart
    )
    if len(eligible) == 0:
        return 0
    failure_rate = work.bank.failure_count[eligible] / np.maximum(
        work.bank.success_count[eligible] + work.bank.failure_count[eligible], 1
    )
    priority = (
        1.0 / np.maximum(work.bank.trust_radius[eligible].astype(float), 1.0e-6)
        + 5.0 * failure_rate
        - 0.25 * work.bank.refinement_count[eligible]
    )
    remaining_capacity = max(0, config.generator.max_charts - len(work.bank))
    if config.generator.refinement_append_only:
        count = min(count, remaining_capacity)
    if count <= 0:
        return 0
    chosen = eligible[np.argsort(priority)[::-1][: min(count, len(eligible))]]
    payloads = _neighbour_payloads(
        work,
        work.bank.launch[chosen],
        work.bank.endpoint[chosen],
        work.bank.diagnostics[chosen],
        config.chart.fit_neighbours,
        config.generator.seed + 900_000_000 + work.state.round_number * 100_000,
    )
    if not payloads:
        return 0
    results: list[tuple[int, Optional[BuiltChart]]] = []
    if config.generator.chart_workers == 1:
        backend = load_backend(solver_path)
        backend_cfg = _backend_config(backend, config.backend_config)
        query_cfg = _query_config(backend, config.query_config)
        bounds = backend._configured_correction_bounds(backend_cfg)
        for chart_index, payload in zip(chosen, payloads):
            built = _build_chart(
                backend, backend_cfg, query_cfg, config.chart,
                work.bank.feature_scale, work.bank.q_scale, bounds, *payload
            )
            results.append((int(chart_index), built))
    else:
        with ProcessPoolExecutor(
            max_workers=config.generator.chart_workers,
            initializer=_chart_worker_init,
            initargs=(
                str(solver_path), config.backend_config, config.query_config,
                asdict(config.chart), config.feature_scale, config.q_scale,
            ),
        ) as executor:
            future_map = {
                executor.submit(_build_chart_worker, payload): int(chart_index)
                for chart_index, payload in zip(chosen, payloads)
            }
            for future in as_completed(future_map):
                results.append((future_map[future], future.result()))
    replaced = 0
    appended = 0
    verify_backend = load_backend(solver_path)
    verify_backend_cfg = _backend_config(verify_backend, config.backend_config)
    verify_query_cfg = _query_config(verify_backend, config.query_config)
    verify_bounds = verify_backend._configured_correction_bounds(verify_backend_cfg)
    for chart_index, built in results:
        if built is None:
            work.bank.refinement_count[chart_index] += 1
            continue
        old_radius = float(work.bank.trust_radius[chart_index])
        old_failures = int(work.bank.failure_count[chart_index])
        gain = built.trust_radius / max(old_radius, 1.0e-12)
        witness_count = int(work.bank.witness_count[chart_index])
        witnesses = work.bank.witness_endpoint[chart_index, :witness_count].astype(float)
        preserves = _built_chart_solves_targets(
            built, witnesses, verify_backend, verify_query_cfg, config.chart, verify_bounds,
            work.bank.feature_scale, work.bank.q_scale,
        )
        improves = gain >= config.generator.refinement_min_radius_gain or old_failures > built.local_failures
        if improves and preserves:
            merged_witnesses: list[FloatArray] = []
            for target in np.vstack((witnesses, built.witnesses)) if len(witnesses) else built.witnesses:
                target = np.asarray(target, dtype=float)
                if not merged_witnesses or all(
                    np.linalg.norm(
                        _normalized_chart_delta(
                            target[None, :], built.endpoint, built.transform_mode,
                            built.transform_scale, config.chart.output_transform_asinh_scale,
                        )[0]
                        - _normalized_chart_delta(
                            np.asarray(old, dtype=float)[None, :], built.endpoint, built.transform_mode,
                            built.transform_scale, config.chart.output_transform_asinh_scale,
                        )[0]
                    ) > 1.0e-6
                    for old in merged_witnesses
                ):
                    merged_witnesses.append(target)
                if len(merged_witnesses) >= config.chart.witness_capacity:
                    break
            built.witnesses = np.vstack(merged_witnesses) if merged_witnesses else built.witnesses
            if config.generator.refinement_append_only:
                work.bank.append([built])
                work.bank.refinement_count[chart_index] += 1
                work.bank.refinement_count[-1] = np.int16(work.bank.refinement_count[chart_index])
                appended += 1
            else:
                work.bank.replace_chart(chart_index, built)
                replaced += 1
        else:
            work.bank.refinement_count[chart_index] += 1
    if replaced or appended:
        _atomic_save(
            output_path, work, config, solver_path,
            compressed=config.generator.checkpoint_compressed,
        )
    print(
        f"    chart refinement: attempted={len(chosen)}, replaced={replaced}, "
        f"appended={appended}"
    )
    return replaced + appended


def _query_probe_set(
    atlas: LocalInverseChartAtlas,
    targets: FloatArray,
    *,
    max_ratio: Optional[float] = None,
    routing_neighbours: Optional[int] = None,
    max_seed_attempts: Optional[int] = None,
) -> tuple[NDArray[np.bool_], FloatArray, NDArray[np.int64], FloatArray, NDArray[np.str_]]:
    success = np.zeros(len(targets), dtype=bool)
    elapsed = np.zeros(len(targets), dtype=float)
    chart_index = np.full(len(targets), -1, dtype=np.int64)
    ratio = np.full(len(targets), np.inf, dtype=float)
    reasons = np.empty(len(targets), dtype="U40")
    for i, target in enumerate(targets):
        corrected, idx, r, reason, seconds = atlas.correct_target(
            target,
            max_ratio=max_ratio,
            return_reason=True,
            routing_neighbours=routing_neighbours,
            max_seed_attempts=max_seed_attempts,
        )
        success[i] = corrected is not None
        elapsed[i] = seconds
        chart_index[i] = idx
        ratio[i] = r
        reasons[i] = reason
    return success, elapsed, chart_index, ratio, reasons


def _reason_counts(reasons: NDArray[np.str_]) -> dict[str, int]:
    if len(reasons) == 0:
        return {}
    values, counts = np.unique(reasons.astype(str), return_counts=True)
    return {str(v): int(c) for v, c in zip(values, counts)}


def _evaluate_and_update(
    work: WorkingSet,
    backend,
    config: ChartAtlasConfig,
    solver_path: Path,
    output_path: Path,
    rng: np.random.Generator,
    fresh_endpoint: FloatArray,
    *,
    source_targets: Optional[FloatArray] = None,
    holdout_before_targets: Optional[FloatArray] = None,
    holdout_before_success: Optional[NDArray[np.bool_]] = None,
) -> dict[str, Any]:
    """Audit repair, fixed holdout, and post-build fresh populations separately."""
    _atomic_save(
        output_path, work, config, solver_path,
        compressed=config.generator.checkpoint_compressed,
    )
    atlas = LocalInverseChartAtlas(output_path, solver_path)

    def choose_targets(rows: FloatArray, limit: int) -> FloatArray:
        if len(rows) == 0:
            return np.empty((0, 7), dtype=float)
        n = min(int(limit), len(rows))
        idx = rng.choice(len(rows), size=n, replace=False)
        return np.asarray(rows[idx], dtype=float)

    repair_targets = choose_targets(work.probe_endpoint, config.generator.validation_test_per_round)
    if holdout_before_targets is not None:
        holdout_targets = np.asarray(holdout_before_targets, dtype=float)
    else:
        holdout_targets = choose_targets(work.holdout_endpoint, config.generator.validation_test_per_round)
    fresh_targets = np.asarray(fresh_endpoint[: config.generator.fresh_validation_rows], dtype=float)
    source_targets = (
        np.asarray(source_targets, dtype=float)
        if source_targets is not None else np.empty((0, 7), dtype=float)
    )

    rep_success, rep_elapsed, rep_chart, rep_ratio, rep_reason = _query_probe_set(atlas, repair_targets)
    hold_success, hold_elapsed, hold_chart, hold_ratio, hold_reason = _query_probe_set(atlas, holdout_targets)
    fresh_success, fresh_elapsed, fresh_chart, fresh_ratio, fresh_reason = _query_probe_set(atlas, fresh_targets)
    source_success, source_elapsed, source_chart, source_ratio, source_reason = _query_probe_set(atlas, source_targets)

    # Independent successes become persistent witnesses; independent failures
    # inside a claimed ellipsoid shrink its certificate.
    for targets, success, chart, ratio in (
        (holdout_targets, hold_success, hold_chart, hold_ratio),
        (fresh_targets, fresh_success, fresh_chart, fresh_ratio),
    ):
        for target, ok, idx, r in zip(targets, success, chart, ratio):
            if idx < 0:
                continue
            if ok:
                work.bank.success_count[idx] += 1
                work.bank.add_witness(int(idx), target)
            elif np.isfinite(r) and r <= 1.0:
                work.bank.failure_count[idx] += 1
                work.bank.shrink_trust(int(idx), max(0.10, 0.90 * float(r)))
    work.bank.invalidate_tree()

    def metrics(success, elapsed, ratio, reasons):
        if len(success) == 0:
            return {
                "tested": 0, "successes": 0, "success_rate": 0.0,
                "p95_seconds": math.inf, "median_ratio": math.inf,
                "unsupported_rate": 1.0, "failure_reasons": {},
            }
        successful_times = elapsed[success]
        finite_ratio = ratio[np.isfinite(ratio)]
        return {
            "tested": int(len(success)),
            "successes": int(np.count_nonzero(success)),
            "success_rate": float(np.mean(success)),
            "p95_seconds": float(np.quantile(successful_times, 0.95)) if len(successful_times) else math.inf,
            "median_ratio": float(np.median(finite_ratio)) if len(finite_ratio) else math.inf,
            "unsupported_rate": float(np.mean(ratio > 1.0)),
            "failure_reasons": _reason_counts(reasons[~success]),
        }

    holdout_gain = 0
    if holdout_before_success is not None and len(holdout_before_success) == len(hold_success):
        holdout_gain = int(np.count_nonzero(hold_success & ~np.asarray(holdout_before_success, dtype=bool)))

    failed_targets = np.vstack((holdout_targets[~hold_success], fresh_targets[~fresh_success])) \
        if (np.any(~hold_success) or np.any(~fresh_success)) else np.empty((0, 7))
    ungated_rows = min(config.generator.diagnostic_ungated_rows, len(failed_targets))
    if ungated_rows:
        choose = rng.choice(len(failed_targets), size=ungated_rows, replace=False)
        ung_success, ung_elapsed, _uc, _ur, ung_reason = _query_probe_set(
            atlas, failed_targets[choose], max_ratio=math.inf,
            routing_neighbours=config.generator.diagnostic_routing_neighbours,
        )
        ungated = {
            "tested": int(ungated_rows), "recovered": int(np.count_nonzero(ung_success)),
            "recovery_rate": float(np.mean(ung_success)),
            "p95_seconds": float(np.quantile(ung_elapsed[ung_success], 0.95)) if np.any(ung_success) else math.inf,
            "failure_reasons": _reason_counts(ung_reason[~ung_success]),
        }
    else:
        ungated = {"tested": 0, "recovered": 0, "recovery_rate": 0.0, "p95_seconds": math.inf, "failure_reasons": {}}

    center_count = min(config.generator.diagnostic_center_self_tests, len(work.bank))
    center_successes = 0
    center_errors: list[float] = []
    if center_count:
        center_indices = rng.choice(len(work.bank), size=center_count, replace=False)
        for idx in center_indices:
            exact = _exact_endpoint(backend, work.bank.launch[int(idx)], atlas.query_cfg)
            if exact is None:
                continue
            err = float(np.linalg.norm((exact - work.bank.endpoint[int(idx)]) / work.bank.feature_scale))
            center_errors.append(err)
            if err <= 1.0e-5:
                center_successes += 1
    center_self_test = {
        "tested": int(center_count), "successes": int(center_successes),
        "success_rate": float(center_successes / center_count) if center_count else 0.0,
        "p95_normalized_endpoint_error": float(np.quantile(center_errors, 0.95)) if center_errors else math.inf,
    }

    local_test = {"tested": 0, "successes": 0, "success_rate": 0.0, "failure_reasons": {}}
    if len(work.fit_endpoint) and len(work.bank) and config.generator.diagnostic_local_training_tests:
        sample_n = min(max(1024, 32 * config.generator.diagnostic_local_training_tests), len(work.fit_endpoint))
        sample_idx = rng.choice(len(work.fit_endpoint), size=sample_n, replace=False)
        sample_targets = work.fit_endpoint[sample_idx]
        sample_ratio, _sample_chart = work.bank.coverage_scores(
            sample_targets, config.routing_neighbours, config.routing_winding_window
        )
        inside = np.flatnonzero(sample_ratio <= 1.0)
        if len(inside):
            if len(inside) > config.generator.diagnostic_local_training_tests:
                inside = rng.choice(inside, size=config.generator.diagnostic_local_training_tests, replace=False)
            loc_success, _le, _lc, _lr, loc_reason = _query_probe_set(atlas, sample_targets[inside])
            local_test = {
                "tested": int(len(inside)), "successes": int(np.count_nonzero(loc_success)),
                "success_rate": float(np.mean(loc_success)),
                "failure_reasons": _reason_counts(loc_reason[~loc_success]),
            }

    scalar = work.bank.trust_radius.astype(float)
    directional = work.bank.trust_radii.astype(float).ravel() if len(work.bank) else np.empty(0)
    trust = {
        "chart_count": int(len(scalar)),
        "p05": float(np.quantile(scalar, 0.05)) if len(scalar) else 0.0,
        "median": float(np.median(scalar)) if len(scalar) else 0.0,
        "p95": float(np.quantile(scalar, 0.95)) if len(scalar) else 0.0,
        "directional_p05": float(np.quantile(directional, 0.05)) if len(directional) else 0.0,
        "directional_median": float(np.median(directional)) if len(directional) else 0.0,
        "directional_p95": float(np.quantile(directional, 0.95)) if len(directional) else 0.0,
    }
    profile_counts: dict[str, int] = {}
    for method_index in work.bank.solver_method.astype(int):
        name = CORRECTOR_PROFILE_NAMES[method_index] if 0 <= method_index < len(CORRECTOR_PROFILE_NAMES) else "unknown"
        profile_counts[str(name)] = profile_counts.get(str(name), 0) + 1
    type_names = {0: "quadratic", 1: "affine", 2: "micro"}
    type_counts: dict[str, int] = {}
    for value in work.bank.chart_type.astype(int):
        name = type_names.get(int(value), "unknown")
        type_counts[name] = type_counts.get(name, 0) + 1
    transform_counts: dict[str, int] = {}
    for value in work.bank.transform_mode.astype(int):
        name = TRANSFORM_NAMES.get(int(value), "unknown")
        transform_counts[name] = transform_counts.get(name, 0) + 1
    patch_points = work.bank.patch_points.astype(float)
    patch_scales = work.bank.patch_scale.astype(float)
    patch_stats = {
        "median_points": float(np.median(patch_points)) if len(patch_points) else 0.0,
        "median_scale": float(np.median(patch_scales)) if len(patch_scales) else 0.0,
        "p05_scale": float(np.quantile(patch_scales, 0.05)) if len(patch_scales) else 0.0,
        "p95_scale": float(np.quantile(patch_scales, 0.95)) if len(patch_scales) else 0.0,
    }

    return {
        "repair": metrics(rep_success, rep_elapsed, rep_ratio, rep_reason),
        "holdout": metrics(hold_success, hold_elapsed, hold_ratio, hold_reason),
        "fresh": metrics(fresh_success, fresh_elapsed, fresh_ratio, fresh_reason),
        "source_repair": metrics(source_success, source_elapsed, source_ratio, source_reason),
        "holdout_newly_covered": holdout_gain,
        "ungated": ungated,
        "center_self_test": center_self_test,
        "local_training_test": local_test,
        "trust_radius": trust,
        "corrector_profiles": profile_counts,
        "chart_types": type_counts,
        "transforms": transform_counts,
        "patch": patch_stats,
    }


def generate_chart_atlas(
    output_path: str | Path,
    config: ChartAtlasConfig,
    solver_path: str | Path,
    *,
    resume: bool = False,
    resume_policy_change: bool = False,
) -> Path:
    """Generate a chart atlas using fixed repair epochs and independent audits."""
    config.validate()
    output_path = Path(output_path).resolve()
    solver_path = Path(solver_path).resolve()
    backend = load_backend(solver_path)
    backend_cfg = _backend_config(backend, config.backend_config)
    query_cfg = _query_config(backend, config.query_config)
    feature_scale = np.asarray(config.feature_scale, dtype=float)

    if resume:
        if not output_path.exists():
            raise FileNotFoundError(f"cannot resume; checkpoint does not exist: {output_path}")
        work = _load_working_set(
            output_path, config, solver_path, allow_policy_change=resume_policy_change
        )
        print(
            f"Resuming chart atlas: charts={len(work.bank):,}, cursor={work.state.sobol_cursor:,}, "
            f"round={work.state.round_number}, epoch={work.state.repair_epoch}, "
            f"pass={work.state.repair_pass}, pending={len(work.pending_launch):,}"
        )
    else:
        work = _empty_working_set(config)
        _atomic_save(
            output_path, work, config, solver_path,
            compressed=config.generator.checkpoint_compressed,
        )
        print(f"Initial empty checkpoint: {output_path}")

    def deterministic_rng(tag: int = 0) -> np.random.Generator:
        seed = (
            int(config.generator.seed)
            + 1_000_003 * int(work.state.round_number)
            + 10_007 * int(work.state.repair_epoch)
            + 101 * int(work.state.repair_pass)
            + int(tag)
        ) & 0xFFFFFFFFFFFFFFFF
        return np.random.default_rng(seed)

    def generate_post_build_fresh() -> tuple[FloatArray, FloatArray, FloatArray]:
        desired = int(config.generator.fresh_validation_rows)
        if desired <= 0:
            return np.empty((0, 7)), np.empty((0, 7)), np.empty((0, 6))
        collected_launch: list[FloatArray] = []
        collected_endpoint: list[FloatArray] = []
        collected_diag: list[FloatArray] = []
        for attempt_round in range(4):
            missing = desired - sum(len(x) for x in collected_launch)
            if missing <= 0:
                break
            attempts = max(
                missing,
                missing * int(config.generator.fresh_audit_attempt_factor),
            )
            launches = sample_launch_chunk(
                backend_cfg,
                int(work.state.audit_cursor),
                attempts,
                int(getattr(backend_cfg, "seed", config.generator.seed)) + 0x51A7,
            )
            raw_launch, _raw_endpoint, _raw_diag = backend._generate_rows_from_launches(
                launches, backend_cfg
            )
            work.state.audit_cursor += attempts
            if len(raw_launch) == 0:
                continue
            rng = deterministic_rng(0xF123 + attempt_round)
            choose = rng.choice(
                len(raw_launch), size=min(max(missing * 2, missing), len(raw_launch)), replace=False
            )
            fl, fe, fd = _exact_probe_rows(
                backend, backend_cfg, query_cfg, raw_launch[choose]
            )
            if len(fl):
                collected_launch.append(fl)
                collected_endpoint.append(fe)
                collected_diag.append(fd)
        if not collected_launch:
            return np.empty((0, 7)), np.empty((0, 7)), np.empty((0, 6))
        launch = np.vstack(collected_launch)
        endpoint = np.vstack(collected_endpoint)
        diag = np.vstack(collected_diag)
        if len(launch) > desired:
            rng = deterministic_rng(0xF999)
            choose = rng.choice(len(launch), size=desired, replace=False)
            launch, endpoint, diag = launch[choose], endpoint[choose], diag[choose]
        return launch, endpoint, diag

    try:
        if len(work.pending_launch):
            print(f"Continuing {len(work.pending_launch):,} pending chart centers")
            _build_pending_charts(work, config, solver_path, output_path)

        while (
            work.state.round_number < config.generator.max_rounds
            and len(work.bank) < config.generator.max_charts
            and not work.state.coverage_converged
        ):
            # Begin a new fixed repair epoch only after the previous target set
            # has been repaired, exhausted, or reached its pass limit.
            if not work.state.repair_epoch_active:
                if work.state.sobol_cursor >= config.generator.max_launch_attempts:
                    break
                remaining = config.generator.max_launch_attempts - work.state.sobol_cursor
                attempts = min(config.generator.launch_batch_size, remaining)
                work.state.repair_epoch += 1
                work.state.repair_pass = 0
                print(
                    f"Epoch {work.state.repair_epoch}: integrating {attempts:,} launch attempts "
                    f"at Sobol cursor {work.state.sobol_cursor:,}"
                )
                launches = sample_launch_chunk(
                    backend_cfg,
                    work.state.sobol_cursor,
                    attempts,
                    int(getattr(backend_cfg, "seed", config.generator.seed)),
                )
                candidate_launch, candidate_endpoint, candidate_diag = backend._generate_rows_from_launches(
                    launches, backend_cfg
                )
                work.state.sobol_cursor += attempts
                print(f"    accepted forward/subarc rows={len(candidate_launch):,}")
                if len(candidate_launch):
                    work.fit_launch, work.fit_endpoint, work.fit_diagnostics = _merge_thinned(
                        backend,
                        work.fit_launch, work.fit_endpoint, work.fit_diagnostics,
                        candidate_launch, candidate_endpoint, candidate_diag,
                        feature_scale,
                        config.generator.fit_pool_cell_size,
                        config.generator.fit_pool_max_rows,
                    )
                    rng = deterministic_rng(0xE001)
                    repair_n = min(config.generator.validation_add_per_round, len(candidate_launch))
                    remaining_rows = max(0, len(candidate_launch) - repair_n)
                    holdout_n = min(config.generator.holdout_add_per_epoch, remaining_rows)
                    total = repair_n + holdout_n
                    chosen = rng.choice(len(candidate_launch), size=total, replace=False) if total else np.empty(0, dtype=int)
                    repair_idx = chosen[:repair_n]
                    holdout_idx = chosen[repair_n:]
                    if len(repair_idx):
                        rl, re, rd = _exact_probe_rows(
                            backend, backend_cfg, query_cfg, candidate_launch[repair_idx]
                        )
                        work.probe_launch, work.probe_endpoint, work.probe_diagnostics = _merge_thinned(
                            backend,
                            work.probe_launch, work.probe_endpoint, work.probe_diagnostics,
                            rl, re, rd,
                            feature_scale,
                            config.generator.candidate_cell_size,
                            config.generator.validation_reservoir_rows,
                        )
                    if len(holdout_idx):
                        hl, he, hd = _exact_probe_rows(
                            backend, backend_cfg, query_cfg, candidate_launch[holdout_idx]
                        )
                        work.holdout_launch, work.holdout_endpoint, work.holdout_diagnostics = _merge_thinned(
                            backend,
                            work.holdout_launch, work.holdout_endpoint, work.holdout_diagnostics,
                            hl, he, hd,
                            feature_scale,
                            config.generator.candidate_cell_size,
                            config.generator.holdout_reservoir_rows,
                        )
                work.state.repair_epoch_active = True
                _atomic_save(
                    output_path, work, config, solver_path,
                    compressed=config.generator.checkpoint_compressed,
                )

            work.state.round_number += 1
            work.state.repair_pass += 1
            rng = deterministic_rng(0xA11D)
            print(
                f"Round {work.state.round_number}: repair epoch {work.state.repair_epoch}, "
                f"pass {work.state.repair_pass}/{config.generator.repair_epoch_max_passes}; "
                f"repair probes={len(work.probe_endpoint):,}, holdout={len(work.holdout_endpoint):,}"
            )

            _atomic_save(
                output_path, work, config, solver_path,
                compressed=config.generator.checkpoint_compressed,
            )
            current_atlas = LocalInverseChartAtlas(output_path, solver_path)

            # Freeze a repair sample for this pass.  Only this population may
            # reveal its known launches and become chart centers.
            repair_n = min(config.generator.validation_test_per_round, len(work.probe_endpoint))
            repair_indices = (
                rng.choice(len(work.probe_endpoint), size=repair_n, replace=False)
                if repair_n else np.empty(0, dtype=int)
            )
            repair_launch = work.probe_launch[repair_indices]
            repair_endpoint = work.probe_endpoint[repair_indices]
            repair_diag = work.probe_diagnostics[repair_indices]
            if len(repair_endpoint) and len(work.bank):
                pre_success, _pe, _pc, pre_ratio, pre_reason = _query_probe_set(
                    current_atlas, repair_endpoint
                )
            elif len(repair_endpoint):
                pre_success = np.zeros(len(repair_endpoint), dtype=bool)
                pre_ratio = np.full(len(repair_endpoint), np.inf)
                pre_reason = np.full(len(repair_endpoint), "empty-atlas", dtype="U40")
            else:
                pre_success = np.empty(0, dtype=bool)
                pre_ratio = np.empty(0, dtype=float)
                pre_reason = np.empty(0, dtype="U40")

            # Fixed independent holdout sample before construction, used to
            # measure actual independent coverage gain from this pass.
            holdout_n = min(config.generator.validation_test_per_round, len(work.holdout_endpoint))
            holdout_indices = (
                rng.choice(len(work.holdout_endpoint), size=holdout_n, replace=False)
                if holdout_n else np.empty(0, dtype=int)
            )
            holdout_targets = work.holdout_endpoint[holdout_indices]
            if len(holdout_targets) and len(work.bank):
                holdout_before_success, *_ = _query_probe_set(current_atlas, holdout_targets)
            else:
                holdout_before_success = np.zeros(len(holdout_targets), dtype=bool)

            failed_idx = np.flatnonzero(~pre_success)
            priority = np.where(np.isfinite(pre_ratio), pre_ratio, 1.0e6)
            for reason, bonus in {
                "correction-failed": 3.0e6,
                "timeout-or-budget": 3.0e6,
                "invalid-chart-prediction": 2.0e6,
                "outside-trust-region": 1.0e6,
                "no-branch-compatible-chart": 1.5e6,
                "empty-atlas": 1.0e6,
            }.items():
                priority[pre_reason == reason] += bonus
            labels = branch_labels(repair_diag) if len(repair_diag) else np.empty((0, 2), dtype=int)
            if len(labels):
                unique, counts = np.unique(labels, axis=0, return_counts=True)
                frequency = {tuple(map(int, key)): int(count) for key, count in zip(unique, counts)}
                priority += np.asarray([
                    1.0 / math.sqrt(max(frequency[tuple(map(int, label))], 1))
                    for label in labels
                ])

            chart_limit = min(
                config.generator.failed_probe_centers_per_round,
                config.generator.charts_per_round,
                config.generator.max_charts - len(work.bank),
            )
            if len(work.bank) == 0:
                chart_limit = min(
                    max(chart_limit, config.generator.initial_charts_per_round),
                    config.generator.max_charts - len(work.bank),
                )
            selected_local = (
                _select_spatially_separated(
                    repair_endpoint[failed_idx], repair_diag[failed_idx], priority[failed_idx],
                    np.asarray(config.chart.search_feature_scale, dtype=float),
                    chart_limit, config.generator.candidate_min_separation,
                )
                if len(failed_idx) else np.empty(0, dtype=np.int64)
            )
            selected = failed_idx[selected_local]
            work.pending_launch = repair_launch[selected].copy()
            work.pending_endpoint = repair_endpoint[selected].copy()
            work.pending_diagnostics = repair_diag[selected].copy()
            source_targets = work.pending_endpoint.copy()
            print(
                f"    pre-build repair failures={len(failed_idx):,}/{len(repair_endpoint):,}; "
                f"selected centers={len(source_targets):,}; reasons={_reason_counts(pre_reason[~pre_success])}"
            )
            _atomic_save(
                output_path, work, config, solver_path,
                compressed=config.generator.checkpoint_compressed,
            )

            charts_before = len(work.bank)
            _build_pending_charts(work, config, solver_path, output_path)
            direct_added = len(work.bank) - charts_before
            refined_added = _refine_existing_charts(work, config, solver_path, output_path)

            # Truly fresh validation is generated only after all construction
            # and refinement, using a disjoint Sobol stream and never entering
            # the fit or repair pools.
            _fl, fresh_endpoint, _fd = generate_post_build_fresh()
            metrics = _evaluate_and_update(
                work, backend, config, solver_path, output_path, rng, fresh_endpoint,
                source_targets=source_targets,
                holdout_before_targets=holdout_targets,
                holdout_before_success=holdout_before_success,
            )
            record = {
                "round": work.state.round_number,
                "repair_epoch": work.state.repair_epoch,
                "repair_pass": work.state.repair_pass,
                "sobol_cursor": work.state.sobol_cursor,
                "audit_cursor": work.state.audit_cursor,
                "charts": len(work.bank),
                "direct_charts_added": int(direct_added),
                "refined_charts_added": int(refined_added),
                **metrics,
            }
            work.state.coverage_history.append(record)

            holdout_ok = (
                metrics["holdout"]["success_rate"] >= config.generator.target_success
                and metrics["holdout"]["p95_seconds"] <= config.generator.target_p95_seconds
            )
            fresh_ok = (
                metrics["fresh"]["success_rate"] >= config.generator.fresh_target_success
                and metrics["fresh"]["p95_seconds"] <= config.generator.target_p95_seconds
            )
            if (
                work.state.round_number >= config.generator.minimum_rounds_before_stop
                and holdout_ok and fresh_ok
            ):
                work.state.successful_rounds += 1
            else:
                work.state.successful_rounds = 0
            if work.state.successful_rounds >= config.generator.patience:
                work.state.coverage_converged = True

            source = metrics["source_repair"]
            local = metrics["local_training_test"]
            trust = metrics["trust_radius"]
            patch = metrics["patch"]
            print(
                "    coverage: "
                f"repair={metrics['repair']['success_rate']:.1%} "
                f"({metrics['repair']['successes']}/{metrics['repair']['tested']}), "
                f"holdout={metrics['holdout']['success_rate']:.1%} "
                f"({metrics['holdout']['successes']}/{metrics['holdout']['tested']}), "
                f"fresh={metrics['fresh']['success_rate']:.1%} "
                f"({metrics['fresh']['successes']}/{metrics['fresh']['tested']}); "
                f"source_repaired={source['successes']}/{source['tested']}; "
                f"independent_gain={metrics['holdout_newly_covered']}; "
                f"ungated={metrics['ungated']['recovery_rate']:.1%}"
            )
            print(
                "    charts: "
                f"direct_added={direct_added}, refined_added={refined_added}, total={len(work.bank):,}; "
                f"types={metrics['chart_types']}; transforms={metrics['transforms']}; "
                f"methods={metrics['corrector_profiles']}; "
                f"patch_scale[p05/med/p95]={patch['p05_scale']:.3g}/{patch['median_scale']:.3g}/{patch['p95_scale']:.3g}"
            )
            print(
                "    trust: "
                f"geomean[p05/med/p95]={trust['p05']:.3g}/{trust['median']:.3g}/{trust['p95']:.3g}; "
                f"directional[p05/med/p95]={trust['directional_p05']:.3g}/"
                f"{trust['directional_median']:.3g}/{trust['directional_p95']:.3g}; "
                f"center_self={metrics['center_self_test']['successes']}/"
                f"{metrics['center_self_test']['tested']}; local_inside={local['successes']}/{local['tested']}"
            )
            print(
                f"    failure reasons: repair={metrics['repair']['failure_reasons']}; "
                f"holdout={metrics['holdout']['failure_reasons']}; "
                f"fresh={metrics['fresh']['failure_reasons']}; "
                f"ungated={metrics['ungated']['failure_reasons']}"
            )

            epoch_complete = (
                metrics["repair"]["success_rate"] >= config.generator.repair_epoch_target_success
                or work.state.repair_pass >= config.generator.repair_epoch_max_passes
                or (direct_added < config.generator.repair_epoch_min_new_charts and len(source_targets) > 0)
                or len(source_targets) == 0
            )
            if epoch_complete:
                print(
                    f"    repair epoch {work.state.repair_epoch} complete at pass "
                    f"{work.state.repair_pass}; next round will add a new feasible probe batch"
                )
                work.state.repair_epoch_active = False
                work.state.repair_pass = 0

            _atomic_save(
                output_path, work, config, solver_path,
                compressed=config.generator.checkpoint_compressed,
            )

        work.state.generation_complete = True
        work.state.phase = "complete"
        _atomic_save(
            output_path, work, config, solver_path,
            compressed=config.generator.checkpoint_compressed,
        )
        print(
            f"Generation finished: charts={len(work.bank):,}, cursor={work.state.sobol_cursor:,}, "
            f"coverage_converged={work.state.coverage_converged}"
        )
        return output_path
    except KeyboardInterrupt:
        work.state.interrupted = True
        work.state.interruption_reason = "KeyboardInterrupt"
        work.state.phase = "interrupted"
        if config.generator.checkpoint_on_interrupt:
            previous_handler = signal.getsignal(signal.SIGINT)
            try:
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                _atomic_save(
                    output_path, work, config, solver_path,
                    compressed=config.generator.checkpoint_compressed,
                )
                print(f"Interrupted checkpoint saved: {output_path}", file=sys.stderr)
            finally:
                signal.signal(signal.SIGINT, previous_handler)
        raise


# ---------------------------------------------------------------------------
# Validation and command line
# ---------------------------------------------------------------------------


def validate_chart_atlas(
    atlas_path: str | Path,
    solver_path: str | Path,
    launches: int,
    rows: int,
    seed_offset: int = 100_000_000,
) -> dict[str, Any]:
    atlas = LocalInverseChartAtlas(atlas_path, solver_path)
    backend_cfg = atlas.backend_cfg
    cursor = int(atlas.metadata.get("state", {}).get("sobol_cursor", 0)) + int(seed_offset)
    launch_attempts = sample_launch_chunk(
        backend_cfg,
        cursor,
        launches,
        int(getattr(backend_cfg, "seed", 12345)),
    )
    validation_cfg = replace(backend_cfg, subarcs_per_trajectory=1)
    launch, endpoint, _diag = atlas.backend._generate_rows_from_launches(launch_attempts, validation_cfg)
    if len(endpoint) > rows:
        rng = np.random.default_rng(cursor)
        idx = rng.choice(len(endpoint), size=rows, replace=False)
        endpoint = endpoint[idx]
    success, elapsed, _chart, ratio, reasons = _query_probe_set(atlas, endpoint)
    successful = elapsed[success]
    return {
        "atlas": str(Path(atlas_path).resolve()),
        "tested": int(len(endpoint)),
        "successes": int(np.count_nonzero(success)),
        "success_rate": float(np.mean(success)) if len(success) else 0.0,
        "unsupported_rate": float(np.mean(ratio > 1.0)) if len(ratio) else 1.0,
        "median_seconds": float(np.median(successful)) if len(successful) else math.inf,
        "p95_seconds": float(np.quantile(successful, 0.95)) if len(successful) else math.inf,
        "chart_count": len(atlas.bank),
        "failure_reasons": _reason_counts(reasons[~success]),
    }


def _load_config(path: str | Path) -> ChartAtlasConfig:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return ChartAtlasConfig.from_dict(payload)


def repack_atlas(input_path: str | Path, output_path: str | Path, *, compressed: bool = True) -> Path:
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    with np.load(input_path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_name(output_path.name + ".tmp")
    save = np.savez_compressed if compressed else np.savez
    with open(temp, "wb") as handle:
        save(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, output_path)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="generate or resume a local inverse chart atlas")
    generate.add_argument("output")
    generate.add_argument("--config", required=True)
    generate.add_argument("--solver", default=str(Path(__file__).with_name("pmp_extremal_atlas.py")))
    generate.add_argument("--resume", action="store_true")
    generate.add_argument(
        "--resume-policy-change",
        action="store_true",
        help="allow compatible chart/generator policy changes while preserving dynamics/query/scales",
    )

    inspect = sub.add_parser("inspect", help="inspect chart-atlas metadata")
    inspect.add_argument("atlas")

    validate = sub.add_parser("validate", help="run an independent feasible-query validation")
    validate.add_argument("atlas")
    validate.add_argument("--solver", default=str(Path(__file__).with_name("pmp_extremal_atlas.py")))
    validate.add_argument("--launches", type=int, default=8192)
    validate.add_argument("--rows", type=int, default=1024)

    repack = sub.add_parser("repack", help="rewrite an atlas with or without ZIP compression")
    repack.add_argument("input")
    repack.add_argument("output")
    repack.add_argument("--uncompressed", action="store_true")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        config = _load_config(args.config)
        generate_chart_atlas(args.output, config, args.solver, resume=args.resume, resume_policy_change=args.resume_policy_change)
        return 0
    if args.command == "inspect":
        with np.load(args.atlas, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
        print(json.dumps(metadata, indent=2))
        return 0
    if args.command == "validate":
        result = validate_chart_atlas(args.atlas, args.solver, args.launches, args.rows)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "repack":
        result = repack_atlas(args.input, args.output, compressed=not args.uncompressed)
        print(result)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
