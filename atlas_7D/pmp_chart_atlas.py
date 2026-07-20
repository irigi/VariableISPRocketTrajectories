#!/usr/bin/env python3
"""Validated local-inverse chart atlas for planar Kepler PMP extremals.

This module implements a chart atlas rather than a point cloud.  Each chart
stores a feasible extremal, a local normalized inverse map from the seven
endpoint coordinates to the five nontrivial launch variables, a low-rank
quadratic correction, and an empirically validated trust radius.

The existing ``pmp_extremal_atlas.py`` module is used only as the authoritative
dynamics and shooting backend.  Generation, chart fitting, coverage logic,
checkpointing, restart, and query routing are implemented here.

NPZ format ``pmp-local-inverse-chart-atlas`` version 1 is continuously and
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
FORMAT_VERSION = 1

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

    validation_reservoir_rows: int = 12_288
    validation_add_per_round: int = 768
    validation_test_per_round: int = 2048
    fresh_validation_rows: int = 512
    target_success: float = 0.97
    fresh_target_success: float = 0.94
    target_p95_seconds: float = 0.50
    patience: int = 5
    minimum_rounds_before_stop: int = 8

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
        data["chart"] = ChartBuildConfig(**data.get("chart", {}))
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
    feature_scale: FloatArray,
    q_scale: FloatArray,
    trust_endpoint_weight: float = 0.04,
) -> tuple[FloatArray, FloatArray, float]:
    z = (np.asarray(target, dtype=float) - np.asarray(center_endpoint, dtype=float)) / feature_scale
    y = np.asarray(affine, dtype=float) @ z
    if len(quad_dirs):
        basis = (np.asarray(quad_dirs, dtype=float) @ z) ** 2
        y = y + np.asarray(quad_coeff, dtype=float) @ basis
    score = math.sqrt(
        float(np.dot(y, y)) + float(trust_endpoint_weight) * float(np.dot(z, z))
    )
    q = launch_to_q(center_launch) + q_scale * y
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
            corrected, _nfev = backend._correct_target_from_explicit_seed(
                exact_target, seed_launch, query_cfg, bounds
            )
        except (ValueError, OverflowError, FloatingPointError):
            # Invalid polynomial extrapolation: classify it as a chart failure
            # and shrink the claimed trust region instead of aborting the run.
            _q_pred, score = work.bank.predict_q(chart_idx, exact_target)
            corrected = None
        exact_ratio = score / max(float(work.bank.trust_radius[chart_idx]), 1.0e-12)
        if corrected is None:
            failed.append(int(candidate_idx))
            work.bank.failure_count[chart_idx] += 1
            if exact_ratio <= 1.0:
                work.bank.trust_radius[chart_idx] *= np.float32(max(0.25, 0.95 * exact_ratio))
        else:
            work.bank.success_count[chart_idx] += 1
            if exact_ratio > 1.0:
                expanded = min(chart_cfg.maximum_trust_radius, score / 0.95)
                work.bank.trust_radius[chart_idx] = np.float32(
                    max(float(work.bank.trust_radius[chart_idx]), expanded)
                )
    if failed:
        work.bank.invalidate_tree()
    return np.asarray(failed, dtype=np.int64)


def _build_chart(
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
) -> Optional[BuiltChart]:
    center_launch = np.asarray(center_launch, dtype=float)
    supplied_center_endpoint = np.asarray(center_endpoint, dtype=float)
    exact_center_endpoint, jac, exact_condition, exact_sigma = _exact_endpoint_jacobian(
        backend, center_launch, query_cfg
    )
    if exact_center_endpoint is None or jac is None:
        return None
    center_endpoint = exact_center_endpoint
    center_diag = np.asarray(center_diag, dtype=float).copy()
    center_diag[5] = float(round(center_endpoint[3] / (2.0 * math.pi)))
    labels = branch_labels(neighbour_diag)
    center_label = branch_labels(np.asarray(center_diag, dtype=float)[None, :])[0]
    same = np.all(labels == center_label[None, :], axis=1)
    neighbour_launch = np.asarray(neighbour_launch, dtype=float)[same]
    neighbour_endpoint = np.asarray(neighbour_endpoint, dtype=float)[same]

    z_all = (neighbour_endpoint - center_endpoint) / feature_scale if len(neighbour_endpoint) else np.empty((0, 7))
    distance = np.linalg.norm(z_all, axis=1) if len(z_all) else np.empty(0)
    keep = np.isfinite(distance) & (distance <= chart_cfg.fit_max_normalized_distance)
    keep &= distance > 1.0e-12
    neighbour_launch = neighbour_launch[keep]
    neighbour_endpoint = neighbour_endpoint[keep]
    z_all = z_all[keep]
    distance = distance[keep]

    # The exact production-tolerance variational Jacobian supplies the five
    # difficult output columns.
    inv_jac, condition, sigma_min = _regularized_inverse(jac, chart_cfg.jacobian_regularization)
    if inv_jac is None or condition > chart_cfg.maximum_jacobian_condition:
        return None

    q0 = launch_to_q(center_launch)
    q_neighbour = launch_to_q(neighbour_launch) if len(neighbour_launch) else np.empty((0, 5))
    y_all = (q_neighbour - q0) / q_scale if len(q_neighbour) else np.empty((0, 5))

    affine = np.zeros((5, 7), dtype=float)
    affine[:, 2:7] = (
        np.diag(1.0 / q_scale)
        @ inv_jac
        @ np.diag(feature_scale[2:7])
    )
    directions = np.zeros((0, 7), dtype=float)
    coefficients = np.zeros((5, 0), dtype=float)
    weights = np.ones(len(z_all), dtype=float)

    if len(neighbour_launch) >= chart_cfg.minimum_fit_neighbours:
        # Estimate the two initial-velocity conditioning columns after removing
        # the exact fixed-(u0,w0) inverse prediction.
        residual_u = y_all - z_all[:, 2:7] @ affine[:, 2:7].T
        zu = z_all[:, :2]
        bandwidth = max(float(np.median(distance)), 1.0e-3)
        weights = np.exp(-0.5 * (distance / bandwidth) ** 2)
        sw = np.sqrt(np.maximum(weights, 1.0e-12))
        lhs = (zu * sw[:, None]).T @ (zu * sw[:, None]) + chart_cfg.fit_ridge * np.eye(2)
        rhs = (zu * sw[:, None]).T @ (residual_u * sw[:, None])
        try:
            affine[:, :2] = np.linalg.solve(lhs, rhs).T
        except np.linalg.LinAlgError:
            affine[:, :2] = 0.0

        initial_prediction = z_all @ affine.T
        residual = y_all - initial_prediction
        residual_norm = np.linalg.norm(residual, axis=1)
        cutoff = float(np.quantile(residual_norm, chart_cfg.fit_trim_quantile))
        fit_keep = residual_norm <= max(cutoff, 1.0e-12)
        if np.count_nonzero(fit_keep) < chart_cfg.minimum_fit_neighbours:
            fit_keep = np.argsort(residual_norm)[: chart_cfg.minimum_fit_neighbours]
        z_fit = z_all[fit_keep]
        residual_fit = residual[fit_keep]
        weights_fit = weights[fit_keep]
        directions, coefficients = _fit_low_rank_quadratic(
            z_fit,
            residual_fit,
            weights_fit,
            chart_cfg.quadratic_rank,
            chart_cfg.quadratic_ridge,
            chart_cfg.coefficient_clip,
        )

    # Validate on held-out neighbours distributed across predicted chart score.
    predicted = z_all @ affine.T if len(z_all) else np.empty((0, 5))
    if len(directions):
        predicted += ((z_all @ directions.T) ** 2) @ coefficients.T
    scores = np.sqrt(
        np.sum(predicted * predicted, axis=1)
        + chart_cfg.trust_endpoint_weight * np.sum(z_all * z_all, axis=1)
    )
    order = np.argsort(scores)
    validation_count = min(chart_cfg.exact_validation_points, len(order))
    if validation_count > 0:
        positions = np.unique(np.linspace(0, len(order) - 1, validation_count).astype(int))
        validation_indices = order[positions]
    else:
        validation_indices = np.empty(0, dtype=int)

    exact_scores: list[float] = []
    exact_success: list[bool] = []
    latencies: list[float] = []
    for idx in validation_indices:
        # Recompute each validation target at production tolerances; bulk
        # generation endpoints may come from the faster approximate backend.
        exact_target, _unused_jac, _unused_cond, _unused_sigma = _exact_endpoint_jacobian(
            backend, neighbour_launch[idx], query_cfg
        )
        if exact_target is None:
            continue
        target = exact_target
        q_pred, _z_pred, score_exact = chart_predict_q_normalized(
            target, center_endpoint, center_launch, affine, directions, coefficients,
            feature_scale, q_scale, chart_cfg.trust_endpoint_weight
        )
        try:
            seed_launch = q_to_launch(
                float(target[0]), float(target[1]), q_pred, bounds=bounds
            )
        except (ValueError, OverflowError, FloatingPointError):
            # An out-of-bounds/non-finite polynomial extrapolation is a failed
            # validation sample, not a fatal generator error.
            exact_scores.append(float(score_exact))
            exact_success.append(False)
            latencies.append(0.0)
            continue
        started = time.monotonic()
        corrected, _nfev = backend._correct_target_from_explicit_seed(
            target, seed_launch, query_cfg, bounds
        )
        elapsed = time.monotonic() - started
        exact_scores.append(float(score_exact))
        exact_success.append(corrected is not None)
        latencies.append(float(elapsed))

    # Cheap known-launch validation supplies a dense conservative initial
    # radius; the exact bounded-Newton samples and later global audits certify
    # and shrink/expand it.
    if len(scores):
        prediction_error = np.linalg.norm(predicted - y_all, axis=1)
        prediction_good = prediction_error <= chart_cfg.prediction_error_limit
        prediction_cfg = replace(
            chart_cfg,
            local_target_success=chart_cfg.prediction_target_success,
            minimum_exact_successes=max(1, min(chart_cfg.minimum_exact_successes, int(np.count_nonzero(prediction_good)))),
        )
        prediction_radius = _validated_radius(scores, prediction_good, prediction_cfg)
    else:
        prediction_radius = chart_cfg.minimum_trust_radius
    if exact_scores:
        exact_radius = _validated_radius(
            np.asarray(exact_scores, dtype=float),
            np.asarray(exact_success, dtype=bool),
            chart_cfg,
        )
        trust_radius = min(prediction_radius, exact_radius)
    else:
        trust_radius = prediction_radius
    trust_radius = float(np.clip(
        trust_radius, chart_cfg.minimum_trust_radius, chart_cfg.maximum_trust_radius
    ))
    successes = int(np.count_nonzero(exact_success))
    failures = int(len(exact_success) - successes)
    if (
        chart_cfg.reject_if_no_local_coverage
        and len(exact_success)
        and successes < chart_cfg.minimum_exact_successes
    ):
        return None
    p95 = float(np.quantile(latencies, 0.95)) if latencies else math.inf
    return BuiltChart(
        launch=center_launch.copy(),
        endpoint=center_endpoint.copy(),
        diagnostics=np.asarray(center_diag, dtype=float).copy(),
        affine=affine,
        quad_dirs=directions,
        quad_coeff=coefficients,
        trust_radius=trust_radius,
        local_successes=successes,
        local_failures=failures,
        local_p95_seconds=p95,
        jacobian_condition=float(max(condition, exact_condition)),
        jacobian_sigma_min=float(min(sigma_min, exact_sigma) if exact_sigma > 0 else sigma_min),
    )


# ---------------------------------------------------------------------------
# In-memory chart bank
# ---------------------------------------------------------------------------


class ChartBank:
    def __init__(
        self, feature_scale: FloatArray, q_scale: FloatArray, quadratic_rank: int,
        trust_endpoint_weight: float = 0.04,
    ):
        self.feature_scale = np.asarray(feature_scale, dtype=float)
        self.q_scale = np.asarray(q_scale, dtype=float)
        self.rank = int(quadratic_rank)
        self.trust_endpoint_weight = float(trust_endpoint_weight)
        self.launch = np.empty((0, 7), dtype=float)
        self.endpoint = np.empty((0, 7), dtype=float)
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
        self.refinement_count = np.empty(0, dtype=np.int16)
        self._tree: Optional[cKDTree] = None

    def __len__(self) -> int:
        return len(self.launch)

    def invalidate_tree(self) -> None:
        self._tree = None

    @property
    def tree(self) -> cKDTree:
        if self._tree is None:
            if len(self) == 0:
                raise RuntimeError("chart bank is empty")
            self._tree = cKDTree(self.endpoint / self.feature_scale)
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
        self.trust_radius = np.concatenate(
            (self.trust_radius, np.asarray([c.trust_radius for c in charts], dtype=np.float32))
        )
        self.success_count = np.concatenate(
            (self.success_count, np.asarray([c.local_successes for c in charts], dtype=np.int32))
        )
        self.failure_count = np.concatenate(
            (self.failure_count, np.asarray([c.local_failures for c in charts], dtype=np.int32))
        )
        self.local_p95_seconds = np.concatenate(
            (self.local_p95_seconds, np.asarray([c.local_p95_seconds for c in charts], dtype=np.float32))
        )
        self.jacobian_condition = np.concatenate(
            (self.jacobian_condition, np.asarray([c.jacobian_condition for c in charts], dtype=np.float32))
        )
        self.jacobian_sigma_min = np.concatenate(
            (self.jacobian_sigma_min, np.asarray([c.jacobian_sigma_min for c in charts], dtype=np.float32))
        )
        self.refinement_count = np.concatenate(
            (self.refinement_count, np.zeros(n, dtype=np.int16))
        )
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
        self.refinement_count[i] = np.int16(min(32767, int(self.refinement_count[i]) + 1))
        self.invalidate_tree()

    def candidate_indices(self, targets: FloatArray, k: int) -> NDArray[np.int64]:
        targets = np.atleast_2d(np.asarray(targets, dtype=float))
        if len(self) == 0:
            return np.empty((len(targets), 0), dtype=np.int64)
        k = min(max(1, int(k)), len(self))
        _distance, index = self.tree.query(targets / self.feature_scale, k=k, workers=-1)
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
            z = (targets - center) / self.feature_scale
            affine = self.affine[idx].astype(float)
            y = np.einsum("nij,nj->ni", affine, z)
            if self.rank:
                dirs = self.quad_dirs[idx].astype(float)
                coeff = self.quad_coeff[idx].astype(float)
                basis = np.einsum("nkj,nj->nk", dirs, z) ** 2
                y += np.einsum("nik,nk->ni", coeff, basis)
            score = np.sqrt(
                np.sum(y * y, axis=1)
                + self.trust_endpoint_weight * np.sum(z * z, axis=1)
            )
            radius = np.maximum(self.trust_radius[idx].astype(float), 1.0e-12)
            ratio = score / radius
            winding = np.rint(self.diagnostics[idx, 5]).astype(int)
            ratio[np.abs(winding - target_winding) > winding_window] = np.inf
            improve = ratio < best_ratio
            best_ratio[improve] = ratio[improve]
            best_index[improve] = idx[improve]
        return best_ratio, best_index

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
            self.feature_scale,
            self.q_scale,
            self.trust_endpoint_weight,
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
            q_to_launch(
                float(target[0]), float(target[1]), q, bounds=bounds, clip=clip
            ),
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
            self.feature_scale = np.asarray(data["feature_scale"], dtype=float)
            self.q_scale = np.asarray(data["q_scale"], dtype=float)
            rank = int(self.metadata["config"]["chart"]["quadratic_rank"])
            trust_weight = float(
                self.metadata["config"]["chart"].get("trust_endpoint_weight", 0.04)
            )
            self.bank = ChartBank(
                self.feature_scale, self.q_scale, rank, trust_weight
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
            self.bank.refinement_count = (
                np.asarray(data["chart_refinement_count"], dtype=np.int16)
                if "chart_refinement_count" in data.files
                else np.zeros(len(self.bank.launch), dtype=np.int16)
            )
        self.config = ChartAtlasConfig.from_dict(self.metadata["config"])
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
    ):
        target = np.asarray(target, dtype=float)
        if target.shape != (7,):
            raise ValueError("target must have seven canonical endpoint coordinates")
        if len(self.bank) == 0:
            result = (None, -1, math.inf, "empty-atlas", 0.0)
            return result if return_reason else None
        indices = self.bank.candidate_indices(target[None, :], self.config.routing_neighbours)[0]
        target_winding = int(round(float(target[3]) / (2.0 * math.pi)))
        candidates: list[tuple[float, int, FloatArray]] = []
        for idx in indices:
            idx = int(idx)
            winding = int(round(float(self.bank.diagnostics[idx, 5])))
            if abs(winding - target_winding) > self.config.routing_winding_window:
                continue
            q_pred, score = self.bank.predict_q(idx, target)
            ratio = score / max(float(self.bank.trust_radius[idx]), 1.0e-12)
            if not math.isfinite(ratio):
                continue
            # Do not exponentiate log(tau), or call the ODE solver, until the
            # chart passes the trust-ratio gate and its prediction is inside
            # the configured physical shooting bounds.
            try:
                seed_launch = q_to_launch(
                    float(target[0]), float(target[1]), q_pred, bounds=self.bounds
                )
            except (ValueError, OverflowError, FloatingPointError):
                continue
            candidates.append((ratio, idx, seed_launch))
        candidates.sort(key=lambda item: item[0])
        limit = self.config.routing_max_ratio if max_ratio is None else float(max_ratio)
        if candidates and candidates[0][0] <= 1.0e-12:
            idx = candidates[0][1]
            result = (self.bank.launch[idx].copy(), idx, 0.0, "chart-center", 0.0)
            return result if return_reason else result[0]
        started = time.monotonic()
        attempts = 0
        for ratio, idx, seed_launch in candidates:
            if ratio > limit:
                break
            corrected, _nfev = self.backend._correct_target_from_explicit_seed(
                target, seed_launch, self.query_cfg, self.bounds
            )
            attempts += 1
            if corrected is not None:
                elapsed = time.monotonic() - started
                result = (corrected, idx, ratio, "converged", elapsed)
                return result if return_reason else corrected
            if attempts >= self.query_cfg.max_seed_attempts:
                break
        elapsed = time.monotonic() - started
        best_ratio = candidates[0][0] if candidates else math.inf
        reason = "unsupported" if best_ratio > limit else "correction-failed"
        result = (None, candidates[0][1] if candidates else -1, best_ratio, reason, elapsed)
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


@dataclass
class WorkingSet:
    bank: ChartBank
    fit_launch: FloatArray
    fit_endpoint: FloatArray
    fit_diagnostics: FloatArray
    probe_launch: FloatArray
    probe_endpoint: FloatArray
    probe_diagnostics: FloatArray
    pending_launch: FloatArray
    pending_endpoint: FloatArray
    pending_diagnostics: FloatArray
    state: GeneratorState


def _empty_working_set(config: ChartAtlasConfig) -> WorkingSet:
    bank = ChartBank(
        np.asarray(config.feature_scale), np.asarray(config.q_scale),
        config.chart.quadratic_rank, config.chart.trust_endpoint_weight,
    )
    return WorkingSet(
        bank=bank,
        fit_launch=np.empty((0, 7), dtype=float),
        fit_endpoint=np.empty((0, 7), dtype=float),
        fit_diagnostics=np.empty((0, 6), dtype=float),
        probe_launch=np.empty((0, 7), dtype=float),
        probe_endpoint=np.empty((0, 7), dtype=float),
        probe_diagnostics=np.empty((0, 6), dtype=float),
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
        "representation": "validated-low-rank-quadratic-local-inverse-charts",
        "solver_path": str(solver_path),
        "solver_digest": _file_digest(solver_path),
        "config": asdict(config),
        "config_digest": _config_digest(config),
        "state": asdict(work.state),
        "chart_count": len(work.bank),
        "fit_pool_rows": len(work.fit_launch),
        "probe_rows": len(work.probe_launch),
        "pending_chart_centers": len(work.pending_launch),
        "endpoint_columns": list(ENDPOINT_COLUMNS),
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
        "chart_refinement_count": work.bank.refinement_count,
        "feature_scale": work.bank.feature_scale,
        "q_scale": work.bank.q_scale,
        "fit_launch": work.fit_launch,
        "fit_endpoint": work.fit_endpoint,
        "fit_diagnostics": work.fit_diagnostics.astype(np.float32),
        "probe_launch": work.probe_launch,
        "probe_endpoint": work.probe_endpoint,
        "probe_diagnostics": work.probe_diagnostics.astype(np.float32),
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


def _load_working_set(path: Path, config: ChartAtlasConfig, solver_path: Path) -> WorkingSet:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        if metadata.get("format") != FORMAT_NAME:
            raise ValueError("resume file is not a chart atlas")
        if metadata.get("config_digest") != _config_digest(config):
            raise ValueError("configuration does not match checkpoint; use the exact original JSON")
        stored_solver_digest = metadata.get("solver_digest")
        if stored_solver_digest and stored_solver_digest != _file_digest(solver_path):
            raise ValueError("dynamics solver does not match the checkpointed solver")
        bank = ChartBank(
            np.asarray(data["feature_scale"], dtype=float),
            np.asarray(data["q_scale"], dtype=float),
            config.chart.quadratic_rank,
            config.chart.trust_endpoint_weight,
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
        bank.refinement_count = (
            np.asarray(data["chart_refinement_count"], dtype=np.int16)
            if "chart_refinement_count" in data.files
            else np.zeros(len(bank.launch), dtype=np.int16)
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
    features = endpoint / feature_scale
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
            tree = cKDTree(work.fit_endpoint[indices] / work.bank.feature_scale) if len(indices) else None
            branch_cache[label] = (indices, tree)
        indices, tree = branch_cache[label]
        if tree is None or len(indices) == 0:
            selected = np.empty(0, dtype=int)
        else:
            k = min(fit_neighbours, len(indices))
            _dist, local = tree.query(center_endpoint[i] / work.bank.feature_scale, k=k)
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
    for chart_index, built in results:
        if built is None:
            work.bank.refinement_count[chart_index] += 1
            continue
        old_radius = float(work.bank.trust_radius[chart_index])
        old_failures = int(work.bank.failure_count[chart_index])
        gain = built.trust_radius / max(old_radius, 1.0e-12)
        if gain >= config.generator.refinement_min_radius_gain or old_failures > built.local_failures:
            work.bank.replace_chart(chart_index, built)
            replaced += 1
        else:
            work.bank.refinement_count[chart_index] += 1
    if replaced:
        _atomic_save(
            output_path, work, config, solver_path,
            compressed=config.generator.checkpoint_compressed,
        )
    print(f"    chart refinement: attempted={len(chosen)}, replaced={replaced}")
    return replaced


def _query_probe_set(
    atlas: LocalInverseChartAtlas,
    targets: FloatArray,
) -> tuple[NDArray[np.bool_], FloatArray, NDArray[np.int64], FloatArray]:
    success = np.zeros(len(targets), dtype=bool)
    elapsed = np.zeros(len(targets), dtype=float)
    chart_index = np.full(len(targets), -1, dtype=np.int64)
    ratio = np.full(len(targets), np.inf, dtype=float)
    for i, target in enumerate(targets):
        corrected, idx, r, _reason, seconds = atlas.correct_target(target, return_reason=True)
        success[i] = corrected is not None
        elapsed[i] = seconds
        chart_index[i] = idx
        ratio[i] = r
    return success, elapsed, chart_index, ratio


def _evaluate_and_update(
    work: WorkingSet,
    backend,
    config: ChartAtlasConfig,
    solver_path: Path,
    output_path: Path,
    rng: np.random.Generator,
    fresh_endpoint: FloatArray,
) -> dict[str, Any]:
    # Query through the exact on-disk representation so validation exercises the
    # same serialization and routing path as production use.
    _atomic_save(
        output_path,
        work,
        config,
        solver_path,
        compressed=config.generator.checkpoint_compressed,
    )
    atlas = LocalInverseChartAtlas(output_path, solver_path)

    if len(work.probe_endpoint):
        n = min(config.generator.validation_test_per_round, len(work.probe_endpoint))
        indices = rng.choice(len(work.probe_endpoint), size=n, replace=False)
        reservoir_targets = work.probe_endpoint[indices]
    else:
        indices = np.empty(0, dtype=int)
        reservoir_targets = np.empty((0, 7))
    fresh_targets = fresh_endpoint[: config.generator.fresh_validation_rows]

    res_success, res_elapsed, res_chart, res_ratio = _query_probe_set(atlas, reservoir_targets)
    fresh_success, fresh_elapsed, fresh_chart, fresh_ratio = _query_probe_set(atlas, fresh_targets)

    # Empirical trust regions are conservative under observed failures.
    for ok, idx, ratio in zip(res_success, res_chart, res_ratio):
        if idx < 0:
            continue
        if ok:
            work.bank.success_count[idx] += 1
        elif np.isfinite(ratio) and ratio <= 1.0:
            work.bank.failure_count[idx] += 1
            # ratio = score/radius; shrink to 95% of the failed score.
            work.bank.trust_radius[idx] *= np.float32(max(0.25, 0.95 * ratio))
    work.bank.invalidate_tree()

    def metrics(success: NDArray[np.bool_], elapsed: FloatArray, ratio: FloatArray) -> dict[str, Any]:
        if len(success) == 0:
            return {"tested": 0, "success_rate": 0.0, "p95_seconds": math.inf, "median_ratio": math.inf}
        successful_times = elapsed[success]
        return {
            "tested": int(len(success)),
            "success_rate": float(np.mean(success)),
            "p95_seconds": float(np.quantile(successful_times, 0.95)) if len(successful_times) else math.inf,
            "median_ratio": float(np.median(ratio[np.isfinite(ratio)])) if np.any(np.isfinite(ratio)) else math.inf,
            "unsupported_rate": float(np.mean(ratio > 1.0)),
        }

    return {
        "reservoir": metrics(res_success, res_elapsed, res_ratio),
        "fresh": metrics(fresh_success, fresh_elapsed, fresh_ratio),
    }


def generate_chart_atlas(
    output_path: str | Path,
    config: ChartAtlasConfig,
    solver_path: str | Path,
    *,
    resume: bool = False,
) -> Path:
    config.validate()
    output_path = Path(output_path).resolve()
    solver_path = Path(solver_path).resolve()
    backend = load_backend(solver_path)
    backend_cfg = _backend_config(backend, config.backend_config)
    query_cfg = _query_config(backend, config.query_config)
    feature_scale = np.asarray(config.feature_scale, dtype=float)
    rng = np.random.default_rng(config.generator.seed)

    if resume:
        if not output_path.exists():
            raise FileNotFoundError(f"cannot resume; checkpoint does not exist: {output_path}")
        work = _load_working_set(output_path, config, solver_path)
        print(
            f"Resuming chart atlas: charts={len(work.bank):,}, cursor={work.state.sobol_cursor:,}, "
            f"round={work.state.round_number}, pending={len(work.pending_launch):,}"
        )
    else:
        work = _empty_working_set(config)
        _atomic_save(
            output_path,
            work,
            config,
            solver_path,
            compressed=config.generator.checkpoint_compressed,
        )
        print(f"Initial empty checkpoint: {output_path}")

    try:
        # Finish any chart centers that were checkpointed during a previous run.
        if len(work.pending_launch):
            print(f"Continuing {len(work.pending_launch):,} pending chart centers")
            _build_pending_charts(work, config, solver_path, output_path)

        while (
            work.state.sobol_cursor < config.generator.max_launch_attempts
            and work.state.round_number < config.generator.max_rounds
            and len(work.bank) < config.generator.max_charts
            and not work.state.coverage_converged
        ):
            work.state.round_number += 1
            remaining = config.generator.max_launch_attempts - work.state.sobol_cursor
            attempts = min(config.generator.launch_batch_size, remaining)
            print(
                f"Round {work.state.round_number}: integrating {attempts:,} launch attempts "
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
            # Advance only after the whole deterministic Sobol batch completed.
            # An interruption during integration therefore reruns, rather than
            # silently skips, the unfinished batch.
            work.state.sobol_cursor += attempts
            print(f"    accepted forward/subarc rows={len(candidate_launch):,}")
            if len(candidate_launch) == 0:
                _atomic_save(
                    output_path, work, config, solver_path,
                    compressed=config.generator.checkpoint_compressed,
                )
                continue

            # A compact rolling fit pool supplies local branch-aware regression
            # neighbours.  It is temporary generation state, not the final atlas.
            work.fit_launch, work.fit_endpoint, work.fit_diagnostics = _merge_thinned(
                backend,
                work.fit_launch,
                work.fit_endpoint,
                work.fit_diagnostics,
                candidate_launch,
                candidate_endpoint,
                candidate_diag,
                feature_scale,
                config.generator.fit_pool_cell_size,
                config.generator.fit_pool_max_rows,
            )

            # Keep an independent feasible probe reservoir.  Its exact launches
            # are hidden from the chart query during validation.
            add_n = min(config.generator.validation_add_per_round, len(candidate_launch))
            if add_n:
                probe_idx = rng.choice(len(candidate_launch), size=add_n, replace=False)
                exact_probe_launch, exact_probe_endpoint, exact_probe_diag = _exact_probe_rows(
                    backend, backend_cfg, query_cfg, candidate_launch[probe_idx]
                )
                work.probe_launch, work.probe_endpoint, work.probe_diagnostics = _merge_thinned(
                    backend,
                    work.probe_launch,
                    work.probe_endpoint,
                    work.probe_diagnostics,
                    exact_probe_launch,
                    exact_probe_endpoint,
                    exact_probe_diag,
                    feature_scale,
                    config.generator.candidate_cell_size,
                    config.generator.validation_reservoir_rows,
                )

            # Determine which feasible candidates lie outside every certified
            # local chart.  At the beginning all candidates are uncovered.
            if len(work.bank):
                ratios, best_index = work.bank.coverage_scores(
                    candidate_endpoint,
                    config.routing_neighbours,
                    config.routing_winding_window,
                )
                audit_failed = _audit_candidate_failures(
                    work, backend, query_cfg, backend._configured_correction_bounds(backend_cfg),
                    candidate_launch, candidate_endpoint, ratios, best_index, rng,
                    config.generator.candidate_audit_rows, config.chart,
                )
            else:
                ratios = np.full(len(candidate_endpoint), np.inf)
                best_index = np.full(len(candidate_endpoint), -1, dtype=np.int64)
                audit_failed = np.empty(0, dtype=np.int64)
            priority = np.where(np.isfinite(ratios), ratios, 1.0e6)
            if len(audit_failed):
                priority[audit_failed] += 1.0e5
            # Prefer rare branch labels in addition to uncovered volume.
            labels = branch_labels(candidate_diag)
            if len(labels):
                unique, counts = np.unique(labels, axis=0, return_counts=True)
                frequency = {tuple(map(int, key)): int(count) for key, count in zip(unique, counts)}
                priority += np.asarray(
                    [1.0 / math.sqrt(max(frequency[tuple(map(int, label))], 1)) for label in labels]
                )

            chart_limit = (
                config.generator.initial_charts_per_round
                if len(work.bank) == 0
                else config.generator.charts_per_round
            )
            chart_limit = min(chart_limit, config.generator.max_charts - len(work.bank))
            # Only uncovered or boundary-near candidates become chart centers.
            eligible = ratios > 0.90
            if len(audit_failed):
                eligible[audit_failed] = True
            eligible_idx = np.flatnonzero(eligible)
            selected_local = _select_spatially_separated(
                candidate_endpoint[eligible_idx],
                candidate_diag[eligible_idx],
                priority[eligible_idx],
                feature_scale,
                chart_limit,
                config.generator.candidate_min_separation,
            )
            selected = eligible_idx[selected_local]
            work.pending_launch = candidate_launch[selected].copy()
            work.pending_endpoint = candidate_endpoint[selected].copy()
            work.pending_diagnostics = candidate_diag[selected].copy()
            print(
                f"    fit pool={len(work.fit_launch):,}, probes={len(work.probe_launch):,}, "
                f"new chart centers={len(work.pending_launch):,}"
            )
            _atomic_save(
                output_path,
                work,
                config,
                solver_path,
                compressed=config.generator.checkpoint_compressed,
            )
            _build_pending_charts(work, config, solver_path, output_path)
            _refine_existing_charts(work, config, solver_path, output_path)

            # Validate using the same bounded production query as the viewer.
            fresh_idx = rng.choice(
                len(candidate_endpoint),
                size=min(config.generator.fresh_validation_rows, len(candidate_endpoint)),
                replace=False,
            )
            _fresh_launch, fresh_endpoint, _fresh_diag = _exact_probe_rows(
                backend, backend_cfg, query_cfg, candidate_launch[fresh_idx]
            )
            metrics = _evaluate_and_update(
                work,
                backend,
                config,
                solver_path,
                output_path,
                rng,
                fresh_endpoint,
            )
            record = {
                "round": work.state.round_number,
                "sobol_cursor": work.state.sobol_cursor,
                "charts": len(work.bank),
                **metrics,
            }
            work.state.coverage_history.append(record)
            reservoir_ok = (
                metrics["reservoir"]["success_rate"] >= config.generator.target_success
                and metrics["reservoir"]["p95_seconds"] <= config.generator.target_p95_seconds
            )
            fresh_ok = (
                metrics["fresh"]["success_rate"] >= config.generator.fresh_target_success
                and metrics["fresh"]["p95_seconds"] <= config.generator.target_p95_seconds
            )
            if (
                work.state.round_number >= config.generator.minimum_rounds_before_stop
                and reservoir_ok
                and fresh_ok
            ):
                work.state.successful_rounds += 1
            else:
                work.state.successful_rounds = 0
            if work.state.successful_rounds >= config.generator.patience:
                work.state.coverage_converged = True
            print(
                "    coverage: "
                f"reservoir={metrics['reservoir']['success_rate']:.1%}, "
                f"p95={metrics['reservoir']['p95_seconds']:.3g}s; "
                f"fresh={metrics['fresh']['success_rate']:.1%}, "
                f"p95={metrics['fresh']['p95_seconds']:.3g}s; "
                f"patience={work.state.successful_rounds}/{config.generator.patience}"
            )
            _atomic_save(
                output_path,
                work,
                config,
                solver_path,
                compressed=config.generator.checkpoint_compressed,
            )

        work.state.generation_complete = True
        work.state.phase = "complete"
        _atomic_save(
            output_path,
            work,
            config,
            solver_path,
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
                # A second Ctrl+C during the short atomic save must not corrupt
                # or prevent the interruption checkpoint.
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                _atomic_save(
                    output_path,
                    work,
                    config,
                    solver_path,
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
    success, elapsed, _chart, ratio = _query_probe_set(atlas, endpoint)
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
        generate_chart_atlas(args.output, config, args.solver, resume=args.resume)
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
