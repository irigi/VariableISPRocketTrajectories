#!/usr/bin/env python3
"""Interactive circular-query viewer for ``pmp_extremal_atlas.py`` atlases.

The seven-dimensional forward atlas stores irregular endpoint rows

    endpoint = (u0, w0, log(rho), theta, ur_f, ut_f, log(kappa)).

The heatmaps show any raw rows lying near the circular-to-circular subspace,
but the complete configured ``(rho, theta, kappa)`` domain is displayed and
every cell is clickable.  A grey cell means only that no raw approximately
circular row was binned there; in corrected mode the viewer still constructs
the exact circular boundary and runs the atlas retrieval/Newton solver.

Mouse controls
--------------
* Left click: solve and replay the shortest converged branch.
* Right click: request several distinct converged branches.

Replay modes
------------
``corrected`` (default)
    Solve the exact clicked circular boundary.  The viewer tries nearby raw
    rows explicitly and the atlas local-regression/nearest-neighbour solver.
    By default the bounded fast Newton method fails quickly; optional robust
    fallback/continuation must be requested explicitly.

``raw``
    Reintegrate a stored approximately circular row.  Raw mode is available
    only in occupied projection bins and is intended for diagnostics.

The corrected replay uses canonical dimensionless physical values
``mu=1, r0=1, mi=2, md=1, P=kappa``.  With these choices the rocket similarity
parameter equals the clicked ``kappa``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from numpy.typing import NDArray
from scipy.spatial import cKDTree


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


METRIC_LABELS = {
    "nearest_distance": "Normalized distance to nearest atlas seed",
    "coverage_ratio": "Distance / validated convergence radius",
    "validated_radius": "Nearest seed validated convergence radius",
    "tau": r"Dimensionless flight time $\tau_f$",
    "circular_error": "Normalized circularity error",
    "count": "Approximately circular atlas rows per bin",
    "minimum_radius": "Minimum dimensionless radius",
    "maximum_acceleration": "Maximum dimensionless acceleration",
    "radial_turns": "Radial turning points",
    "normal_constant": "Normal Hamiltonian constant",
}


class ViewerError(RuntimeError):
    """User-facing atlas or replay error."""


@dataclass(slots=True)
class ForwardAtlas:
    path: Path
    launch: FloatArray
    endpoint: FloatArray
    diagnostics: FloatArray
    feature_scale: FloatArray
    coverage_radius: FloatArray
    coverage_success_count: NDArray[np.int32]
    coverage_failure_count: NDArray[np.int32]
    metadata: dict

    @property
    def rows(self) -> int:
        return int(self.launch.shape[0])

    @property
    def rho(self) -> FloatArray:
        return np.exp(self.endpoint[:, 2])

    @property
    def theta(self) -> FloatArray:
        return self.endpoint[:, 3]

    @property
    def kappa(self) -> FloatArray:
        return np.exp(self.endpoint[:, 6])

    @property
    def tau(self) -> FloatArray:
        return self.launch[:, 6]


@dataclass(slots=True)
class CircularTolerances:
    initial_radial: float = 0.03
    initial_tangential: float = 0.03
    final_radial_relative: float = 0.05
    final_tangential_relative: float = 0.05

    def validate(self) -> None:
        values = (
            self.initial_radial,
            self.initial_tangential,
            self.final_radial_relative,
            self.final_tangential_relative,
        )
        if not all(math.isfinite(v) and v > 0.0 for v in values):
            raise ViewerError("All circularity tolerances must be finite and positive")


@dataclass(slots=True)
class QueryOptions:
    method: str = "fast_newton"
    max_iterations: int = 6
    max_seed_attempts: int = 3
    wall_time_seconds: float = 3.0
    line_search_steps: int = 3
    step_limit: float = 0.30
    robust_fallback: bool = False
    allow_continuation: bool = False


@dataclass(slots=True)
class CircularSelection:
    row_indices: IntArray
    error: FloatArray
    relax_factor: float
    tolerances: CircularTolerances


@dataclass(slots=True)
class Projection:
    rho_edges: FloatArray
    theta_edges: FloatArray
    kappa_edges: FloatArray
    selected_row: IntArray
    count: NDArray[np.int32]
    selected_error: FloatArray
    cell_rows: dict[int, IntArray]
    selection: CircularSelection

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.selected_row.shape

    def flat_cell(self, i: int, j: int, k: int) -> int:
        nr, nt, nk = self.shape
        return (i * nt + j) * nk + k


@dataclass(slots=True)
class Replay:
    label: str
    tau_f: float
    time: FloatArray
    position: FloatArray
    velocity: FloatArray
    acceleration: FloatArray
    resource_fraction: FloatArray
    residual_text: str
    launch: FloatArray


# ---------------------------------------------------------------------------
# Loading and circular-subspace selection
# ---------------------------------------------------------------------------


def load_module_from_path(path: Path | str) -> ModuleType:
    resolved = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location("pmp_extremal_atlas_viewer_module", resolved)
    if spec is None or spec.loader is None:
        raise ViewerError(f"Could not import solver module from {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_forward_atlas(path: Path | str) -> ForwardAtlas:
    resolved = Path(path).expanduser().resolve()
    with np.load(resolved, allow_pickle=False) as bundle:
        required = ("launch", "endpoint", "diagnostics", "feature_scale", "metadata_json")
        missing = [name for name in required if name not in bundle.files]
        if missing:
            raise ViewerError(f"Atlas is missing required arrays: {', '.join(missing)}")
        launch = np.asarray(bundle["launch"], dtype=np.float64)
        endpoint = np.asarray(bundle["endpoint"], dtype=np.float64)
        diagnostics = np.asarray(bundle["diagnostics"], dtype=np.float64)
        feature_scale = np.asarray(bundle["feature_scale"], dtype=np.float64)
        coverage_radius = (
            np.asarray(bundle["coverage_radius"], dtype=np.float64)
            if "coverage_radius" in bundle.files
            else np.zeros(launch.shape[0], dtype=np.float64)
        )
        coverage_success_count = (
            np.asarray(bundle["coverage_success_count"], dtype=np.int32)
            if "coverage_success_count" in bundle.files
            else np.zeros(launch.shape[0], dtype=np.int32)
        )
        coverage_failure_count = (
            np.asarray(bundle["coverage_failure_count"], dtype=np.int32)
            if "coverage_failure_count" in bundle.files
            else np.zeros(launch.shape[0], dtype=np.int32)
        )
        try:
            metadata = json.loads(str(bundle["metadata_json"].item()))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ViewerError("Could not parse atlas metadata_json") from exc

    if launch.ndim != 2 or launch.shape[1] != 7:
        raise ViewerError(f"launch must have shape (N,7); found {launch.shape}")
    if endpoint.shape != launch.shape:
        raise ViewerError(f"endpoint shape {endpoint.shape} does not match launch {launch.shape}")
    if diagnostics.ndim != 2 or diagnostics.shape[0] != launch.shape[0] or diagnostics.shape[1] < 6:
        raise ViewerError(f"diagnostics must have shape (N,>=6); found {diagnostics.shape}")
    if feature_scale.shape != (7,) or np.any(feature_scale <= 0.0):
        raise ViewerError("feature_scale must contain seven positive values")
    for name, value in (
        ("coverage_radius", coverage_radius),
        ("coverage_success_count", coverage_success_count),
        ("coverage_failure_count", coverage_failure_count),
    ):
        if value.shape != (launch.shape[0],):
            raise ViewerError(f"{name} must have shape (N,); found {value.shape}")
    if not np.all(np.isfinite(launch)) or not np.all(np.isfinite(endpoint)):
        raise ViewerError("Atlas contains non-finite launch or endpoint values")

    return ForwardAtlas(
        path=resolved,
        launch=launch,
        endpoint=endpoint,
        diagnostics=diagnostics,
        feature_scale=feature_scale,
        coverage_radius=coverage_radius,
        coverage_success_count=coverage_success_count,
        coverage_failure_count=coverage_failure_count,
        metadata=metadata,
    )


def circular_components(atlas: ForwardAtlas) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    rho = atlas.rho
    if np.any(rho <= 0.0):
        raise ViewerError("Atlas contains non-positive radius ratios")
    circular_speed = rho ** -0.5
    initial_radial = np.abs(atlas.endpoint[:, 0])
    initial_tangential = np.abs(atlas.endpoint[:, 1] - 1.0)
    final_radial_relative = np.abs(atlas.endpoint[:, 4]) / circular_speed
    final_tangential_relative = np.abs(atlas.endpoint[:, 5] / circular_speed - 1.0)
    return (
        initial_radial,
        initial_tangential,
        final_radial_relative,
        final_tangential_relative,
    )


def select_circular_rows(
    atlas: ForwardAtlas,
    tolerances: CircularTolerances,
    *,
    min_rows: int,
    auto_relax: bool,
    max_relax: float,
    relax_step: float = 1.5,
) -> CircularSelection:
    tolerances.validate()
    if min_rows < 1:
        raise ViewerError("min_rows must be positive")
    if max_relax < 1.0:
        raise ViewerError("max_relax must be at least one")

    components = circular_components(atlas)
    base = np.array(
        [
            tolerances.initial_radial,
            tolerances.initial_tangential,
            tolerances.final_radial_relative,
            tolerances.final_tangential_relative,
        ],
        dtype=float,
    )

    factor = 1.0
    mask = np.zeros(atlas.rows, dtype=bool)
    while True:
        scaled = np.vstack(components).T / (base * factor)
        mask = np.all(scaled <= 1.0, axis=1)
        if np.count_nonzero(mask) >= min_rows:
            break
        if not auto_relax or factor >= max_relax * (1.0 - 1.0e-12):
            break
        factor = min(max_relax, factor * relax_step)

    indices = np.flatnonzero(mask).astype(np.int64)
    if indices.size == 0:
        return CircularSelection(
            row_indices=np.empty(0, dtype=np.int64),
            error=np.empty(0, dtype=float),
            relax_factor=float(factor),
            tolerances=tolerances,
        )
    scaled_selected = np.vstack([component[indices] for component in components]).T / (base * factor)
    # RMS makes 1 roughly the tolerance boundary while retaining a smooth score.
    error = np.sqrt(np.mean(scaled_selected**2, axis=1))
    return CircularSelection(
        row_indices=indices,
        error=np.asarray(error, dtype=float),
        relax_factor=float(factor),
        tolerances=tolerances,
    )


# ---------------------------------------------------------------------------
# Binning and catalogue construction
# ---------------------------------------------------------------------------


def log_edges(low: float, high: float, bins: int) -> FloatArray:
    if not (math.isfinite(low) and math.isfinite(high) and 0.0 < low < high):
        raise ViewerError(f"Invalid logarithmic bounds ({low}, {high})")
    if bins <= 0:
        raise ViewerError("Bin counts must be positive")
    return np.geomspace(low, high, bins + 1)


def linear_edges(low: float, high: float, bins: int) -> FloatArray:
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ViewerError(f"Invalid linear bounds ({low}, {high})")
    if bins <= 0:
        raise ViewerError("Bin counts must be positive")
    return np.linspace(low, high, bins + 1)


def _metadata_bounds(atlas: ForwardAtlas, name: str) -> Optional[tuple[float, float]]:
    config = atlas.metadata.get("config", {})
    value = config.get(name)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            low, high = float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
        if math.isfinite(low) and math.isfinite(high) and low < high:
            return low, high
    return None


def resolve_bounds(
    atlas: ForwardAtlas,
    selection: CircularSelection,
    explicit: tuple[Optional[float], Optional[float]],
    metadata_name: str,
    values: FloatArray,
    *,
    positive: bool,
) -> tuple[float, float]:
    low_arg, high_arg = explicit
    metadata = _metadata_bounds(atlas, metadata_name)
    selected = values[selection.row_indices] if selection.row_indices.size else values
    low = float(low_arg) if low_arg is not None else (metadata[0] if metadata else float(np.min(selected)))
    high = float(high_arg) if high_arg is not None else (metadata[1] if metadata else float(np.max(selected)))
    if positive and low <= 0.0:
        positive_values = selected[selected > 0.0]
        if positive_values.size == 0:
            raise ViewerError(f"No positive values available for {metadata_name}")
        low = float(np.min(positive_values))
    if not low < high:
        raise ViewerError(f"Resolved {metadata_name} bounds are invalid: {low}, {high}")
    return low, high


def build_projection(
    atlas: ForwardAtlas,
    selection: CircularSelection,
    *,
    rho_edges: FloatArray,
    theta_edges: FloatArray,
    kappa_edges: FloatArray,
    max_cell_candidates: int,
) -> Projection:
    nr = rho_edges.size - 1
    nt = theta_edges.size - 1
    nk = kappa_edges.size - 1
    indices = selection.row_indices
    rho = atlas.rho[indices]
    theta = atlas.theta[indices]
    kappa = atlas.kappa[indices]

    i = np.searchsorted(rho_edges, rho, side="right") - 1
    j = np.searchsorted(theta_edges, theta, side="right") - 1
    k = np.searchsorted(kappa_edges, kappa, side="right") - 1
    # Include values exactly equal to the final edge in the final bin.
    i[rho == rho_edges[-1]] = nr - 1
    j[theta == theta_edges[-1]] = nt - 1
    k[kappa == kappa_edges[-1]] = nk - 1
    inside = (i >= 0) & (i < nr) & (j >= 0) & (j < nt) & (k >= 0) & (k < nk)
    if not np.any(inside):
        return Projection(
            rho_edges=rho_edges,
            theta_edges=theta_edges,
            kappa_edges=kappa_edges,
            selected_row=np.full((nr, nt, nk), -1, dtype=np.int64),
            count=np.zeros((nr, nt, nk), dtype=np.int32),
            selected_error=np.full((nr, nt, nk), np.nan, dtype=float),
            cell_rows={},
            selection=selection,
        )

    rows = indices[inside]
    errors = selection.error[inside]
    ii, jj, kk = i[inside], j[inside], k[inside]
    flat = (ii * nt + jj) * nk + kk

    selected_row = np.full((nr, nt, nk), -1, dtype=np.int64)
    selected_error = np.full((nr, nt, nk), np.nan, dtype=float)
    count = np.zeros((nr, nt, nk), dtype=np.int32)
    cell_rows: dict[int, IntArray] = {}

    # Group by flat cell.  Within each cell, keep shortest trajectories first;
    # circular error breaks near-equal-time ties.
    order = np.argsort(flat, kind="stable")
    flat_sorted = flat[order]
    split = np.flatnonzero(np.diff(flat_sorted)) + 1
    groups = np.split(order, split)
    for group in groups:
        cell = int(flat[group[0]])
        group_rows = rows[group]
        group_errors = errors[group]
        ranking = np.lexsort((group_errors, atlas.tau[group_rows]))
        ranked_rows = group_rows[ranking]
        ranked_errors = group_errors[ranking]
        if max_cell_candidates > 0:
            ranked_rows = ranked_rows[:max_cell_candidates]
            ranked_errors = ranked_errors[:max_cell_candidates]
        ci = cell // (nt * nk)
        remainder = cell % (nt * nk)
        cj = remainder // nk
        ck = remainder % nk
        selected_row[ci, cj, ck] = int(ranked_rows[0])
        selected_error[ci, cj, ck] = float(ranked_errors[0])
        count[ci, cj, ck] = int(group_rows.size)
        cell_rows[cell] = np.asarray(ranked_rows, dtype=np.int64)

    return Projection(
        rho_edges=rho_edges,
        theta_edges=theta_edges,
        kappa_edges=kappa_edges,
        selected_row=selected_row,
        count=count,
        selected_error=selected_error,
        cell_rows=cell_rows,
        selection=selection,
    )


def _coverage_cache_key(
    atlas: ForwardAtlas,
    projection: Projection,
    theta_indices: IntArray,
) -> str:
    stat = atlas.path.stat()
    digest = hashlib.sha256()
    digest.update(str(atlas.path.resolve()).encode())
    digest.update(str(stat.st_size).encode())
    digest.update(str(stat.st_mtime_ns).encode())
    for array in (
        atlas.feature_scale,
        projection.rho_edges,
        projection.theta_edges,
        projection.kappa_edges,
        np.asarray(theta_indices, dtype=np.int64),
    ):
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def _coverage_metric_slices(
    atlas: ForwardAtlas,
    projection: Projection,
    theta_indices: IntArray,
    *,
    workers: int = -1,
    cache_path: Optional[Path] = None,
    use_cache: bool = True,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Evaluate only displayed circular slices in one batched KD-tree query."""
    theta_indices = np.asarray(theta_indices, dtype=np.int64)
    key = _coverage_cache_key(atlas, projection, theta_indices)
    if use_cache and cache_path is not None and cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                if str(data["cache_key"].item()) == key:
                    return (
                        np.asarray(data["distance"], dtype=float),
                        np.asarray(data["radius"], dtype=float),
                        np.asarray(data["ratio"], dtype=float),
                    )
        except Exception:
            pass

    nr, _nt, nk = projection.shape
    rho_centres = np.sqrt(projection.rho_edges[:-1] * projection.rho_edges[1:])
    theta_centres = 0.5 * (projection.theta_edges[:-1] + projection.theta_edges[1:])
    kappa_centres = np.sqrt(projection.kappa_edges[:-1] * projection.kappa_edges[1:])
    rr, kk = np.meshgrid(rho_centres, kappa_centres, indexing="ij")
    plane_size = rr.size
    targets = np.empty((len(theta_indices) * plane_size, 7), dtype=float)
    base_rho = rr.ravel()
    base_kappa = kk.ravel()
    for panel, theta_index in enumerate(theta_indices):
        sl = slice(panel * plane_size, (panel + 1) * plane_size)
        targets[sl, 0] = 0.0
        targets[sl, 1] = 1.0
        targets[sl, 2] = np.log(base_rho)
        targets[sl, 3] = theta_centres[int(theta_index)]
        targets[sl, 4] = 0.0
        targets[sl, 5] = base_rho ** -0.5
        targets[sl, 6] = np.log(base_kappa)

    normalized_endpoint = np.asarray(atlas.endpoint / atlas.feature_scale, dtype=np.float64)
    tree = cKDTree(normalized_endpoint, compact_nodes=True, balanced_tree=True)
    distances, indices = tree.query(
        targets / atlas.feature_scale,
        k=1,
        workers=int(workers),
    )
    distances = np.asarray(distances, dtype=float).reshape(len(theta_indices), nr, nk)
    indices = np.asarray(indices, dtype=int).reshape(len(theta_indices), nr, nk)
    radii = atlas.coverage_radius[indices]
    ratio = np.full_like(distances, np.nan, dtype=float)
    valid = radii > 0.0
    ratio[valid] = distances[valid] / radii[valid]
    distance_cube = np.transpose(distances, (1, 0, 2))
    radius_cube = np.transpose(radii, (1, 0, 2))
    ratio_cube = np.transpose(ratio, (1, 0, 2))

    if use_cache and cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_name(cache_path.name + ".tmp.npz")
            np.savez(
                temporary,
                cache_key=np.array(key),
                distance=distance_cube.astype(np.float32),
                radius=radius_cube.astype(np.float32),
                ratio=ratio_cube.astype(np.float32),
            )
            temporary.replace(cache_path)
        except Exception as exc:
            print(f"Viewer distance cache could not be saved: {exc}")
    return distance_cube, radius_cube, ratio_cube


def projection_metric(
    atlas: ForwardAtlas,
    projection: Projection,
    metric: str,
    *,
    theta_indices: Optional[IntArray] = None,
    distance_workers: int = -1,
    distance_cache: Optional[Path] = None,
    use_distance_cache: bool = True,
) -> FloatArray:
    selected_theta = (
        np.arange(projection.shape[1], dtype=np.int64)
        if theta_indices is None
        else np.asarray(theta_indices, dtype=np.int64)
    )
    if metric in {"nearest_distance", "validated_radius", "coverage_ratio"}:
        distance, radius, ratio = _coverage_metric_slices(
            atlas,
            projection,
            selected_theta,
            workers=distance_workers,
            cache_path=distance_cache,
            use_cache=use_distance_cache,
        )
        if metric == "nearest_distance":
            return distance
        if metric == "validated_radius":
            radius[radius <= 0.0] = np.nan
            return radius
        return ratio

    if metric == "count":
        out = projection.count[:, selected_theta, :].astype(float)
        out[out <= 0.0] = np.nan
        return out

    selected = projection.selected_row[:, selected_theta, :]
    out = np.full(selected.shape, np.nan, dtype=float)
    occupied = selected >= 0
    rows = selected[occupied]
    if metric == "tau":
        out[occupied] = atlas.tau[rows]
    elif metric == "circular_error":
        out[occupied] = projection.selected_error[:, selected_theta, :][occupied]
    elif metric == "minimum_radius":
        out[occupied] = atlas.diagnostics[rows, 1]
    elif metric == "maximum_acceleration":
        out[occupied] = atlas.diagnostics[rows, 3]
    elif metric == "radial_turns":
        out[occupied] = atlas.diagnostics[rows, 4]
    elif metric == "normal_constant":
        out[occupied] = atlas.diagnostics[rows, 0]
    else:
        raise ViewerError(f"Unknown metric {metric!r}")
    return out


def metric_norm(values: FloatArray, metric: str):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    low = float(np.min(finite))
    high = float(np.max(finite))
    if metric in {
        "nearest_distance", "coverage_ratio", "validated_radius", "tau", "count",
        "minimum_radius", "maximum_acceleration", "normal_constant"
    }:
        positive = finite[finite > 0.0]
        if positive.size and high / float(np.min(positive)) > 100.0:
            return LogNorm(vmin=float(np.min(positive)), vmax=high)
    if math.isclose(low, high, rel_tol=1.0e-12, abs_tol=1.0e-15):
        pad = max(abs(low) * 0.05, 1.0e-12)
        return Normalize(vmin=low - pad, vmax=high + pad)
    return Normalize(vmin=low, vmax=high)


def infer_theta_panels(projection: Projection, panel_count: int) -> IntArray:
    """Choose panels across the complete requested theta domain.

    Earlier versions displayed only theta bins already containing approximately
    circular raw rows.  That hid unsupported regions and made the catalogue a
    projection viewer rather than a query/coverage viewer.
    """
    count = projection.theta_edges.size - 1
    if count <= panel_count:
        return np.arange(count, dtype=np.int64)
    positions = np.rint(np.linspace(0, count - 1, panel_count)).astype(int)
    return np.unique(positions).astype(np.int64)


def cell_index(edges: FloatArray, value: float) -> Optional[int]:
    index = int(np.searchsorted(edges, value, side="right") - 1)
    if index == edges.size - 1 and math.isclose(value, float(edges[-1])):
        index -= 1
    if index < 0 or index >= edges.size - 1:
        return None
    return index


# ---------------------------------------------------------------------------
# Replay helpers
# ---------------------------------------------------------------------------


def query_config_from_metadata(
    solver: ModuleType,
    atlas: ForwardAtlas,
    samples: int,
    options: QueryOptions,
):
    fields = getattr(solver.QueryConfig, "__dataclass_fields__", {})
    source = atlas.metadata.get("config", {})
    coverage = source.get("coverage", {}) if isinstance(source, dict) else {}
    kwargs = {}
    for name in ("rtol", "atol", "max_step", "r_collision", "r_escape", "max_acceleration"):
        if name in fields and name in source:
            kwargs[name] = source[name]
    mapped = {
        "method": options.method,
        "max_iterations": options.max_iterations,
        "max_seed_attempts": options.max_seed_attempts,
        "wall_time_seconds": options.wall_time_seconds,
        "line_search_steps": options.line_search_steps,
        "step_limit": options.step_limit,
        "robust_fallback": options.robust_fallback,
        "allow_continuation": options.allow_continuation,
        "neighbours": int(coverage.get("neighbours", 24)),
        "direct_seeds": int(coverage.get("direct_seeds", 4)),
        "regression_neighbours": int(coverage.get("regression_neighbours", 16)),
        "max_nfev": int(coverage.get("max_nfev", 35)),
        "trajectory_points": max(32, int(samples)),
    }
    for name, value in mapped.items():
        if name in fields:
            kwargs[name] = value
    return solver.QueryConfig(**kwargs)


def theta_revolution_decomposition(theta: float) -> tuple[float, int]:
    principal = math.atan2(math.sin(theta), math.cos(theta))
    revolutions = int(round((theta - principal) / (2.0 * math.pi)))
    return principal, revolutions


def _explicit_corrected_replay(
    solver: ModuleType,
    solver_atlas,
    atlas: ForwardAtlas,
    seed_row: int,
    *,
    rho: float,
    theta: float,
    kappa: float,
    samples: int,
    label: str,
    query_options: QueryOptions,
) -> Optional[Replay]:
    """Correct one exact circular transfer using the production fast solver."""
    config = query_config_from_metadata(solver, atlas, samples, query_options)
    target = np.array(
        [0.0, 1.0, math.log(rho), theta, 0.0, rho ** -0.5, math.log(kappa)],
        dtype=float,
    )
    seed_launch = np.asarray(atlas.launch[seed_row], dtype=float)
    bounds = solver_atlas._bounds_for_correction(config)
    if hasattr(solver, "_correct_target_from_explicit_seed"):
        launch, _nfev = solver._correct_target_from_explicit_seed(
            target, seed_launch, config, bounds
        )
    else:
        launch = None
    if launch is None:
        return None
    t_eval = np.linspace(0.0, float(launch[6]), max(32, int(samples)))
    dense, normal_constant, status = solver._integrate_launch(
        launch,
        rtol=config.rtol,
        atol=config.atol,
        max_step=config.max_step,
        r_collision=config.r_collision,
        r_escape=config.r_escape,
        max_acceleration=config.max_acceleration,
        t_eval=t_eval,
    )
    if dense is None or status != "ok" or normal_constant <= 0.0:
        return None
    y = np.asarray(dense.y, dtype=float)
    radius, theta_end, ur, ut, resource = solver._endpoint_from_state(y[:, -1])
    raw = np.array(
        [math.log(radius) - math.log(rho), theta_end - theta, ur,
         ut - rho ** -0.5, math.log(resource) - math.log(kappa)], dtype=float
    )
    if np.any(np.abs(raw) > np.asarray(config.acceptance, dtype=float)):
        return None
    return Replay(
        label=label,
        tau_f=float(launch[6]),
        time=np.asarray(dense.t, dtype=float),
        position=y[0:2, :].T,
        velocity=y[2:4, :].T,
        acceleration=y[4:6, :].T,
        resource_fraction=np.asarray(y[8, :] / kappa, dtype=float),
        residual_text=f"fast residual={np.linalg.norm(raw):.2e}",
        launch=np.asarray(launch, dtype=float),
    )


def _nearest_seed_rows_for_target(
    solver_atlas,
    atlas: ForwardAtlas,
    rho: float,
    theta: float,
    kappa: float,
    count: int,
) -> IntArray:
    target = np.array(
        [0.0, 1.0, math.log(rho), theta, 0.0, rho ** -0.5, math.log(kappa)],
        dtype=float,
    )
    if hasattr(solver_atlas, "_nearest_indices"):
        try:
            return np.asarray(solver_atlas._nearest_indices(target, count), dtype=np.int64)
        except Exception:
            pass
    from scipy.spatial import cKDTree

    tree = cKDTree(atlas.endpoint / atlas.feature_scale)
    _distance, index = tree.query(target / atlas.feature_scale, k=min(count, atlas.rows))
    return np.atleast_1d(index).astype(np.int64)


def corrected_replays_target(
    solver: ModuleType,
    solver_atlas,
    atlas: ForwardAtlas,
    *,
    rho: float,
    theta: float,
    kappa: float,
    seed_rows: Sequence[int],
    samples: int,
    all_branches: bool,
    max_branches: int,
    query_options: QueryOptions,
) -> list[Replay]:
    """Solve an exact circular query at an arbitrary clicked target."""
    if not (rho > 0.0 and kappa > 0.0):
        raise ViewerError("rho and kappa must be positive")

    # Explicit nearby seeds are tried first, including rows outside the clicked
    # projection bin.  This makes empty cells fully queryable.
    replays: list[Replay] = []
    for seed_row in seed_rows:
        replay = _explicit_corrected_replay(
            solver,
            solver_atlas,
            atlas,
            int(seed_row),
            rho=rho,
            theta=theta,
            kappa=kappa,
            samples=samples,
            label=f"corrected seed row {int(seed_row)}",
            query_options=query_options,
        )
        if replay is None:
            continue
        duplicate = any(
            np.linalg.norm(
                (replay.launch - old.launch)
                / np.array([1, 1, 1, 1, 1, 1, max(1.0, old.tau_f)], dtype=float)
            )
            < 2.0e-5
            for old in replays
        )
        if not duplicate:
            replays.append(replay)
        if not all_branches and replays:
            break
        if max_branches > 0 and len(replays) >= max_branches:
            break

    # The public atlas query performs local regression, several direct seeds,
    # and exact Newton/least-squares correction for the requested target.
    if not replays or (all_branches and (max_branches <= 0 or len(replays) < max_branches)):
        principal, revolutions = theta_revolution_decomposition(theta)
        initial = solver.circular_state(1.0, 1.0, angle=0.0, prograde=True)
        final = solver.circular_state(rho, 1.0, angle=principal, prograde=True)
        capability = solver.RocketCapability(
            useful_power=kappa,
            initial_mass=2.0,
            dry_mass=1.0,
            mu=1.0,
        )
        config = query_config_from_metadata(solver, atlas, samples, query_options)
        try:
            result = solver_atlas.solve(
                initial,
                final,
                capability,
                revolutions=revolutions,
                config=config,
                return_all=all_branches,
            )
        except RuntimeError:
            result = []
        solutions = result if isinstance(result, list) else ([result] if result else [])
        for index, solution in enumerate(solutions):
            trajectory = solution.trajectory
            candidate = Replay(
                label=f"corrected atlas branch {index}",
                tau_f=float(solution.launch[6]),
                time=np.asarray(trajectory["time"], dtype=float),
                position=np.asarray(trajectory["position"], dtype=float),
                velocity=np.asarray(trajectory["velocity"], dtype=float),
                acceleration=np.asarray(trajectory["acceleration"], dtype=float),
                resource_fraction=np.asarray(trajectory["resource_fraction"], dtype=float),
                residual_text=f"||scaled residual||={solution.residual_norm:.2e}",
                launch=np.asarray(solution.launch, dtype=float),
            )
            duplicate = any(
                np.linalg.norm(
                    (candidate.launch - old.launch)
                    / np.array([1, 1, 1, 1, 1, 1, max(1.0, old.tau_f)], dtype=float)
                )
                < 2.0e-5
                for old in replays
            )
            if not duplicate:
                replays.append(candidate)
            if max_branches > 0 and len(replays) >= max_branches:
                break

    # Final fallback: continue gradually from nearby endpoint/launch pairs to
    # the exact clicked circular target.  This is more expensive than one
    # Newton solve but substantially enlarges the practical convergence basin.
    if (
        not replays
        and query_options.allow_continuation
        and hasattr(solver, "_continuation_correct_target")
    ):
        target = np.array(
            [0.0, 1.0, math.log(rho), theta, 0.0, rho ** -0.5, math.log(kappa)],
            dtype=float,
        )
        config = query_config_from_metadata(solver, atlas, samples, query_options)
        try:
            bounds = solver._launch_correction_bounds(
                atlas.launch, config.launch_bound_margin
            )
            continuation_rows = _nearest_seed_rows_for_target(
                solver_atlas, atlas, rho, theta, kappa, max(8, max_branches)
            )
            for seed_row in continuation_rows:
                launch, _nfev = solver._continuation_correct_target(
                    atlas.launch[int(seed_row)],
                    atlas.endpoint[int(seed_row)],
                    target,
                    config,
                    bounds,
                    12,
                )
                if launch is None:
                    continue
                t_eval = np.linspace(0.0, float(launch[6]), max(32, int(samples)))
                dense, normal_constant, status = solver._integrate_launch(
                    launch,
                    rtol=config.rtol,
                    atol=config.atol,
                    max_step=config.max_step,
                    r_collision=config.r_collision,
                    r_escape=config.r_escape,
                    max_acceleration=config.max_acceleration,
                    t_eval=t_eval,
                )
                if dense is None or status != "ok" or normal_constant <= 0.0:
                    continue
                y = np.asarray(dense.y, dtype=float)
                radius, theta_end, ur, ut, resource = solver._endpoint_from_state(y[:, -1])
                raw = np.array(
                    [
                        math.log(radius) - math.log(rho),
                        theta_end - theta,
                        ur,
                        ut - rho ** -0.5,
                        math.log(resource) - math.log(kappa),
                    ],
                    dtype=float,
                )
                if np.any(np.abs(raw) > np.asarray(config.acceptance, dtype=float)):
                    continue
                replays.append(
                    Replay(
                        label=f"continued seed row {int(seed_row)}",
                        tau_f=float(launch[6]),
                        time=np.asarray(dense.t, dtype=float),
                        position=y[0:2, :].T,
                        velocity=y[2:4, :].T,
                        acceleration=y[4:6, :].T,
                        resource_fraction=np.asarray(y[8, :] / kappa, dtype=float),
                        residual_text=f"continued residual={np.linalg.norm(raw):.2e}",
                        launch=np.asarray(launch, dtype=float),
                    )
                )
                if not all_branches or (max_branches > 0 and len(replays) >= max_branches):
                    break
        except Exception:
            pass

    if not replays:
        certificate_text = ""
        try:
            principal, revolutions = theta_revolution_decomposition(theta)
            initial = solver.circular_state(1.0, 1.0, angle=0.0, prograde=True)
            final = solver.circular_state(rho, 1.0, angle=principal, prograde=True)
            capability = solver.RocketCapability(
                useful_power=kappa,
                initial_mass=2.0,
                dry_mass=1.0,
                mu=1.0,
            )
            certificate = solver_atlas.coverage_certificate(
                initial, final, capability, revolutions
            )
            certificate_text = (
                f" Nearest normalized distance={certificate['distance']:.3g}; "
                f"validated radius={certificate['validated_radius']:.3g}; "
                f"inside={certificate['inside_validated_cell']}."
            )
        except Exception:
            pass
        raise ViewerError(
            "No exact circular branch converged for the clicked target." + certificate_text
        )

    replays.sort(key=lambda replay: replay.tau_f)
    if not all_branches:
        return replays[:1]
    return replays[:max_branches] if max_branches > 0 else replays


def raw_replay(
    solver: ModuleType,
    atlas: ForwardAtlas,
    row: int,
    *,
    samples: int,
) -> Replay:
    launch = np.asarray(atlas.launch[row], dtype=float)
    source = atlas.metadata.get("config", {})
    t_eval = np.linspace(0.0, float(launch[6]), max(32, int(samples)))
    sol, normal_constant, status = solver._integrate_launch(
        launch,
        rtol=float(source.get("rtol", 2.0e-9)),
        atol=float(source.get("atol", 2.0e-11)),
        max_step=float(source.get("max_step", 0.04)),
        r_collision=float(source.get("r_collision", 0.01)),
        r_escape=float(source.get("r_escape", 200.0)),
        max_acceleration=float(source.get("max_acceleration", 5000.0)),
        t_eval=t_eval,
    )
    if sol is None or status != "ok":
        raise ViewerError(f"Raw trajectory replay failed: {status}")
    y = np.asarray(sol.y, dtype=float)
    rho = float(atlas.rho[row])
    circular_speed = rho ** -0.5
    ur = float(atlas.endpoint[row, 4])
    ut = float(atlas.endpoint[row, 5])
    residual = math.hypot(ur, ut - circular_speed)
    kappa = float(atlas.kappa[row])
    return Replay(
        label=f"raw atlas row {row}",
        tau_f=float(launch[6]),
        time=np.asarray(sol.t, dtype=float),
        position=y[0:2, :].T,
        velocity=y[2:4, :].T,
        acceleration=y[4:6, :].T,
        resource_fraction=np.asarray(y[8, :] / kappa, dtype=float),
        residual_text=f"circular velocity error={residual:.2e}; K={normal_constant:.3g}",
        launch=launch,
    )


def plot_replays(replays: Sequence[Replay], rho: float, theta: float, kappa: float) -> None:
    if not replays:
        return
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    ax_traj, ax_acc, ax_budget, ax_vel = axes.ravel()
    palette = plt.get_cmap("tab10")
    max_extent = max(1.0, rho)

    for order, replay in enumerate(replays):
        color = palette(order % 10)
        rxy = replay.position
        vxy = replay.velocity
        axy = replay.acceleration
        radius = np.linalg.norm(rxy, axis=1)
        acceleration = np.linalg.norm(axy, axis=1)
        radial_velocity = np.einsum("ij,ij->i", rxy, vxy) / radius
        tangential_velocity = (rxy[:, 0] * vxy[:, 1] - rxy[:, 1] * vxy[:, 0]) / radius
        circular_velocity = radius ** -0.5
        label = f"{replay.label}: $\\tau_f$={replay.tau_f:.6g}"

        ax_traj.plot(rxy[:, 0], rxy[:, 1], color=color, label=label)
        ax_traj.scatter(rxy[0, 0], rxy[0, 1], color=color, marker="o", s=28)
        ax_traj.scatter(rxy[-1, 0], rxy[-1, 1], color=color, marker="x", s=48)
        ax_acc.plot(replay.time, acceleration, color=color, label=f"|A|, {replay.label}")
        ax_acc.plot(replay.time, axy[:, 0], color=color, linestyle="--", alpha=0.65)
        ax_acc.plot(replay.time, axy[:, 1], color=color, linestyle=":", alpha=0.65)
        ax_budget.plot(replay.time, replay.resource_fraction, color=color, label=replay.label)
        ax_vel.plot(replay.time, radial_velocity, color=color, linestyle="--", label=f"$v_r$, {replay.label}")
        ax_vel.plot(replay.time, tangential_velocity, color=color, label=f"$v_t$, {replay.label}")
        if order == 0:
            ax_vel.plot(replay.time, circular_velocity, color="black", alpha=0.45, label="$1/\\sqrt{r}$")
        max_extent = max(max_extent, float(np.nanmax(radius)))

    for orbit_radius, style in ((1.0, "--"), (rho, ":")):
        ax_traj.add_patch(plt.Circle((0.0, 0.0), orbit_radius, fill=False, linestyle=style, alpha=0.5))
    ax_traj.scatter([0.0], [0.0], marker="*", s=180, label="central body")
    target = rho * np.array([math.cos(theta), math.sin(theta)])
    ax_traj.scatter([target[0]], [target[1]], marker="+", s=100, label="target")
    extent = 1.08 * max_extent
    ax_traj.set_xlim(-extent, extent)
    ax_traj.set_ylim(-extent, extent)
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.set_xlabel("Dimensionless x")
    ax_traj.set_ylabel("Dimensionless y")
    ax_traj.set_title("Trajectory")
    ax_traj.legend(fontsize=8)

    ax_acc.set_xlabel("Dimensionless time $\\tau$")
    ax_acc.set_ylabel("Dimensionless acceleration")
    ax_acc.set_title("Acceleration: magnitude and Cartesian components")
    ax_acc.grid(alpha=0.25)
    ax_acc.legend(fontsize=8)

    ax_budget.axhline(1.0, linestyle="--", alpha=0.5)
    ax_budget.set_xlabel("Dimensionless time $\\tau$")
    ax_budget.set_ylabel("Consumed resource fraction")
    ax_budget.set_ylim(bottom=0.0)
    ax_budget.set_title("Inverse-mass expenditure")
    ax_budget.grid(alpha=0.25)
    ax_budget.legend(fontsize=8)

    ax_vel.axhline(0.0, linewidth=0.8, alpha=0.4)
    ax_vel.set_xlabel("Dimensionless time $\\tau$")
    ax_vel.set_ylabel("Dimensionless velocity")
    ax_vel.set_title("Radial and tangential velocity")
    ax_vel.grid(alpha=0.25)
    ax_vel.legend(fontsize=8, ncols=2)

    residuals = "; ".join(f"{r.label}: {r.residual_text}" for r in replays)
    fig.suptitle(
        f"Circular transfer: rho={rho:.6g}, theta={math.degrees(theta):.2f}°, "
        f"kappa={kappa:.6g}\n{residuals}",
        fontsize=11,
    )
    fig.canvas.manager.set_window_title(
        f"PMP circular trajectory rho={rho:.4g}, theta={math.degrees(theta):.1f}°, kappa={kappa:.4g}"
    )
    plt.show(block=False)


# ---------------------------------------------------------------------------
# Interactive catalogue
# ---------------------------------------------------------------------------


def make_catalogue(
    atlas: ForwardAtlas,
    solver: ModuleType,
    solver_atlas,
    projection: Projection,
    *,
    metric: str,
    nrows: int,
    ncols: int,
    samples: int,
    figsize: tuple[float, float],
    replay_mode: str,
    max_replay_branches: int,
    query_options: QueryOptions,
    distance_workers: int,
    distance_cache: Optional[Path],
    use_distance_cache: bool,
):
    theta_indices = infer_theta_panels(projection, nrows * ncols)
    values = projection_metric(
        atlas,
        projection,
        metric,
        theta_indices=theta_indices,
        distance_workers=distance_workers,
        distance_cache=distance_cache,
        use_distance_cache=use_distance_cache,
    )
    selected_values = values
    norm = metric_norm(selected_values, metric)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.88")

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    mesh = None
    for panel, ax in enumerate(axes_flat):
        if panel >= theta_indices.size:
            ax.set_visible(False)
            continue
        j = int(theta_indices[panel])
        panel_values = np.ma.masked_invalid(values[:, panel, :].T)
        mesh = ax.pcolormesh(
            projection.rho_edges,
            projection.kappa_edges,
            panel_values,
            shading="auto",
            cmap=cmap,
            norm=norm,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\rho=r_f/r_0$")
        ax.set_ylabel(r"$\kappa$")
        center = 0.5 * (projection.theta_edges[j] + projection.theta_edges[j + 1])
        ax.set_title(rf"$\Theta\approx{math.degrees(center):.1f}^\circ$")
        ax._pmp_theta_index = j  # type: ignore[attr-defined]

    if mesh is None:
        raise ViewerError("No catalogue panels were created")
    visible_axes = [ax for ax in axes_flat if ax.get_visible()]
    cbar = fig.colorbar(mesh, ax=visible_axes, pad=0.02, shrink=0.93)
    cbar.set_label(METRIC_LABELS[metric])

    selected_rows = projection.selection.row_indices.size
    occupied = int(np.count_nonzero(projection.selected_row >= 0))
    total_bins = int(np.prod(projection.shape))
    factor = projection.selection.relax_factor
    fig.suptitle(
        f"Circular-to-circular projection of {atlas.path.name} — {METRIC_LABELS[metric]}\n"
        f"selected rows={selected_rows:,}, occupied bins={occupied:,}/{total_bins:,}, "
        f"tolerance relaxation={factor:.3g}×; every cell queryable; left: one, right: several",
        fontsize=12,
    )
    fig.canvas.manager.set_window_title(f"PMP circular subspace: {atlas.path.name}")
    status = fig.text(
        0.01,
        0.005,
        f"Replay mode: {replay_mode}. Every displayed cell is queryable; grey means no raw circular sample.",
        ha="left",
        va="bottom",
        fontsize=9,
    )

    query_cache: dict[tuple[int, int, int, bool], object] = {}

    def mark_cell(ax, x: float, y: float, *, success: bool) -> None:
        ax.scatter(
            [x], [y], marker=("o" if success else "x"), s=(52 if success else 72),
            linewidths=1.8, facecolors="none" if success else None,
            edgecolors="black" if success else None,
            color=None if success else "crimson", zorder=20,
        )

    def on_click(event) -> None:
        ax = event.inaxes
        if ax is None or not hasattr(ax, "_pmp_theta_index"):
            return
        if event.xdata is None or event.ydata is None:
            return
        i = cell_index(projection.rho_edges, float(event.xdata))
        k = cell_index(projection.kappa_edges, float(event.ydata))
        j = int(ax._pmp_theta_index)  # type: ignore[attr-defined]
        if i is None or k is None:
            status.set_text("Clicked outside catalogue bounds.")
            fig.canvas.draw_idle()
            return

        # Use the exact clicked rho/kappa and the panel's theta-bin centre.  The
        # target no longer depends on a raw approximately circular atlas row.
        rho = float(event.xdata)
        kappa = float(event.ydata)
        theta = float(0.5 * (projection.theta_edges[j] + projection.theta_edges[j + 1]))
        row = int(projection.selected_row[i, j, k])
        cell = projection.flat_cell(i, j, k)
        cell_candidates = projection.cell_rows.get(cell, np.empty(0, dtype=np.int64))
        nearest = _nearest_seed_rows_for_target(
            solver_atlas,
            atlas,
            rho,
            theta,
            kappa,
            max(16, 4 * max_replay_branches),
        )
        seed_rows = np.unique(np.concatenate((cell_candidates, nearest))).astype(np.int64)
        all_requested = event.button == 3
        cache_key = (i, j, k, bool(all_requested))
        cached = query_cache.get(cache_key)
        if cached is not None:
            if isinstance(cached, str):
                status.set_text(cached + " (cached)")
                mark_cell(ax, rho, kappa, success=False)
                fig.canvas.draw_idle()
                return
            replays = cached
            status.set_text(
                f"Using cached solution for cell ({i},{j},{k}); "
                f"rho={rho:.6g}, theta={math.degrees(theta):.2f}°, kappa={kappa:.6g}"
            )
            mark_cell(ax, rho, kappa, success=True)
            fig.canvas.draw_idle()
            plot_replays(replays, rho=rho, theta=theta, kappa=kappa)
            return

        raw_note = "occupied raw bin" if row >= 0 else "empty raw bin"
        status.set_text(
            f"Solving exact cell ({i},{j},{k}), {raw_note}, mode={replay_mode}; "
            f"rho={rho:.6g}, theta={math.degrees(theta):.2f}°, kappa={kappa:.6g}"
        )
        print(status.get_text())
        fig.canvas.draw_idle()

        try:
            if replay_mode == "corrected":
                replays = corrected_replays_target(
                    solver,
                    solver_atlas,
                    atlas,
                    rho=rho,
                    theta=theta,
                    kappa=kappa,
                    seed_rows=[int(value) for value in seed_rows],
                    samples=samples,
                    all_branches=all_requested,
                    max_branches=max_replay_branches,
                    query_options=query_options,
                )
            else:
                if row < 0:
                    raise ViewerError(
                        "Raw replay requires a stored approximately circular row. "
                        "Use --replay-mode corrected for empty cells."
                    )
                chosen = cell_candidates[:max_replay_branches] if all_requested else np.array([row])
                replays = [
                    raw_replay(solver, atlas, int(candidate), samples=samples)
                    for candidate in chosen
                ]
        except Exception as exc:
            message = f"Replay failed for cell ({i},{j},{k}): {exc}"
            query_cache[cache_key] = message
            status.set_text(message)
            print(message)
            mark_cell(ax, rho, kappa, success=False)
            fig.canvas.draw_idle()
            return
        query_cache[cache_key] = replays
        mark_cell(ax, rho, kappa, success=True)
        fig.canvas.draw_idle()
        plot_replays(replays, rho=rho, theta=theta, kappa=kappa)

    fig.canvas.mpl_connect("button_press_event", on_click)
    return fig


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def parse_figsize(text: str) -> tuple[float, float]:
    try:
        width_text, height_text = text.lower().split("x", maxsplit=1)
        width, height = float(width_text), float(height_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("figsize must look like 16x10") from exc
    if width <= 0.0 or height <= 0.0:
        raise argparse.ArgumentTypeError("figsize dimensions must be positive")
    return width, height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("atlas", type=Path, help="NPZ atlas produced by pmp_extremal_atlas.py")
    parser.add_argument(
        "--solver",
        type=Path,
        default=Path(__file__).with_name("pmp_extremal_atlas.py"),
        help="Path to pmp_extremal_atlas.py",
    )
    parser.add_argument("--nrows", type=int, default=3)
    parser.add_argument("--ncols", type=int, default=4)
    parser.add_argument("--rho-bins", type=int, default=80)
    parser.add_argument("--theta-bins", type=int, default=120)
    parser.add_argument("--kappa-bins", type=int, default=60)
    parser.add_argument("--rho-min", type=float, default=None)
    parser.add_argument("--rho-max", type=float, default=None)
    parser.add_argument("--theta-min", type=float, default=None)
    parser.add_argument("--theta-max", type=float, default=None)
    parser.add_argument("--kappa-min", type=float, default=None)
    parser.add_argument("--kappa-max", type=float, default=None)
    parser.add_argument("--initial-radial-tol", type=float, default=0.03)
    parser.add_argument("--initial-tangential-tol", type=float, default=0.03)
    parser.add_argument("--final-radial-relative-tol", type=float, default=0.05)
    parser.add_argument("--final-tangential-relative-tol", type=float, default=0.05)
    parser.add_argument(
        "--min-circular-rows",
        type=int,
        default=1000,
        help="Uniformly relax tolerances until at least this many rows pass",
    )
    parser.add_argument("--max-relax", type=float, default=8.0)
    parser.add_argument("--no-auto-relax", action="store_true")
    parser.add_argument(
        "--metric",
        choices=tuple(METRIC_LABELS),
        default="nearest_distance",
    )
    parser.add_argument("--replay-mode", choices=("corrected", "raw"), default="corrected")
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--max-cell-candidates", type=int, default=12)
    parser.add_argument("--max-replay-branches", type=int, default=6)
    parser.add_argument(
        "--query-method", choices=("fast_newton", "robust_least_squares"),
        default="fast_newton", help="Numerical corrector used for each clicked pixel",
    )
    parser.add_argument("--query-seconds", type=float, default=3.0)
    parser.add_argument("--query-iterations", type=int, default=6)
    parser.add_argument("--query-seeds", type=int, default=3)
    parser.add_argument("--query-line-search", type=int, default=3)
    parser.add_argument("--query-step-limit", type=float, default=0.30)
    parser.add_argument("--robust-fallback", action="store_true")
    parser.add_argument("--allow-continuation", action="store_true")
    parser.add_argument("--figsize", type=parse_figsize, default=(16.0, 10.0))
    parser.add_argument("--save-catalogue", type=Path, default=None)
    parser.add_argument(
        "--distance-workers", type=int, default=-1,
        help="cKDTree query workers for displayed coverage slices (-1 uses all cores)",
    )
    parser.add_argument(
        "--distance-cache", type=Path, default=None,
        help="Optional sidecar NPZ for displayed nearest-distance slices",
    )
    parser.add_argument("--no-distance-cache", action="store_true")
    parser.add_argument("--no-show", action="store_true", help="Build/save without opening GUI")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.nrows <= 0 or args.ncols <= 0:
        raise SystemExit("--nrows and --ncols must be positive")
    if args.samples < 32:
        raise SystemExit("--samples must be at least 32")
    if min(args.rho_bins, args.theta_bins, args.kappa_bins) <= 0:
        raise SystemExit("All bin counts must be positive")

    atlas = load_forward_atlas(args.atlas)
    solver = load_module_from_path(args.solver)
    if not hasattr(solver, "ExtremalAtlas"):
        raise ViewerError("Solver module does not expose ExtremalAtlas")
    solver_atlas = solver.ExtremalAtlas(atlas.path)

    tolerances = CircularTolerances(
        initial_radial=args.initial_radial_tol,
        initial_tangential=args.initial_tangential_tol,
        final_radial_relative=args.final_radial_relative_tol,
        final_tangential_relative=args.final_tangential_relative_tol,
    )
    selection = select_circular_rows(
        atlas,
        tolerances,
        min_rows=args.min_circular_rows,
        auto_relax=not args.no_auto_relax,
        max_relax=args.max_relax,
    )

    rho_bounds = resolve_bounds(
        atlas,
        selection,
        (args.rho_min, args.rho_max),
        "rho_bounds",
        atlas.rho,
        positive=True,
    )
    theta_bounds = resolve_bounds(
        atlas,
        selection,
        (args.theta_min, args.theta_max),
        "theta_bounds",
        atlas.theta,
        positive=False,
    )
    kappa_bounds = resolve_bounds(
        atlas,
        selection,
        (args.kappa_min, args.kappa_max),
        "kappa_bounds",
        atlas.kappa,
        positive=True,
    )

    projection = build_projection(
        atlas,
        selection,
        rho_edges=log_edges(*rho_bounds, args.rho_bins),
        theta_edges=linear_edges(*theta_bounds, args.theta_bins),
        kappa_edges=log_edges(*kappa_bounds, args.kappa_bins),
        max_cell_candidates=args.max_cell_candidates,
    )

    components = circular_components(atlas)
    selected = selection.row_indices
    print(f"Loaded {atlas.path}")
    print(f"Forward atlas rows: {atlas.rows:,}")
    if "coverage_converged" in atlas.metadata:
        print(f"Coverage certificate converged: {atlas.metadata.get('coverage_converged')}")
        if "coverage_final_general_success_rate" in atlas.metadata:
            print(
                "Final certified rates: "
                f"general={atlas.metadata.get('coverage_final_general_success_rate', float('nan')):.3%}, "
                f"circular={atlas.metadata.get('coverage_final_circular_success_rate', float('nan')):.3%}, "
                f"p95={atlas.metadata.get('coverage_final_p95_nearest_distance', float('nan')):.3g}, "
                f"p99={atlas.metadata.get('coverage_final_p99_nearest_distance', float('nan')):.3g}"
            )
    print(f"Approximately circular rows: {selected.size:,}")
    print(f"Tolerance relaxation factor: {selection.relax_factor:.6g}x")
    print(
        "Effective tolerances: "
        f"|u0|<={args.initial_radial_tol * selection.relax_factor:.4g}, "
        f"|w0-1|<={args.initial_tangential_tol * selection.relax_factor:.4g}, "
        f"|ur_f|/v_c<={args.final_radial_relative_tol * selection.relax_factor:.4g}, "
        f"|ut_f/v_c-1|<={args.final_tangential_relative_tol * selection.relax_factor:.4g}"
    )
    if selected.size:
        print(
            "Median selected deviations: "
            f"u0={np.median(components[0][selected]):.3g}, "
            f"w0={np.median(components[1][selected]):.3g}, "
            f"ur_f/vc={np.median(components[2][selected]):.3g}, "
            f"ut_f/vc={np.median(components[3][selected]):.3g}"
        )
    else:
        print("Median selected deviations: no raw approximately circular rows")
    print(f"Occupied projection bins: {np.count_nonzero(projection.selected_row >= 0):,}")

    query_options = QueryOptions(
        method=args.query_method,
        max_iterations=args.query_iterations,
        max_seed_attempts=args.query_seeds,
        wall_time_seconds=args.query_seconds,
        line_search_steps=args.query_line_search,
        step_limit=args.query_step_limit,
        robust_fallback=args.robust_fallback,
        allow_continuation=args.allow_continuation,
    )
    print(
        "Pixel solver: "
        f"{query_options.method}, {query_options.max_seed_attempts} seeds, "
        f"{query_options.max_iterations} iterations, "
        f"{query_options.wall_time_seconds:.3g}s budget; "
        f"fallback={query_options.robust_fallback}, "
        f"continuation={query_options.allow_continuation}"
    )

    distance_cache = args.distance_cache
    if distance_cache is None:
        distance_cache = atlas.path.with_name(atlas.path.stem + ".viewer_distance_cache.npz")

    fig = make_catalogue(
        atlas,
        solver,
        solver_atlas,
        projection,
        metric=args.metric,
        nrows=args.nrows,
        ncols=args.ncols,
        samples=args.samples,
        figsize=args.figsize,
        replay_mode=args.replay_mode,
        max_replay_branches=args.max_replay_branches,
        query_options=query_options,
        distance_workers=args.distance_workers,
        distance_cache=distance_cache,
        use_distance_cache=not args.no_distance_cache,
    )
    if args.save_catalogue is not None:
        output = args.save_catalogue.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
        print(f"Saved catalogue image to {output}")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
