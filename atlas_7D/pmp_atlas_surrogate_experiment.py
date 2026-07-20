#!/usr/bin/env python3
"""Compression and surrogate experiment for a planar PMP extremal atlas.

The script answers two questions before a compressed atlas representation is
chosen:

1. Which endpoint directions and cross-directions are locally nonlinear?
2. Can many stored rows be replaced by local inverse charts or a compact
   branch-conditioned regression model without losing fast exact-query success?

It deliberately treats exact shooting convergence under the production query
budget as the decisive metric.  Direct launch-parameter prediction errors are
reported as diagnostics only.

The experiment never inserts circular trajectories into the training data.
Circular-to-circular targets are generated only as a lower-dimensional probe of
how well a generic seven-dimensional representation generalizes.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from scipy.stats import qmc

try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.ensemble import ExtraTreesRegressor
    from threadpoolctl import threadpool_limits
except Exception as exc:  # pragma: no cover - dependency failure is user-facing
    raise SystemExit(
        "pmp_atlas_surrogate_experiment.py requires scikit-learn"
    ) from exc

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plots are optional
    plt = None

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

ENDPOINT_NAMES = (
    "u0",
    "w0",
    "log_rho",
    "theta_unwrapped",
    "ur_final",
    "ut_final",
    "log_kappa",
)
Q_NAMES = ("Ar0", "At0", "Jr0", "ell", "log_tau")


@dataclass
class SplitData:
    train: IntArray
    random_holdout: IntArray
    cell_holdout: IntArray


@dataclass
class ChartBank:
    model_name: str
    centers_x: FloatArray
    centers_q: FloatArray
    branches: IntArray
    coordinate_scale: FloatArray
    q_scale: FloatArray
    q_min: FloatArray
    q_max: FloatArray
    affine: FloatArray
    quadratic: Optional[FloatArray]
    local_radius: FloatArray
    affine_test_rmse: FloatArray
    quadratic_test_rmse: FloatArray
    build_seconds: float
    storage_bytes: int

    def __post_init__(self) -> None:
        self.tree = cKDTree(self.centers_x)

    def predict_candidates(
        self,
        x: FloatArray,
        *,
        winding: int,
        candidate_count: int,
        search_charts: int = 24,
    ) -> list[FloatArray]:
        k = min(max(candidate_count, search_charts), len(self.centers_x))
        _distance, index = self.tree.query(x, k=k)
        indices = np.atleast_1d(index).astype(int)
        # Winding is visible from the unwrapped angle. Radial turns are not, so
        # they remain alternative local experts rather than a hard filter.
        matching = [i for i in indices if decode_branch(int(self.branches[i]))[0] == winding]
        ordered = matching + [i for i in indices if i not in set(matching)]
        output: list[FloatArray] = []
        for i in ordered:
            dx = (x - self.centers_x[i]) / self.coordinate_scale[i]
            normalized_dq = self.affine[i] @ dx
            if self.quadratic is not None:
                normalized_dq = normalized_dq + self.quadratic[i] @ quadratic_features(dx)
            q = self.centers_q[i] + self.q_scale * normalized_dq
            q = np.clip(q, self.q_min, self.q_max)
            if np.all(np.isfinite(q)):
                output.append(np.asarray(q, dtype=float))
            if len(output) >= candidate_count:
                break
        return output


@dataclass
class TreeMixture:
    global_model: ExtraTreesRegressor
    experts: dict[int, ExtraTreesRegressor]
    expert_counts: dict[int, int]
    build_seconds: float
    storage_bytes: int

    @staticmethod
    def _predict_one(model: ExtraTreesRegressor, x: FloatArray) -> FloatArray:
        original_n_jobs = model.n_jobs
        try:
            # Single-target predictions are latency-bound here; serial dispatch
            # avoids sklearn/joblib config propagation warnings without changing
            # the trained model or prediction result.
            model.n_jobs = 1
            return np.asarray(model.predict(x[None, :])[0], dtype=float)
        finally:
            model.n_jobs = original_n_jobs

    def predict_candidates(
        self,
        x: FloatArray,
        *,
        winding: int,
        candidate_count: int,
    ) -> list[FloatArray]:
        output: list[FloatArray] = []
        candidates = [
            code
            for code in self.experts
            if decode_branch(code)[0] == winding
        ]
        candidates.sort(key=lambda code: self.expert_counts[code], reverse=True)
        for code in candidates:
            pred = self._predict_one(self.experts[code], x)
            if np.all(np.isfinite(pred)):
                output.append(pred)
            if len(output) >= candidate_count:
                return output
        global_pred = self._predict_one(self.global_model, x)
        if np.all(np.isfinite(global_pred)):
            output.append(global_pred)
        return output[:candidate_count]


@dataclass
class ExactMetric:
    model: str
    dataset: str
    tested: int
    successes: int
    success_rate: float
    median_seconds: float
    p95_seconds: float
    median_evaluations: float
    p95_evaluations: float
    timeout_or_failure_rate: float


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            value = float(obj)
            return value if math.isfinite(value) else None
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def status(message: str) -> None:
    print(message, flush=True)


def load_solver(path: Path):
    spec = importlib.util.spec_from_file_location("pmp_extremal_atlas_experiment_solver", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import solver module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def branch_codes(endpoint: FloatArray, diagnostics: FloatArray, max_turns: int) -> IntArray:
    if diagnostics.ndim == 2 and diagnostics.shape[1] >= 6:
        turns = np.rint(diagnostics[:, 4]).astype(np.int64)
        winding = np.rint(diagnostics[:, 5]).astype(np.int64)
    else:
        turns = np.zeros(len(endpoint), dtype=np.int64)
        winding = np.rint(endpoint[:, 3] / (2.0 * math.pi)).astype(np.int64)
    turns = np.clip(turns, 0, max_turns)
    # Zig-zag encode winding so codes remain nonnegative.
    wind_code = np.where(winding >= 0, 2 * winding, -2 * winding - 1)
    return (wind_code * (max_turns + 1) + turns).astype(np.int64)


def decode_branch(code: int, max_turns: int = 12) -> tuple[int, int]:
    wind_code, turns = divmod(int(code), max_turns + 1)
    winding = wind_code // 2 if wind_code % 2 == 0 else -(wind_code + 1) // 2
    return int(winding), int(turns)


def make_branch_decoder(max_turns: int):
    global decode_branch

    def decoder(code: int) -> tuple[int, int]:
        wind_code, turns = divmod(int(code), max_turns + 1)
        winding = wind_code // 2 if wind_code % 2 == 0 else -(wind_code + 1) // 2
        return int(winding), int(turns)

    decode_branch = decoder  # type: ignore[assignment]


def q_from_launch(launch: FloatArray) -> FloatArray:
    return np.column_stack((launch[:, 2:6], np.log(np.maximum(launch[:, 6], 1.0e-15))))


def quadratic_features(dx: FloatArray) -> FloatArray:
    dx = np.asarray(dx, dtype=float)
    values = [dx[i] * dx[j] for i in range(len(dx)) for j in range(i, len(dx))]
    return np.asarray(values, dtype=float)


def quadratic_design(dx: FloatArray) -> FloatArray:
    columns = [dx[:, i] * dx[:, j] for i in range(dx.shape[1]) for j in range(i, dx.shape[1])]
    return np.column_stack(columns)


def stable_cell_hash(cells: IntArray, branch: Optional[IntArray] = None) -> NDArray[np.uint64]:
    h = np.full(len(cells), np.uint64(1469598103934665603), dtype=np.uint64)
    primes = (
        1099511628211,
        14029467366897019727,
        1609587929392839161,
        9650029242287828579,
        2870177450012600261,
        11400714785074694791,
        7046029254386353131,
    )
    for j in range(cells.shape[1]):
        value = cells[:, j].astype(np.int64).view(np.uint64)
        h ^= value + np.uint64(primes[j % len(primes)])
        h *= np.uint64(1099511628211)
    if branch is not None:
        h ^= branch.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        h *= np.uint64(1099511628211)
    return h


def diversified_sample_indices(
    endpoint: FloatArray,
    branch: IntArray,
    feature_scale: FloatArray,
    sample_rows: int,
    rng: np.random.Generator,
    candidate_factor: int,
    cell_size: float,
) -> IntArray:
    n = len(endpoint)
    sample_rows = min(sample_rows, n)
    candidate_n = min(n, max(sample_rows, sample_rows * candidate_factor))
    candidate = rng.choice(n, size=candidate_n, replace=False)
    cells = np.floor(endpoint[candidate] / feature_scale / cell_size).astype(np.int64)
    h = stable_cell_hash(cells, branch[candidate])
    order = np.argsort(h, kind="stable")
    sorted_hash = h[order]
    first = np.concatenate(([True], sorted_hash[1:] != sorted_hash[:-1]))
    diverse = candidate[order[first]]
    if len(diverse) >= sample_rows:
        return np.sort(rng.choice(diverse, size=sample_rows, replace=False)).astype(np.int64)
    remaining_mask = np.ones(candidate_n, dtype=bool)
    chosen_lookup = set(map(int, diverse))
    remaining_mask[:] = [int(i) not in chosen_lookup for i in candidate]
    remaining = candidate[remaining_mask]
    need = sample_rows - len(diverse)
    extra = rng.choice(remaining, size=min(need, len(remaining)), replace=False)
    result = np.concatenate((diverse, extra))
    return np.sort(result[:sample_rows]).astype(np.int64)


def split_data(
    x: FloatArray,
    branch: IntArray,
    rng: np.random.Generator,
    random_holdout_fraction: float,
    cell_holdout_fraction: float,
    holdout_cell_size: float,
) -> SplitData:
    cells = np.floor(x / holdout_cell_size).astype(np.int64)
    h = stable_cell_hash(cells, branch)
    cell_threshold = int(round(cell_holdout_fraction * 10_000))
    cell_mask = (h % np.uint64(10_000)) < np.uint64(cell_threshold)
    cell_holdout = np.flatnonzero(cell_mask)
    remaining = np.flatnonzero(~cell_mask)
    random_n = min(int(round(random_holdout_fraction * len(x))), max(len(remaining) // 3, 1))
    random_holdout = rng.choice(remaining, size=random_n, replace=False)
    train_mask = np.ones(len(x), dtype=bool)
    train_mask[cell_holdout] = False
    train_mask[random_holdout] = False
    train = np.flatnonzero(train_mask)
    if len(train) < 100 or len(cell_holdout) < 20 or len(random_holdout) < 20:
        permutation = rng.permutation(len(x))
        n_cell = max(20, int(0.15 * len(x)))
        n_random = max(20, int(0.10 * len(x)))
        cell_holdout = permutation[:n_cell]
        random_holdout = permutation[n_cell : n_cell + n_random]
        train = permutation[n_cell + n_random :]
    return SplitData(
        train=np.asarray(train, dtype=np.int64),
        random_holdout=np.asarray(random_holdout, dtype=np.int64),
        cell_holdout=np.asarray(cell_holdout, dtype=np.int64),
    )


def choose_anchor_indices(
    x_train: FloatArray,
    chart_count: int,
    rng_seed: int,
    workers: int,
) -> IntArray:
    chart_count = min(chart_count, len(x_train))
    model = MiniBatchKMeans(
        n_clusters=chart_count,
        random_state=rng_seed,
        batch_size=min(8192, max(1024, 4 * chart_count)),
        n_init=1,
        max_iter=80,
        reassignment_ratio=0.01,
        verbose=0,
    )
    with threadpool_limits(limits=1):
        model.fit(x_train)
    tree = cKDTree(x_train)
    _distance, index = tree.query(model.cluster_centers_, workers=workers)
    unique = np.unique(np.asarray(index, dtype=np.int64))
    if len(unique) < chart_count:
        rng = np.random.default_rng(rng_seed + 1009)
        missing = chart_count - len(unique)
        pool = np.setdiff1d(np.arange(len(x_train), dtype=np.int64), unique, assume_unique=False)
        if len(pool):
            unique = np.concatenate((unique, rng.choice(pool, min(missing, len(pool)), replace=False)))
    return unique[:chart_count]


def ridge_fit(design: FloatArray, target: FloatArray, weights: FloatArray, ridge: float) -> FloatArray:
    sw = np.sqrt(np.maximum(weights, 1.0e-12))
    aw = design * sw[:, None]
    bw = target * sw[:, None]
    lhs = aw.T @ aw + ridge * np.eye(aw.shape[1])
    rhs = aw.T @ bw
    try:
        coef = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(lhs, rhs, rcond=1.0e-10)[0]
    return np.asarray(coef, dtype=float)


def fit_chart_bank(
    x_train: FloatArray,
    q_train: FloatArray,
    branch_train: IntArray,
    q_scale: FloatArray,
    *,
    chart_count: int,
    neighbours: int,
    ridge: float,
    quadratic: bool,
    rng_seed: int,
    workers: int,
) -> ChartBank:
    started = time.perf_counter()
    tree = cKDTree(x_train)
    anchor_local = choose_anchor_indices(x_train, chart_count, rng_seed, workers)
    centers_x = x_train[anchor_local].copy()
    centers_q = q_train[anchor_local].copy()
    centers_branch = branch_train[anchor_local].copy()
    coordinate_scale = np.ones((len(anchor_local), 7), dtype=np.float32)
    q_min = np.min(q_train, axis=0).astype(float)
    q_max = np.max(q_train, axis=0).astype(float)
    affine = np.zeros((len(anchor_local), 5, 7), dtype=np.float32)
    quad_count = 7 * 8 // 2
    quadratic_coef = (
        np.zeros((len(anchor_local), 5, quad_count), dtype=np.float32) if quadratic else None
    )
    radius = np.zeros(len(anchor_local), dtype=np.float32)
    affine_rmse = np.full(len(anchor_local), np.nan, dtype=np.float32)
    quadratic_rmse = np.full(len(anchor_local), np.nan, dtype=np.float32)

    query_k = min(len(x_train), max(neighbours * 5, neighbours + 8))
    progress_step = max(1, len(anchor_local) // 8)
    for a, center_index in enumerate(anchor_local):
        _dist, candidates = tree.query(x_train[center_index], k=query_k)
        candidates = np.atleast_1d(candidates).astype(np.int64)
        same_branch = candidates[branch_train[candidates] == branch_train[center_index]]
        same_winding = np.asarray(
            [
                i
                for i in candidates
                if decode_branch(int(branch_train[i]))[0]
                == decode_branch(int(branch_train[center_index]))[0]
            ],
            dtype=np.int64,
        )
        if len(same_branch) >= max(24, neighbours // 2):
            local = same_branch[:neighbours]
        elif len(same_winding) >= max(24, neighbours // 2):
            local = same_winding[:neighbours]
        else:
            local = candidates[:neighbours]
        dx_raw = x_train[local] - x_train[center_index]
        local_scale = np.quantile(np.abs(dx_raw), 0.70, axis=0)
        local_scale = np.maximum(local_scale, 0.05)
        coordinate_scale[a] = local_scale.astype(np.float32)
        dx = dx_raw / local_scale
        dq = (q_train[local] - q_train[center_index]) / q_scale
        distance = np.linalg.norm(dx, axis=1)
        bandwidth = max(float(np.quantile(distance, 0.65)), 1.0e-5)
        weights = np.exp(-0.5 * (distance / bandwidth) ** 2)
        radius[a] = float(np.quantile(distance, 0.85))

        order = np.argsort(stable_cell_hash(np.floor(dx / 0.15).astype(np.int64)))
        test_mask = np.zeros(len(local), dtype=bool)
        test_mask[order[::5]] = True
        train_mask = ~test_mask
        if np.count_nonzero(train_mask) < 12 or np.count_nonzero(test_mask) < 4:
            train_mask[:] = True
            test_mask[:] = False

        linear_coef = ridge_fit(dx[train_mask], dq[train_mask], weights[train_mask], ridge)
        if np.any(test_mask):
            prediction = dx[test_mask] @ linear_coef
            affine_rmse[a] = float(np.sqrt(np.mean((prediction - dq[test_mask]) ** 2)))

        full_linear = ridge_fit(dx, dq, weights, ridge)
        affine[a] = full_linear.T.astype(np.float32)

        if quadratic:
            qdx = quadratic_design(dx)
            design = np.column_stack((dx, qdx))
            coef_cv = ridge_fit(design[train_mask], dq[train_mask], weights[train_mask], ridge)
            if np.any(test_mask):
                prediction = design[test_mask] @ coef_cv
                quadratic_rmse[a] = float(np.sqrt(np.mean((prediction - dq[test_mask]) ** 2)))
            full_coef = ridge_fit(design, dq, weights, ridge)
            affine[a] = full_coef[:7].T.astype(np.float32)
            assert quadratic_coef is not None
            quadratic_coef[a] = full_coef[7:].T.astype(np.float32)

        if (a + 1) % progress_step == 0 or a + 1 == len(anchor_local):
            status(f"    fitted {a + 1:,}/{len(anchor_local):,} local charts")

    storage = (
        centers_x.nbytes
        + centers_q.nbytes
        + centers_branch.nbytes
        + coordinate_scale.nbytes
        + np.asarray(q_scale).nbytes
        + q_min.nbytes
        + q_max.nbytes
        + affine.nbytes
        + radius.nbytes
        + affine_rmse.nbytes
        + quadratic_rmse.nbytes
        + (0 if quadratic_coef is None else quadratic_coef.nbytes)
    )
    return ChartBank(
        model_name="quadratic" if quadratic else "affine",
        centers_x=np.asarray(centers_x, dtype=np.float32),
        centers_q=np.asarray(centers_q, dtype=np.float32),
        branches=np.asarray(centers_branch, dtype=np.int64),
        coordinate_scale=coordinate_scale,
        q_scale=np.asarray(q_scale, dtype=float),
        q_min=q_min,
        q_max=q_max,
        affine=affine,
        quadratic=quadratic_coef,
        local_radius=radius,
        affine_test_rmse=affine_rmse,
        quadratic_test_rmse=quadratic_rmse,
        build_seconds=time.perf_counter() - started,
        storage_bytes=int(storage),
    )


def fit_tree_mixture(
    x_train: FloatArray,
    q_train: FloatArray,
    branch_train: IntArray,
    *,
    trees: int,
    max_depth: Optional[int],
    min_leaf: int,
    max_experts: int,
    min_expert_rows: int,
    workers: int,
    seed: int,
) -> TreeMixture:
    started = time.perf_counter()
    global_model = ExtraTreesRegressor(
        n_estimators=trees,
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        max_features=1.0,
        n_jobs=workers,
        random_state=seed,
    )
    with threadpool_limits(limits=1):
        global_model.fit(x_train, q_train)

    codes, counts = np.unique(branch_train, return_counts=True)
    eligible = [
        (int(code), int(count))
        for code, count in zip(codes, counts)
        if count >= min_expert_rows
    ]
    eligible.sort(key=lambda item: item[1], reverse=True)
    experts: dict[int, ExtraTreesRegressor] = {}
    expert_counts: dict[int, int] = {}
    for position, (code, count) in enumerate(eligible[:max_experts]):
        indices = np.flatnonzero(branch_train == code)
        model = ExtraTreesRegressor(
            n_estimators=max(12, trees // 2),
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            max_features=1.0,
            n_jobs=workers,
            random_state=seed + 101 + position,
        )
        with threadpool_limits(limits=1):
            model.fit(x_train[indices], q_train[indices])
        experts[code] = model
        expert_counts[code] = count
    storage = len(pickle.dumps((global_model, experts), protocol=pickle.HIGHEST_PROTOCOL))
    return TreeMixture(
        global_model=global_model,
        experts=experts,
        expert_counts=expert_counts,
        build_seconds=time.perf_counter() - started,
        storage_bytes=storage,
    )


def prediction_metrics(
    name: str,
    dataset: str,
    truth_q: FloatArray,
    predictions: FloatArray,
    q_scale: FloatArray,
    storage_bytes: int,
    build_seconds: float,
) -> dict:
    valid = np.all(np.isfinite(predictions), axis=1)
    error = np.full(len(truth_q), np.nan, dtype=float)
    if np.any(valid):
        error[valid] = np.linalg.norm((predictions[valid] - truth_q[valid]) / q_scale, axis=1)
    finite = error[np.isfinite(error)]
    return {
        "model": name,
        "dataset": dataset,
        "rows": int(len(truth_q)),
        "prediction_valid_rate": float(np.mean(valid)),
        "median_q_error": float(np.median(finite)) if len(finite) else math.inf,
        "p95_q_error": float(np.quantile(finite, 0.95)) if len(finite) else math.inf,
        "rmse_q_error": float(np.sqrt(np.mean(finite * finite))) if len(finite) else math.inf,
        "storage_bytes": int(storage_bytes),
        "build_seconds": float(build_seconds),
    }


def model_predictions(
    model_name: str,
    x: FloatArray,
    branch: IntArray,
    *,
    train_tree: cKDTree,
    q_train: FloatArray,
    chart: Optional[ChartBank] = None,
    mixture: Optional[TreeMixture] = None,
) -> FloatArray:
    output = np.full((len(x), 5), np.nan, dtype=float)
    if model_name == "nearest":
        _distance, index = train_tree.query(x, k=1)
        return q_train[np.asarray(index, dtype=int)]
    if model_name in {"affine", "quadratic"}:
        assert chart is not None
        for i, point in enumerate(x):
            winding = decode_branch(int(branch[i]))[0]
            candidates = chart.predict_candidates(point, winding=winding, candidate_count=1)
            if candidates:
                output[i] = candidates[0]
        return output
    if model_name == "mixture":
        assert mixture is not None
        for i, point in enumerate(x):
            winding = decode_branch(int(branch[i]))[0]
            candidates = mixture.predict_candidates(point, winding=winding, candidate_count=1)
            if candidates:
                output[i] = candidates[0]
        return output
    raise ValueError(model_name)


def exact_benchmark(
    solver,
    model_name: str,
    dataset: str,
    targets: FloatArray,
    branches: IntArray,
    predictor: Callable[[FloatArray, int, int], list[FloatArray]],
    *,
    query_config,
    bounds: tuple[FloatArray, FloatArray],
    candidates_per_query: int,
) -> ExactMetric:
    successes = 0
    elapsed_values: list[float] = []
    evaluation_values: list[int] = []
    lo, hi = bounds
    for row, (target, code) in enumerate(zip(targets, branches)):
        started = time.perf_counter()
        deadline = time.monotonic() + max(float(query_config.wall_time_seconds), 1.0e-3)
        winding = decode_branch(int(code))[0]
        candidates = predictor(target, winding, candidates_per_query)
        total_evaluations = 0
        success = False
        for q in candidates[:candidates_per_query]:
            if time.monotonic() >= deadline:
                break
            corrected, nfev, _reason = solver._correct_one_seed(
                np.asarray(q, dtype=float),
                lo,
                hi,
                float(target[0]),
                float(target[1]),
                np.asarray(target[2:7], dtype=float),
                query_config,
                deadline=deadline,
            )
            total_evaluations += int(nfev)
            if corrected is not None:
                success = True
                break
        elapsed = time.perf_counter() - started
        successes += int(success)
        elapsed_values.append(elapsed)
        evaluation_values.append(total_evaluations)
        if (row + 1) % max(1, len(targets) // 5) == 0:
            status(
                f"      {model_name}/{dataset}: {row + 1:,}/{len(targets):,}, "
                f"success={successes / (row + 1):.1%}"
            )
    elapsed_array = np.asarray(elapsed_values, dtype=float)
    evaluation_array = np.asarray(evaluation_values, dtype=float)
    return ExactMetric(
        model=model_name,
        dataset=dataset,
        tested=int(len(targets)),
        successes=int(successes),
        success_rate=float(successes / max(len(targets), 1)),
        median_seconds=float(np.median(elapsed_array)) if len(elapsed_array) else math.nan,
        p95_seconds=float(np.quantile(elapsed_array, 0.95)) if len(elapsed_array) else math.nan,
        median_evaluations=float(np.median(evaluation_array)) if len(evaluation_array) else math.nan,
        p95_evaluations=float(np.quantile(evaluation_array, 0.95)) if len(evaluation_array) else math.nan,
        timeout_or_failure_rate=float(1.0 - successes / max(len(targets), 1)),
    )


def circular_targets(metadata: dict, rows: int, seed: int, max_turns: int) -> tuple[FloatArray, IntArray]:
    config = metadata.get("config", {})
    rho_bounds = config.get("rho_bounds", [0.01, 100.0])
    theta_bounds = config.get("theta_bounds", [-2.0 * math.pi, 2.0 * math.pi])
    kappa_bounds = config.get("kappa_bounds", [0.1, 2.0e5])
    engine = qmc.Sobol(d=3, scramble=True, seed=seed)
    sample = engine.random(rows)
    log_rho = math.log(rho_bounds[0]) + sample[:, 0] * math.log(rho_bounds[1] / rho_bounds[0])
    theta = theta_bounds[0] + sample[:, 1] * (theta_bounds[1] - theta_bounds[0])
    log_kappa = math.log(kappa_bounds[0]) + sample[:, 2] * math.log(kappa_bounds[1] / kappa_bounds[0])
    rho = np.exp(log_rho)
    endpoint = np.column_stack(
        (
            np.zeros(rows),
            np.ones(rows),
            log_rho,
            theta,
            np.zeros(rows),
            rho ** -0.5,
            log_kappa,
        )
    )
    winding = np.rint(theta / (2.0 * math.pi)).astype(np.int64)
    wind_code = np.where(winding >= 0, 2 * winding, -2 * winding - 1)
    branch = wind_code * (max_turns + 1)  # radial turns unknown, default zero
    return endpoint.astype(float), branch.astype(np.int64)


def anisotropy_report(chart: ChartBank) -> tuple[list[dict], FloatArray]:
    linear_norm = np.linalg.norm(chart.affine.astype(float), axis=1)
    if chart.quadratic is None:
        quadratic_diag = np.full((len(chart.centers_x), 7), np.nan)
        cross_matrix = np.full((7, 7), np.nan)
    else:
        diag_indices: list[int] = []
        pair_indices: dict[tuple[int, int], int] = {}
        p = 0
        for i in range(7):
            for j in range(i, 7):
                pair_indices[(i, j)] = p
                if i == j:
                    diag_indices.append(p)
                p += 1
        quadratic_diag = np.linalg.norm(chart.quadratic[:, :, diag_indices].astype(float), axis=1)
        cross_matrix = np.zeros((7, 7), dtype=float)
        for i in range(7):
            for j in range(i, 7):
                values = np.linalg.norm(chart.quadratic[:, :, pair_indices[(i, j)]].astype(float), axis=1)
                cross_matrix[i, j] = cross_matrix[j, i] = float(np.nanmedian(values))
    records: list[dict] = []
    for j, name in enumerate(ENDPOINT_NAMES):
        records.append(
            {
                "coordinate": name,
                "median_linear_sensitivity": float(np.nanmedian(linear_norm[:, j])),
                "p95_linear_sensitivity": float(np.nanquantile(linear_norm[:, j], 0.95)),
                "median_quadratic_curvature": float(np.nanmedian(quadratic_diag[:, j])),
                "p95_quadratic_curvature": float(np.nanquantile(quadratic_diag[:, j], 0.95)),
            }
        )
    return records, cross_matrix


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)



def write_markdown_report(
    path: Path,
    *,
    full_rows: int,
    sample_rows: int,
    anisotropy: list[dict],
    prediction_rows: list[dict],
    exact_rows: list[dict],
) -> None:
    ranked = sorted(
        anisotropy,
        key=lambda row: row["median_quadratic_curvature"]
        if math.isfinite(row["median_quadratic_curvature"])
        else -math.inf,
        reverse=True,
    )
    cell_rows = [row for row in prediction_rows if row["dataset"] == "cell_holdout"]
    best_prediction = min(cell_rows, key=lambda row: row["median_q_error"]) if cell_rows else None
    lines = [
        "# PMP atlas surrogate experiment",
        "",
        f"Atlas rows: **{full_rows:,}**  ",
        f"Experiment sample: **{sample_rows:,}**",
        "",
        "## Local anisotropy",
        "",
    ]
    if ranked:
        lines.append(
            "Highest median normalized quadratic curvature: "
            + ", ".join(f"`{row['coordinate']}`" for row in ranked[:3])
            + "."
        )
    lines.extend(["", "| Coordinate | Linear sensitivity | Quadratic curvature |", "|---|---:|---:|"])
    for row in anisotropy:
        lines.append(
            f"| {row['coordinate']} | {row['median_linear_sensitivity']:.4g} | "
            f"{row['median_quadratic_curvature']:.4g} |"
        )
    lines.extend(["", "## Interpolation across held-out endpoint cells", ""])
    if best_prediction is not None:
        lines.append(
            f"Lowest median normalized launch error: **{best_prediction['model']}** "
            f"({best_prediction['median_q_error']:.4g})."
        )
    lines.extend(["", "| Model | Median error | p95 error | Storage MiB |", "|---|---:|---:|---:|"])
    for row in sorted(cell_rows, key=lambda item: item["median_q_error"]):
        lines.append(
            f"| {row['model']} | {row['median_q_error']:.4g} | {row['p95_q_error']:.4g} | "
            f"{row['storage_bytes'] / 1024**2:.3g} |"
        )
    if exact_rows:
        lines.extend(["", "## Exact bounded-Newton benchmark", "", "| Model | Dataset | Success | Median s | p95 s |", "|---|---|---:|---:|---:|"])
        for row in exact_rows:
            lines.append(
                f"| {row['model']} | {row['dataset']} | {row['success_rate']:.1%} | "
                f"{row['median_seconds']:.4g} | {row['p95_seconds']:.4g} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation rules",
            "",
            "- Prefer cell-holdout and exact-query results over random-row prediction error.",
            "- A compact surrogate is useful only if exact convergence remains close to the point-atlas baseline.",
            "- Circular targets are validation-only; failure may combine poor coverage and target infeasibility.",
            "- The current atlas format does not include parent trajectory IDs, so parent-independent splitting is unavailable.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def plot_outputs(
    output_dir: Path,
    anisotropy: list[dict],
    prediction_rows: list[dict],
    exact_rows: list[dict],
) -> None:
    if plt is None:
        return
    names = [row["coordinate"] for row in anisotropy]
    curvature = [row["median_quadratic_curvature"] for row in anisotropy]
    sensitivity = [row["median_linear_sensitivity"] for row in anisotropy]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    positions = np.arange(len(names))
    width = 0.38
    ax.bar(positions - width / 2, sensitivity, width, label="linear sensitivity")
    ax.bar(positions + width / 2, curvature, width, label="quadratic curvature")
    ax.set_xticks(positions, names, rotation=30, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("median normalized coefficient norm")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "anisotropy.png", dpi=160)
    plt.close(fig)

    if exact_rows:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        grouped: dict[str, list[dict]] = {}
        for row in exact_rows:
            if row["dataset"] == "cell_holdout":
                grouped.setdefault(row["model"], []).append(row)
        storage = {
            row["model"]: row["storage_bytes"]
            for row in prediction_rows
            if row["dataset"] == "cell_holdout"
        }
        for model, rows in grouped.items():
            success = max(float(row["success_rate"]) for row in rows)
            size_mb = storage.get(model, 0) / (1024.0**2)
            ax.scatter([size_mb], [success], s=70)
            ax.annotate(model, (size_mb, success), xytext=(4, 4), textcoords="offset points")
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xlabel("surrogate storage (MiB)")
        ax.set_ylabel("exact fast-query success: cell holdout")
        ax.set_ylim(0.0, 1.02)
        fig.tight_layout()
        fig.savefig(output_dir / "compression_tradeoff.png", dpi=160)
        plt.close(fig)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("atlas", type=Path)
    p.add_argument("--solver", type=Path, default=Path("pmp_extremal_atlas.py"))
    p.add_argument("--output-dir", type=Path, default=Path("surrogate_experiment"))
    p.add_argument("--sample-rows", type=int, default=250_000)
    p.add_argument("--sample-candidate-factor", type=int, default=6)
    p.add_argument("--sample-cell-size", type=float, default=0.22)
    p.add_argument("--random-holdout-fraction", type=float, default=0.10)
    p.add_argument("--cell-holdout-fraction", type=float, default=0.15)
    p.add_argument("--holdout-cell-size", type=float, default=0.35)
    p.add_argument("--max-radial-turns", type=int, default=12)
    p.add_argument("--chart-counts", type=int, nargs="+", default=[500, 2_000, 8_000])
    p.add_argument("--chart-neighbours", type=int, default=96)
    p.add_argument("--chart-ridge", type=float, default=1.0e-5)
    p.add_argument("--trees", type=int, default=48)
    p.add_argument("--tree-max-depth", type=int, default=24)
    p.add_argument("--tree-min-leaf", type=int, default=2)
    p.add_argument("--max-experts", type=int, default=12)
    p.add_argument("--min-expert-rows", type=int, default=1_000)
    p.add_argument("--prediction-holdout-rows", type=int, default=20_000)
    p.add_argument("--exact-holdout-rows", type=int, default=200)
    p.add_argument("--circular-probe-rows", type=int, default=128)
    p.add_argument("--query-seconds", type=float, default=0.5)
    p.add_argument("--query-iterations", type=int, default=3)
    p.add_argument("--query-seeds", type=int, default=2)
    p.add_argument("--query-line-search", type=int, default=2)
    p.add_argument("--query-step-limit", type=float, default=0.25)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--seed", type=int, default=20260720)
    p.add_argument("--skip-mixture", action="store_true")
    p.add_argument("--skip-exact", action="store_true")
    p.add_argument("--skip-circular", action="store_true")
    p.add_argument("--save-models", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    solver_path = args.solver.resolve()
    if not solver_path.exists():
        sibling = Path(__file__).resolve().with_name(args.solver.name)
        if sibling.exists():
            solver_path = sibling
        else:
            raise FileNotFoundError(args.solver)
    solver = load_solver(solver_path)
    make_branch_decoder(args.max_radial_turns)

    status("[1/6] Loading atlas arrays and selecting a spatially diverse sample")
    with np.load(args.atlas, allow_pickle=False) as data:
        launch_all = np.asarray(data["launch"], dtype=float)
        endpoint_all = np.asarray(data["endpoint"], dtype=float)
        diagnostics_all = np.asarray(data["diagnostics"], dtype=float)
        feature_scale = np.asarray(data["feature_scale"], dtype=float)
        metadata = json.loads(str(data["metadata_json"].item()))
        atlas_format = int(metadata.get("format_version", 0))
        full_rows = len(launch_all)
        branch_all = branch_codes(endpoint_all, diagnostics_all, args.max_radial_turns)
        selected = diversified_sample_indices(
            endpoint_all,
            branch_all,
            feature_scale,
            args.sample_rows,
            rng,
            args.sample_candidate_factor,
            args.sample_cell_size,
        )
        launch = launch_all[selected]
        endpoint = endpoint_all[selected]
        diagnostics = diagnostics_all[selected]
        branch = branch_all[selected]
        del launch_all, endpoint_all, diagnostics_all, branch_all

    x = endpoint / feature_scale
    q = q_from_launch(launch)
    generation_config = solver._atlas_config_from_dict(metadata.get("config", {}))
    q_scale = np.asarray(solver._launch_q_scale(generation_config), dtype=float)
    split = split_data(
        x,
        branch,
        rng,
        args.random_holdout_fraction,
        args.cell_holdout_fraction,
        args.holdout_cell_size,
    )
    status(
        f"    atlas rows={full_rows:,}; sample={len(x):,}; train={len(split.train):,}; "
        f"random holdout={len(split.random_holdout):,}; cell holdout={len(split.cell_holdout):,}"
    )
    x_train = np.asarray(x[split.train], dtype=np.float32)
    q_train = np.asarray(q[split.train], dtype=np.float32)
    branch_train = branch[split.train]
    train_tree = cKDTree(x_train)

    status("[2/6] Fitting local affine and quadratic inverse charts")
    chart_models: dict[tuple[str, int], ChartBank] = {}
    for count in sorted(set(args.chart_counts)):
        status(f"  chart count {count:,}: affine")
        affine = fit_chart_bank(
            x_train,
            q_train,
            branch_train,
            q_scale,
            chart_count=count,
            neighbours=args.chart_neighbours,
            ridge=args.chart_ridge,
            quadratic=False,
            rng_seed=args.seed + count,
            workers=args.workers,
        )
        chart_models[("affine", count)] = affine
        status(f"  chart count {count:,}: quadratic")
        quadratic = fit_chart_bank(
            x_train,
            q_train,
            branch_train,
            q_scale,
            chart_count=count,
            neighbours=args.chart_neighbours,
            ridge=args.chart_ridge,
            quadratic=True,
            rng_seed=args.seed + count,
            workers=args.workers,
        )
        chart_models[("quadratic", count)] = quadratic

    status("[3/6] Fitting the branch-conditioned global regression benchmark")
    mixture: Optional[TreeMixture] = None
    if not args.skip_mixture:
        mixture = fit_tree_mixture(
            x_train,
            q_train,
            branch_train,
            trees=args.trees,
            max_depth=None if args.tree_max_depth <= 0 else args.tree_max_depth,
            min_leaf=args.tree_min_leaf,
            max_experts=args.max_experts,
            min_expert_rows=args.min_expert_rows,
            workers=args.workers,
            seed=args.seed,
        )
        status(
            f"    global tree plus {len(mixture.experts)} experts; "
            f"serialized size={mixture.storage_bytes / 1024**2:.1f} MiB"
        )
    else:
        status("    skipped by command line")

    status("[4/6] Measuring anisotropy and interpolation across held-out cells")
    largest_count = max(args.chart_counts)
    anisotropy, cross_curvature = anisotropy_report(chart_models[("quadratic", largest_count)])
    prediction_rows: list[dict] = []
    dataset_indices = {
        "random_holdout": split.random_holdout,
        "cell_holdout": split.cell_holdout,
    }
    for dataset, indices in dataset_indices.items():
        if len(indices) > args.prediction_holdout_rows:
            indices = rng.choice(indices, args.prediction_holdout_rows, replace=False)
        x_test = x[indices]
        q_test = q[indices]
        branch_test = branch[indices]
        nearest_prediction = model_predictions(
            "nearest", x_test, branch_test, train_tree=train_tree, q_train=q_train
        )
        prediction_rows.append(
            prediction_metrics(
                "nearest",
                dataset,
                q_test,
                nearest_prediction,
                q_scale,
                int(x_train.nbytes + q_train.nbytes),
                0.0,
            )
        )
        for count in sorted(set(args.chart_counts)):
            for kind in ("affine", "quadratic"):
                chart = chart_models[(kind, count)]
                prediction = model_predictions(
                    kind,
                    x_test,
                    branch_test,
                    train_tree=train_tree,
                    q_train=q_train,
                    chart=chart,
                )
                prediction_rows.append(
                    prediction_metrics(
                        f"{kind}_{count}",
                        dataset,
                        q_test,
                        prediction,
                        q_scale,
                        chart.storage_bytes,
                        chart.build_seconds,
                    )
                )
        if mixture is not None:
            prediction = model_predictions(
                "mixture",
                x_test,
                branch_test,
                train_tree=train_tree,
                q_train=q_train,
                mixture=mixture,
            )
            prediction_rows.append(
                prediction_metrics(
                    "mixture",
                    dataset,
                    q_test,
                    prediction,
                    q_scale,
                    mixture.storage_bytes,
                    mixture.build_seconds,
                )
            )

    exact_metrics: list[ExactMetric] = []
    if not args.skip_exact:
        status("[5/6] Testing exact bounded-Newton convergence from surrogate predictions")
        query_config = solver.QueryConfig(
            method="fast_newton",
            neighbours=24,
            direct_seeds=args.query_seeds,
            regression_neighbours=20,
            max_iterations=args.query_iterations,
            max_seed_attempts=args.query_seeds,
            wall_time_seconds=args.query_seconds,
            line_search_steps=args.query_line_search,
            step_limit=args.query_step_limit,
            regularization=1.0e-8,
            robust_fallback=False,
            allow_continuation=False,
            rtol=8.0e-10,
            atol=8.0e-12,
            max_step=float(metadata.get("config", {}).get("max_step", 0.04)),
            trajectory_points=8,
            r_collision=float(metadata.get("config", {}).get("r_collision", 0.01)),
            r_escape=float(metadata.get("config", {}).get("r_escape", 100.0)),
            max_acceleration=float(metadata.get("config", {}).get("max_acceleration", 300.0)),
        )
        bounds = solver._configured_correction_bounds(generation_config)

        best_affine_count = min(
            args.chart_counts,
            key=lambda count: next(
                row["median_q_error"]
                for row in prediction_rows
                if row["model"] == f"affine_{count}" and row["dataset"] == "cell_holdout"
            ),
        )
        best_quadratic_count = min(
            args.chart_counts,
            key=lambda count: next(
                row["median_q_error"]
                for row in prediction_rows
                if row["model"] == f"quadratic_{count}" and row["dataset"] == "cell_holdout"
            ),
        )
        exact_models: list[tuple[str, Callable[[FloatArray, int, int], list[FloatArray]]]] = []

        def nearest_predictor(target: FloatArray, winding: int, count: int) -> list[FloatArray]:
            _distance, indices = train_tree.query(target / feature_scale, k=min(count, len(x_train)))
            return [np.asarray(q_train[i], dtype=float) for i in np.atleast_1d(indices).astype(int)]

        exact_models.append(("nearest", nearest_predictor))
        affine_bank = chart_models[("affine", best_affine_count)]
        quadratic_bank = chart_models[("quadratic", best_quadratic_count)]
        exact_models.append(
            (
                f"affine_{best_affine_count}",
                lambda target, winding, count: affine_bank.predict_candidates(
                    target / feature_scale,
                    winding=winding,
                    candidate_count=count,
                ),
            )
        )
        exact_models.append(
            (
                f"quadratic_{best_quadratic_count}",
                lambda target, winding, count: quadratic_bank.predict_candidates(
                    target / feature_scale,
                    winding=winding,
                    candidate_count=count,
                ),
            )
        )
        if mixture is not None:
            exact_models.append(
                (
                    "mixture",
                    lambda target, winding, count: mixture.predict_candidates(
                        target / feature_scale,
                        winding=winding,
                        candidate_count=count,
                    ),
                )
            )

        for dataset, indices in dataset_indices.items():
            exact_n = min(args.exact_holdout_rows, len(indices))
            exact_indices = rng.choice(indices, exact_n, replace=False)
            exact_targets = endpoint[exact_indices]
            exact_branch = branch[exact_indices]
            for model_name, predictor in exact_models:
                status(f"    exact benchmark: {model_name}, {dataset}")
                exact_metrics.append(
                    exact_benchmark(
                        solver,
                        model_name,
                        dataset,
                        exact_targets,
                        exact_branch,
                        predictor,
                        query_config=query_config,
                        bounds=bounds,
                        candidates_per_query=args.query_seeds,
                    )
                )

        if not args.skip_circular and args.circular_probe_rows > 0:
            probe_targets, probe_branch = circular_targets(
                metadata, args.circular_probe_rows, args.seed + 77_777, args.max_radial_turns
            )
            for model_name, predictor in exact_models:
                status(f"    exact benchmark: {model_name}, circular probe")
                exact_metrics.append(
                    exact_benchmark(
                        solver,
                        model_name,
                        "circular_probe",
                        probe_targets,
                        probe_branch,
                        predictor,
                        query_config=query_config,
                        bounds=bounds,
                        candidates_per_query=args.query_seeds,
                    )
                )
    else:
        status("[5/6] Exact shooting benchmark skipped")

    status("[6/6] Writing reports")
    exact_rows = [asdict(metric) for metric in exact_metrics]
    write_csv(args.output_dir / "anisotropy.csv", anisotropy)
    write_csv(args.output_dir / "prediction_metrics.csv", prediction_rows)
    write_csv(args.output_dir / "exact_query_metrics.csv", exact_rows)
    np.savetxt(
        args.output_dir / "cross_curvature.csv",
        cross_curvature,
        delimiter=",",
        header=",".join(ENDPOINT_NAMES),
        comments="",
    )

    branch_counts = []
    codes, counts = np.unique(branch_train, return_counts=True)
    for code, count in sorted(zip(codes, counts), key=lambda item: item[1], reverse=True):
        winding, turns = decode_branch(int(code))
        branch_counts.append(
            {
                "branch_code": int(code),
                "winding": winding,
                "radial_turns_capped": turns,
                "training_rows": int(count),
            }
        )
    write_csv(args.output_dir / "branch_counts.csv", branch_counts)

    summary = {
        "atlas": str(args.atlas.resolve()),
        "atlas_format": atlas_format,
        "full_atlas_rows": full_rows,
        "sample_rows": len(x),
        "train_rows": len(split.train),
        "random_holdout_rows": len(split.random_holdout),
        "cell_holdout_rows": len(split.cell_holdout),
        "feature_scale": feature_scale,
        "q_scale": q_scale,
        "arguments": vars(args),
        "anisotropy": anisotropy,
        "prediction_metrics": prediction_rows,
        "exact_query_metrics": exact_rows,
        "notes": [
            "The NPZ schema does not store parent trajectory identifiers; exact parent-independent splitting is therefore unavailable.",
            "Cell holdout excludes complete normalized endpoint cells and is the primary interpolation test.",
            "Circular targets are validation-only and are not inserted into any model or chart bank.",
            "Exact success means convergence under the configured bounded fast-Newton budget.",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, cls=JsonEncoder), encoding="utf-8"
    )
    write_markdown_report(
        args.output_dir / "report.md",
        full_rows=full_rows,
        sample_rows=len(x),
        anisotropy=anisotropy,
        prediction_rows=prediction_rows,
        exact_rows=exact_rows,
    )
    plot_outputs(args.output_dir, anisotropy, prediction_rows, exact_rows)

    if args.save_models:
        with (args.output_dir / "chart_models.pkl").open("wb") as handle:
            pickle.dump(chart_models, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if mixture is not None:
            with (args.output_dir / "tree_mixture.pkl").open("wb") as handle:
                pickle.dump(mixture, handle, protocol=pickle.HIGHEST_PROTOCOL)

    status(f"Experiment complete: {args.output_dir.resolve()}")
    if anisotropy:
        ranked = sorted(
            anisotropy,
            key=lambda row: row["median_quadratic_curvature"]
            if math.isfinite(row["median_quadratic_curvature"])
            else -math.inf,
            reverse=True,
        )
        status(
            "Most curved normalized endpoint directions: "
            + ", ".join(row["coordinate"] for row in ranked[:3])
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
