"""Phase-2 generator for the trajectory atlas.

Implements roadmap Step 2:
- 3D grid construction in (rho, kappa_tilde, theta)
- anchor-thread generation along theta
- wavefront propagation across rho/kappa with predictor/fallback seeds
- infeasible/failed status bookkeeping
- compressed atlas persistence
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from rocketHamilton import (
    AU,
    DEFAULT_CONFIG,
    MU_SI,
    SOLUTION0,
    integrate_fixed_time,
    solve_target_fast,
)
from trajectory_scaling import solve_power_for_kappa

ATLAS_VECTOR_SIZE = 6  # [lam_r0, lam_vr0, lam_vtheta0, C_m, C_theta, t_f_days]


@dataclass(frozen=True)
class AtlasGridSpec:
    rho_min: float = 0.05
    rho_max: float = 100.0
    rho_points: int = 25
    kappa_min: float = 1.0e-1
    kappa_max: float = 1.0e3
    kappa_points: int = 10
    theta_min: float = 0.0
    theta_max: float = float(8.0 * np.pi)
    theta_points: int = 25


@dataclass(frozen=True)
class AtlasMeta:
    version: str
    anchor_i: int
    anchor_j: int
    anchor_completed: bool
    wavefront_completed: bool
    config_power_w: float
    config_m0_kg: float
    config_m_dry_kg: float
    config_mu_si: float
    notes: str


class AtlasStatus:
    EMPTY = 0
    SOLVED = 1
    INFEASIBLE = 2
    FAILED = 3


class SeedSource:
    NONE = 0
    RHO_NEIGHBOR = 1
    KAPPA_NEIGHBOR = 2
    OTHER_PHYSICAL = 3
    THETA_FALLBACK = 4


def build_axes(spec: AtlasGridSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build atlas axes: rho/log, kappa/log, theta/linear."""
    if spec.rho_min <= 0 or spec.kappa_min <= 0:
        raise ValueError("rho_min and kappa_min must be positive")
    rho_axis = np.logspace(np.log10(spec.rho_min), np.log10(spec.rho_max), spec.rho_points)
    kappa_axis = np.logspace(np.log10(spec.kappa_min), np.log10(spec.kappa_max), spec.kappa_points)
    theta_axis = np.linspace(spec.theta_min, spec.theta_max, spec.theta_points)
    return rho_axis, kappa_axis, theta_axis


def init_atlas_tensor(spec: AtlasGridSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initialize atlas values and diagnostic tensors."""
    values = np.full((spec.rho_points, spec.kappa_points, spec.theta_points, ATLAS_VECTOR_SIZE), np.nan)
    status = np.full((spec.rho_points, spec.kappa_points, spec.theta_points), AtlasStatus.EMPTY, dtype=np.int8)
    seed_source = np.full((spec.rho_points, spec.kappa_points, spec.theta_points), SeedSource.NONE, dtype=np.int8)
    residual_norm = np.full((spec.rho_points, spec.kappa_points, spec.theta_points), np.nan)
    return values, status, seed_source, residual_norm


def select_anchor_indices(spec: AtlasGridSpec) -> tuple[int, int]:
    """Select center anchor index in rho and kappa dimensions."""
    return spec.rho_points // 2, spec.kappa_points // 2


def seed_from_solution0() -> np.ndarray:
    return np.asarray(SOLUTION0, dtype=float).reshape(-1)


def _is_physically_feasible(params: np.ndarray, t_days: float, config) -> bool:
    """Feasibility check: final mass must stay above dry mass."""
    sol = integrate_fixed_time(params, t_days, config=config)
    return bool(sol.y[4, -1] >= config.m_dry)


def _config_for_kappa(kappa_target: float):
    """Construct local trajectory config whose capability matches kappa_target at 1 AU."""
    power = solve_power_for_kappa(
        target_kappa=float(kappa_target),
        m_dry=DEFAULT_CONFIG.m_dry,
        m0=DEFAULT_CONFIG.m0,
        r0=AU,
        mu=MU_SI,
    )
    return replace(DEFAULT_CONFIG, power=power)


def _solve_cell(
    rho_target: float,
    theta_target: float,
    kappa_target: float,
    seed_params: np.ndarray,
    t_guess_days: float,
) -> tuple[int, np.ndarray | None, float | None, float | None]:
    """Attempt solving one cell and return status + solution payload."""
    local_config = _config_for_kappa(kappa_target)
    params, t_days, info = solve_target_fast(
        r_target=float(rho_target),
        theta_target=float(theta_target),
        seed_params=np.asarray(seed_params, dtype=float),
        t_guess_days=float(t_guess_days),
        config=local_config,
    )
    vec = np.concatenate([np.asarray(params, dtype=float), [float(t_days)]])
    if vec.shape[0] != ATLAS_VECTOR_SIZE:
        raise ValueError("atlas vector shape mismatch")

    res_norm = float(np.linalg.norm(getattr(info, "fun", np.zeros(1))))
    feasible = _is_physically_feasible(np.asarray(params, dtype=float), float(t_days), config=local_config)
    status = AtlasStatus.SOLVED if feasible else AtlasStatus.INFEASIBLE
    return status, vec, float(t_days), res_norm


def _try_seed_candidate(
    values: np.ndarray,
    status: np.ndarray,
    i: int,
    j: int,
    k: int,
    candidate_i: int,
    candidate_j: int,
    source: int,
) -> tuple[np.ndarray, float, int] | None:
    if candidate_i < 0 or candidate_i >= status.shape[0]:
        return None
    if candidate_j < 0 or candidate_j >= status.shape[1]:
        return None
    if status[candidate_i, candidate_j, k] != AtlasStatus.SOLVED:
        return None
    vec = values[candidate_i, candidate_j, k, :]
    return np.asarray(vec[:5], dtype=float), float(vec[5]), source


def _select_predictor_seed(
    values: np.ndarray,
    status: np.ndarray,
    i: int,
    j: int,
    k: int,
    anchor_i: int,
    anchor_j: int,
) -> tuple[np.ndarray, float, int] | None:
    """Pick predictor seed, preferring rho/kappa neighbors as described in roadmap."""
    candidates: list[tuple[np.ndarray, float, int] | None] = []

    rho_neighbor = i - 1 if i > anchor_i else i + 1 if i < anchor_i else None
    if rho_neighbor is not None:
        candidates.append(_try_seed_candidate(values, status, i, j, k, rho_neighbor, j, SeedSource.RHO_NEIGHBOR))

    kappa_neighbor = j - 1 if j > anchor_j else j + 1 if j < anchor_j else None
    if kappa_neighbor is not None:
        candidates.append(_try_seed_candidate(values, status, i, j, k, i, kappa_neighbor, SeedSource.KAPPA_NEIGHBOR))

    candidates.append(_try_seed_candidate(values, status, i, j, k, i - 1, j, SeedSource.OTHER_PHYSICAL))
    candidates.append(_try_seed_candidate(values, status, i, j, k, i + 1, j, SeedSource.OTHER_PHYSICAL))
    candidates.append(_try_seed_candidate(values, status, i, j, k, i, j - 1, SeedSource.OTHER_PHYSICAL))
    candidates.append(_try_seed_candidate(values, status, i, j, k, i, j + 1, SeedSource.OTHER_PHYSICAL))

    for c in candidates:
        if c is not None:
            return c

    if k > 0 and status[i, j, k - 1] == AtlasStatus.SOLVED:
        vec = values[i, j, k - 1, :]
        return np.asarray(vec[:5], dtype=float), float(vec[5]), SeedSource.THETA_FALLBACK

    return None


def run_anchor_thread(
    values: np.ndarray,
    status: np.ndarray,
    seed_source: np.ndarray,
    residual_norm: np.ndarray,
    rho_axis: np.ndarray,
    kappa_axis: np.ndarray,
    theta_axis: np.ndarray,
    anchor_i: int,
    anchor_j: int,
    initial_seed: np.ndarray | None = None,
    t_guess_days: float = 500.0,
) -> None:
    """Populate the anchor column along theta for fixed (rho, kappa)."""
    rho_target = float(rho_axis[anchor_i])
    kappa_target = float(kappa_axis[anchor_j])
    seed = seed_from_solution0() if initial_seed is None else np.asarray(initial_seed, dtype=float)
    current_params = seed.copy()
    current_t = float(t_guess_days)

    for k, theta_target in enumerate(theta_axis):
        try:
            s, vec, new_t, res = _solve_cell(rho_target, float(theta_target), kappa_target, current_params, current_t)
            values[anchor_i, anchor_j, k, :] = vec
            status[anchor_i, anchor_j, k] = s
            seed_source[anchor_i, anchor_j, k] = SeedSource.THETA_FALLBACK if k > 0 else SeedSource.NONE
            residual_norm[anchor_i, anchor_j, k] = res
            if s == AtlasStatus.SOLVED:
                current_params = np.asarray(vec[:5], dtype=float)
                current_t = float(new_t)
        except Exception:
            status[anchor_i, anchor_j, k] = AtlasStatus.FAILED


def _wavefront_pairs(shape_i: int, shape_j: int, anchor_i: int, anchor_j: int):
    max_d = max(anchor_i, shape_i - 1 - anchor_i) + max(anchor_j, shape_j - 1 - anchor_j)
    for d in range(max_d + 1):
        for i in range(shape_i):
            for j in range(shape_j):
                if abs(i - anchor_i) + abs(j - anchor_j) == d:
                    yield i, j


def run_wavefront_propagation(
    values: np.ndarray,
    status: np.ndarray,
    seed_source: np.ndarray,
    residual_norm: np.ndarray,
    rho_axis: np.ndarray,
    kappa_axis: np.ndarray,
    theta_axis: np.ndarray,
    anchor_i: int,
    anchor_j: int,
) -> None:
    """Fill all grid cells with 3D homotopy wavefront propagation."""
    ni, nj, nk = status.shape

    for i, j in _wavefront_pairs(ni, nj, anchor_i, anchor_j):
        if i == anchor_i and j == anchor_j:
            continue
        for k in range(nk):
            if status[i, j, k] != AtlasStatus.EMPTY:
                continue

            seed_payload = _select_predictor_seed(values, status, i, j, k, anchor_i, anchor_j)
            if seed_payload is None:
                status[i, j, k] = AtlasStatus.FAILED
                continue

            seed_params, t_guess, source = seed_payload
            try:
                s, vec, _, res = _solve_cell(
                    rho_target=float(rho_axis[i]),
                    theta_target=float(theta_axis[k]),
                    kappa_target=float(kappa_axis[j]),
                    seed_params=seed_params,
                    t_guess_days=t_guess,
                )
                values[i, j, k, :] = vec
                status[i, j, k] = s
                seed_source[i, j, k] = source
                residual_norm[i, j, k] = res
            except Exception:
                # explicit roadmap fallback: regrow from angular neighbor if available
                if k > 0 and status[i, j, k - 1] == AtlasStatus.SOLVED:
                    try:
                        fallback = values[i, j, k - 1, :]
                        s, vec, _, res = _solve_cell(
                            rho_target=float(rho_axis[i]),
                            theta_target=float(theta_axis[k]),
                            kappa_target=float(kappa_axis[j]),
                            seed_params=np.asarray(fallback[:5], dtype=float),
                            t_guess_days=float(fallback[5]),
                        )
                        values[i, j, k, :] = vec
                        status[i, j, k] = s
                        seed_source[i, j, k] = SeedSource.THETA_FALLBACK
                        residual_norm[i, j, k] = res
                    except Exception:
                        status[i, j, k] = AtlasStatus.FAILED
                else:
                    status[i, j, k] = AtlasStatus.FAILED


def save_atlas(
    out_path: Path,
    spec: AtlasGridSpec,
    rho_axis: np.ndarray,
    kappa_axis: np.ndarray,
    theta_axis: np.ndarray,
    values: np.ndarray,
    status: np.ndarray,
    seed_source: np.ndarray,
    residual_norm: np.ndarray,
    meta: AtlasMeta,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        rho_axis=rho_axis,
        kappa_axis=kappa_axis,
        theta_axis=theta_axis,
        values=values,
        status=status,
        seed_source=seed_source,
        residual_norm=residual_norm,
        spec_json=json.dumps(asdict(spec)),
        meta_json=json.dumps(asdict(meta)),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate trajectory atlas")
    p.add_argument("--out", type=Path, default=Path("artifacts/trajectory_atlas.npz"))
    p.add_argument("--rho-points", type=int, default=25)
    p.add_argument("--kappa-points", type=int, default=10)
    p.add_argument("--theta-points", type=int, default=25)
    p.add_argument("--skip-anchor", action="store_true", help="Skip solving anchor theta-thread")
    p.add_argument("--run-wavefront", action="store_true", help="Run full wavefront propagation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spec = AtlasGridSpec(
        rho_points=args.rho_points,
        kappa_points=args.kappa_points,
        theta_points=args.theta_points,
    )
    rho_axis, kappa_axis, theta_axis = build_axes(spec)
    values, status, seed_source, residual_norm = init_atlas_tensor(spec)
    anchor_i, anchor_j = select_anchor_indices(spec)

    if not args.skip_anchor:
        run_anchor_thread(values, status, seed_source, residual_norm, rho_axis, kappa_axis, theta_axis, anchor_i, anchor_j)

    if args.run_wavefront:
        run_wavefront_propagation(
            values,
            status,
            seed_source,
            residual_norm,
            rho_axis,
            kappa_axis,
            theta_axis,
            anchor_i,
            anchor_j,
        )

    meta = AtlasMeta(
        version="phase2-complete-v1",
        anchor_i=anchor_i,
        anchor_j=anchor_j,
        anchor_completed=bool(np.any(status[anchor_i, anchor_j, :] != AtlasStatus.EMPTY)),
        wavefront_completed=bool(np.all(status != AtlasStatus.EMPTY)) if args.run_wavefront else False,
        config_power_w=DEFAULT_CONFIG.power,
        config_m0_kg=DEFAULT_CONFIG.m0,
        config_m_dry_kg=DEFAULT_CONFIG.m_dry,
        config_mu_si=DEFAULT_CONFIG.mu,
        notes="Wavefront implemented with predictor/fallback seed selection.",
    )
    save_atlas(args.out, spec, rho_axis, kappa_axis, theta_axis, values, status, seed_source, residual_norm, meta)
    print(f"Wrote atlas: {args.out}")


if __name__ == "__main__":
    main()
