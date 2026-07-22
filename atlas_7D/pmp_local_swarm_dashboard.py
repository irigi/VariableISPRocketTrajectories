#!/usr/bin/env python3
"""Visualize and tune analytical linearizations of the PMP endpoint map.

The script perturbs a chosen launch on a normalized 7D sphere (or ball),
integrates the swarm, applies a selectable analytical transformation to the
final endpoint coordinates, and renders all 21 pairwise projections in 4x3
dashboards.

It produces three complementary views:

1. Raw endpoint coordinates.
2. Analytically transformed endpoint coordinates.
3. Local affine pullback of the transformed outputs into normalized launch
   perturbation coordinates.  If the analytical transformation has removed
   most nonlinearity, this third view should resemble the original nested
   sphere/ball closely.

Raw endpoint coordinates:
    (u0, w0, log_rho, theta_unwrapped, ur_final, ut_final, log_kappa)

Launch perturbation coordinates:
    (u0, w0, Ar0, At0, Jr0, ell, log_tau)

The USER-TUNABLE TRANSFORM section below is deliberately isolated so the
analytical transformation can be revised without touching the integration,
plotting, or diagnostics machinery.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


RAW_ENDPOINT_LABELS = [
    "u0",
    "w0",
    "log_rho",
    "theta_unwrapped",
    "ur_final",
    "ut_final",
    "log_kappa",
]
LAUNCH_LABELS = [
    "u0",
    "w0",
    "Ar0",
    "At0",
    "Jr0",
    "ell",
    "log_tau",
]
DEFAULT_INTEGRATION = dict(
    rtol=2e-9,
    atol=2e-11,
    max_step=0.04,
    r_collision=0.005,
    r_escape=200.0,
    max_acceleration=5000.0,
)
DEFAULT_BOUNDS = {
    "u0_bounds": (-1.5, 1.5),
    "w0_bounds": (-0.5, 2.0),
    "ar0_bounds": (-32.0, 32.0),
    "at0_bounds": (-32.0, 32.0),
    "jr0_bounds": (-128.0, 128.0),
    "ell_bounds": (-128.0, 128.0),
    "tau_bounds": (0.02, 250.0),
}


@dataclass
class IntegrationConfig:
    rtol: float
    atol: float
    max_step: float
    r_collision: float
    r_escape: float
    max_acceleration: float


# ===========================================================================
# USER-TUNABLE ANALYTICAL OUTPUT TRANSFORM
# ===========================================================================

def custom_analytical_transform(
    raw_endpoints: np.ndarray,
    center_raw_endpoint: np.ndarray,
    params: dict[str, float],
) -> tuple[np.ndarray, list[str]]:
    """User-editable analytical transformation from raw endpoint to 7D output.

    Parameters
    ----------
    raw_endpoints:
        Array of shape (N, 7) with columns
        (u0, w0, log_rho, theta, ur, ut, log_kappa).
    center_raw_endpoint:
        Raw endpoint of the unperturbed center launch.
    params:
        Numeric values supplied through ``--transform-param NAME=VALUE``.

    Returns
    -------
    transformed, labels
        ``transformed`` must have shape (N, 7), and ``labels`` must contain
        seven plotting labels.

    Current editable candidate
    --------------------------
    The default custom map uses final Cartesian position/velocity in a frame
    rotated by the center trajectory's final angle.  This removes the polar
    radius/angle coupling while retaining a locally fixed orientation.  The
    discrete winding number can later remain separate atlas branch metadata.

    Optional ``asinh_scale`` applies a smooth compressive nonlinearity to the
    four Cartesian state coordinates after centering and scaling.  Its default
    is zero, which disables that extra operation.
    """
    raw = np.asarray(raw_endpoints, dtype=float)
    center = np.asarray(center_raw_endpoint, dtype=float)

    rho = np.exp(raw[:, 2])
    theta = raw[:, 3]
    ur = raw[:, 4]
    ut = raw[:, 5]
    theta_ref = float(center[3])
    delta = theta - theta_ref
    c = np.cos(delta)
    s = np.sin(delta)

    # Cartesian final state in the nominal final-angle frame.
    x = rho * c
    y = rho * s
    vx = ur * c - ut * s
    vy = ur * s + ut * c
    state = np.column_stack((x, y, vx, vy))

    asinh_scale = float(params.get("asinh_scale", 0.0))
    if asinh_scale > 0.0:
        # Centering keeps the nominal point at zero.  The same scalar scale is
        # intentionally simple; later experiments can replace it with four
        # physically selected scales or a dimensionless invariant map.
        rho0 = math.exp(float(center[2]))
        state0 = np.array([rho0, 0.0, float(center[4]), float(center[5])])
        state = np.arcsinh((state - state0[None, :]) / asinh_scale)
        state_labels = ["asinh_x", "asinh_y", "asinh_vx", "asinh_vy"]
    else:
        state_labels = ["x_rot", "y_rot", "vx_rot", "vy_rot"]

    transformed = np.column_stack(
        (raw[:, 0], raw[:, 1], state, raw[:, 6])
    )
    labels = ["u0", "w0", *state_labels, "log_kappa"]
    return transformed, labels


# ===========================================================================
# BUILT-IN TRANSFORMS FOR COMPARISON
# ===========================================================================

def _transform_raw(raw: np.ndarray, center: np.ndarray, params: dict[str, float]):
    return raw.copy(), list(RAW_ENDPOINT_LABELS)


def _transform_cartesian(raw: np.ndarray, center: np.ndarray, params: dict[str, float]):
    rho = np.exp(raw[:, 2])
    theta = raw[:, 3]
    ur = raw[:, 4]
    ut = raw[:, 5]
    c = np.cos(theta)
    s = np.sin(theta)
    x = rho * c
    y = rho * s
    vx = ur * c - ut * s
    vy = ur * s + ut * c
    return (
        np.column_stack((raw[:, 0], raw[:, 1], x, y, vx, vy, raw[:, 6])),
        ["u0", "w0", "x_final", "y_final", "vx_final", "vy_final", "log_kappa"],
    )


def _transform_cartesian_rotated(raw: np.ndarray, center: np.ndarray, params: dict[str, float]):
    rho = np.exp(raw[:, 2])
    theta = raw[:, 3]
    ur = raw[:, 4]
    ut = raw[:, 5]
    delta = theta - float(center[3])
    c = np.cos(delta)
    s = np.sin(delta)
    x = rho * c
    y = rho * s
    vx = ur * c - ut * s
    vy = ur * s + ut * c
    return (
        np.column_stack((raw[:, 0], raw[:, 1], x, y, vx, vy, raw[:, 6])),
        ["u0", "w0", "x_rot", "y_rot", "vx_rot", "vy_rot", "log_kappa"],
    )


def _transform_equinoctial(raw: np.ndarray, center: np.ndarray, params: dict[str, float]):
    """Planar modified-equinoctial-like state coordinates.

    With dimensionless mu=1, h=r*u_t, p=h^2, and the eccentricity vector is
        e = ((v^2 - 1/r) r_vec - (r dot v) v_vec).
    The discrete winding remains branch metadata; theta is kept locally
    unwrapped as the seventh continuous orbital-state coordinate before kappa.
    """
    rho = np.exp(raw[:, 2])
    theta = raw[:, 3]
    ur = raw[:, 4]
    ut = raw[:, 5]
    c = np.cos(theta)
    s = np.sin(theta)
    rx = rho * c
    ry = rho * s
    vx = ur * c - ut * s
    vy = ur * s + ut * c
    v2 = vx * vx + vy * vy
    rv = rx * vx + ry * vy
    ex = (v2 - 1.0 / rho) * rx - rv * vx
    ey = (v2 - 1.0 / rho) * ry - rv * vy
    h = rho * ut
    p = np.maximum(h * h, np.finfo(float).tiny)
    local_theta = theta - float(center[3])
    return (
        np.column_stack((raw[:, 0], raw[:, 1], np.log(p), ex, ey, local_theta, raw[:, 6])),
        ["u0", "w0", "log_p", "ecc_x", "ecc_y", "delta_theta", "log_kappa"],
    )


def apply_analytical_transform(
    raw_endpoints: np.ndarray,
    center_raw_endpoint: np.ndarray,
    mode: str,
    params: dict[str, float],
) -> tuple[np.ndarray, list[str]]:
    transforms = {
        "raw": _transform_raw,
        "cartesian": _transform_cartesian,
        "cartesian_rotated": _transform_cartesian_rotated,
        "equinoctial": _transform_equinoctial,
        "custom": custom_analytical_transform,
    }
    try:
        values, labels = transforms[mode](raw_endpoints, center_raw_endpoint, params)
    except KeyError as exc:
        raise ValueError(f"Unknown transform mode: {mode}") from exc
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError(f"Transform '{mode}' returned shape {values.shape}; expected (N, 7)")
    if len(labels) != 7:
        raise ValueError(f"Transform '{mode}' returned {len(labels)} labels; expected 7")
    return values, list(labels)


def _load_solver_module(path: str):
    module_name = f"pmp_solver_{abs(hash(os.path.abspath(path))) & 0xFFFFFFFF:x}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load solver module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_config(config_path: Optional[str]) -> tuple[IntegrationConfig, np.ndarray]:
    payload: dict[str, Any] = {}
    if config_path:
        with open(config_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    merged = dict(DEFAULT_INTEGRATION)
    merged.update({k: payload[k] for k in DEFAULT_INTEGRATION.keys() if k in payload})
    integ = IntegrationConfig(**merged)

    bounds = dict(DEFAULT_BOUNDS)
    bounds.update({k: tuple(payload[k]) for k in DEFAULT_BOUNDS.keys() if k in payload})
    tau_lo, tau_hi = bounds["tau_bounds"]
    scales = np.array(
        [
            0.5 * (bounds["u0_bounds"][1] - bounds["u0_bounds"][0]),
            0.5 * (bounds["w0_bounds"][1] - bounds["w0_bounds"][0]),
            0.5 * (bounds["ar0_bounds"][1] - bounds["ar0_bounds"][0]),
            0.5 * (bounds["at0_bounds"][1] - bounds["at0_bounds"][0]),
            0.5 * (bounds["jr0_bounds"][1] - bounds["jr0_bounds"][0]),
            0.5 * (bounds["ell_bounds"][1] - bounds["ell_bounds"][0]),
            0.5 * (math.log(tau_hi) - math.log(tau_lo)),
        ],
        dtype=float,
    )
    return integ, scales


def _load_center_from_atlas(npz_path: str, row: int) -> np.ndarray:
    with np.load(npz_path, allow_pickle=True) as data:
        if "launches" in data:
            launches = np.asarray(data["launches"], dtype=float)
        elif "forward_launches" in data:
            launches = np.asarray(data["forward_launches"], dtype=float)
        elif "chart_center_launch" in data:
            launches = np.asarray(data["chart_center_launch"], dtype=float)
        else:
            raise KeyError(
                "Could not find 'launches', 'forward_launches', or "
                "'chart_center_launch' in atlas npz"
            )
    if row < 0 or row >= len(launches):
        raise IndexError(f"Atlas row {row} out of bounds for {len(launches)} launches")
    return launches[row].copy()


def _make_launch_from_q(q: np.ndarray) -> np.ndarray:
    launch = np.empty(7, dtype=float)
    launch[:6] = q[:6]
    launch[6] = math.exp(float(q[6]))
    return launch


def _launch_to_q(launch: np.ndarray) -> np.ndarray:
    q = np.empty(7, dtype=float)
    q[:6] = launch[:6]
    q[6] = math.log(float(launch[6]))
    return q


def _generate_sphere_directions(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    vecs = rng.normal(size=(n, dim))
    norms = np.linalg.norm(vecs, axis=1)
    mask = norms > 0.0
    vecs[mask] /= norms[mask, None]
    return vecs


def _generate_ball_points(n: int, dim: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    dirs = _generate_sphere_directions(n, dim, rng)
    radii = rng.random(n) ** (1.0 / dim)
    return dirs * radii[:, None], radii


def _generate_shell_points(n: int, dim: int, shells: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if shells <= 1:
        dirs = _generate_sphere_directions(n, dim, rng)
        return dirs, np.ones(n, dtype=float)
    shell_radii = np.linspace(1.0 / shells, 1.0, shells)
    counts = np.full(shells, n // shells, dtype=int)
    counts[: n % shells] += 1
    pts: list[np.ndarray] = []
    tags: list[np.ndarray] = []
    for radius, count in zip(shell_radii, counts):
        if count <= 0:
            continue
        directions = _generate_sphere_directions(int(count), dim, rng)
        pts.append(directions * radius)
        tags.append(np.full(int(count), radius, dtype=float))
    return np.vstack(pts), np.concatenate(tags)


_WORKER_MODULE = None
_WORKER_CFG = None


def _worker_init(solver_path: str, cfg_dict: dict[str, float]):
    global _WORKER_MODULE, _WORKER_CFG
    _WORKER_MODULE = _load_solver_module(solver_path)
    _WORKER_CFG = cfg_dict


def _integrate_one(task: tuple[int, np.ndarray]):
    idx, launch = task
    cfg = _WORKER_CFG
    mod = _WORKER_MODULE
    sol, _normal_constant, status = mod._integrate_launch(
        launch,
        rtol=cfg["rtol"],
        atol=cfg["atol"],
        max_step=cfg["max_step"],
        r_collision=cfg["r_collision"],
        r_escape=cfg["r_escape"],
        max_acceleration=cfg["max_acceleration"],
        dense_output=False,
    )
    if sol is None:
        return idx, False, status, np.full(7, np.nan, dtype=float)
    radius, theta, ur, ut, kappa = mod._endpoint_from_state(sol.y[:, -1])
    endpoint = np.array(
        [
            float(launch[0]),
            float(launch[1]),
            math.log(radius),
            theta,
            ur,
            ut,
            math.log(kappa),
        ],
        dtype=float,
    )
    return idx, True, status, endpoint


def _integrate_swarm(
    launches: np.ndarray,
    solver_path: str,
    integ: IntegrationConfig,
    workers: int,
):
    endpoints = np.full((len(launches), 7), np.nan, dtype=float)
    ok = np.zeros(len(launches), dtype=bool)
    statuses = np.empty(len(launches), dtype=object)
    cfg_dict = asdict(integ)
    tasks = [(i, launches[i].copy()) for i in range(len(launches))]
    if workers <= 1:
        _worker_init(solver_path, cfg_dict)
        for idx, success, status, endpoint in map(_integrate_one, tasks):
            ok[idx] = success
            statuses[idx] = status
            endpoints[idx] = endpoint
        return ok, statuses, endpoints

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(solver_path, cfg_dict),
    ) as pool:
        futures = [pool.submit(_integrate_one, task) for task in tasks]
        done = 0
        report_stride = max(1, len(tasks) // 10)
        for future in as_completed(futures):
            idx, success, status, endpoint = future.result()
            ok[idx] = success
            statuses[idx] = status
            endpoints[idx] = endpoint
            done += 1
            if done % report_stride == 0 or done == len(tasks):
                print(f"  integrated {done:,}/{len(tasks):,}")
    return ok, statuses, endpoints


def _parse_transform_params(entries: list[str]) -> dict[str, float]:
    params: dict[str, float] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Transform parameter must be NAME=VALUE, got: {entry}")
        name, value = entry.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty transform parameter name in: {entry}")
        params[name] = float(value)
    return params


def _affine_linearity_analysis(
    normalized_input: np.ndarray,
    outputs: np.ndarray,
    center_output: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Fit local affine output map and compute pullback coordinates.

    Fits
        output - center_output = intercept + normalized_input @ A
    then reconstructs normalized input via the pseudoinverse of A.
    """
    x = np.asarray(normalized_input, dtype=float)
    z = np.asarray(outputs, dtype=float)
    z0 = np.asarray(center_output, dtype=float)
    dz = z - z0[None, :]
    design = np.column_stack((np.ones(len(x)), x))
    coeff, *_ = np.linalg.lstsq(design, dz, rcond=None)
    intercept = coeff[0]
    jacobian_row = coeff[1:]  # dz ~= x @ jacobian_row
    predicted = design @ coeff
    residual = dz - predicted
    pinv = np.linalg.pinv(jacobian_row)
    pullback = (dz - intercept[None, :]) @ pinv

    output_rms = float(np.sqrt(np.mean(np.sum(dz * dz, axis=1))))
    residual_norm = np.linalg.norm(residual, axis=1)
    residual_rms = float(np.sqrt(np.mean(residual_norm * residual_norm)))
    relative_rms = residual_rms / max(output_rms, np.finfo(float).tiny)
    input_error = np.linalg.norm(pullback - x, axis=1)

    per_coordinate = []
    for j in range(7):
        centered = dz[:, j] - np.mean(dz[:, j])
        ss_tot = float(np.dot(centered, centered))
        ss_res = float(np.dot(residual[:, j], residual[:, j]))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
        scale = float(np.sqrt(np.mean(dz[:, j] ** 2)))
        rms = float(np.sqrt(np.mean(residual[:, j] ** 2)))
        per_coordinate.append(
            {
                "coordinate": j,
                "r2": r2,
                "relative_rms_residual": rms / max(scale, np.finfo(float).tiny),
            }
        )

    singular_values = np.linalg.svd(jacobian_row, compute_uv=False)
    condition = float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 0 else float("inf")
    metrics: dict[str, Any] = {
        "relative_rms_nonlinearity": relative_rms,
        "p50_output_residual_norm": float(np.percentile(residual_norm, 50)),
        "p95_output_residual_norm": float(np.percentile(residual_norm, 95)),
        "p50_pullback_input_error": float(np.percentile(input_error, 50)),
        "p95_pullback_input_error": float(np.percentile(input_error, 95)),
        "affine_jacobian_condition": condition,
        "affine_jacobian_singular_values": singular_values.tolist(),
        "per_coordinate": per_coordinate,
    }
    return metrics, pullback, predicted + z0[None, :]


def _plot_dashboards(
    points: np.ndarray,
    center_point: np.ndarray,
    colors: np.ndarray,
    labels: list[str],
    out_prefix: Path,
    *,
    title_prefix: str,
):
    pairs = list(itertools.combinations(range(7), 2))
    nrows, ncols = 4, 3
    per_page = nrows * ncols
    pages = int(math.ceil(len(pairs) / per_page))
    cmap = plt.get_cmap("viridis")

    pdf_path = out_prefix.with_suffix(".pdf")
    with PdfPages(pdf_path) as pdf:
        for page in range(pages):
            fig, axes = plt.subplots(nrows, ncols, figsize=(15, 18), constrained_layout=True)
            axes = np.asarray(axes).ravel()
            start = page * per_page
            subset = pairs[start : start + per_page]
            for ax, (i, j) in zip(axes, subset):
                ax.scatter(
                    points[:, i],
                    points[:, j],
                    c=colors,
                    cmap=cmap,
                    s=4,
                    alpha=0.45,
                    linewidths=0,
                    rasterized=True,
                )
                ax.scatter([center_point[i]], [center_point[j]], marker="x", s=80)
                ax.set_xlabel(labels[i])
                ax.set_ylabel(labels[j])
                ax.set_title(f"{labels[i]} vs {labels[j]}")
                ax.grid(True, alpha=0.25)
            for ax in axes[len(subset):]:
                ax.axis("off")
            fig.suptitle(f"{title_prefix} — dashboard {page + 1}/{pages}")
            png_path = out_prefix.parent / f"{out_prefix.stem}_dashboard_{page + 1:02d}.png"
            fig.savefig(png_path, dpi=220)
            pdf.savefig(fig)
            plt.close(fig)
    return pdf_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", required=True, help="Path to pmp_extremal_atlas.py")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional atlas/generator JSON config for integration settings and default launch scales",
    )
    center_group = parser.add_mutually_exclusive_group(required=False)
    center_group.add_argument(
        "--center",
        nargs=7,
        type=float,
        metavar=("u0", "w0", "Ar0", "At0", "Jr0", "ell", "tau"),
        help="Launch center as seven numbers",
    )
    center_group.add_argument(
        "--atlas-row",
        nargs=2,
        metavar=("ATLAS_NPZ", "ROW"),
        help="Load center launch from a point or chart atlas npz and row index",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.05,
        help="Normalized swarm radius in launch-space units (default: 0.05)",
    )
    parser.add_argument(
        "--scale",
        nargs=7,
        type=float,
        default=None,
        metavar=LAUNCH_LABELS,
        help=(
            "Characteristic scales for (u0,w0,Ar0,At0,Jr0,ell,log_tau). "
            "If omitted, inferred from config bounds."
        ),
    )
    parser.add_argument("--points", type=int, default=4096, help="Number of swarm points")
    parser.add_argument(
        "--mode",
        choices=["sphere", "ball"],
        default="sphere",
        help="Sample on nested sphere shells or in a solid ball",
    )
    parser.add_argument(
        "--shells",
        type=int,
        default=4,
        help="For sphere mode, number of nested shells (default: 4)",
    )
    parser.add_argument(
        "--transform",
        choices=["raw", "cartesian", "cartesian_rotated", "equinoctial", "custom"],
        default="custom",
        help="Analytical output transformation to evaluate (default: custom)",
    )
    parser.add_argument(
        "--transform-param",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Numeric parameter passed to the analytical transform; may be repeated",
    )
    parser.add_argument(
        "--no-raw-dashboard",
        action="store_true",
        help="Skip the raw endpoint dashboard",
    )
    parser.add_argument(
        "--no-pullback-dashboard",
        action="store_true",
        help="Skip the affine-pullback dashboard",
    )
    parser.add_argument("--seed", type=int, default=12345, help="RNG seed")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Parallel workers (default: CPU count - 1)",
    )
    parser.add_argument(
        "--output-prefix",
        default="swarm_dashboard",
        help="Output prefix for dashboards, metrics, and optional NPZ",
    )
    parser.add_argument(
        "--save-npz",
        action="store_true",
        help="Also save launches, raw/transformed endpoints and pullback coordinates",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.center is None and args.atlas_row is None:
        parser.error("Provide either --center or --atlas-row")

    integ, default_scales = _load_config(args.config)
    scales = np.array(args.scale, dtype=float) if args.scale is not None else default_scales
    if np.any(scales <= 0.0) or not np.all(np.isfinite(scales)):
        raise ValueError("All launch-space scales must be positive and finite")

    if args.center is not None:
        center_launch = np.array(args.center, dtype=float)
    else:
        atlas_npz, row_text = args.atlas_row
        center_launch = _load_center_from_atlas(atlas_npz, int(row_text))
    if center_launch.shape != (7,):
        raise ValueError("Center launch must have seven components")
    if center_launch[6] <= 0.0:
        raise ValueError("Center tau must be positive")

    solver_module = _load_solver_module(args.solver)
    sol0, normal0, status0 = solver_module._integrate_launch(
        center_launch,
        rtol=integ.rtol,
        atol=integ.atol,
        max_step=integ.max_step,
        r_collision=integ.r_collision,
        r_escape=integ.r_escape,
        max_acceleration=integ.max_acceleration,
        dense_output=False,
    )
    if sol0 is None:
        raise RuntimeError(
            f"Center launch does not integrate successfully: status={status0}, "
            f"normal_constant={normal0}"
        )
    radius0, theta0, ur0, ut0, kappa0 = solver_module._endpoint_from_state(sol0.y[:, -1])
    center_raw = np.array(
        [
            center_launch[0],
            center_launch[1],
            math.log(radius0),
            theta0,
            ur0,
            ut0,
            math.log(kappa0),
        ],
        dtype=float,
    )

    rng = np.random.default_rng(args.seed)
    if args.mode == "ball":
        normalized_points, shell_values = _generate_ball_points(args.points, 7, rng)
    else:
        normalized_points, shell_values = _generate_shell_points(
            args.points, 7, max(1, args.shells), rng
        )
    normalized_input = args.radius * normalized_points

    q_center = _launch_to_q(center_launch)
    q_swarm = q_center[None, :] + normalized_input * scales[None, :]
    launches = np.vstack([_make_launch_from_q(q) for q in q_swarm])

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Integrating swarm: center={center_launch.tolist()}, radius={args.radius}, "
        f"points={len(launches):,}, workers={args.workers}"
    )
    success, statuses, raw_endpoints = _integrate_swarm(
        launches, args.solver, integ, args.workers
    )
    n_success = int(np.count_nonzero(success))
    print(f"Successful trajectories: {n_success:,}/{len(launches):,}")
    if n_success < 16:
        raise RuntimeError(
            "Too few successful trajectories for a stable 7D affine analysis; "
            "reduce --radius or choose a more robust center"
        )

    unique_status, counts = np.unique(statuses.astype(str), return_counts=True)
    status_counts = {str(k): int(v) for k, v in zip(unique_status, counts)}
    print(f"Status counts: {status_counts}")

    raw_ok = raw_endpoints[success]
    x_ok = normalized_input[success]
    colors_ok = shell_values[success]
    params = _parse_transform_params(args.transform_param)
    transformed_ok, transformed_labels = apply_analytical_transform(
        raw_ok, center_raw, args.transform, params
    )
    center_transformed, _ = apply_analytical_transform(
        center_raw[None, :], center_raw, args.transform, params
    )
    center_transformed = center_transformed[0]

    raw_metrics, raw_pullback, _ = _affine_linearity_analysis(x_ok, raw_ok, center_raw)
    transformed_metrics, transformed_pullback, _ = _affine_linearity_analysis(
        x_ok, transformed_ok, center_transformed
    )
    raw_metrics["labels"] = list(RAW_ENDPOINT_LABELS)
    transformed_metrics["labels"] = transformed_labels
    for metric, label in zip(raw_metrics["per_coordinate"], RAW_ENDPOINT_LABELS):
        metric["label"] = label
    for metric, label in zip(transformed_metrics["per_coordinate"], transformed_labels):
        metric["label"] = label

    metrics_payload = {
        "center_launch": center_launch.tolist(),
        "center_raw_endpoint": center_raw.tolist(),
        "launch_scales": scales.tolist(),
        "radius": float(args.radius),
        "mode": args.mode,
        "shells": int(args.shells),
        "requested_points": int(args.points),
        "successful_points": n_success,
        "status_counts": status_counts,
        "transform": args.transform,
        "transform_params": params,
        "raw": raw_metrics,
        "transformed": transformed_metrics,
        "improvement_factor": (
            raw_metrics["relative_rms_nonlinearity"]
            / max(transformed_metrics["relative_rms_nonlinearity"], np.finfo(float).tiny)
        ),
    }
    metrics_path = out_prefix.parent / f"{out_prefix.stem}_linearity_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    common_title = (
        f"success={n_success}/{len(launches)}, radius={args.radius}, "
        f"mode={args.mode}, transform={args.transform}"
    )
    if not args.no_raw_dashboard:
        _plot_dashboards(
            raw_ok,
            center_raw,
            colors_ok,
            list(RAW_ENDPOINT_LABELS),
            out_prefix.parent / f"{out_prefix.stem}_raw",
            title_prefix=f"Raw endpoint swarm; {common_title}",
        )
    _plot_dashboards(
        transformed_ok,
        center_transformed,
        colors_ok,
        transformed_labels,
        out_prefix.parent / f"{out_prefix.stem}_{args.transform}",
        title_prefix=(
            f"Analytically transformed endpoint swarm; {common_title}; "
            f"relative nonlinear RMS={transformed_metrics['relative_rms_nonlinearity']:.3g}"
        ),
    )
    if not args.no_pullback_dashboard:
        _plot_dashboards(
            transformed_pullback,
            np.zeros(7, dtype=float),
            colors_ok,
            [f"recovered_{name}" for name in LAUNCH_LABELS],
            out_prefix.parent / f"{out_prefix.stem}_{args.transform}_pullback",
            title_prefix=(
                "Local affine pullback after analytical transform; "
                f"p95 input error={transformed_metrics['p95_pullback_input_error']:.3g}"
            ),
        )

    if args.save_npz:
        npz_path = out_prefix.with_suffix(".npz")
        np.savez_compressed(
            npz_path,
            center_launch=center_launch,
            center_raw_endpoint=center_raw,
            center_transformed_endpoint=center_transformed,
            launch_scales=scales,
            radius=float(args.radius),
            normalized_input=normalized_input,
            launches=launches,
            raw_endpoints=raw_endpoints,
            transformed_endpoints_success=transformed_ok,
            transformed_pullback_success=transformed_pullback,
            raw_pullback_success=raw_pullback,
            success_mask=success,
            shell_values=shell_values,
            statuses=statuses,
            transform=np.array(args.transform),
            transform_params_json=np.array(json.dumps(params)),
        )
        print(f"Saved swarm data: {npz_path}")

    print(
        "Linearity comparison: "
        f"raw relative RMS={raw_metrics['relative_rms_nonlinearity']:.4g}; "
        f"transformed={transformed_metrics['relative_rms_nonlinearity']:.4g}; "
        f"improvement={metrics_payload['improvement_factor']:.3g}x"
    )
    print(f"Saved metrics: {metrics_path}")
    print(
        f"Saved dashboards with prefixes {out_prefix.stem}_raw, "
        f"{out_prefix.stem}_{args.transform}, and "
        f"{out_prefix.stem}_{args.transform}_pullback"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
