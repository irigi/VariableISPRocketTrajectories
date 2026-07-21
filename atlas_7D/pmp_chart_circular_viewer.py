#!/usr/bin/env python3
"""Circular-section coverage viewer for a local inverse PMP chart atlas.

Every pixel is an exact circular-to-circular canonical query

    [u0,w0,log(rho),theta,ur_f,ut_f,log(kappa)]
    = [0,1,log(rho),theta,0,rho**(-1/2),log(kappa)].

The background is the minimum validated chart trust ratio.  Values <= 1 are
inside at least one empirical chart trust region.  Clicking a pixel runs the
same bounded exact corrector used by generator validation.  Successes and
failures are marked and cached for the session.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
CORRECTOR_PROFILE_NAMES = (
    "newton_balanced",
    "newton_aggressive",
    "newton_damped",
    "trf_fast",
)


def load_module(path: str | Path, name_prefix: str) -> ModuleType:
    path = Path(path).resolve()
    name = f"{name_prefix}_{hashlib.sha1(str(path).encode()).hexdigest()[:10]}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def log_edges(low: float, high: float, bins: int) -> FloatArray:
    return np.geomspace(low, high, bins + 1)


def linear_edges(low: float, high: float, bins: int) -> FloatArray:
    return np.linspace(low, high, bins + 1)


def centers(edges: FloatArray) -> FloatArray:
    if np.all(edges > 0.0) and np.allclose(np.diff(np.log(edges)), np.diff(np.log(edges))[0], rtol=1e-5, atol=1e-8):
        return np.sqrt(edges[:-1] * edges[1:])
    return 0.5 * (edges[:-1] + edges[1:])


def circular_targets(rho: FloatArray, theta: float, kappa: FloatArray) -> FloatArray:
    rr, kk = np.meshgrid(rho, kappa, indexing="xy")
    n = rr.size
    target = np.empty((n, 7), dtype=float)
    target[:, 0] = 0.0
    target[:, 1] = 1.0
    target[:, 2] = np.log(rr.ravel())
    target[:, 3] = float(theta)
    target[:, 4] = 0.0
    target[:, 5] = rr.ravel() ** -0.5
    target[:, 6] = np.log(kk.ravel())
    return target


def cache_key(
    atlas_path: Path,
    rho_edges: FloatArray,
    kappa_edges: FloatArray,
    theta_values: FloatArray,
    chart_count: int,
) -> str:
    stat = atlas_path.stat()
    payload = {
        "path": str(atlas_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "chart_count": int(chart_count),
        "rho": rho_edges.tolist(),
        "kappa": kappa_edges.tolist(),
        "theta": theta_values.tolist(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_or_compute_coverage(
    atlas,
    atlas_path: Path,
    rho_edges: FloatArray,
    kappa_edges: FloatArray,
    theta_values: FloatArray,
    cache_path: Optional[Path],
) -> tuple[FloatArray, NDArray[np.int64]]:
    key = cache_key(atlas_path, rho_edges, kappa_edges, theta_values, len(atlas.bank))
    if cache_path is not None and cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                if str(data["key"].item()) == key:
                    print(f"Loaded coverage cache: {cache_path}")
                    return np.asarray(data["ratio"], dtype=float), np.asarray(data["chart_index"], dtype=np.int64)
        except Exception:
            pass

    rho = centers(rho_edges)
    kappa = centers(kappa_edges)
    ratios = np.empty((len(theta_values), len(kappa), len(rho)), dtype=float)
    chart_index = np.empty_like(ratios, dtype=np.int64)
    for panel, theta in enumerate(theta_values):
        target = circular_targets(rho, float(theta), kappa)
        ratio, index = atlas.coverage_scores(target)
        ratios[panel] = ratio.reshape(len(kappa), len(rho))
        chart_index[panel] = index.reshape(len(kappa), len(rho))
        print(
            f"Coverage slice {panel + 1}/{len(theta_values)}: "
            f"theta={math.degrees(theta):.2f} deg, covered={np.mean(ratio <= 1.0):.1%}"
        )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp = cache_path.with_name(cache_path.name + ".tmp")
        with open(temp, "wb") as handle:
            np.savez_compressed(handle, key=np.array(key), ratio=ratios, chart_index=chart_index)
        temp.replace(cache_path)
        print(f"Saved coverage cache: {cache_path}")
    return ratios, chart_index


def plot_canonical_trajectory(atlas, launch: FloatArray, target: FloatArray, chart_index: int, ratio: float) -> None:
    points = max(300, int(atlas.query_cfg.trajectory_points))
    t_eval = np.linspace(0.0, launch[6], points)
    sol, normal_constant, status = atlas.backend._integrate_launch(
        launch,
        rtol=atlas.query_cfg.rtol,
        atol=atlas.query_cfg.atol,
        max_step=atlas.query_cfg.max_step,
        r_collision=atlas.query_cfg.r_collision,
        r_escape=atlas.query_cfg.r_escape,
        max_acceleration=atlas.query_cfg.max_acceleration,
        t_eval=t_eval,
    )
    if sol is None or status != "ok":
        raise RuntimeError(f"corrected trajectory integration failed: {status}")
    radius = np.linalg.norm(sol.y[0:2], axis=0)
    acceleration = np.linalg.norm(sol.y[4:6], axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ax = axes[0]
    ax.plot(sol.y[0], sol.y[1])
    circle0 = plt.Circle((0, 0), 1.0, fill=False, linestyle="--", alpha=0.5)
    circlef = plt.Circle((0, 0), math.exp(target[2]), fill=False, linestyle=":", alpha=0.6)
    ax.add_patch(circle0)
    ax.add_patch(circlef)
    ax.scatter([sol.y[0, 0], sol.y[0, -1]], [sol.y[1, 0], sol.y[1, -1]], s=24)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("canonical x")
    ax.set_ylabel("canonical y")
    ax.set_title("Corrected extremal")

    axes[1].plot(sol.t, radius)
    axes[1].set_xlabel("canonical time")
    axes[1].set_ylabel("radius")
    axes[1].set_title("Radius history")

    axes[2].plot(sol.t, acceleration)
    axes[2].set_xlabel("canonical time")
    axes[2].set_ylabel("|A|")
    axes[2].set_title("Acceleration history")

    method_index = int(atlas.bank.solver_method[chart_index]) if len(atlas.bank.solver_method) else 0
    profiles = CORRECTOR_PROFILE_NAMES
    method_name = profiles[method_index] if 0 <= method_index < len(profiles) else "unknown"
    fig.suptitle(
        f"chart={chart_index}, trust ratio={ratio:.3g}, method={method_name}, "
        f"tau={launch[6]:.5g}, normal={normal_constant:.5g}"
    )
    fig.tight_layout()
    fig.show()


def make_viewer(
    atlas,
    atlas_path: Path,
    rho_edges: FloatArray,
    kappa_edges: FloatArray,
    theta_values: FloatArray,
    ratios: FloatArray,
    *,
    max_display_ratio: float,
    query_max_ratio: float,
    save_path: Optional[Path],
    show: bool,
) -> None:
    n = len(theta_values)
    columns = min(4, max(1, int(math.ceil(math.sqrt(n)))))
    rows = int(math.ceil(n / columns))
    fig, axes_array = plt.subplots(
        rows, columns, figsize=(4.3 * columns, 3.7 * rows), squeeze=False,
        constrained_layout=True,
    )
    axes = axes_array.ravel()
    panel_by_axis = {}
    query_marks: dict[tuple[int, int, int], object] = {}

    finite = ratios[np.isfinite(ratios)]
    displayed = np.log10(np.clip(ratios, 1.0e-3, max_display_ratio))
    norm = Normalize(vmin=-1.0, vmax=math.log10(max_display_ratio))
    image = None
    for panel, theta in enumerate(theta_values):
        ax = axes[panel]
        panel_by_axis[ax] = panel
        image = ax.pcolormesh(
            rho_edges,
            kappa_edges,
            displayed[panel],
            shading="auto",
            norm=norm,
            cmap="viridis",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"$\kappa$")
        covered = float(np.mean(ratios[panel] <= 1.0))
        ax.set_title(rf"$\Theta={math.degrees(theta):.1f}^\circ$; certified {covered:.1%}")
    for ax in axes[n:]:
        ax.set_visible(False)
    if image is not None:
        cbar = fig.colorbar(image, ax=axes[:n].tolist(), shrink=0.85)
        cbar.set_label(r"$\log_{10}$(best chart trust ratio); covered when $\leq0$")
    fig.suptitle(
        f"Validated local-inverse chart coverage — {atlas_path.name}\n"
        "Click any pixel to run the bounded exact correction"
    )

    rho_center = centers(rho_edges)
    kappa_center = centers(kappa_edges)

    def on_click(event) -> None:
        if event.inaxes not in panel_by_axis or event.xdata is None or event.ydata is None:
            return
        panel = panel_by_axis[event.inaxes]
        rho = float(event.xdata)
        kappa = float(event.ydata)
        if not (rho_edges[0] <= rho <= rho_edges[-1] and kappa_edges[0] <= kappa <= kappa_edges[-1]):
            return
        ir = int(np.clip(np.searchsorted(rho_edges, rho, side="right") - 1, 0, len(rho_center) - 1))
        ik = int(np.clip(np.searchsorted(kappa_edges, kappa, side="right") - 1, 0, len(kappa_center) - 1))
        key = (panel, ik, ir)
        rho_q = float(rho_center[ir])
        kappa_q = float(kappa_center[ik])
        theta = float(theta_values[panel])
        target = circular_targets(np.array([rho_q]), theta, np.array([kappa_q]))[0]
        corrected, chart_idx, ratio, reason, elapsed = atlas.correct_target(
            target, max_ratio=query_max_ratio, return_reason=True
        )
        print(
            f"Query rho={rho_q:.7g}, theta={math.degrees(theta):.4g} deg, "
            f"kappa={kappa_q:.7g}: {reason}, ratio={ratio:.4g}, "
            f"chart={chart_idx}, elapsed={elapsed:.4g}s"
        )
        old = query_marks.pop(key, None)
        if old is not None:
            try:
                old.remove()
            except Exception:
                pass
        if corrected is not None:
            mark = event.inaxes.plot(rho_q, kappa_q, marker="o", markersize=7, markerfacecolor="none", markeredgewidth=1.8)[0]
            query_marks[key] = mark
            plot_canonical_trajectory(atlas, corrected, target, chart_idx, ratio)
        else:
            marker = "x" if reason in {
                "correction-failed", "timeout-or-budget", "invalid-chart-prediction"
            } else "+"
            mark = event.inaxes.plot(rho_q, kappa_q, marker=marker, markersize=8, markeredgewidth=2.0)[0]
            query_marks[key] = mark
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    if save_path is not None:
        fig.savefig(save_path, dpi=160)
        print(f"Saved viewer image: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("atlas")
    parser.add_argument("--chart-module", default=str(Path(__file__).with_name("pmp_chart_atlas.py")))
    parser.add_argument("--solver", default=str(Path(__file__).with_name("pmp_extremal_atlas.py")))
    parser.add_argument("--rho-min", type=float)
    parser.add_argument("--rho-max", type=float)
    parser.add_argument("--rho-bins", type=int, default=80)
    parser.add_argument("--kappa-min", type=float)
    parser.add_argument("--kappa-max", type=float)
    parser.add_argument("--kappa-bins", type=int, default=60)
    parser.add_argument("--theta-min", type=float)
    parser.add_argument("--theta-max", type=float)
    parser.add_argument("--theta-panels", type=int, default=12)
    parser.add_argument("--max-display-ratio", type=float, default=100.0)
    parser.add_argument("--query-max-ratio", type=float, default=1.0)
    parser.add_argument("--cache")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--save")
    parser.add_argument("--no-show", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    chart_module = load_module(args.chart_module, "pmp_chart_module")
    atlas_path = Path(args.atlas).resolve()
    atlas = chart_module.LocalInverseChartAtlas(atlas_path, args.solver)
    backend_cfg = atlas.config.backend_config
    rho_bounds = backend_cfg.get("rho_bounds", [0.05, 30.0])
    kappa_bounds = backend_cfg.get("kappa_bounds", [1.0e-5, 1.0e4])
    theta_bounds = backend_cfg.get("theta_bounds", [-2.0 * math.pi, 2.0 * math.pi])
    rho_min = float(args.rho_min if args.rho_min is not None else rho_bounds[0])
    rho_max = float(args.rho_max if args.rho_max is not None else rho_bounds[1])
    kappa_min = float(args.kappa_min if args.kappa_min is not None else kappa_bounds[0])
    kappa_max = float(args.kappa_max if args.kappa_max is not None else kappa_bounds[1])
    theta_min = float(args.theta_min if args.theta_min is not None else theta_bounds[0])
    theta_max = float(args.theta_max if args.theta_max is not None else theta_bounds[1])
    rho_edges = log_edges(rho_min, rho_max, args.rho_bins)
    kappa_edges = log_edges(kappa_min, kappa_max, args.kappa_bins)
    theta_edges = linear_edges(theta_min, theta_max, args.theta_panels)
    theta_values = centers(theta_edges)
    if args.no_cache:
        cache_path = None
    elif args.cache:
        cache_path = Path(args.cache).resolve()
    else:
        cache_path = atlas_path.with_suffix(".chart_viewer_cache.npz")
    ratios, _chart_index = load_or_compute_coverage(
        atlas, atlas_path, rho_edges, kappa_edges, theta_values, cache_path
    )
    make_viewer(
        atlas,
        atlas_path,
        rho_edges,
        kappa_edges,
        theta_values,
        ratios,
        max_display_ratio=args.max_display_ratio,
        query_max_ratio=args.query_max_ratio,
        save_path=Path(args.save).resolve() if args.save else None,
        show=not args.no_show,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
