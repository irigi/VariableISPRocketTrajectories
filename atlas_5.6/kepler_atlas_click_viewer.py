#!/usr/bin/env python3
"""Interactive click viewer for ``kepler_atlas_solver.py`` NPZ atlases.

The catalogue window shows a grid of two-dimensional ``(rho, kappa)`` slices.
Each panel fixes one unwrapped transfer angle ``theta``.  By default, colour
shows the dimensionless flight time ``tau_f`` of the shortest valid branch.

Mouse controls
--------------
* Left click: replay the selected branch (shortest by default).
* Right click: replay all valid branches stored in the clicked cell.

The replay window contains the dimensionless Cartesian trajectory, acceleration
history, inverse-mass budget usage, and velocity diagnostics.  No dimensional
spacecraft assumptions are introduced; dimensional reconstruction can be added
later without changing the atlas format.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from numpy.typing import NDArray
from scipy.integrate import solve_ivp


METRIC_LABELS = {
    "tau": "Dimensionless flight time $\\tau_f$",
    "residual": "Terminal residual norm",
    "condition": "Shooting Jacobian condition number",
    "iterations": "Newton iterations",
    "nfev": "ODE RHS evaluations",
}


@dataclass(slots=True)
class AtlasData:
    path: Path
    rho: NDArray[np.float64]
    theta: NDArray[np.float64]
    kappa: NDArray[np.float64]
    shooting: NDArray[np.float64]
    valid: NDArray[np.bool_]
    residual_norm: NDArray[np.float64]
    condition: NDArray[np.float64]
    iterations: NDArray[np.int16]
    nfev: NDArray[np.int32]
    status: NDArray[np.int8]
    metadata: dict

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.rho.size, self.theta.size, self.kappa.size)

    @property
    def branch_count(self) -> int:
        return int(self.valid.shape[-1])


@dataclass(slots=True)
class Replay:
    branch: int
    shooting: NDArray[np.float64]
    tau: NDArray[np.float64]
    state: NDArray[np.float64]
    residual: NDArray[np.float64]

    @property
    def tau_f(self) -> float:
        return float(self.shooting[4])


class ViewerError(RuntimeError):
    """User-facing atlas or trajectory replay error."""


def load_module_from_path(path: Path) -> ModuleType:
    resolved = path.expanduser().resolve()
    spec = importlib.util.spec_from_file_location("kepler_atlas_solver_viewer", resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import solver module from {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _require_keys(bundle: np.lib.npyio.NpzFile, names: Iterable[str]) -> None:
    missing = [name for name in names if name not in bundle.files]
    if missing:
        raise ViewerError(f"Atlas is missing required NPZ arrays: {', '.join(missing)}")


def load_atlas(path: Path) -> AtlasData:
    resolved = path.expanduser().resolve()
    with np.load(resolved, allow_pickle=False) as bundle:
        _require_keys(bundle, ("rho", "theta", "kappa", "shooting", "valid"))
        rho = np.asarray(bundle["rho"], dtype=np.float64)
        theta = np.asarray(bundle["theta"], dtype=np.float64)
        kappa = np.asarray(bundle["kappa"], dtype=np.float64)
        shooting = np.asarray(bundle["shooting"], dtype=np.float64)
        valid = np.asarray(bundle["valid"], dtype=bool)

        expected_prefix = (rho.size, theta.size, kappa.size)
        if shooting.ndim == 4 and shooting.shape[-1] == 5:
            # Compatibility with a possible single-branch atlas without an
            # explicit branch axis.
            shooting = shooting[..., None, :]
        if valid.ndim == 3:
            valid = valid[..., None]
        if shooting.shape[:3] != expected_prefix or shooting.shape[-1] != 5:
            raise ViewerError(
                "shooting must have shape (n_rho, n_theta, n_kappa, n_branch, 5); "
                f"found {shooting.shape}"
            )
        if valid.shape != shooting.shape[:-1]:
            raise ViewerError(
                f"valid shape {valid.shape} does not match shooting prefix {shooting.shape[:-1]}"
            )

        diagnostic_shape = valid.shape

        def diagnostic(name: str, dtype, default):
            if name not in bundle.files:
                return np.full(diagnostic_shape, default, dtype=dtype)
            value = np.asarray(bundle[name], dtype=dtype)
            if value.ndim == 3:
                value = value[..., None]
            if value.shape != diagnostic_shape:
                raise ViewerError(
                    f"{name} shape {value.shape} does not match valid shape {diagnostic_shape}"
                )
            return value

        residual_norm = diagnostic("residual_norm", np.float64, np.nan)
        condition = diagnostic("condition", np.float64, np.nan)
        iterations = diagnostic("iterations", np.int16, 0)
        nfev = diagnostic("nfev", np.int32, 0)
        status = (
            np.asarray(bundle["status"], dtype=np.int8)
            if "status" in bundle.files
            else np.where(np.any(valid, axis=-1), 1, 0).astype(np.int8)
        )
        if status.shape != expected_prefix:
            raise ViewerError(f"status shape {status.shape} does not match grid {expected_prefix}")

        metadata: dict = {}
        if "metadata_json" in bundle.files:
            try:
                metadata = json.loads(str(bundle["metadata_json"]))
            except (TypeError, ValueError, json.JSONDecodeError):
                metadata = {"metadata_parse_error": True}

    if np.any(rho <= 0.0) or np.any(kappa <= 0.0):
        raise ViewerError("rho and kappa axes must be strictly positive")
    if not (np.all(np.diff(rho) > 0.0) and np.all(np.diff(theta) > 0.0) and np.all(np.diff(kappa) > 0.0)):
        raise ViewerError("rho, theta, and kappa axes must be strictly increasing")

    return AtlasData(
        path=resolved,
        rho=rho,
        theta=theta,
        kappa=kappa,
        shooting=shooting,
        valid=valid,
        residual_norm=residual_norm,
        condition=condition,
        iterations=iterations,
        nfev=nfev,
        status=status,
        metadata=metadata,
    )


def solver_config_from_metadata(solver: ModuleType, atlas: AtlasData):
    if not hasattr(solver, "SolverConfig"):
        raise ViewerError("Solver module does not expose SolverConfig")
    fields = getattr(solver.SolverConfig, "__dataclass_fields__", {})
    saved = atlas.metadata.get("solver_config", {})
    kwargs = {name: saved[name] for name in fields if name in saved}
    # Viewer replays should not be stopped by the atlas builder's short
    # wall-clock budget.  The stored trajectory is already converged.
    if "max_integration_seconds" in fields:
        kwargs["max_integration_seconds"] = max(
            float(kwargs.get("max_integration_seconds", 3.0)), 60.0
        )
    if "max_integration_nfev" in fields:
        kwargs["max_integration_nfev"] = max(
            int(kwargs.get("max_integration_nfev", 12000)), 250000
        )
    return solver.SolverConfig(**kwargs)


def compute_edges_from_centers(values: NDArray[np.float64], logarithmic: bool) -> NDArray[np.float64]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ViewerError("Axis values must be a nonempty one-dimensional array")
    if values.size == 1:
        value = float(values[0])
        if logarithmic:
            return np.array((value / math.sqrt(2.0), value * math.sqrt(2.0)))
        width = max(abs(value) * 0.15, 0.1)
        return np.array((value - width, value + width))

    edges = np.empty(values.size + 1, dtype=np.float64)
    if logarithmic:
        edges[1:-1] = np.sqrt(values[:-1] * values[1:])
        edges[0] = values[0] ** 2 / edges[1]
        edges[-1] = values[-1] ** 2 / edges[-2]
    else:
        edges[1:-1] = 0.5 * (values[:-1] + values[1:])
        edges[0] = values[0] - 0.5 * (values[1] - values[0])
        edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def digitize_cell(edges: NDArray[np.float64], value: float) -> Optional[int]:
    index = int(np.searchsorted(edges, value, side="right") - 1)
    if index < 0 or index >= edges.size - 1:
        return None
    return index


def infer_theta_indices(theta: NDArray[np.float64], panel_count: int) -> NDArray[np.int64]:
    if panel_count <= 0:
        raise ViewerError("The catalogue must contain at least one panel")
    if theta.size <= panel_count:
        return np.arange(theta.size, dtype=np.int64)
    # Unique guards against accidental duplicate indices for small axes.
    return np.unique(np.rint(np.linspace(0, theta.size - 1, panel_count)).astype(np.int64))


def shortest_branch_map(atlas: AtlasData) -> NDArray[np.int16]:
    times = np.where(atlas.valid, atlas.shooting[..., 4], np.inf)
    chosen = np.argmin(times, axis=-1).astype(np.int16)
    chosen[~np.any(atlas.valid, axis=-1)] = -1
    return chosen


def fixed_branch_map(atlas: AtlasData, branch: int) -> NDArray[np.int16]:
    if branch < 0 or branch >= atlas.branch_count:
        raise ViewerError(
            f"Requested branch {branch}, but atlas stores branches 0..{atlas.branch_count - 1}"
        )
    chosen = np.full(atlas.shape, branch, dtype=np.int16)
    chosen[~atlas.valid[..., branch]] = -1
    return chosen


def choose_branch_map(atlas: AtlasData, branch_spec: str) -> NDArray[np.int16]:
    if branch_spec.lower() == "shortest":
        return shortest_branch_map(atlas)
    try:
        branch = int(branch_spec)
    except ValueError as exc:
        raise ViewerError("--branch must be 'shortest' or a nonnegative integer") from exc
    return fixed_branch_map(atlas, branch)


def gather_branch_values(array: NDArray, branch_map: NDArray[np.int16], fill=np.nan) -> NDArray:
    output = np.full(branch_map.shape, fill, dtype=np.result_type(array.dtype, type(fill)))
    valid_cells = branch_map >= 0
    indices = np.nonzero(valid_cells)
    output[indices] = array[indices + (branch_map[indices],)]
    return output


def metric_cube(atlas: AtlasData, branch_map: NDArray[np.int16], metric: str) -> NDArray[np.float64]:
    if metric == "tau":
        return gather_branch_values(atlas.shooting[..., 4], branch_map).astype(np.float64)
    if metric == "residual":
        return gather_branch_values(atlas.residual_norm, branch_map).astype(np.float64)
    if metric == "condition":
        return gather_branch_values(atlas.condition, branch_map).astype(np.float64)
    if metric == "iterations":
        return gather_branch_values(atlas.iterations, branch_map).astype(np.float64)
    if metric == "nfev":
        return gather_branch_values(atlas.nfev, branch_map).astype(np.float64)
    raise ViewerError(f"Unsupported metric {metric!r}")


def metric_norm(values: NDArray[np.float64], metric: str):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    low = float(np.nanmin(finite))
    high = float(np.nanmax(finite))
    if metric in {"residual", "condition", "nfev"} and low > 0.0 and high / low > 30.0:
        return LogNorm(vmin=low, vmax=max(high, low * (1.0 + 1.0e-12)))
    if math.isclose(low, high, rel_tol=1e-12, abs_tol=1e-15):
        padding = max(abs(low) * 0.05, 1.0e-12)
        return Normalize(vmin=low - padding, vmax=high + padding)
    return Normalize(vmin=low, vmax=high)


def integrate_replay(
    solver: ModuleType,
    config,
    shooting: NDArray[np.float64],
    rho: float,
    theta: float,
    kappa: float,
    branch: int,
    samples: int,
) -> Replay:
    required = ("initial_state", "augmented_rhs", "terminal_residual_from_state")
    missing = [name for name in required if not hasattr(solver, name)]
    if missing:
        raise ViewerError(f"Solver module lacks replay functions: {', '.join(missing)}")

    vector = np.asarray(shooting, dtype=np.float64)
    if vector.shape != (5,) or not np.all(np.isfinite(vector)):
        raise ViewerError("Stored shooting vector is missing or non-finite")
    tau_f = float(vector[4])
    if tau_f <= 0.0:
        raise ViewerError("Stored flight time is not positive")

    y0 = np.asarray(solver.initial_state(vector[:4]), dtype=np.float64)
    t_eval = np.linspace(0.0, tau_f, max(32, int(samples)))

    def rhs(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(solver.augmented_rhs(t, y, 1.0), dtype=np.float64)

    solution = solve_ivp(
        rhs,
        (0.0, tau_f),
        y0,
        method="DOP853",
        t_eval=t_eval,
        rtol=float(getattr(config, "rtol", 2.0e-9)),
        atol=float(getattr(config, "atol", 2.0e-11)),
        max_step=float(getattr(config, "max_step", math.inf)),
    )
    if not solution.success or solution.t.size == 0 or solution.t[-1] < tau_f * (1.0 - 1.0e-9):
        raise ViewerError(f"Trajectory replay failed: {solution.message}")

    target = np.array((math.log(rho), theta, math.log(kappa)), dtype=np.float64)
    residual = np.asarray(
        solver.terminal_residual_from_state(solution.y[:, -1], target, config),
        dtype=np.float64,
    )
    return Replay(branch=branch, shooting=vector.copy(), tau=solution.t, state=solution.y, residual=residual)


def closed_curve_turns_about_origin(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    if x.size < 2 or y.size != x.size:
        return 0.0
    if np.any(np.hypot(x, y) < 1.0e-10):
        return 0.0
    closed_x = np.concatenate((x, x[:1]))
    closed_y = np.concatenate((y, y[:1]))
    return float((np.unwrap(np.arctan2(closed_y, closed_x))[-1] - np.arctan2(y[0], x[0])) / (2.0 * np.pi))


def plot_replays(
    replays: Sequence[Replay],
    rho: float,
    theta: float,
    kappa: float,
    atlas: AtlasData,
    clicked_indices: tuple[int, int, int],
) -> None:
    if not replays:
        return

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    ax_trajectory, ax_acceleration, ax_budget, ax_velocity = axes.ravel()
    palette = plt.get_cmap("tab10")

    max_extent = max(1.0, rho)
    for order, replay in enumerate(replays):
        colour = palette(order % 10)
        state = replay.state
        Rxy = state[0:2]
        Vxy = state[2:4]
        Axy = state[4:6]
        radius = np.linalg.norm(Rxy, axis=0)
        acceleration = np.linalg.norm(Axy, axis=0)
        q = state[8]
        radial_velocity = np.einsum("ij,ij->j", Rxy, Vxy) / radius
        tangential_velocity = (Rxy[0] * Vxy[1] - Rxy[1] * Vxy[0]) / radius
        circular_velocity = radius ** -0.5
        label = f"branch {replay.branch}: $\\tau_f$={replay.tau_f:.6g}"

        ax_trajectory.plot(Rxy[0], Rxy[1], label=label, color=colour)
        ax_trajectory.scatter(Rxy[0, 0], Rxy[1, 0], marker="o", color=colour, s=28)
        ax_trajectory.scatter(Rxy[0, -1], Rxy[1, -1], marker="x", color=colour, s=48)
        ax_acceleration.plot(replay.tau, acceleration, label=f"|A|, b{replay.branch}", color=colour)
        ax_acceleration.plot(replay.tau, Axy[0], linestyle="--", alpha=0.65, color=colour)
        ax_acceleration.plot(replay.tau, Axy[1], linestyle=":", alpha=0.65, color=colour)
        ax_budget.plot(replay.tau, q / kappa, label=f"branch {replay.branch}", color=colour)
        ax_velocity.plot(replay.tau, radial_velocity, linestyle="--", color=colour, label=f"$v_r$, b{replay.branch}")
        ax_velocity.plot(replay.tau, tangential_velocity, color=colour, label=f"$v_t$, b{replay.branch}")
        if order == 0:
            ax_velocity.plot(replay.tau, circular_velocity, color="black", alpha=0.45, label="$1/\\sqrt{r}$")

        max_extent = max(max_extent, float(np.nanmax(radius)))

    # Orbit guides are deliberately dimensionless.
    for orbit_radius, style, alpha in ((1.0, "--", 0.45), (rho, ":", 0.55)):
        ax_trajectory.add_patch(
            plt.Circle((0.0, 0.0), orbit_radius, fill=False, linestyle=style, alpha=alpha)
        )
    ax_trajectory.scatter((0.0,), (0.0,), marker="*", s=180, label="central body")
    target_xy = rho * np.array((math.cos(theta), math.sin(theta)))
    ax_trajectory.scatter(target_xy[0], target_xy[1], marker="+", s=100, label="target state")
    extent = max_extent * 1.08
    ax_trajectory.set_xlim(-extent, extent)
    ax_trajectory.set_ylim(-extent, extent)
    ax_trajectory.set_aspect("equal", adjustable="box")
    ax_trajectory.set_xlabel("Dimensionless x")
    ax_trajectory.set_ylabel("Dimensionless y")
    ax_trajectory.set_title("Trajectory")
    ax_trajectory.legend(fontsize=8)

    ax_acceleration.set_xlabel("Dimensionless time $\\tau$")
    ax_acceleration.set_ylabel("Dimensionless acceleration")
    ax_acceleration.set_title("Acceleration: solid magnitude, dashed/dotted components")
    ax_acceleration.legend(fontsize=8)
    ax_acceleration.grid(alpha=0.25)

    ax_budget.axhline(1.0, linestyle="--", alpha=0.5)
    ax_budget.set_xlabel("Dimensionless time $\\tau$")
    ax_budget.set_ylabel("Consumed budget $q/\\kappa$")
    ax_budget.set_ylim(bottom=0.0)
    ax_budget.set_title("Inverse-mass expenditure")
    ax_budget.legend(fontsize=8)
    ax_budget.grid(alpha=0.25)

    ax_velocity.axhline(0.0, linewidth=0.8, alpha=0.4)
    ax_velocity.set_xlabel("Dimensionless time $\\tau$")
    ax_velocity.set_ylabel("Dimensionless velocity")
    ax_velocity.set_title("Radial and tangential velocity")
    ax_velocity.legend(fontsize=8, ncols=2)
    ax_velocity.grid(alpha=0.25)

    i, j, k = clicked_indices
    residual_text = ", ".join(
        f"b{replay.branch}: ||R||={np.linalg.norm(replay.residual):.2e}"
        for replay in replays
    )
    turns_text = ", ".join(
        f"b{replay.branch}: closed turns={closed_curve_turns_about_origin(replay.state[0], replay.state[1]):+.2f}"
        for replay in replays
    )
    fig.suptitle(
        f"Atlas cell (rho={rho:.6g}, theta={math.degrees(theta):.2f}°, kappa={kappa:.6g})\n"
        f"indices=({i},{j},{k}) | {residual_text} | {turns_text}",
        fontsize=12,
    )
    fig.canvas.manager.set_window_title(
        f"Kepler trajectory rho={rho:.4g}, theta={math.degrees(theta):.1f}°, kappa={kappa:.4g}"
    )
    plt.show(block=False)


def make_catalogue(
    atlas: AtlasData,
    solver: ModuleType,
    config,
    branch_map: NDArray[np.int16],
    metric: str,
    nrows: int,
    ncols: int,
    samples: int,
    figsize: tuple[float, float],
):
    values = metric_cube(atlas, branch_map, metric)
    theta_indices = infer_theta_indices(atlas.theta, nrows * ncols)
    selected_values = values[:, theta_indices, :]
    norm = metric_norm(selected_values, metric)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.88")

    rho_edges = compute_edges_from_centers(atlas.rho, logarithmic=True)
    kappa_edges = compute_edges_from_centers(atlas.kappa, logarithmic=True)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    mesh = None
    for panel, ax in enumerate(axes_flat):
        if panel >= theta_indices.size:
            ax.set_visible(False)
            continue
        theta_index = int(theta_indices[panel])
        panel_values = np.ma.masked_invalid(values[:, theta_index, :].T)
        mesh = ax.pcolormesh(
            rho_edges,
            kappa_edges,
            panel_values,
            shading="auto",
            cmap=cmap,
            norm=norm,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\rho=r_f/r_0$")
        ax.set_ylabel(r"$\kappa$")
        ax.set_title(
            rf"$\Theta[{theta_index}]={math.degrees(atlas.theta[theta_index]):.1f}^\circ$"
        )
        ax._atlas_theta_index = theta_index  # type: ignore[attr-defined]

    if mesh is None:
        raise ViewerError("No visible catalogue panels were created")
    cbar = fig.colorbar(mesh, ax=[ax for ax in axes_flat if ax.get_visible()], pad=0.02, shrink=0.93)
    cbar.set_label(METRIC_LABELS[metric])

    solved = int(np.count_nonzero(np.any(atlas.valid, axis=-1)))
    selected = int(np.count_nonzero(branch_map >= 0))
    total = int(np.prod(atlas.shape))
    branch_description = "shortest valid" if np.any(
        branch_map != np.where(np.any(atlas.valid, axis=-1), 0, -1)
    ) else "branch 0 / shortest"
    title = (
        f"Kepler extremal atlas — {METRIC_LABELS[metric]}\n"
        f"selected={selected}, any-branch solved={solved}, total={total}; "
        f"left click: selected branch, right click: all branches"
    )
    fig.suptitle(title, fontsize=13)
    fig.canvas.manager.set_window_title(f"Kepler atlas: {atlas.path.name}")
    status_text = fig.text(
        0.01,
        0.005,
        "Click a solved pixel to replay its dimensionless trajectory.",
        ha="left",
        va="bottom",
        fontsize=9,
    )

    def on_click(event) -> None:
        ax = event.inaxes
        if ax is None or not hasattr(ax, "_atlas_theta_index"):
            return
        if event.xdata is None or event.ydata is None:
            return
        i = digitize_cell(rho_edges, float(event.xdata))
        k = digitize_cell(kappa_edges, float(event.ydata))
        theta_index = int(ax._atlas_theta_index)  # type: ignore[attr-defined]
        if i is None or k is None:
            status_text.set_text("Clicked outside atlas bounds.")
            fig.canvas.draw_idle()
            return

        rho = float(atlas.rho[i])
        theta = float(atlas.theta[theta_index])
        kappa = float(atlas.kappa[k])
        valid_branches = np.flatnonzero(atlas.valid[i, theta_index, k])
        if valid_branches.size == 0:
            message = (
                f"Cell ({i},{theta_index},{k}): rho={rho:.6g}, "
                f"theta={math.degrees(theta):.2f}°, kappa={kappa:.6g} — unsolved"
            )
            status_text.set_text(message)
            print(message)
            fig.canvas.draw_idle()
            return

        if event.button == 3:
            branches = [int(value) for value in valid_branches]
        else:
            selected = int(branch_map[i, theta_index, k])
            branches = [selected] if selected >= 0 else [int(valid_branches[0])]

        status_text.set_text(
            f"Replaying cell ({i},{theta_index},{k}), branches {branches}: "
            f"rho={rho:.6g}, theta={math.degrees(theta):.2f}°, kappa={kappa:.6g}"
        )
        fig.canvas.draw_idle()
        print(status_text.get_text())

        replays: list[Replay] = []
        try:
            for branch in branches:
                replays.append(
                    integrate_replay(
                        solver=solver,
                        config=config,
                        shooting=atlas.shooting[i, theta_index, k, branch],
                        rho=rho,
                        theta=theta,
                        kappa=kappa,
                        branch=branch,
                        samples=samples,
                    )
                )
        except Exception as exc:
            message = f"Replay failed for cell ({i},{theta_index},{k}): {exc}"
            status_text.set_text(message)
            print(message)
            fig.canvas.draw_idle()
            return

        plot_replays(
            replays,
            rho=rho,
            theta=theta,
            kappa=kappa,
            atlas=atlas,
            clicked_indices=(i, theta_index, k),
        )

    fig.canvas.mpl_connect("button_press_event", on_click)
    return fig


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
    parser.add_argument("atlas", type=Path, help="NPZ atlas produced by kepler_atlas_solver.py")
    parser.add_argument(
        "--solver",
        type=Path,
        default=Path(__file__).with_name("kepler_atlas_solver.py"),
        help="Path to kepler_atlas_solver.py",
    )
    parser.add_argument("--nrows", type=int, default=3)
    parser.add_argument("--ncols", type=int, default=4)
    parser.add_argument(
        "--metric",
        choices=tuple(METRIC_LABELS),
        default="tau",
        help="Quantity shown by the heatmaps",
    )
    parser.add_argument(
        "--branch",
        default="shortest",
        help="'shortest' or a fixed zero-based branch index",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1200,
        help="Trajectory samples in each replay window",
    )
    parser.add_argument("--figsize", type=parse_figsize, default=(16.0, 10.0))
    parser.add_argument(
        "--save-catalogue",
        type=Path,
        default=None,
        help="Also save a static image of the catalogue before opening it",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.nrows <= 0 or args.ncols <= 0:
        raise SystemExit("--nrows and --ncols must be positive")
    if args.samples < 32:
        raise SystemExit("--samples must be at least 32")

    atlas = load_atlas(args.atlas)
    solver = load_module_from_path(args.solver)
    config = solver_config_from_metadata(solver, atlas)
    branch_map = choose_branch_map(atlas, args.branch)

    print(f"Loaded {atlas.path}")
    print(f"Grid: rho={atlas.rho.size}, theta={atlas.theta.size}, kappa={atlas.kappa.size}")
    print(f"Branches stored: {atlas.branch_count}")
    print(f"Solved grid points: {np.count_nonzero(np.any(atlas.valid, axis=-1))}/{np.prod(atlas.shape)}")

    fig = make_catalogue(
        atlas=atlas,
        solver=solver,
        config=config,
        branch_map=branch_map,
        metric=args.metric,
        nrows=args.nrows,
        ncols=args.ncols,
        samples=args.samples,
        figsize=args.figsize,
    )
    if args.save_catalogue is not None:
        output = args.save_catalogue.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
        print(f"Saved catalogue image to {output}")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
