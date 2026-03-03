import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from generate_atlas import get_canonical_mission_config


STATE_UNSEEN = np.uint8(0)
STATE_QUEUED = np.uint8(1)
STATE_SOLVED = np.uint8(2)
STATE_RETRYABLE_FAILED = np.uint8(3)
STATE_DEAD_FAILED = np.uint8(4)

STATE_NAMES = {
    STATE_UNSEEN: "unseen",
    STATE_QUEUED: "queued",
    STATE_SOLVED: "solved",
    STATE_RETRYABLE_FAILED: "retryable failed",
    STATE_DEAD_FAILED: "dead failed",
}


def load_module_from_path(module_path: str, module_name: str = "user_solver_module"):
    module_path = str(Path(module_path).expanduser().resolve())
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_edges_from_centers(values, log_spacing=False):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("values must be a 1D array with at least 2 elements")

    edges = np.empty(len(values) + 1, dtype=float)
    if log_spacing:
        if np.any(values <= 0):
            raise ValueError("log-spaced edges require strictly positive values")
        edges[1:-1] = np.sqrt(values[:-1] * values[1:])
        edges[0] = values[0] ** 2 / edges[1]
        edges[-1] = values[-1] ** 2 / edges[-2]
    else:
        edges[1:-1] = 0.5 * (values[:-1] + values[1:])
        edges[0] = values[0] - 0.5 * (values[1] - values[0])
        edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def infer_theta_indices(theta_grid, n_panels):
    if len(theta_grid) <= n_panels:
        return np.arange(len(theta_grid))
    return np.linspace(0, len(theta_grid) - 1, n_panels, dtype=int)


def digitize_to_cell_index(edges, value):
    idx = np.searchsorted(edges, value, side="right") - 1
    if idx < 0 or idx >= len(edges) - 1:
        return None
    return int(idx)


def make_status_figure(state, rho_grid, kappa_grid, theta_grid, nrows=3, ncols=4, figsize=(16, 10)):
    expected_shape = (len(rho_grid), len(kappa_grid), len(theta_grid))
    if state.shape != expected_shape:
        raise ValueError(f"state.shape={state.shape}, expected {expected_shape}")

    cmap = ListedColormap([
        "#f0f0f0",  # unseen
        "#4c78a8",  # queued
        "#54a24b",  # solved
        "#f2cf5b",  # retryable failed
        "#e45756",  # dead failed
    ])
    norm = BoundaryNorm(np.arange(-0.5, 5.5, 1.0), cmap.N)

    rho_edges = compute_edges_from_centers(rho_grid, log_spacing=True)
    kappa_edges = compute_edges_from_centers(kappa_grid, log_spacing=True)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    theta_indices = infer_theta_indices(theta_grid, nrows * ncols)
    mesh = None

    for panel_idx, ax in enumerate(axes):
        if panel_idx >= len(theta_indices):
            ax.set_visible(False)
            continue

        k = int(theta_indices[panel_idx])
        z = state[:, :, k].T
        mesh = ax.pcolormesh(
            rho_edges,
            kappa_edges,
            z,
            cmap=cmap,
            norm=norm,
            shading="auto",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("rho = r_target / r_start")
        ax.set_ylabel("kappa")
        ax.set_title(f"theta[{k}] = {np.degrees(theta_grid[k]):.1f}°")
        ax._atlas_theta_index = k

    cbar = fig.colorbar(
        mesh,
        ax=axes.tolist(),
        ticks=[0, 1, 2, 3, 4],
        shrink=0.92,
        pad=0.02,
    )
    cbar.ax.set_yticklabels([STATE_NAMES[np.uint8(i)] for i in range(5)])
    cbar.set_label("Cell state")

    solved = int(np.count_nonzero(state == STATE_SOLVED))
    queued = int(np.count_nonzero(state == STATE_QUEUED))
    retryable = int(np.count_nonzero(state == STATE_RETRYABLE_FAILED))
    dead = int(np.count_nonzero(state == STATE_DEAD_FAILED))
    unseen = int(np.count_nonzero(state == STATE_UNSEEN))
    total = int(state.size)

    fig.suptitle("Atlas state slices — click a green cell to replay its trajectory\n"
        f"solved={solved}, queued={queued}, retryable={retryable}, dead={dead}, unseen={unseen}, total={total}",
        fontsize=14,
    )
    return fig, axes, rho_edges, kappa_edges


def main():
    parser = argparse.ArgumentParser(
        description="Interactive atlas viewer: click solved cells to integrate and plot the stored trajectory."
    )
    parser.add_argument("--npz_path", help="Path to trajectory_atlas.npz / trajectory_atlas_final.npz")
    parser.add_argument(
        "--solver",
        default="../rocketHamilton.py",
        help="Path to the solver script that defines TrajectoryConfig, integrate_fixed_time, and make_plots",
    )
    parser.add_argument("--nrows", type=int, default=3, help="Number of subplot rows")
    parser.add_argument("--ncols", type=int, default=4, help="Number of subplot columns")
    args = parser.parse_args()

    solver = load_module_from_path(args.solver)
    required_names = ["TrajectoryConfig", "AU", "MU_SI", "integrate_fixed_time", "make_plots"]
    missing = [name for name in required_names if not hasattr(solver, name)]
    if missing:
        raise AttributeError(f"Solver module is missing required names: {missing}")

    bundle = np.load(args.npz_path)
    rho_grid = bundle["rho"]
    kappa_grid = bundle["kappa"]
    theta_grid = bundle["theta"]
    data = bundle["data"]
    state = bundle["state"]

    if data.shape[:3] != state.shape:
        raise ValueError(f"data.shape[:3]={data.shape[:3]} does not match state.shape={state.shape}")
    if data.shape[-1] < 5:
        raise ValueError(f"Expected last axis of data to hold at least 5 parameters, got shape {data.shape}")

    fig, axes, rho_edges, kappa_edges = make_status_figure(
        state, rho_grid, kappa_grid, theta_grid, nrows=args.nrows, ncols=args.ncols
    )

    status_text = fig.text(
        0.01,
        0.01,
        "Click a cell. Green cells replay the stored trajectory.",
        ha="left",
        va="bottom",
        fontsize=10,
    )

    def on_click(event):
        ax = event.inaxes
        if ax is None or not hasattr(ax, "_atlas_theta_index"):
            return
        if event.xdata is None or event.ydata is None:
            return

        i = digitize_to_cell_index(rho_edges, event.xdata)
        j = digitize_to_cell_index(kappa_edges, event.ydata)
        k = int(ax._atlas_theta_index)

        if i is None or j is None:
            status_text.set_text("Clicked outside atlas bounds.")
            fig.canvas.draw_idle()
            return

        cell_state = np.uint8(state[i, j, k])
        rho = float(rho_grid[i])
        kappa = float(kappa_grid[j])
        theta = float(theta_grid[k])

        summary = (
            f"Cell (i={i}, j={j}, k={k}) | rho={rho:.6g}, kappa={kappa:.6g}, "
            f"theta={np.degrees(theta):.2f}° | state={STATE_NAMES.get(cell_state, str(cell_state))}"
        )
        print(summary)

        if cell_state != STATE_SOLVED:
            status_text.set_text(summary + " — not solved, nothing to replay.")
            fig.canvas.draw_idle()
            return

        row = np.asarray(data[i, j, k], dtype=float)
        if row.size < 5 or not np.all(np.isfinite(row[:5])):
            status_text.set_text(summary + " — stored solution is missing or non-finite.")
            fig.canvas.draw_idle()
            return

        t_days = float(row[5])
        _, config = get_canonical_mission_config(rho, kappa)
        status_text.set_text(summary + f" — replaying, stored t={t_days:.3f} d")
        fig.canvas.draw_idle()
        params = row[:5]

        sol_opt = solver.integrate_fixed_time(params, t_days, config=config)
        solver.make_plots(sol_opt, params, show=True, config=config)

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()


if __name__ == "__main__":
    main()
