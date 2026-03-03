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

# Display classes for the plotted state map.
DISPLAY_UNSEEN = 0
DISPLAY_QUEUED = 1
DISPLAY_SOLVED_LEFT = 2
DISPLAY_SOLVED_RIGHT = 3
DISPLAY_SOLVED_UNDEFINED = 4
DISPLAY_RETRYABLE_FAILED = 5
DISPLAY_DEAD_FAILED = 6

DISPLAY_NAMES = {
    DISPLAY_UNSEEN: "unseen",
    DISPLAY_QUEUED: "queued",
    DISPLAY_SOLVED_LEFT: "solved: Sun left",
    DISPLAY_SOLVED_RIGHT: "solved: Sun right",
    DISPLAY_SOLVED_UNDEFINED: "solved: undefined/far",
    DISPLAY_RETRYABLE_FAILED: "retryable failed",
    DISPLAY_DEAD_FAILED: "dead failed",
}

BRANCH_LEFT = 1
BRANCH_RIGHT = -1
BRANCH_UNDEFINED = 0

BRANCH_NAMES = {
    BRANCH_LEFT: "Sun left",
    BRANCH_RIGHT: "Sun right",
    BRANCH_UNDEFINED: "undefined / not enclosing Sun",
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


def closed_curve_winding_number(x, y):
    """
    Winding number of the closed curve formed by the trajectory plus the straight
    chord from the endpoint back to the start. Positive/negative sign labels the
    two topological families around the Sun; zero means the Sun is not enclosed.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
        return 0

    # If the path gets extremely close to the Sun, the side classification is not stable.
    r = np.hypot(x, y)
    if np.any(r < 1e-6):
        return 0

    x_closed = np.concatenate([x, [x[0]]])
    y_closed = np.concatenate([y, [y[0]]])
    angles = np.unwrap(np.arctan2(y_closed, x_closed))
    total_turn = angles[-1] - angles[0]
    return int(np.rint(total_turn / (2.0 * np.pi)))


def classify_solution_branch(sol, solver):
    x = (sol.y[0] * np.cos(sol.y[1])) / solver.AU
    y = (sol.y[0] * np.sin(sol.y[1])) / solver.AU
    winding = closed_curve_winding_number(x, y)

    if winding > 0:
        return BRANCH_LEFT, winding
    if winding < 0:
        return BRANCH_RIGHT, winding
    return BRANCH_UNDEFINED, winding


def classify_all_solved_points(state, data, rho_grid, kappa_grid, solver):
    branch_map = np.full(state.shape, BRANCH_UNDEFINED, dtype=np.int8)
    winding_map = np.zeros(state.shape, dtype=np.int8)

    solved_indices = np.argwhere(state == STATE_SOLVED)
    total = int(len(solved_indices))
    print(f"Classifying {total} solved trajectories into Sun-left / Sun-right / undefined...")

    for n, (i, j, k) in enumerate(solved_indices, start=1):
        row = np.asarray(data[i, j, k], dtype=float)
        if row.size < 6 or not np.all(np.isfinite(row[:6])):
            branch_map[i, j, k] = BRANCH_UNDEFINED
            winding_map[i, j, k] = 0
            continue

        rho = float(rho_grid[i])
        kappa = float(kappa_grid[j])
        t_days = float(row[5])
        params = row[:5]
        _, config = get_canonical_mission_config(rho, kappa)

        try:
            sol = solver.integrate_fixed_time(params, t_days, config=config)
            branch, winding = classify_solution_branch(sol, solver)
        except Exception as exc:
            print(f"  Warning: classification failed for cell {(int(i), int(j), int(k))}: {exc}")
            branch, winding = BRANCH_UNDEFINED, 0

        branch_map[i, j, k] = np.int8(branch)
        winding_map[i, j, k] = np.int8(np.clip(winding, -9, 9))

        if (n % 100 == 0) or (n == total):
            print(f"  classified {n}/{total}")

    return branch_map, winding_map


def make_display_state(state, branch_map):
    display = np.full(state.shape, DISPLAY_UNSEEN, dtype=np.uint8)
    display[state == STATE_UNSEEN] = DISPLAY_UNSEEN
    display[state == STATE_QUEUED] = DISPLAY_QUEUED
    display[state == STATE_RETRYABLE_FAILED] = DISPLAY_RETRYABLE_FAILED
    display[state == STATE_DEAD_FAILED] = DISPLAY_DEAD_FAILED

    solved = (state == STATE_SOLVED)
    display[solved & (branch_map == BRANCH_LEFT)] = DISPLAY_SOLVED_LEFT
    display[solved & (branch_map == BRANCH_RIGHT)] = DISPLAY_SOLVED_RIGHT
    display[solved & (branch_map == BRANCH_UNDEFINED)] = DISPLAY_SOLVED_UNDEFINED
    return display


def make_status_figure(display_state, state, branch_map, rho_grid, kappa_grid, theta_grid, nrows=3, ncols=4, figsize=(16, 10)):
    expected_shape = (len(rho_grid), len(kappa_grid), len(theta_grid))
    if display_state.shape != expected_shape:
        raise ValueError(f"display_state.shape={display_state.shape}, expected {expected_shape}")

    cmap = ListedColormap([
        "#f0f0f0",  # unseen
        "#4c78a8",  # queued
        "#0b6e3a",  # solved: Sun left
        "#54a24b",  # solved: Sun right
        "#b7e4c7",  # solved: undefined / far
        "#f2cf5b",  # retryable failed
        "#e45756",  # dead failed
    ])
    norm = BoundaryNorm(np.arange(-0.5, 7.5, 1.0), cmap.N)

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
        z = display_state[:, :, k].T
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
        ticks=np.arange(7),
        shrink=0.92,
        pad=0.02,
    )
    cbar.ax.set_yticklabels([DISPLAY_NAMES[i] for i in range(7)])
    cbar.set_label("Cell state / trajectory family")

    solved = int(np.count_nonzero(state == STATE_SOLVED))
    solved_left = int(np.count_nonzero((state == STATE_SOLVED) & (branch_map == BRANCH_LEFT)))
    solved_right = int(np.count_nonzero((state == STATE_SOLVED) & (branch_map == BRANCH_RIGHT)))
    solved_undefined = int(np.count_nonzero((state == STATE_SOLVED) & (branch_map == BRANCH_UNDEFINED)))
    queued = int(np.count_nonzero(state == STATE_QUEUED))
    retryable = int(np.count_nonzero(state == STATE_RETRYABLE_FAILED))
    dead = int(np.count_nonzero(state == STATE_DEAD_FAILED))
    unseen = int(np.count_nonzero(state == STATE_UNSEEN))
    total = int(state.size)

    fig.suptitle(
        "Atlas state slices — solved cells are shaded by trajectory family around the Sun\n"
        f"solved={solved} (left={solved_left}, right={solved_right}, undefined={solved_undefined}), "
        f"queued={queued}, retryable={retryable}, dead={dead}, unseen={unseen}, total={total}",
        fontsize=14,
    )
    return fig, axes, rho_edges, kappa_edges


def main():
    parser = argparse.ArgumentParser(
        description="Interactive atlas viewer: solved cells are shaded by whether the trajectory goes Sun-left, Sun-right, or neither."
    )
    parser.add_argument("--npz_path", help="Path to trajectory_atlas.npz / trajectory_atlas_final.npz")
    parser.add_argument(
        "--solver",
        default="../rocketHamilton.py",
        help="Path to the solver script that defines TrajectoryConfig, AU, integrate_fixed_time, and make_plots",
    )
    parser.add_argument("--nrows", type=int, default=3, help="Number of subplot rows")
    parser.add_argument("--ncols", type=int, default=4, help="Number of subplot columns")
    args = parser.parse_args()

    solver = load_module_from_path(args.solver)
    required_names = ["TrajectoryConfig", "AU", "MU_SI", "integrate_fixed_time", "make_plots"]
    missing = [name for name in required_names if not hasattr(solver, name)]
    if missing:
        raise AttributeError(f"Solver module is missing required names: {missing}")

    skip_mod = 5
    bundle = np.load(args.npz_path)
    rho_grid = bundle["rho"][::skip_mod]
    kappa_grid = bundle["kappa"][::skip_mod]
    theta_grid = bundle["theta"][::skip_mod]
    data = bundle["data"][::skip_mod, ::skip_mod, ::skip_mod, :]
    state = bundle["state"][::skip_mod, ::skip_mod, ::skip_mod]

    if data.shape[:3] != state.shape:
        raise ValueError(f"data.shape[:3]={data.shape[:3]} does not match state.shape={state.shape}")
    if data.shape[-1] < 6:
        raise ValueError(f"Expected last axis of data to hold at least 6 values [params..., t_days], got shape {data.shape}")

    branch_map, winding_map = classify_all_solved_points(state, data, rho_grid, kappa_grid, solver)
    display_state = make_display_state(state, branch_map)

    fig, axes, rho_edges, kappa_edges = make_status_figure(
        display_state, state, branch_map, rho_grid, kappa_grid, theta_grid, nrows=args.nrows, ncols=args.ncols
    )

    status_text = fig.text(
        0.01,
        0.01,
        "Click a cell. Solved cells replay the stored fixed-time trajectory.",
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
        if row.size < 6 or not np.all(np.isfinite(row[:6])):
            status_text.set_text(summary + " — stored solution is missing or non-finite.")
            fig.canvas.draw_idle()
            return

        t_days = float(row[5])
        _, config = get_canonical_mission_config(rho, kappa)
        params = row[:5]
        branch = int(branch_map[i, j, k])
        winding = int(winding_map[i, j, k])

        status_text.set_text(
            summary + f" — {BRANCH_NAMES.get(branch, 'unknown')} (winding={winding}), replaying t={t_days:.3f} d"
        )
        fig.canvas.draw_idle()

        sol_opt = solver.integrate_fixed_time(params, t_days, config=config)
        solver.make_plots(sol_opt, params, show=True, config=config)

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()


if __name__ == "__main__":
    main()
