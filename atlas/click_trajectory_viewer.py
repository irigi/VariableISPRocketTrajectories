import argparse
import importlib.util
from pathlib import Path

import multiprocessing as mp

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
    DISPLAY_RETRYABLE_FAILED: "solved: boundary mismatch",
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

BRANCH_UNDEFINED_TURNS_EPS = 0.15  # smaller => more willing to decide
RHO_TOL_REL = 1e-2     # relative tolerance on rho (r_end / AU)
# RHO_TOL_ABS = 1e-2     # absolute tolerance on rho
THETA_TOL_DEG = 1.0    # degrees (wrapped to [-180, 180])


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


def wrap_to_pi(angle_rad: float) -> float:
    return float(np.arctan2(np.sin(angle_rad), np.cos(angle_rad)))


def closed_curve_turns_about_origin(x, y):
    """
    Returns the net turns (total angle change / 2π) of the closed curve formed by
    the trajectory plus the straight chord from the endpoint back to the start.

    Sign of turns labels the two families around the Sun; near-zero means the Sun
    is not enclosed (or enclosure is ambiguous).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
        return 0.0

    # If the path gets extremely close to the Sun, any "side" classification is unstable.
    r = np.hypot(x, y)
    if np.any(r < 1e-9):
        return 0.0

    x_closed = np.concatenate([x, [x[0]]])
    y_closed = np.concatenate([y, [y[0]]])
    angles = np.unwrap(np.arctan2(y_closed, x_closed))
    total_turn = angles[-1] - angles[0]
    return float(total_turn / (2.0 * np.pi))


def classify_solution_branch(sol, solver):
    x = (sol.y[0] * np.cos(sol.y[1])) / solver.AU
    y = (sol.y[0] * np.sin(sol.y[1])) / solver.AU

    turns = closed_curve_turns_about_origin(x, y)

    if turns > BRANCH_UNDEFINED_TURNS_EPS:
        branch = BRANCH_LEFT
    elif turns < -BRANCH_UNDEFINED_TURNS_EPS:
        branch = BRANCH_RIGHT
    else:
        branch = BRANCH_UNDEFINED

    # For display/debug, keep a small signed integer.
    # If rounding would give 0 but we classified left/right, keep ±1.
    winding = int(np.rint(turns))
    if winding == 0 and branch != BRANCH_UNDEFINED:
        winding = int(np.sign(turns))
    return branch, winding, turns


def check_boundary_mismatch(sol, rho_target, theta_target_rad, solver):
    r_end = float(sol.y[0, -1] / solver.AU)
    theta_end = float(sol.y[1, -1])

    dr = r_end - float(rho_target)
    rho_tol = max(0.0, RHO_TOL_REL * float(rho_target))

    dtheta = wrap_to_pi(theta_end - float(theta_target_rad))
    theta_tol = np.deg2rad(THETA_TOL_DEG)

    mismatch = (abs(dr) > rho_tol) or (abs(dtheta) > theta_tol)
    return mismatch, r_end, dr, float(dtheta)


# --- Multiprocessing helpers (classification) ----------------------------

_CLASSIFY_SOLVER = None
_CLASSIFY_RHO_GRID = None
_CLASSIFY_KAPPA_GRID = None
_CLASSIFY_THETA_GRID = None


def _init_classify_pool(solver_path: str, rho_grid, kappa_grid, theta_grid):
    """Initializer for multiprocessing workers."""
    global _CLASSIFY_SOLVER, _CLASSIFY_RHO_GRID, _CLASSIFY_KAPPA_GRID, _CLASSIFY_THETA_GRID
    _CLASSIFY_SOLVER = load_module_from_path(solver_path, module_name="user_solver_module_worker")
    _CLASSIFY_RHO_GRID = np.asarray(rho_grid, dtype=float)
    _CLASSIFY_KAPPA_GRID = np.asarray(kappa_grid, dtype=float)
    _CLASSIFY_THETA_GRID = np.asarray(theta_grid, dtype=float)


def _classify_one_solved_cell(task):
    """Worker: classify a single solved cell. Returns (i, j, k, branch, winding, mismatch)."""
    i, j, k, row6 = task
    row = np.asarray(row6, dtype=float)

    if row.size < 6 or (not np.all(np.isfinite(row[:6]))):
        return int(i), int(j), int(k), int(BRANCH_UNDEFINED), 0, True

    rho = float(_CLASSIFY_RHO_GRID[int(i)])
    kappa = float(_CLASSIFY_KAPPA_GRID[int(j)])
    theta_target = float(_CLASSIFY_THETA_GRID[int(k)])

    t_days = float(row[5])
    params = row[:5]
    _, config = get_canonical_mission_config(rho, kappa)

    try:
        sol = _CLASSIFY_SOLVER.integrate_fixed_time(params, t_days, config=config)
        branch, winding, _turns = classify_solution_branch(sol, _CLASSIFY_SOLVER)
        mismatch, _r_end, _dr, _dtheta = check_boundary_mismatch(sol, rho, theta_target, _CLASSIFY_SOLVER)
    except Exception:
        branch, winding, mismatch = BRANCH_UNDEFINED, 0, True

    winding = int(np.clip(int(winding), -9, 9))
    return int(i), int(j), int(k), int(branch), int(winding), bool(mismatch)


def classify_all_solved_points(state, data, rho_grid, kappa_grid, theta_grid, solver):
    branch_map = np.full(state.shape, BRANCH_UNDEFINED, dtype=np.int8)
    winding_map = np.zeros(state.shape, dtype=np.int8)
    mismatch_map = np.zeros(state.shape, dtype=bool)

    # return branch_map, winding_map, mismatch_map            # skip everything

    from_two_files = True
    if from_two_files:
        second_npz_path = r"c:\Programs\VariableISPRocketTrajectories\atlas\run005\trajectory_atlas.npz"
        other_bundle = np.load(second_npz_path)
        other_state = other_bundle["state"]
        other_data = other_bundle["data"]

        if other_state.shape != state.shape:
            raise ValueError(
                f"other_state.shape={other_state.shape} does not match state.shape={state.shape}"
            )
        if other_data.shape[:3] != state.shape:
            raise ValueError(
                f"other_data.shape[:3]={other_data.shape[:3]} does not match state.shape={state.shape}"
            )
        if other_data.shape[-1] < 6:
            raise ValueError(
                f"Expected other_data last axis to hold at least 6 values [params..., t_days], "
                f"got shape {other_data.shape}"
            )

        # In this two-file mode:
        #   BRANCH_LEFT      -> this file is faster
        #   BRANCH_RIGHT     -> other file is faster
        #   BRANCH_UNDEFINED -> tie / undecidable among solved cases
        #
        # We will also rewrite `state` for plotting:
        #   STATE_SOLVED          -> at least one file solved, branch_map decides color
        #   STATE_DEAD_FAILED     -> failed in both
        #   STATE_UNSEEN          -> unknown overall / no usable result
        #   STATE_QUEUED          -> unused here
        #   STATE_RETRYABLE_FAILED-> unused here
        #
        # mismatch_map is unused here.

        solved_here = (state == STATE_SOLVED)
        solved_other = (other_state == STATE_SOLVED)

        failed_here = (state == STATE_RETRYABLE_FAILED) | (state == STATE_DEAD_FAILED)
        failed_other = (other_state == STATE_RETRYABLE_FAILED) | (other_state == STATE_DEAD_FAILED)

        unseen_here = (state == STATE_UNSEEN)
        unseen_other = (other_state == STATE_UNSEEN)

        queued_here = (state == STATE_QUEUED)
        queued_other = (other_state == STATE_QUEUED)

        # Start from a clean comparison-state for plotting.
        state[:] = STATE_UNSEEN
        branch_map[:] = BRANCH_UNDEFINED
        winding_map[:] = 0
        mismatch_map[:] = False

        # 1) Solved in exactly one file -> that file wins.
        solved_here_only = solved_here & (~solved_other)
        solved_other_only = solved_other & (~solved_here)

        state[solved_here_only] = STATE_SOLVED
        branch_map[solved_here_only] = BRANCH_LEFT

        state[solved_other_only] = STATE_SOLVED
        branch_map[solved_other_only] = BRANCH_RIGHT

        # 2) Solved in both -> compare times of flight.
        solved_both = solved_here & solved_other

        t_here = np.asarray(data[..., 5], dtype=float)
        t_other = np.asarray(other_data[..., 5], dtype=float)

        finite_t_here = np.isfinite(t_here)
        finite_t_other = np.isfinite(t_other)
        comparable = solved_both & finite_t_here & finite_t_other

        this_faster = comparable & (t_here < t_other)
        other_faster = comparable & (t_other < t_here)
        tied = comparable & (t_here == t_other)

        state[this_faster | other_faster | tied] = STATE_SOLVED
        branch_map[this_faster] = BRANCH_LEFT
        branch_map[other_faster] = BRANCH_RIGHT
        branch_map[tied] = BRANCH_UNDEFINED

        # Solved in both but non-finite time in one/both -> unknown overall.
        bad_time = solved_both & (~finite_t_here | ~finite_t_other)
        state[bad_time] = STATE_UNSEEN
        branch_map[bad_time] = BRANCH_UNDEFINED

        # 3) Failed in both -> red.
        failed_both = failed_here & failed_other
        state[failed_both] = STATE_DEAD_FAILED

        # 4) Everything else remains unknown overall.
        # This includes:
        #    - unseen/queued combinations with no solved result
        #    - failed in one file and unseen/queued in the other
        #    - any other combination not handled above

        return branch_map, winding_map, mismatch_map

    solved_indices = np.argwhere(state == STATE_SOLVED)
    total = int(len(solved_indices))
    print(f"Classifying {total} solved trajectories into Sun-left / Sun-right / undefined...")

    if total == 0:
        return branch_map, winding_map, mismatch_map

    solver_path = getattr(solver, "__file__", None)
    if not solver_path:
        raise ValueError("Solver module does not have a __file__ attribute; cannot spawn worker processes safely")

    tasks = (
        (int(i), int(j), int(k), np.asarray(data[i, j, k, :6], dtype=float))
        for (i, j, k) in solved_indices
    )

    with mp.Pool(
        initializer=_init_classify_pool,
        initargs=(str(solver_path), rho_grid, kappa_grid, theta_grid),
    ) as pool:
        for n, (i, j, k, branch, winding, mismatch) in enumerate(
            pool.imap_unordered(_classify_one_solved_cell, tasks, chunksize=50), start=1
        ):
            branch_map[i, j, k] = np.int8(branch)
            winding_map[i, j, k] = np.int8(winding)
            mismatch_map[i, j, k] = bool(mismatch)

            if (n % 100 == 0) or (n == total):
                print(f"  classified {n}/{total}")

    return branch_map, winding_map, mismatch_map


def make_display_state(state, branch_map, mismatch_map):
    display = np.full(state.shape, DISPLAY_UNSEEN, dtype=np.uint8)
    display[state == STATE_UNSEEN] = DISPLAY_UNSEEN
    display[state == STATE_QUEUED] = DISPLAY_QUEUED
    # If these exist in your atlas, treat as "dead failed" for display purposes.
    display[state == STATE_RETRYABLE_FAILED] = DISPLAY_DEAD_FAILED
    display[state == STATE_DEAD_FAILED] = DISPLAY_DEAD_FAILED

    solved = (state == STATE_SOLVED)

    # Mark boundary-mismatched solved cells with the special color (repurposed DISPLAY_RETRYABLE_FAILED).
    display[solved & mismatch_map] = DISPLAY_RETRYABLE_FAILED

    # Normal solved cells: shade by branch family.
    ok = solved & (~mismatch_map)
    display[ok & (branch_map == BRANCH_LEFT)] = DISPLAY_SOLVED_LEFT
    display[ok & (branch_map == BRANCH_RIGHT)] = DISPLAY_SOLVED_RIGHT
    display[ok & (branch_map == BRANCH_UNDEFINED)] = DISPLAY_SOLVED_UNDEFINED
    return display


def make_status_figure(display_state, state, branch_map, mismatch_map, rho_grid, kappa_grid, theta_grid, nrows=3, ncols=4, figsize=(16, 10)):
    expected_shape = (len(rho_grid), len(kappa_grid), len(theta_grid))
    if display_state.shape != expected_shape:
        raise ValueError(f"display_state.shape={display_state.shape}, expected {expected_shape}")

    cmap = ListedColormap([
        "#f0f0f0",  # unseen
        "#4c78a8",  # queued
        "#0b6e3a",  # solved: Sun left
        "#54a24b",  # solved: Sun right
        "#b7e4c7",  # solved: undefined / far
        "#f2cf5b",  # solved: boundary mismatch
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
    mismatch = int(np.count_nonzero((state == STATE_SOLVED) & mismatch_map))
    ok = solved - mismatch

    solved_left = int(np.count_nonzero((state == STATE_SOLVED) & (~mismatch_map) & (branch_map == BRANCH_LEFT)))
    solved_right = int(np.count_nonzero((state == STATE_SOLVED) & (~mismatch_map) & (branch_map == BRANCH_RIGHT)))
    solved_undefined = int(np.count_nonzero((state == STATE_SOLVED) & (~mismatch_map) & (branch_map == BRANCH_UNDEFINED)))

    queued = int(np.count_nonzero(state == STATE_QUEUED))
    dead = int(np.count_nonzero((state == STATE_DEAD_FAILED) | (state == STATE_RETRYABLE_FAILED)))
    unseen = int(np.count_nonzero(state == STATE_UNSEEN))
    total = int(state.size)

    fig.suptitle(
        "Atlas state slices — solved cells shaded by trajectory family; yellow marks boundary mismatch"
        f"solved={solved} (ok={ok}: left={solved_left}, right={solved_right}, undefined={solved_undefined}; mismatch={mismatch}), "
        f"queued={queued}, dead={dead}, unseen={unseen}, total={total}",
        fontsize=14,
    )
    return fig, axes, rho_edges, kappa_edges


def main():
    parser = argparse.ArgumentParser(
        description="Interactive atlas viewer: solved cells are shaded by whether the trajectory goes Sun-left, Sun-right, or neither; boundary mismatches are highlighted."
    )
    parser.add_argument("--npz_path", help="Path to trajectory_atlas.npz / trajectory_atlas_final.npz")
    parser.add_argument(
        "--solver",
        default="../rocketHamilton.py",
        help="Path to the solver script that defines TrajectoryConfig, AU, integrate_fixed_time, and make_plots",
    )
    parser.add_argument("--nrows", type=int, default=3, help="Number of subplot rows")
    parser.add_argument("--ncols", type=int, default=4, help="Number of subplot columns")
    parser.add_argument("--skip_mod", type=int, default=1,
                        help="Subsample factor for rho/kappa/theta axes (e.g., 5 shows every 5th)")
    args = parser.parse_args()

    solver = load_module_from_path(args.solver)
    required_names = ["TrajectoryConfig", "AU", "MU_SI", "integrate_fixed_time", "make_plots"]
    missing = [name for name in required_names if not hasattr(solver, name)]
    if missing:
        raise AttributeError(f"Solver module is missing required names: {missing}")

    bundle = np.load(args.npz_path)
    skip_mod = int(args.skip_mod)

    rho_grid = bundle["rho"][::skip_mod]
    kappa_grid = bundle["kappa"][::skip_mod]
    theta_grid = bundle["theta"][::skip_mod]
    data = bundle["data"][::skip_mod, ::skip_mod, ::skip_mod, :]
    state = bundle["state"][::skip_mod, ::skip_mod, ::skip_mod]

    if data.shape[:3] != state.shape:
        raise ValueError(f"data.shape[:3]={data.shape[:3]} does not match state.shape={state.shape}")
    if data.shape[-1] < 6:
        raise ValueError(f"Expected last axis of data to hold at least 6 values [params..., t_days], got shape {data.shape}")

    branch_map, winding_map, mismatch_map = classify_all_solved_points(state, data, rho_grid, kappa_grid, theta_grid, solver)
    display_state = make_display_state(state, branch_map, mismatch_map)

    write_patched_file = False
    if write_patched_file:
        mismatch_solved = (state == STATE_SOLVED) & mismatch_map
        n_bad = int(np.count_nonzero(mismatch_solved))

        out_path = Path(args.npz_path)
        cleaned_path = out_path.with_name(out_path.stem + "_solution_cleanup" + out_path.suffix)

        if n_bad > 0:
            state_full = state.copy()
            data_full = data.copy()
            state_full[mismatch_solved] = STATE_UNSEEN
            data_full[mismatch_solved, :] = np.nan

            save_payload = {}
            for key in bundle.files:
                if key == "state":
                    save_payload[key] = state_full
                elif key == "data":
                    save_payload[key] = data_full
                else:
                    save_payload[key] = bundle[key]

            np.savez_compressed(cleaned_path, **save_payload)
            print(f"Saved cleaned atlas to: {cleaned_path}  (reverted {n_bad} boundary-mismatch cells)")

    fig, axes, rho_edges, kappa_edges = make_status_figure(
        display_state, state, branch_map, mismatch_map, rho_grid, kappa_grid, theta_grid, nrows=args.nrows, ncols=args.ncols
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
        theta_target = float(theta_grid[k])

        summary = (
            f"Cell (i={i}, j={j}, k={k}) | rho={rho:.6g}, kappa={kappa:.6g}, "
            f"theta={np.degrees(theta_target):.2f}° | state={STATE_NAMES.get(cell_state, str(cell_state))}"
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
        is_mismatch = bool(mismatch_map[i, j, k])

        status_text.set_text(
            summary
            + f" — {BRANCH_NAMES.get(branch, 'unknown')} (winding={winding}), "
            + ("BOUNDARY MISMATCH, " if is_mismatch else "")
            + f"replaying t={t_days:.3f} d"
        )
        fig.canvas.draw_idle()

        sol_opt = solver.integrate_fixed_time(params, t_days, config=config)

        # Print endpoint residuals when replaying.
        mismatch, r_end, dr, dtheta = check_boundary_mismatch(sol_opt, rho, theta_target, solver)
        print(
            f"  endpoint check: r_end={r_end:.6g} AU (dr={dr:+.3e}), "
            f"dtheta={np.degrees(dtheta):+.3f} deg, mismatch={mismatch}"
        )

        solver.make_plots(sol_opt, params, show=True, config=config)

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()


if __name__ == "__main__":
    main()
