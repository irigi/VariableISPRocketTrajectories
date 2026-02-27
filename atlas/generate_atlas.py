"""
generate_atlas.py

Generates the "Time-Optimal Trajectory Atlas" (TOTA).
This script explores the 3D parameter space (Radius Ratio, Capability, Angle)
using a parallelized 3D flood-fill (wavefront) strategy.

Output: 'trajectory_atlas.npz'
"""

import time
import multiprocessing as mp
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

import rocketHamilton as rh


# -------------------------------------------------------
# 1. Grid Configuration
# -------------------------------------------------------

# Radius Ratio (rho = r_target / r_start)
RHO_MIN, RHO_MAX = 0.01, 100.0
N_RHO = 80

# Capability Parameter (kappa)
KAPPA_MIN, KAPPA_MAX = 0.1, 200000.0
N_KAPPA = 60

# Angle (theta) in Radians
THETA_MAX_REV = 1.1
N_THETA = 120

# Retry / batching controls
MAX_RETRIES_PER_CELL = 0
MIN_CHUNKSIZE = 4
MAX_CHUNKSIZE = 32
PROGRESS_INTERVAL = 1

# Cell states
STATE_UNSEEN = np.uint8(0)
STATE_QUEUED = np.uint8(1)
STATE_SOLVED = np.uint8(2)
STATE_RETRYABLE_FAILED = np.uint8(3)
STATE_DEAD_FAILED = np.uint8(4)

# Physical constants reused in every worker invocation
R0_SI = rh.AU
MU_SI = rh.MU_SI
M0_KG = 3000.0
M_DRY_KG = 1000.0
DELTA_INV_M = (1.0 / M_DRY_KG) - (1.0 / M0_KG)
KAPPA_SCALE_FACTOR = (R0_SI ** 2.5) / (MU_SI ** 1.5)
DEFAULT_SEED_PARAMS = rh.unpack([-8.33529969, -99.6312038, 0.43134401, 0.66967974])
NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
]


# -------------------------------------------------------
# 2. Dimensional Analysis Utilities
# -------------------------------------------------------


def get_grids():
    """Returns the defining axes of the Atlas."""
    rho_grid = np.logspace(np.log10(RHO_MIN), np.log10(RHO_MAX), N_RHO)
    kappa_grid = np.logspace(np.log10(KAPPA_MIN), np.log10(KAPPA_MAX), N_KAPPA)
    theta_grid = np.linspace(-THETA_MAX_REV * 2 * np.pi, THETA_MAX_REV * 2 * np.pi, N_THETA)
    return rho_grid, kappa_grid, theta_grid


def get_canonical_mission_config(rho, kappa):
    """
    Constructs a 'Canonical Mission' (starting at 1 AU) that
    physically represents the dimensionless point (rho, kappa).
    """
    # 1. Geometry
    r_target = R0_SI * rho

    # 2. Ship Capability
    # Reversing the kappa formula:
    # kappa = [2P * (1/m_dry - 1/m0)] * (r0^2.5 / mu^1.5)
    j_capacity = kappa / KAPPA_SCALE_FACTOR
    power = j_capacity / (2.0 * DELTA_INV_M)

    config = rh.TrajectoryConfig(
        mu=MU_SI,
        power=power,
        m_dry=M_DRY_KG,
        m0=M0_KG,
        r0=R0_SI,
        vr0=0.0,
        vtheta0=None,
    )

    return r_target, config


# -------------------------------------------------------
# 3. The Solver Kernel (Worker Function)
# -------------------------------------------------------

def worker_task(task_data):
    """
    The function executed by worker processes.
    Args:
        task_data: tuple (target_indices, rho_val, kappa_val, theta_val, seed_params, seed_time)
    Returns:
        (indices, success, params, time_days)
    """
    indices, rho, kappa, theta, seed_params, seed_time = task_data

    # Re-construct config inside worker (objects might not pickle perfectly otherwise)
    r_target_si, config = get_canonical_mission_config(rho, kappa)
    r_target_au = r_target_si / rh.AU

    # Heuristic guess if seed is None (only for anchor / retries without good timing)
    if seed_time is None:
        avg_r_au = 0.5 * (1.0 + r_target_au)
        period_days = 365.25 * (avg_r_au ** 1.5)
        # Using abs() because theta can be negative
        guess_time = period_days * (abs(theta) / (2 * np.pi))
        if guess_time < 10:
            guess_time = 50.0
    else:
        guess_time = seed_time

    # Default seed params
    if seed_params is None:
        seed_params = DEFAULT_SEED_PARAMS

    try:
        # Run solver
        # We use a modest max_nfev. If the seed is good (neighbor), it converges fast.
        # If the physics changed too much, we fail fast and let another successful neighbor retry.
        params, t_days, info = rh.solve_target_fast(
            r_target=r_target_au,
            theta_target=theta,
            seed_params=seed_params,
            t_guess_days=guess_time,
            n_starts=1,
            max_nfev=80,
            config=config,
        )

        # print(info.fun, info.success)
        success = info.success
        return (indices, success, params, t_days)

    except Exception:
        return (indices, False, None, None)


# -------------------------------------------------------
# 4. Parallel Wavefront Generator
# -------------------------------------------------------


def choose_chunksize(frontier_size, num_workers):
    """Heuristic chunksize to reduce scheduling overhead without starving workers."""
    if frontier_size <= 0:
        return 1

    target = frontier_size // max(1, num_workers * 4)
    return max(1, min(MAX_CHUNKSIZE, max(MIN_CHUNKSIZE, target)))


def make_task(indices, rho_grid, kappa_grid, theta_grid, seed_params, seed_time):
    """Build a solver task tuple from grid indices and seed information."""
    i, j, k = indices
    return (
        indices,
        rho_grid[i],
        kappa_grid[j],
        theta_grid[k],
        seed_params,
        seed_time,
    )


def in_bounds(i, j, k):
    return (0 <= i < N_RHO) and (0 <= j < N_KAPPA) and (0 <= k < N_THETA)


def generate():
    # Windows support for multiprocessing
    mp.freeze_support()

    rho_grid, kappa_grid, theta_grid = get_grids()

    # Tensor shape: (N_rho, N_kappa, N_theta, 6)
    data_shape = (N_RHO, N_KAPPA, N_THETA, 6)

    # Using float64 for numerical safety in downstream physics lookups
    atlas = np.full(data_shape, np.nan, dtype=np.float64)

    # State map and retry counters
    state = np.zeros(data_shape[:3], dtype=np.uint8)
    retry_count = np.zeros(data_shape[:3], dtype=np.uint8)

    print(f"[-] Initializing Parallel Atlas: {data_shape} points.")

    # --- Parallel Pool Setup ---
    num_workers = max(1, mp.cpu_count() - 1)  # Leave one for the OS/Manager
    print(f"[-] Spawning {num_workers} worker processes...")

    # --- Anchor Setup ---
    idx_rho_start = np.abs(rho_grid - 1.0).argmin()
    idx_kappa_start = np.abs(kappa_grid - 9.58).argmin()
    idx_theta_start = np.abs(theta_grid - np.deg2rad(-95.0)).argmin()
    start_node = (idx_rho_start, idx_kappa_start, idx_theta_start)

    frontier = [make_task(start_node, rho_grid, kappa_grid, theta_grid, None, None)]
    state[start_node] = STATE_QUEUED

    print("[-] Anchor queued. Starting frontier expansion...")

    total_points = N_RHO * N_KAPPA * N_THETA
    solved_count = 0
    frontier_round = 0
    failed_count = 0
    start_time = time.time()

    viz_dir = "atlas_state_plots"
    os.makedirs(viz_dir, exist_ok=True)
    last_solved_count, last_elapsed = 0,0

    with mp.Pool(processes=num_workers) as pool:
        while frontier:
            frontier_round += 1
            chunksize = choose_chunksize(len(frontier), num_workers)
            next_frontier = []

            for indices, success, params, t_days in pool.imap_unordered(worker_task, frontier, chunksize=chunksize):
                i, j, k = indices

                if success:
                    atlas[i, j, k, :] = np.append(params, t_days)
                    state[i, j, k] = STATE_SOLVED
                    solved_count += 1

                    for di, dj, dk in NEIGHBOR_OFFSETS:
                        ni, nj, nk = i + di, j + dj, k + dk
                        if not in_bounds(ni, nj, nk):
                            continue

                        neighbor_state = state[ni, nj, nk]
                        can_retry = (
                            neighbor_state == STATE_RETRYABLE_FAILED
                            and retry_count[ni, nj, nk] < MAX_RETRIES_PER_CELL
                        )

                        if neighbor_state == STATE_UNSEEN or can_retry:
                            state[ni, nj, nk] = STATE_QUEUED
                            next_frontier.append(
                                make_task(
                                    (ni, nj, nk),
                                    rho_grid,
                                    kappa_grid,
                                    theta_grid,
                                    params,
                                    t_days,
                                )
                            )
                else:
                    failed_count += 1
                    retry_count[i, j, k] += 1
                    if retry_count[i, j, k] <= MAX_RETRIES_PER_CELL:
                        state[i, j, k] = STATE_RETRYABLE_FAILED
                    else:
                        state[i, j, k] = STATE_DEAD_FAILED

            frontier = next_frontier

            if solved_count and (
                solved_count % PROGRESS_INTERVAL == 0 or frontier_round == 1 or not frontier
            ):
                elapsed = time.time() - start_time
                rate = solved_count / max(elapsed, 1e-9)
                rate_now = (solved_count - last_solved_count) / max(elapsed - last_elapsed, 1e-9)
                print(
                    "    "
                    f"Round: {frontier_round} | "
                    f"Solved: {solved_count} | "
                    f"Queued next: {len(frontier)} | "
                    f"Chunksize: {chunksize} | "
                    f"Rate: {rate:.1f} pts/s | "
                    f"Rate now: {rate_now:.1f} pts/s | "
                )
                last_solved_count, last_elapsed = solved_count, elapsed

                output_filename = "trajectory_atlas.npz"
                np.savez_compressed(
                    output_filename,
                    rho=rho_grid,
                    kappa=kappa_grid,
                    theta=theta_grid,
                    data=atlas,
                )

                plot_filename = os.path.join(viz_dir, f"atlas_state_round_{frontier_round:05d}.png")
                visualize_atlas_state_slices(
                    state=state,
                    rho_grid=rho_grid,
                    kappa_grid=kappa_grid,
                    theta_grid=theta_grid,
                    round_idx=frontier_round,
                    output_path=plot_filename,
                    show=False,
                )

    elapsed = time.time() - start_time
    print(f"\n[+] Parallel Atlas Generation Complete in {elapsed:.1f}s")
    print(f"[+] Coverage: {solved_count}/{total_points} ({solved_count / total_points * 100:.1f}%)")
    print(f"[+] Failed solver calls: {failed_count}")

    output_filename = "trajectory_atlas_final.npz"
    np.savez_compressed(
        output_filename,
        rho=rho_grid,
        kappa=kappa_grid,
        theta=theta_grid,
        data=atlas,
    )
    print(f"[+] Saved to {output_filename}")


def _compute_edges_from_centers(values, log_spacing=False):
    """
    Convert 1D cell centers to cell edges for pcolormesh.
    Works well for monotonic grids.
    """
    values = np.asarray(values, dtype=float)

    if values.ndim != 1 or len(values) < 2:
        raise ValueError("values must be a 1D array with at least 2 elements")

    if log_spacing:
        if np.any(values <= 0):
            raise ValueError("log-spaced edges require strictly positive values")

        edges = np.empty(len(values) + 1, dtype=float)
        edges[1:-1] = np.sqrt(values[:-1] * values[1:])
        edges[0] = values[0] ** 2 / edges[1]
        edges[-1] = values[-1] ** 2 / edges[-2]
    else:
        edges = np.empty(len(values) + 1, dtype=float)
        edges[1:-1] = 0.5 * (values[:-1] + values[1:])
        edges[0] = values[0] - 0.5 * (values[1] - values[0])
        edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])

    return edges


def visualize_atlas_state_slices(
    state,
    rho_grid,
    kappa_grid,
    theta_grid,
    round_idx=None,
    output_path=None,
    nrows=3,
    ncols=4,
    figsize=(16, 10),
    dpi=150,
    show=False,
):
    """
    Visualize the 3D atlas state as a grid of 2D (rho, kappa) cuts at selected theta slices.

    Parameters
    ----------
    state : ndarray, shape (N_RHO, N_KAPPA, N_THETA)
        Integer state tensor using the STATE_* codes.
    rho_grid, kappa_grid, theta_grid : 1D ndarrays
        Grid center coordinates from get_grids().
    round_idx : int or None
        Optional round number for the figure title.
    output_path : str or None
        If provided, save the figure here.
    nrows, ncols : int
        Layout of subplot grid.
    figsize : tuple
        Matplotlib figure size.
    dpi : int
        Figure DPI for saving.
    show : bool
        Whether to display interactively.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of Axes
    """
    # --- Validate shapes ---
    expected_shape = (len(rho_grid), len(kappa_grid), len(theta_grid))
    if state.shape != expected_shape:
        raise ValueError(f"state.shape={state.shape}, expected {expected_shape}")

    # --- Discrete colormap for your state machine ---
    # 0 unseen, 1 queued, 2 solved, 3 retryable_failed, 4 dead_failed
    cmap = ListedColormap([
        "#f0f0f0",  # unseen
        "#4c78a8",  # queued
        "#54a24b",  # solved
        "#f2cf5b",  # retryable failed
        "#e45756",  # dead failed
    ])
    norm = BoundaryNorm(np.arange(-0.5, 5.5, 1.0), cmap.N)

    state_names = {
        0: "unseen",
        1: "queued",
        2: "solved",
        3: "retryable failed",
        4: "dead failed",
    }

    # --- Choose theta slices evenly across the theta axis ---
    n_panels = nrows * ncols
    if len(theta_grid) <= n_panels:
        theta_indices = np.arange(len(theta_grid))
    else:
        theta_indices = np.linspace(0, len(theta_grid) - 1, n_panels, dtype=int)

    # --- Cell edges for pcolormesh ---
    rho_edges = _compute_edges_from_centers(rho_grid, log_spacing=True)
    kappa_edges = _compute_edges_from_centers(kappa_grid, log_spacing=True)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    for panel_idx, ax in enumerate(axes):
        if panel_idx >= len(theta_indices):
            ax.set_visible(False)
            continue

        k = theta_indices[panel_idx]

        # state[:, :, k] has shape (N_RHO, N_KAPPA)
        # pcolormesh expects Z shape (len(y)-1, len(x)-1), so transpose to (N_KAPPA, N_RHO)
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
        ax.set_title(
            f"theta[{k}] = {theta_grid[k]:.2f} rad\n({np.degrees(theta_grid[k]):.1f}°)"
        )

    # One shared colorbar
    cbar = fig.colorbar(
        mesh,
        ax=axes.tolist(),
        ticks=[0, 1, 2, 3, 4],
        shrink=0.92,
        pad=0.02,
    )
    cbar.ax.set_yticklabels([state_names[i] for i in range(5)])
    cbar.set_label("Cell state")

    # Overall title
    solved = np.count_nonzero(state == 2)
    queued = np.count_nonzero(state == 1)
    retryable = np.count_nonzero(state == 3)
    dead = np.count_nonzero(state == 4)
    unseen = np.count_nonzero(state == 0)
    total = state.size

    title = (
        f"Atlas state slices"
        + (f" — round {round_idx}" if round_idx is not None else "")
        + f"\nsolved={solved}, queued={queued}, retryable={retryable}, dead={dead}, unseen={unseen}, total={total}"
    )
    fig.suptitle(title, fontsize=14)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes


if __name__ == "__main__":
    generate()
