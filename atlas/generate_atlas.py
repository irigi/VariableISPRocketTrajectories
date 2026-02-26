"""
generate_atlas.py

Generates the "Time-Optimal Trajectory Atlas" (TOTA).
This script explores the 3D parameter space (Radius Ratio, Capability, Angle)
using a parallelized 3D flood-fill (wavefront) strategy.

Output: 'trajectory_atlas.npz'
"""

import time
import multiprocessing as mp

import numpy as np

import rocketHamilton as rh


# -------------------------------------------------------
# 1. Grid Configuration
# -------------------------------------------------------

# Radius Ratio (rho = r_target / r_start)
RHO_MIN, RHO_MAX = 0.05, 100.0
N_RHO = 40

# Capability Parameter (kappa)
KAPPA_MIN, KAPPA_MAX = 0.1, 200000.0
N_KAPPA = 30

# Angle (theta) in Radians
THETA_MAX_REV = 1.1
N_THETA = 60

# Retry / batching controls
MAX_RETRIES_PER_CELL = 1
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

    with mp.Pool(processes=num_workers) as pool:
        while frontier:
            frontier_round += 1
            chunksize = choose_chunksize(len(frontier), num_workers)
            next_frontier = []

            for indices, success, params, t_days in pool.imap_unordered(
                worker_task,
                frontier,
                chunksize=chunksize,
            ):
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
                print(
                    "    "
                    f"Round: {frontier_round} | "
                    f"Solved: {solved_count} | "
                    f"Queued next: {len(frontier)} | "
                    f"Chunksize: {chunksize} | "
                    f"Rate: {rate:.1f} pts/s"
                )

    elapsed = time.time() - start_time
    print(f"\n[+] Parallel Atlas Generation Complete in {elapsed:.1f}s")
    print(f"[+] Coverage: {solved_count}/{total_points} ({solved_count / total_points * 100:.1f}%)")
    print(f"[+] Failed solver calls: {failed_count}")

    output_filename = "trajectory_atlas.npz"
    np.savez_compressed(
        output_filename,
        rho=rho_grid,
        kappa=kappa_grid,
        theta=theta_grid,
        data=atlas,
    )
    print(f"[+] Saved to {output_filename}")


if __name__ == "__main__":
    generate()
