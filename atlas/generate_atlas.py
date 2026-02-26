"""
generate_atlas.py

Generates the "Time-Optimal Trajectory Atlas" (TOTA).
This script explores the 3D parameter space (Radius Ratio, Capability, Angle)
using a parallelized 3D flood-fill (wavefront) strategy.

Output: 'trajectory_atlas.npz'
"""

import numpy as np
import time
import sys
import multiprocessing as mp
from collections import deque

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
    # Fix Reference scales
    r0 = rh.AU
    mu = rh.MU_SI

    # 1. Geometry
    r_target = r0 * rho

    # 2. Ship Capability
    # Reversing the kappa formula:
    # kappa = [2P * (1/m_dry - 1/m0)] * (r0^2.5 / mu^1.5)

    m0 = 3000.0  # kg (Arbitrary, cancels out in dimensionless form)
    m_dry = 1000.0  # kg
    delta_inv_m = (1.0 / m_dry) - (1.0 / m0)

    scale_factor = (r0 ** 2.5) / (mu ** 1.5)
    J_capacity = kappa / scale_factor
    P = J_capacity / (2.0 * delta_inv_m)

    config = rh.TrajectoryConfig(
        mu=mu,
        power=P,
        m_dry=m_dry,
        m0=m0,
        r0=r0,
        vr0=0.0,
        vtheta0=None
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
    (indices, rho, kappa, theta, seed_params, seed_time) = task_data

    # Re-construct config inside worker (objects might not pickle perfectly otherwise)
    r_target_si, config = get_canonical_mission_config(rho, kappa)
    r_target_au = r_target_si / rh.AU

    # Heuristic guess if seed is None (only for anchor)
    if seed_time is None:
        avg_r_au = 0.5 * (1.0 + r_target_au)
        period_days = 365.25 * (avg_r_au ** 1.5)
        # Using abs() because theta can be negative
        guess_time = period_days * (abs(theta) / (2 * np.pi))
        if guess_time < 10: guess_time = 50.0
    else:
        guess_time = seed_time

    # Default seed params
    if seed_params is None:
        # User's default solution
        seed_params = rh.unpack([-8.33529969, -99.6312038, 0.43134401, 0.66967974])

    try:
        # Run solver
        # We use a modest max_nfev. If the seed is good (neighbor), it converges fast.
        # If the physics changed too much, we fail fast and stop the wave in that direction.
        params, t_days, info = rh.solve_target_fast(
            r_target=r_target_au,
            theta_target=theta,
            seed_params=seed_params,
            t_guess_days=guess_time,
            n_starts=1,
            max_nfev=80,
            config=config
        )

        success = info.success

        return (indices, success, params, t_days)

    except Exception:
        return (indices, False, None, None)


# -------------------------------------------------------
# 4. Parallel Wavefront Generator
# -------------------------------------------------------

def generate():
    # Windows support for multiprocessing
    mp.freeze_support()

    rho_grid, kappa_grid, theta_grid = get_grids()

    # Tensor shape: (N_rho, N_kappa, N_theta, 6)
    data_shape = (N_RHO, N_KAPPA, N_THETA, 6)

    # Shared Array for results (optional, but we'll use main memory to collect)
    # Using float32 to save RAM if grids get huge, but float64 is safer for physics
    atlas = np.full(data_shape, np.nan, dtype=np.float64)

    # Visited Map: 0 = Unvisited, 1 = In Progress/Solved/Failed
    visited = np.zeros(data_shape[:3], dtype=bool)

    print(f"[-] Initializing Parallel Atlas: {data_shape} points.")

    # --- Parallel Pool Setup ---
    num_workers = mp.cpu_count() - 1  # Leave one for the OS/Manager
    if num_workers < 1: num_workers = 1
    print(f"[-] Spawning {num_workers} worker processes...")

    pool = mp.Pool(processes=num_workers)

    # --- Anchor Setup ---
    # We find indices for a known good starting point
    idx_rho_start = np.abs(rho_grid - 1.0).argmin()
    idx_kappa_start = np.abs(kappa_grid - 9.58).argmin()
    idx_theta_start = np.abs(theta_grid - np.deg2rad(-95.0)).argmin()

    start_node = (idx_rho_start, idx_kappa_start, idx_theta_start)

    # We prime the pump with the anchor task
    # (indices, rho, kappa, theta, seed_params, seed_time)
    anchor_task = (
        start_node,
        rho_grid[idx_rho_start],
        kappa_grid[idx_kappa_start],
        theta_grid[idx_theta_start],
        None, None
    )

    # Queue stores FUTURE tasks to be computed
    # But since we need the results of neighbors to create tasks,
    # we actually manage the "Frontier" here.

    # 1. Submit Anchor
    visited[start_node] = True
    active_jobs = []

    # Submit asynchronous job
    job = pool.apply_async(worker_task, (anchor_task,))
    active_jobs.append(job)

    print(f"[-] Anchor submitted. Starting Event Loop...")

    total_points = N_RHO * N_KAPPA * N_THETA
    solved_count = 0
    start_time = time.time()

    # Directions for 3D neighbors
    neighbor_offsets = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]

    # --- Event Loop ---
    # We continually check for finished jobs, harvest results,
    # and spawn new jobs for their neighbors.

    while active_jobs:
        # Check for completed jobs
        # We iterate backwards to allow removing items safely
        still_active = []
        new_tasks = []

        for job in active_jobs:
            if job.ready():
                # Harvest Result
                try:
                    indices, success, params, t_days = job.get()
                    i, j, k = indices

                    if success:
                        # 1. Save Result
                        sol_vec = np.append(params, t_days)
                        atlas[i, j, k, :] = sol_vec
                        solved_count += 1

                        # 2. Identify Neighbors
                        for di, dj, dk in neighbor_offsets:
                            ni, nj, nk = i + di, j + dj, k + dk

                            # Check Bounds
                            if (0 <= ni < N_RHO) and (0 <= nj < N_KAPPA) and (0 <= nk < N_THETA):
                                if not visited[ni, nj, nk]:
                                    # Mark visited immediately so no other job claims it
                                    visited[ni, nj, nk] = True

                                    # Prepare Task
                                    task = (
                                        (ni, nj, nk),
                                        rho_grid[ni],
                                        kappa_grid[nj],
                                        theta_grid[nk],
                                        params, # Use parent's params as seed
                                        t_days  # Use parent's time as seed
                                    )
                                    new_tasks.append(task)
                    else:
                        # Failed to converge.
                        # We do nothing. The wavefront stops here.
                        pass

                except Exception as e:
                    print(f"[!] Job failed with error: {e}")
            else:
                still_active.append(job)

        # Dispatch New Tasks
        if new_tasks:
            # Batch submission logic could go here if overhead is high,
            # but apply_async is usually fast enough for N=70k.
            for task in new_tasks:
                job = pool.apply_async(worker_task, (task,))
                still_active.append(job)

        active_jobs = still_active

        # Logging & Throttling
        if solved_count % 100 == 0 and len(new_tasks) > 0:
            elapsed = time.time() - start_time
            rate = solved_count / (elapsed + 1e-5)
            print(f"    Solved: {solved_count} | Active Workers: {len(active_jobs)} | Rate: {rate:.1f} pts/s")

        # Prevent CPU spin if waiting for long jobs
        if not new_tasks and active_jobs:
            time.sleep(0.05)

    # --- Shutdown ---
    pool.close()
    pool.join()

    # --- Save ---
    elapsed = time.time() - start_time
    print(f"\n[+] Parallel Atlas Generation Complete in {elapsed:.1f}s")
    print(f"[+] Coverage: {solved_count}/{total_points} ({solved_count/total_points*100:.1f}%)")

    output_filename = 'trajectory_atlas.npz'
    np.savez_compressed(
        output_filename,
        rho=rho_grid,
        kappa=kappa_grid,
        theta=theta_grid,
        data=atlas
    )
    print(f"[+] Saved to {output_filename}")


if __name__ == "__main__":
    generate()
