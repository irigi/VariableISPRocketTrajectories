"""
generate_atlas.py

Generates the "Time-Optimal Trajectory Atlas" (TOTA).
This script explores the 3D parameter space (Radius Ratio, Capability, Angle)
using a 3D flood-fill (wavefront) strategy to pre-compute optimal controls.

Output: 'trajectory_atlas.npz'
"""

import numpy as np
import time
import sys
import multiprocessing as mp
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from scipy.interpolate import RegularGridInterpolator

import rocketHamilton as rh


# -------------------------------------------------------
# 1. Grid Configuration (Physics Verification Applied)
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
    # We fix m0, m_dry, and solve for Power P to match kappa.

    m0 = 3000.0  # kg (Arbitrary, cancels out in dimensionless form)
    m_dry = 1000.0  # kg
    delta_inv_m = (1.0 / m_dry) - (1.0 / m0)

    scale_factor = (r0 ** 2.5) / (mu ** 1.5)

    # Required J_capacity (integral a^2 dt)
    J_capacity = kappa / scale_factor

    # Required Power
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
# 3. The Solver Kernel
# -------------------------------------------------------

def solve_point(rho, kappa, theta_target, guess_params=None, guess_time=None):
    """
    Solves a single point in the grid using the user's rocketHamilton solver.
    Returns: (success, params_array, time_days)
    """
    r_target_si, config = get_canonical_mission_config(rho, kappa)
    r_target_au = r_target_si / rh.AU

    # Heuristic guess if none provided
    if guess_time is None:
        avg_r_au = 0.5 * (1.0 + r_target_au)
        period_days = 365.25 * (avg_r_au ** 1.5)
        guess_time = period_days * (abs(theta_target) / (2 * np.pi)) # abs() for symmetric grid
        if guess_time < 10: guess_time = 50.0

    # Default seed params if none provided (from user's SOLUTION0)
    if guess_params is None:
        guess_params = rh.unpack([-8.33529969, -99.6312038, 0.43134401, 0.66967974])

    try:
        # We reduce max_nfev because we expect good guesses from neighbors
        params, t_days, info = rh.solve_target_fast(
            r_target=r_target_au,
            theta_target=theta_target,
            seed_params=guess_params,
            t_guess_days=guess_time,
            n_starts=1,  # Single start because we trust the homotopy guess
            max_nfev=60,
            config=config
        )

        success = info.success
        return success, params, t_days

    except Exception:
        return False, None, None


def _solve_neighbor_task(args):
    """Worker helper for solving a neighbor atlas point in a spawned process."""
    ni, nj, nk, n_rho, n_kappa, n_theta, curr_params, curr_time = args
    success, n_params, n_time = solve_point(
        n_rho,
        n_kappa,
        n_theta,
        guess_params=curr_params,
        guess_time=curr_time
    )
    return ni, nj, nk, success, n_params, n_time


# -------------------------------------------------------
# 4. Wavefront Propagation Generator
# -------------------------------------------------------

def generate():
    rho_grid, kappa_grid, theta_grid = get_grids()

    # Tensor shape: (N_rho, N_kappa, N_theta, 6)
    # Data: [lambda_r, lambda_vr, lambda_vtheta, C_m, C_theta, t_flight_days]
    data_shape = (N_RHO, N_KAPPA, N_THETA, 6)
    atlas = np.full(data_shape, np.nan)

    # Track visited status separately to distinguish between "not reached" and "failed"
    visited = np.zeros(data_shape[:3], dtype=bool)

    print(f"[-] Initializing Atlas: {data_shape} points.")
    print(f"[-] Grid Bounds: Rho[{RHO_MIN}-{RHO_MAX}], Kappa[{KAPPA_MIN:.1f}-{KAPPA_MAX:.1f}]")

    start_time = time.time()
    total_points = N_RHO * N_KAPPA * N_THETA
    solved_count = 0

    # --- Anchor Setup ---
    # Find the indices for the seed point (Saturn-ish transfer)
    idx_rho_start = np.abs(rho_grid - 1.0).argmin()
    idx_kappa_start = np.abs(kappa_grid - 9.58).argmin()
    idx_theta_start = np.abs(theta_grid - np.deg2rad(-95.0)).argmin()

    print(f"[-] Starting Anchor at indices [{idx_rho_start}, {idx_kappa_start}, {idx_theta_start}]...")

    # Solve the anchor first
    rho_start = rho_grid[idx_rho_start]
    kappa_start = kappa_grid[idx_kappa_start]
    theta_start = theta_grid[idx_theta_start]

    success, params, t_days = solve_point(rho_start, kappa_start, theta_start)

    # Initialize Queue for the wave
    # Queue stores tuples: (i, j, k)
    queue = deque()

    if success:
        sol_vec = np.append(params, t_days)
        atlas[idx_rho_start, idx_kappa_start, idx_theta_start, :] = sol_vec
        visited[idx_rho_start, idx_kappa_start, idx_theta_start] = True
        queue.append((idx_rho_start, idx_kappa_start, idx_theta_start))
        solved_count += 1
        print("[-] Anchor Solved. Starting 3D Wave...")
    else:
        print("[!] Anchor failed. Cannot start propagation.")
        return

    # --- 3D Flood Fill Loop ---

    # Neighbors: 6 directions in 3D (up/down/left/right/forward/back)
    directions = [
        (1, 0, 0), (-1, 0, 0),  # Rho neighbors
        (0, 1, 0), (0, -1, 0),  # Kappa neighbors
        (0, 0, 1), (0, 0, -1)   # Theta neighbors
    ]

    max_workers = max(1, mp.cpu_count() - 1)
    spawn_context = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=spawn_context) as executor:
        while queue:
            # Pop the seed node
            curr_i, curr_j, curr_k = queue.popleft()

            # Get the solution at the current node (to use as guess for neighbors)
            curr_sol = atlas[curr_i, curr_j, curr_k]
            curr_params = curr_sol[:5]
            curr_time = curr_sol[5]

            neighbor_tasks = []

            # Try to expand to all immediate neighbors
            for di, dj, dk in directions:
                ni, nj, nk = curr_i + di, curr_j + dj, curr_k + dk

                # 1. Check Bounds
                if not (0 <= ni < N_RHO and 0 <= nj < N_KAPPA and 0 <= nk < N_THETA):
                    continue

                # 2. Check if already visited (solved or attempted & failed)
                if visited[ni, nj, nk]:
                    continue

                # 3. Mark as visited immediately to prevent duplicates in queue
                visited[ni, nj, nk] = True

                # 4. Queue neighbor solve task (using current node's solution as seed guess)
                neighbor_tasks.append(
                    (ni, nj, nk, rho_grid[ni], kappa_grid[nj], theta_grid[nk], curr_params, curr_time)
                )

            if neighbor_tasks:
                results = executor.map(_solve_neighbor_task, neighbor_tasks)

                for ni, nj, nk, success, n_params, n_time in results:
                    if success:
                        # Store solution
                        n_sol_vec = np.append(n_params, n_time)
                        atlas[ni, nj, nk, :] = n_sol_vec

                        # Add to queue to propagate further
                        queue.append((ni, nj, nk))
                        solved_count += 1

                    # If failed: do nothing.
                    # It is marked 'visited', so we won't try again.
                    # We do NOT add to queue, so the wave stops in this direction.

            # Progress logging
            if solved_count % 10 == 0:
                 print(f"    Solved: {solved_count}/{total_points} (Queue size: {len(queue)})")

    # -------------------------------------------------------
    # 5. Save
    # -------------------------------------------------------
    output_filename = 'trajectory_atlas.npz'
    np.savez_compressed(
        output_filename,
        rho=rho_grid,
        kappa=kappa_grid,
        theta=theta_grid,
        data=atlas
    )

    elapsed = time.time() - start_time
    print(f"\n[+] Atlas Generation Complete in {elapsed:.1f}s")
    print(f"[+] Saved to {output_filename}")
    print(f"[+] Coverage: {solved_count / total_points * 100:.1f}% of grid solved.")


if __name__ == "__main__":
    generate()
