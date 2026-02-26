"""
generate_atlas.py

Generates the "Time-Optimal Trajectory Atlas" (TOTA).
This script explores the 3D parameter space (Radius Ratio, Capability, Angle)
using a wavefront propagation strategy to pre-compute optimal controls.

Output: 'trajectory_atlas.npz'
"""

import numpy as np
import time
import sys
from scipy.interpolate import RegularGridInterpolator

import rocketHamilton as rh


# -------------------------------------------------------
# 1. Grid Configuration (Physics Verification Applied)
# -------------------------------------------------------
# Adjusted based on the "Neptune High-Power" case analysis.

# Radius Ratio (rho = r_target / r_start)
# 0.05 (Sun dive) to 100.0 (Kuiper Belt)
RHO_MIN, RHO_MAX = 0.05, 100.0
N_RHO = 40

# Capability Parameter (kappa)
# Range expanded to 200,000 to cover GW-class ships at outer planets.
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
    theta_grid = np.linspace(-THETA_MAX_REV * 2 * np.pi, THETA_MAX_REV * 2 * np.pi, N_THETA)  # Start slightly > 0
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
        vtheta0=None  # Defaults to circular
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

    # Heuristic guess if none provided (Basic Hohmann-ish time)
    if guess_time is None:
        # Simple heuristic: Time ~ distance^1.5
        avg_r_au = 0.5 * (1.0 + r_target_au)
        period_days = 365.25 * (avg_r_au ** 1.5)
        guess_time = period_days * (theta_target / (2 * np.pi))
        if guess_time < 10: guess_time = 50.0

    # Default seed params if none provided (from user's SOLUTION0)
    if guess_params is None:
        guess_params = rh.unpack([-8.33529969, -99.6312038, 0.43134401, 0.66967974])

        # Run the user's fast solver
    # We reduce max_nfev because we expect good guesses from neighbors
    try:
        params, t_days, info = rh.solve_target_fast(
            r_target=r_target_au,
            theta_target=theta_target,
            seed_params=guess_params,
            t_guess_days=guess_time,
            n_starts=1,  # Single start because we trust the homotopy guess
            max_nfev=60,
            config=config
        )

        success = np.linalg.norm(info.fun) < 1e-3
        return success, params, t_days

    except NotImplementedError as e:
        return False, None, None


# -------------------------------------------------------
# 4. Wavefront Propagation Generator
# -------------------------------------------------------

def generate():
    rho_grid, kappa_grid, theta_grid = get_grids()

    # Tensor shape: (N_rho, N_kappa, N_theta, DATA_DIM)
    # Data: [lambda_r, lambda_vr, lambda_vtheta, C_m, C_theta, t_flight_days]
    # We store 6 values.
    # Note: These are specific to the "Canonical Mission" (1 AU start).
    data_shape = (N_RHO, N_KAPPA, N_THETA, 6)
    atlas = np.full(data_shape, np.nan)

    print(f"[-] Initializing Atlas: {data_shape} points.")
    print(f"[-] Grid Bounds: Rho[{RHO_MIN}-{RHO_MAX}], Kappa[{KAPPA_MIN:.1f}-{KAPPA_MAX:.1f}]")

    start_time = time.time()
    total_points = N_RHO * N_KAPPA * N_THETA
    solved_count = 0

    idx_rho_start = np.abs(rho_grid - 1.0).argmin()
    idx_kappa_start = np.abs(kappa_grid - 9.58).argmin()
    idx_theta_start = np.abs(theta_grid - np.deg2rad(-95.0)).argmin()

    print(f"[-] Starting Anchor Column at indices [{idx_rho_start}, {idx_kappa_start}, {idx_theta_start}]...")

    last_params = None
    last_time = None
    success, params, t_days = solve_point(
        rho_grid[idx_rho_start],
        kappa_grid[idx_kappa_start],
        theta_grid[idx_theta_start],
        last_params,
        last_time
    )

    if success:
        sol_vec = np.append(params, t_days)
        atlas[idx_rho_start, idx_kappa_start, idx_theta_start, :] = sol_vec
        last_params, last_time = params, t_days
        solved_count += 1
    else:
        print(f"[!] Anchor failed")

    # --- B. Wavefront Propagation ---
    # We expand outwards from the anchor in concentric "shells" of radius d
    # Distance metric: simple Manhattan distance in grid indices

    # Create a list of all (i, j) coordinates sorted by distance from center
    coords = []
    for i in range(N_RHO):
        for j in range(N_KAPPA):
            if i == idx_rho_start and j == idx_kappa_start: continue
            dist = abs(i - idx_rho_start) + abs(j - idx_kappa_start)
            coords.append((dist, i, j))

    # Sort by distance to grow outwards
    coords.sort()

    print("[-] Starting Wavefront Propagation...")

    for dist, i, j in coords:
        rho = rho_grid[i]
        kappa = kappa_grid[j]

        # Angle Loop
        # We perform the angle loop for this (rho, kappa) pair.
        # Crucial: We get our initial guess from a NEIGHBOR in (rho, kappa) space.

        # 1. Find a valid neighbor to seed the theta=0 start
        seed_params = None
        seed_time = None

        # Check neighbors (i-1, j), (i+1, j), (i, j-1), etc.
        # We prioritize neighbors that are closer to the anchor (already solved)
        potential_seeds = [
            (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)
        ]

        for ni, nj in potential_seeds:
            if 0 <= ni < N_RHO and 0 <= nj < N_KAPPA:
                # Check if neighbor has a solution at first angle
                if not np.isnan(atlas[ni, nj, 0, 0]):
                    # Valid seed found!
                    seed_params = atlas[ni, nj, 0, :5]
                    seed_time = atlas[ni, nj, 0, 5]
                    break

        # If no neighbor (shouldn't happen with sorted expansion), use default

        # 2. Run the Angular Thread
        # We reuse the previous angle's solution as we step through k
        curr_params = seed_params
        curr_time = seed_time

        for k in range(N_THETA):
            theta = theta_grid[k]

            # If previous angle failed, we can try to recover using a spatial neighbor
            # at this specific angle k (Cross-linking the mesh)
            if curr_params is None:
                for ni, nj in potential_seeds:
                    if 0 <= ni < N_RHO and 0 <= nj < N_KAPPA:
                        if not np.isnan(atlas[ni, nj, k, 0]):
                            curr_params = atlas[ni, nj, k, :5]
                            curr_time = atlas[ni, nj, k, 5]
                            break

            # Solve
            success, params, t_days = solve_point(rho, kappa, theta, curr_params, curr_time)

            if success:
                sol_vec = np.append(params, t_days)
                atlas[i, j, k, :] = sol_vec
                curr_params = params
                curr_time = t_days
                solved_count += 1
            else:
                # If we fail, we leave as NaN.
                # Future points in the angle loop might recover via spatial neighbors,
                # but usually failure implies a physical limit (e.g. max thrust exceeded).
                curr_params = None

        if i % 5 == 0 and j % 5 == 0:
            print(f"    Processed Grid ({i}, {j}). Solved: {solved_count}/{total_points}")

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
