"""
trajectory_loader.py

The runtime interface for the Time-Optimal Trajectory Atlas.
Loads pre-computed solutions and scales them to specific mission parameters
to provide "Instant" warm-starts for the optimizer.
"""

import numpy as np
import os
import sys
from scipy.interpolate import RegularGridInterpolator
import atlas_utils as utils

# Import user's physics solver
try:
    import rocketHamilton as rh
except ImportError:
    print("Error: rocketHamilton.py must be in the same directory.")
    sys.exit(1)

ATLAS_FILENAME = 'trajectory_atlas.npz'


class TrajectoryAtlas:
    def __init__(self, atlas_path=ATLAS_FILENAME):
        if not os.path.exists(atlas_path):
            raise FileNotFoundError(f"Atlas file '{atlas_path}' not found. Run generate_atlas.py first.")

        print(f"[-] Loading Trajectory Atlas from {atlas_path}...")
        raw = np.load(atlas_path)

        self.rho_grid = raw['rho']
        self.kappa_grid = raw['kappa']
        self.theta_grid = raw['theta']
        self.data = raw['data']  # Shape: (N_rho, N_kappa, N_theta, 6)

        # Create interpolator
        # We use nearest-neighbor or linear. Linear is safer for continuous gradients.
        # But we must handle NaNs (failed solutions).
        # Strategy: We interpolate; if we hit NaNs, the solver might struggle,
        # but the 'smart guess' usually ensures we are in valid regions.
        self.interp = RegularGridInterpolator(
            (self.rho_grid, self.kappa_grid, self.theta_grid),
            self.data,
            method='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        print("[-] Atlas loaded successfully.")

    def get_scaled_guess(self, r0, r_target, P, m0, m_dry, theta_target):
        """
        Retrieves an initial guess from the Atlas and scales it to the
        physical dimensions of the requested mission.
        """
        # 1. Map to Dimensionless Atlas Coordinates
        rho, kappa = utils.physical_to_dimensionless(P, m0, m_dry, r0, r_target)

        # 2. Handle Angle Wrapping
        # The Atlas stores 0 to 6*pi. The user might ask for 45 deg (0.78 rad).
        # We need to find the "best" revolution count.
        # For now, we assume the user wants the simplest transfer (k=0)
        # unless they specify a large angle.
        # Ideally, we should check k=0, 1, 2 revolutions and pick the fastest valid one.
        # Here we just look up the specific angle requested.

        query_point = np.array([rho, kappa, theta_target])

        # 3. Interpolate Canonical Solution
        # result is [lam_r, lam_vr, lam_vth, Cm, Cth, t_days]
        canonical_vec = self.interp(query_point)[0]

        if np.isnan(canonical_vec).any():
            return False, None, None

        # 4. Scaling Laws (Canonical -> Physical)
        # The Atlas was generated with:
        #   R_ref = 1.0 AU
        #   M_ref = 3000.0 kg
        #   T_ref derived from R_ref (Gaussian)

        R_c = utils.AU
        M_c = 3000.0

        # Scaling Factors (S_L = Length, S_M = Mass, S_T = Time)
        S_L = r0 / R_c
        S_M = m0 / M_c
        S_T = S_L ** 1.5  # Keplerian time scaling

        # Unpack Canonical Vector
        lam_r_c, lam_vr_c, lam_vth_c, Cm_c, Cth_c, t_days_c = canonical_vec

        # Apply Scaling based on adjoint units (H + 1 = 0 formulation)
        # lam_r  ~ T/L   -> scale by S_T / S_L = S_L^0.5
        # lam_v  ~ T^2/L -> scale by S_T^2 / S_L = S_L^2
        # Cm     ~ M*T   -> scale by S_M * S_T
        # Cth    ~ T     -> scale by S_T
        # Time   ~ T     -> scale by S_T

        lam_r = lam_r_c * (S_L ** 0.5)
        lam_vr = lam_vr_c * (S_L ** 2.0)
        lam_vth = lam_vth_c * (S_L ** 2.0)
        Cm = Cm_c * (S_M * S_T)
        Cth = Cth_c * (S_T)
        t_days = t_days_c * (S_T)  # Time in days scales same as time in seconds

        guess_params = np.array([lam_r, lam_vr, lam_vth, Cm, Cth])

        return True, guess_params, t_days

    def solve_mission(self, r0_au, r_target_au, P, m0, m_dry, theta_target_deg, optimize_revolutions=True):
        """
        The "Instant Solver" main entry point.
        """
        r0 = r0_au * utils.AU
        r_target = r_target_au * utils.AU
        theta_rad = np.radians(theta_target_deg)

        print(f"\n--- Instant Solver Request ---")
        print(f"Transfer: {r0_au} AU -> {r_target_au} AU")
        print(f"Ship: {P / 1e6:.1f} MW, {m0 / 1000:.1f}t -> {m_dry / 1000:.1f}t")

        # Determine Angles to check (0, 1, 2 revolutions?)
        candidates = []
        if optimize_revolutions:
            # Check base angle, +1 rev, +2 revs
            revs_to_check = [0, 1, 2]
        else:
            revs_to_check = [0]

        for k in revs_to_check:
            angle = theta_rad + (2 * np.pi * k)

            found, params, t_guess = self.get_scaled_guess(r0, r_target, P, m0, m_dry, angle)

            if found:
                print(f"[-] Found Atlas guess for {k} revs ({np.degrees(angle):.1f} deg). T_est={t_guess:.1f}d")
                candidates.append((t_guess, angle, params))
            else:
                print(f"[!] No Atlas solution for {k} revs (Out of bounds or physically impossible).")

        if not candidates:
            print("[!] Atlas lookup failed for all requested revolutions.")
            return None

        # Pick the fastest candidate as the primary guess
        candidates.sort(key=lambda x: x[0])  # Sort by time
        best_t, best_angle, best_params = candidates[0]

        print(f"[-] Selected best candidate: {np.degrees(best_angle):.1f} deg, ~{best_t:.1f} days.")
        print("[-] Fine-tuning with physics solver...")

        # Setup Physics Config
        config = rh.TrajectoryConfig(
            mu=rh.MU_SI, power=P, m_dry=m_dry, m0=m0,
            r0=r0, vr0=0.0, vtheta0=None
        )

        # FINAL SOLVE
        # We use n_starts=1 because the guess should be excellent.
        try:
            params_opt, t_final, info = rh.solve_target_fast(
                r_target=r_target_au,
                theta_target=best_angle,
                seed_params=best_params,
                t_guess_days=best_t,
                n_starts=1,
                max_nfev=100,  # Should converge fast
                config=config
            )

            print(f"[+] CONVERGED! Final Flight Time: {t_final:.2f} days")
            return params_opt, t_final, info

        except Exception as e:
            print(f"[!] Solver Error: {e}")
            return None


# Simple CLI test
if __name__ == "__main__":
    loader = TrajectoryAtlas()

    # Test: The user's original Earth -> Mars case (approx)
    loader.solve_mission(
        r0_au=1.0, r_target_au=1.52,
        P=1e9, m0=3e6, m_dry=1e6,
        theta_target_deg=60.0
    )
