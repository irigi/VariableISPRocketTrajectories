"""
Constant-power, variable-Isp heliocentric transfers (indirect/PMP solver)

This script implements the time-optimal control formulation from
“Time-Optimal Heliocentric Transfers With a Constant-Power, Variable-Isp Engine”.

Model & equations referenced in comments use the paper’s notation:

State      x = (r, θ, v_r, v_θ, m)
Control    a_T = (a_r, a_θ)
Mass flow  \\dot m = - m^2 ||a_T||^2 / (2P)
Hamiltonian optimality ⇒ a_T = k_T λ_v
with constant gain  k_T = P / (λ_m m^2) = P/C  (C = λ_m m^2 = const)

Trajectory search:
  Stage 1: global (heuristic) to get *any* admissible endpoint
  Stage 2: continuation (homotopy) + Powell to walk the endpoint to the target
(§ Numerical Procedure)

Units: SI throughout (m, kg, s). AU and day are convenience scales for I/O/plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, replace
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize, least_squares

from numba import njit


# ----------------------------- #
#  Constants in SI (m, kg, s)   #
# ----------------------------- #
AU = 1.495978707e11           # m
DAY = 86400.0                 # s
MU_SI = 1.32712440018e20      # m^3 s⁻²  (GM☉)

@dataclass(frozen=True)
class TrajectoryConfig:
    """All physical constants and initial conditions for one trajectory solve."""
    mu: float = MU_SI
    power: float = 1.0e9
    m_dry: float = 1.0e6
    m0: float = 3.0e6
    r0: float = AU
    vr0: float = 0.0
    vtheta0: float | None = None
    k_gain: float = -3.725e-6 - 4.91294688e-06

    def with_initial_radius(self, r0_au):
        """Return a new config with a new circular-orbit initial radius."""
        r0_si = r0_au * AU
        return replace(self, r0=r0_si, vtheta0=np.sqrt(self.mu / r0_si))


DEFAULT_CONFIG = TrajectoryConfig()

# Backward-compatible aliases for existing callers/tests
P = DEFAULT_CONFIG.power
M_DRY = DEFAULT_CONFIG.m_dry
M0 = DEFAULT_CONFIG.m0
R0 = DEFAULT_CONFIG.r0
VR0 = DEFAULT_CONFIG.vr0
VTHETA0 = np.sqrt(DEFAULT_CONFIG.mu / DEFAULT_CONFIG.r0)
K_GAIN_FIXED = DEFAULT_CONFIG.k_gain


@njit(cache=True)
def _ode_system_kernel(y, mu, power, k_gain, C_theta):
    r = y[0]
    v_r = y[2]
    v_theta = y[3]
    m = y[4]
    lam_r = y[5]
    lam_vr = y[6]
    lam_vtheta = y[7]

    a_r = k_gain * lam_vr
    a_th = k_gain * lam_vtheta
    accel_sq = a_r * a_r + a_th * a_th

    out = np.empty(8, dtype=np.float64)
    out[0] = v_r
    out[1] = v_theta / r
    out[2] = v_theta * v_theta / r - mu / (r * r) + a_r
    out[3] = -v_r * v_theta / r + a_th
    out[4] = -(m * m) / (2.0 * power) * accel_sq
    out[5] = (C_theta * v_theta) / (r * r) + (lam_vr * v_theta * v_theta) / (r * r) - 2.0 * lam_vr * mu / (r * r * r) - lam_vtheta * v_r * v_theta / (r * r)
    out[6] = -lam_r + lam_vtheta * v_theta / r
    out[7] = -C_theta / r - 2.0 * lam_vr * v_theta / r + lam_vtheta * v_r / r
    return out


def _ode_system_numba(t, y, config, C_m, C_theta):
    # t and C_m are part of solver signature but not used by dynamics as currently formulated.
    _ = (t, C_m)
    y_arr = y if isinstance(y, np.ndarray) else np.asarray(y, dtype=np.float64)
    return _ode_system_kernel(y_arr, config.mu, config.power, config.k_gain, C_theta)


def ode_system(t, y, config, C_m, C_theta):
    """
    The ODE system implements equations in the paper:
      \\dot r = v_r
      \\dot θ = v_θ / r
      \\dot v_r = v_θ^2 / r − μ/r^2 + a_r
      \\dot v_θ = −(v_r v_θ)/r + a_θ
      \\dot m = − m^2 ||a_T||^2 / (2P)

    Optimal control (Eq. (39)):
      a_T = k_T λ_v  with  λ_v = (λ_{v_r}, λ_{v_θ})^T  and  k_T = const.
    Here we pass two “constants” as parameters:
      C_theta  = λ_θ  (first integral from θ cyclic symmetry)
      C_m      ~ k_T  (naming carries over from experiments; see below)

    IMPORTANT: In the theory, the constant is k_T = P / (λ_m m^2) = P/C.
    In this implementation we *don’t integrate λ_m*; instead we treat the
    gain as a single scalar unknown that we call C_m for historical reasons.
    For clarity, read C_m in the params as k_T (constant thrust gain).

    y = [r, theta, v_r, v_theta, m, lambda_r, lambda_vr, lambda_vtheta]
    returns dy/dt in the same order, working in **SI units**
    """
    return _ode_system_numba(t, y, config, C_m, C_theta)


def make_fuel_out_event(config):
    def fuel_out_event(t, y, *args):
        """Event: stop if propellant exhausted (m == m_dry)."""
        return y[4] - config.m_dry
    fuel_out_event.terminal = True
    fuel_out_event.direction = -1
    return fuel_out_event


def make_circular_velocity_event(config):
    def circular_velocity_event(t, y, *args):
        """Circular velocity of planet at given radius reached."""
        v_circ_target = np.sqrt(config.mu / y[0])
        velocity_err = np.hypot(y[2], y[3] - v_circ_target)
        return velocity_err + (500 if y[0] / AU < 1.5 else 0)
    circular_velocity_event.terminal = True
    circular_velocity_event.direction = -1
    return circular_velocity_event


def integrate_trajectory(params, t_max_days=365*3, max_step_days=0.5, record=True, config=DEFAULT_CONFIG):
    """
    Integrate trajectory (one‑shot)
    params = [λ_r0, λ_vr0, λ_vθ0, C_m, C_theta]
    Returns SciPy solution object from solve_ivp
    """
    lam_r0, lam_vr0, lam_vtheta0, C_m, C_theta = params

    vtheta0 = np.sqrt(config.mu / config.r0) if config.vtheta0 is None else config.vtheta0
    y0 = [config.r0, 0.0, config.vr0, vtheta0, config.m0, lam_r0, lam_vr0, lam_vtheta0]

    t_span = (0.0, t_max_days * DAY)
    sol = solve_ivp(_ode_system_numba, t_span, y0,
                    events=(make_fuel_out_event(config), make_circular_velocity_event(config)),
                    args=(config, C_m, C_theta),
                    rtol=1e-8, atol=1e-9,
                    max_step=max_step_days * DAY,
                    dense_output=False)

    if record:
        print(f"Integration started at t = {sol.t[0]/DAY:.2f} days "
              f"with m = {sol.y[4, 0] / 1e6:.1f} kT, theta = {np.rad2deg(sol.y[1, 0]):.3f} deg "
              f"and r = {sol.y[0, 0] / AU:.3f} AU")
        print(f"Integration finished at t = {sol.t[-1]/DAY:.2f} days "
              f"with m = {sol.y[4, -1] / 1e6:.1f} kT, theta = {np.rad2deg(sol.y[1, -1]):.3f} deg "
              f"and r = {sol.y[0, -1] / AU:.3f} AU")
    return sol


def integrate_fixed_time(params, t_days, max_step_days=0.5, rtol=1e-8, atol=1e-9, config=DEFAULT_CONFIG):
    """
    Integrate the trajectory to a *fixed* final time with no terminal events.

    Compared to integrate_trajectory(), this is smooth in the optimization
    variables and therefore much better suited for Newton/least-squares solvers.
    """
    lam_r0, lam_vr0, lam_vtheta0, C_m, C_theta = params
    vtheta0 = np.sqrt(config.mu / config.r0) if config.vtheta0 is None else config.vtheta0
    y0 = [config.r0, 0.0, config.vr0, vtheta0, config.m0, lam_r0, lam_vr0, lam_vtheta0]

    sol = solve_ivp(
        _ode_system_numba,
        (0.0, t_days * DAY),
        y0,
        args=(config, C_m, C_theta),
        rtol=rtol,
        atol=atol,
        max_step=max_step_days * DAY,
        dense_output=False,
    )
    return sol


# saving some solutions that worked, as starting points
# SOLUTION0 = (-9.39556446e-05, -2.30484959e+02, -2.10191027e+03, 0.00000000e+00, -2.75116990e+07)
SOLUTION0 = ([-9.04177133e-05, -2.23208767e+01, -2.82272150e+03, 0.00000000e+00, -1.56907920e+08])
SCALE = np.delete(np.array(SOLUTION0), -2)


def pack(w):
    """
    physical → scaled ([-1..1])
    This is to avoid large exponents and keep all parameters around one. At the same time, it is a wrapper around
    already trusted implementation that I did not want to touch. (It would be cleaner to rewrite everything into
    re-scaled from.
    """
    w = np.delete(w, -2)        # remove parameter that is always -1 by fixed gauge
    return w / SCALE


def unpack(z):
    """
    scaled → physical
    Inversion of the previous operation.
    """
    w = z * SCALE
    return np.insert(w, -1, 0)


def objective_scaled(z, rt, tht, config=DEFAULT_CONFIG):
    return objective(unpack(z), rt, tht, config=config)


def angle_wrap(x):
    """Wrap angle to [-pi, pi] for smooth residuals."""
    return (x + np.pi) % (2 * np.pi) - np.pi


@njit(cache=True)
def _angle_wrap_numba(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


@njit(cache=True)
def _closest_seed_index_numba(seed_r, seed_t, target_r, target_t):
    best_idx = 0
    best_dist = 1e300
    for i in range(seed_r.shape[0]):
        dr = seed_r[i] - target_r
        dt = _angle_wrap_numba(seed_t[i] - target_t)
        dist = dr * dr + dt * dt
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def scaled_power_for_mass_scaling(base_power, base_m0, new_m0):
    """
    Scaling law that preserves dynamics when all masses are scaled together.

    If m -> s m for both wet and dry mass, then P must scale as P -> s P,
    with s = new_m0 / base_m0.
    """
    return base_power * (new_m0 / base_m0)


@dataclass(frozen=True)
class CacheSpec:
    r0_grid_au: tuple[float, ...]
    rf_grid_au: tuple[float, ...]
    theta_samples: int = 15
    max_abs_theta_deg: float = 170.0
    n_homotopy_steps: int = 5
    m0_scales: tuple[float, ...] = (1.0,)
    dry_mass_fractions: tuple[float, ...] = (1.0 / 3.0,)
    power_factors: tuple[float, ...] = (1.0,)
    max_workers: int = 1


def boundary_residual(z, r_target, theta_target, config=DEFAULT_CONFIG):
    """
    Residual for direct boundary solve.

    Unknowns z = [scaled_costates..., t_days_scale], where first 4 values are
    in the same pack()/unpack() scaling as the legacy code and the last value
    is transfer time in years.
    """
    params = unpack(z[:4])
    t_days = np.clip(z[4] * 365.0, 5.0, 3650.0)
    sol = integrate_fixed_time(params, t_days, config=config)
    y = sol.y[:, -1]

    r_end, th_end, vr_end, vth_end, m_end = y[:5]
    r_end_au = r_end / AU
    v_circ = np.sqrt(config.mu / r_end)

    # scales keep all residual components around O(1)
    v_scale = np.sqrt(config.mu / AU)
    fuel_scale = config.m0 - config.m_dry
    # Penalize dry-mass violations strongly: this term must be zero when
    # feasible (m_end >= M_DRY) and grow with any propellant overuse.
    dry_mass_deficit = max(0.0, config.m_dry - m_end)
    return np.array([
        (r_end_au - r_target) / 0.2,
        angle_wrap(th_end - theta_target) / 0.3,
        vr_end / v_scale,
        (vth_end - v_circ) / v_scale,
        dry_mass_deficit / (0.05 * fuel_scale),
    ])


def solve_target_fast(r_target, theta_target, seed_params, t_guess_days=500.0,
                      n_starts=5, max_nfev=140, config=DEFAULT_CONFIG):
    """
    Fast and robust target solver using least_squares + multi-start.

    This replaces the old "shift target and half-step on failure" strategy.
    """
    seed = np.concatenate([pack(seed_params), [t_guess_days / 365.0]])

    starts = [seed]
    rng = np.random.default_rng(1234)
    for _ in range(n_starts - 1):
        jitter = np.array([0.1, 0.15, 0.15, 0.1, 0.2]) * rng.normal(size=5)
        starts.append(seed + jitter)

    best = None
    for s in starts:
        res = least_squares(
            boundary_residual,
            s,
            args=(r_target, theta_target, config),
            method="trf",
            loss="soft_l1",
            f_scale=0.2,
            x_scale="jac",
            max_nfev=max_nfev,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        if (best is None) or (np.linalg.norm(res.fun) < np.linalg.norm(best.fun)):
            best = res
        if np.linalg.norm(res.fun) < 3e-4:
            break

    best_params = unpack(best.x[:4])
    best_time_days = np.clip(best.x[4] * 365.0, 5.0, 3650.0)
    return best_params, best_time_days, best


def solve_arbitrary_transfer(r0_au, r_target_au, theta_target_rad, seed_params=SOLUTION0,
                             n_homotopy_steps=8, config=DEFAULT_CONFIG):
    """
    Solve transfer for arbitrary initial radius, target radius and target angle.

    Strategy:
      1) Set initial radius directly.
      2) Build a homotopy path from known endpoint to desired endpoint.
      3) At each step, solve a smooth boundary system with least_squares.

    This is significantly faster than the legacy nested Powell + step-halving
    loops and much more reliable for far targets.
    """
    local_config = config.with_initial_radius(r0_au)

    seed_params = np.asarray(seed_params, dtype=float)
    seed_sol = integrate_trajectory(seed_params, record=False, config=local_config)
    r_seed = seed_sol.y[0, -1] / AU
    th_seed = seed_sol.y[1, -1]

    params = seed_params.copy()
    t_guess = seed_sol.t[-1] / DAY

    for alpha in np.linspace(0.0, 1.0, n_homotopy_steps + 1)[1:]:
        r_step = (1 - alpha) * r_seed + alpha * r_target_au
        th_step = angle_wrap((1 - alpha) * th_seed + alpha * theta_target_rad)
        params, t_guess, info = solve_target_fast(r_step, th_step, params, t_guess_days=t_guess, config=local_config)
        print(f"alpha={alpha:.2f}, target=({r_step:.4f} AU, {np.rad2deg(th_step):.2f} deg), "
              f"res={np.linalg.norm(info.fun):.3e}, nfev={info.nfev}, t={t_guess:.1f} d")

    sol = integrate_fixed_time(params, t_guess, config=local_config)
    return params, t_guess, sol, local_config


def estimate_reachable_theta_bounds(
    r0_au,
    r_target_au,
    seed_params,
    config=DEFAULT_CONFIG,
    max_abs_theta_deg=170.0,
    n_scan=9,
):
    """Estimate positive/negative reachable theta bounds via continuation scan."""
    local_cfg = config.with_initial_radius(r0_au)
    seed = np.asarray(seed_params, dtype=float)
    max_abs = np.deg2rad(max_abs_theta_deg)

    def _scan(sign):
        params = seed.copy()
        t_guess = 450.0
        last_ok = 0.0
        for th in np.linspace(0.0, sign * max_abs, n_scan)[1:]:
            try:
                params, t_guess, info = solve_target_fast(
                    r_target_au, th, params, t_guess_days=t_guess, config=local_cfg, max_nfev=90
                )
                if np.linalg.norm(info.fun) < 2e-2:
                    last_ok = th
                else:
                    break
            except Exception:
                break
        return last_ok

    return _scan(+1.0), _scan(-1.0)


def _solve_parameter_case(case):
    (
        case_idx,
        config,
        spec,
        theta_grid,
        seed_params,
    ) = case
    r0_grid = np.asarray(spec.r0_grid_au, dtype=float)
    rf_grid = np.asarray(spec.rf_grid_au, dtype=float)

    params_cache = np.full((r0_grid.size, rf_grid.size, theta_grid.size, 5), np.nan)
    time_cache = np.full((r0_grid.size, rf_grid.size, theta_grid.size), np.nan)
    resid_cache = np.full((r0_grid.size, rf_grid.size, theta_grid.size), np.nan)
    success = np.zeros((r0_grid.size, rf_grid.size, theta_grid.size), dtype=bool)
    theta_bounds = np.zeros((r0_grid.size, rf_grid.size, 2), dtype=float)

    seed_params = np.asarray(seed_params, dtype=float)
    for i, r0 in enumerate(r0_grid):
        for j, rf in enumerate(rf_grid):
            th_pos, th_neg = estimate_reachable_theta_bounds(
                r0, rf, seed_params, config=config, max_abs_theta_deg=spec.max_abs_theta_deg
            )
            theta_bounds[i, j, 0] = th_neg
            theta_bounds[i, j, 1] = th_pos

            solved_r = []
            solved_t = []
            solved_params = []
            t_guess = 450.0
            for k, th in enumerate(theta_grid):
                if th < th_neg or th > th_pos:
                    continue

                if solved_params:
                    idx = _closest_seed_index_numba(
                        np.asarray(solved_r), np.asarray(solved_t), rf, th
                    )
                    seed = solved_params[idx]
                else:
                    seed = seed_params

                try:
                    out_params, out_t, info = solve_target_fast(
                        rf,
                        th,
                        seed,
                        t_guess_days=t_guess,
                        n_starts=3,
                        max_nfev=100,
                        config=config.with_initial_radius(r0),
                    )
                    params_cache[i, j, k, :] = out_params
                    time_cache[i, j, k] = out_t
                    resid_cache[i, j, k] = np.linalg.norm(info.fun)
                    success[i, j, k] = resid_cache[i, j, k] < 2e-2
                    if success[i, j, k]:
                        solved_r.append(rf)
                        solved_t.append(th)
                        solved_params.append(out_params)
                        t_guess = out_t
                except Exception:
                    continue

    return case_idx, params_cache, time_cache, resid_cache, success, theta_bounds


def build_trajectory_cache_npz(
    output_path,
    spec,
    base_config=DEFAULT_CONFIG,
    seed_params=SOLUTION0,
    show_progress=True,
):
    """
    Build and save a trajectory cache over (r0, rf, theta, propulsion parameters).

    The 3 physical ship parameters are reduced with scaling symmetry:
      m0 scale (absolute), dry-mass fraction, and power factor at fixed scaled power.
    """
    theta_grid = np.linspace(-np.deg2rad(spec.max_abs_theta_deg), np.deg2rad(spec.max_abs_theta_deg), spec.theta_samples)
    cases = []
    parameter_table = []

    for m0_scale in spec.m0_scales:
        m0 = base_config.m0 * m0_scale
        for dry_frac in spec.dry_mass_fractions:
            m_dry = m0 * dry_frac
            for power_factor in spec.power_factors:
                power_scaled = scaled_power_for_mass_scaling(base_config.power, base_config.m0, m0)
                cfg = replace(base_config, m0=m0, m_dry=m_dry, power=power_scaled * power_factor)
                idx = len(parameter_table)
                parameter_table.append((m0, m_dry, cfg.power, m0_scale, dry_frac, power_factor))
                cases.append((idx, cfg, spec, theta_grid, np.array(seed_params, dtype=float).reshape(-1)))

    shape = (
        len(parameter_table),
        len(spec.r0_grid_au),
        len(spec.rf_grid_au),
        spec.theta_samples,
    )
    params_cache = np.full(shape + (5,), np.nan)
    time_cache = np.full(shape, np.nan)
    resid_cache = np.full(shape, np.nan)
    success = np.zeros(shape, dtype=bool)
    theta_bounds = np.zeros((len(parameter_table), len(spec.r0_grid_au), len(spec.rf_grid_au), 2))

    workers = max(1, int(spec.max_workers))
    total_cases = len(cases)
    started = perf_counter()

    def _ingest_result(result, completed):
        case_idx, p, t, r, s, b = result
        params_cache[case_idx] = p
        time_cache[case_idx] = t
        resid_cache[case_idx] = r
        success[case_idx] = s
        theta_bounds[case_idx] = b

        if show_progress:
            elapsed = perf_counter() - started
            rate = completed / elapsed if elapsed > 0 else 0.0
            remaining = (total_cases - completed) / rate if rate > 0 else float("inf")
            eta_str = f"{remaining/60.0:6.1f} min" if np.isfinite(remaining) else "   inf"
            print(
                f"[cache] case {completed:>3}/{total_cases} "
                f"({100.0*completed/total_cases:5.1f}%) "
                f"elapsed {elapsed/60.0:6.1f} min, eta {eta_str}",
                flush=True,
            )

    if show_progress:
        print(f"[cache] starting {total_cases} parameter cases with {workers} worker(s)", flush=True)

    if workers == 1:
        for completed, result in enumerate(map(_solve_parameter_case, cases), start=1):
            _ingest_result(result, completed)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for completed, result in enumerate(ex.map(_solve_parameter_case, cases), start=1):
                _ingest_result(result, completed)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        parameter_table=np.asarray(parameter_table, dtype=float),
        r0_grid_au=np.asarray(spec.r0_grid_au, dtype=float),
        rf_grid_au=np.asarray(spec.rf_grid_au, dtype=float),
        theta_grid_rad=theta_grid,
        theta_bounds_rad=theta_bounds,
        time_days=time_cache,
        params=params_cache,
        residual_norm=resid_cache,
        success=success,
    )
    return output_path


def objective(params, r_target=None, th_target=None, config=DEFAULT_CONFIG):
    """
    Objective mirrors the two-stage strategy from the paper (§ Numerical Procedure):
     - If r_target / th_target are None ⇒ global "any circular orbit far enough" search:
         • penalize fuel exhaustion, encourage r ≥ 1.5 AU, minimize circular-velocity mismatch
     - If targets are set ⇒ local refinement to a precise (r_target, θ_target):
         • add quadratic penalties on r and θ errors

    The velocity error is ||(v_r, v_θ) − (0, sqrt(μ/r))|| at the final state,
    i.e., distance to the local circular velocity vector (Eq. (28) definition).
    """
    sol = integrate_trajectory(params, record=False, config=config)
    r_end = sol.y[0, -1] / AU        # AU
    th_end = sol.y[1, -1]
    v_r_end = sol.y[2, -1]
    v_th_end = sol.y[3, -1]
    m_end = sol.y[4, -1]

    # ---- penalty if fuel exhausted early ----
    fuel_frac = (m_end - config.m_dry) / (config.m0 - config.m_dry)
    # fuel_frac*100 is a shaping term so near-feasible runs still prefer saving fuel.
    penalty_fuel = 1e6 if fuel_frac < 0 else fuel_frac*100

    # ---- radial penalty (needs to reach >1.5 AU) ----
    if r_end < 1.5 and (th_target is None):
        penalty_r = (1.5 - r_end)**2 * 1e3
    else:
        penalty_r = 0.0

    # ---- “reward” for matching any outer planet circular orbit ----
    v_circ_target = np.sqrt(config.mu / (r_end * AU))
    velocity_err = np.hypot(v_r_end, v_th_end - v_circ_target)

    if (r_target is None) or (th_target is None):
        reward = (velocity_err / 1e4)**2   #   + np.abs(r_end-3)**2   # no theta constrain, for easiser solution finding
    else:
        reward = (velocity_err / 1e4)**2 + np.abs(r_end - r_target)**2 + np.abs(th_end - th_target)**2*10

    return penalty_fuel + penalty_r + reward


def make_plots(sol, params, show=False, config=DEFAULT_CONFIG):
    t_days = sol.t / DAY
    r = sol.y[0]
    theta = sol.y[1]
    m = sol.y[4]
    lam_vr = sol.y[6]
    lam_vth = sol.y[7]

    # Recover lam_m and thrust → exhaust velocity
    k_gain = config.k_gain

    a_mag = np.abs(k_gain) * np.sqrt(lam_vr**2 + lam_vth**2)
    v_e = 2.0 * config.power / (m * a_mag)      # m/s

    # Cartesian trajectory for plotting
    x_au = (r * np.cos(theta)) / AU
    y_au = (r * np.sin(theta)) / AU

    fig, (ax_traj, ax_fuel, ax_ve) = plt.subplots(
        1, 3, figsize=(14, 4), gridspec_kw={'width_ratios': [2, 1, 1]})

    # --- orbital path
    ax_traj.plot(x_au, y_au, label='spacecraft path')
    ax_traj.scatter([0], [0], color='yellow', marker='*', s=200, label='Sun')
    circle = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle='--', label='Earth orbit')
    ax_traj.add_patch(circle)
    ax_traj.set_aspect('equal')
    ax_traj.set_xlabel('x [AU]')
    ax_traj.set_ylabel('y [AU]')
    ax_traj.set_title(f'{config.r0/AU:.1f} AU → {np.round(r[-1]/AU, 1):.1f} AU, {np.round(t_days[-1], 0):.0f} days, '
                      f'{np.round(np.rad2deg(theta[-1]), 1):.1f} deg')
    # ax_traj.legend()

    # --- propellant mass
    ax_fuel.plot(t_days, (m - config.m_dry)/1e6)
    ax_fuel.set_xlabel('Time [days]')
    ax_fuel.set_ylabel('Propellant mass [kT]')
    ax_fuel.set_title('Fuel on board')

    # --- effective exhaust velocity
    # ax_ve.plot(t_days, v_e_kms)   # v_e_kms
    ax_ve.plot(t_days, a_mag)
    ax_ve.set_xlabel('Time [days]')
    ax_ve.set_ylabel('a [m/s²]')
    ax_ve.set_title('Acceleration magnitude')

    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(r'c:\target-directory' +
                    # f'{np.round(r[-1]/AU, 1):.1f}-{np.round(np.rad2deg(theta[-1]), 1):.1f}.png', dpi=600)
                    f'{np.round(config.r0/AU, 1):.1f}-{np.round(np.rad2deg(theta[-1]), 1):.1f}.png', dpi=300)
    plt.close()


def main():
    print("\n--- Constant‑power electric transfer demo ---")
    print("Initial conditions: Earth orbit, m₀ = 3000 kT (payload 1000 kT + prop 2000 kT)")
    print("Thruster power     : 1 GW\n")

    # Default branch: solve and plot one trajectory.
    run_parameter_search = False

    if run_parameter_search:
        print("Running parameter-space cache search...")
        spec = CacheSpec(
            r0_grid_au=(0.05, 0.2, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0),
            rf_grid_au=(0.05, 0.2, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0),
            theta_samples=31,
            max_abs_theta_deg=170.0,
            m0_scales=(0.5, 1.0, 2.0),
            dry_mass_fractions=(0.25, 1.0 / 3.0, 0.5),
            power_factors=(0.75, 1.0, 1.25),
            max_workers=4,
        )
        cache_path = build_trajectory_cache_npz(
            output_path="cache/trajectory_cache.npz",
            spec=spec,
            base_config=DEFAULT_CONFIG,
            seed_params=np.array(SOLUTION0, dtype=float).reshape(-1),
        )
        print(f"Cache written to: {cache_path}")
    else:
        # Example: Saturn radius -> Earth radius with fixed arrival angle
        params_opt, t_opt_days, sol_opt, transfer_config = solve_arbitrary_transfer(
            r0_au=9.58,
            r_target_au=1.0,
            theta_target_rad=np.deg2rad(-95.0),
            seed_params=unpack([-8.33529969, -99.6312038, 0.43134401, 0.66967974]),
        )
        print("\nSolved transfer")
        print("params:", params_opt)
        print(f"t_f = {t_opt_days:.2f} days")
        integrate_trajectory(params_opt, config=transfer_config)
        make_plots(sol_opt, params_opt, show=True, config=transfer_config)


if __name__ == "__main__":
    main()
