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
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize


# ----------------------------- #
#  Constants in SI (m, kg, s)   #
# ----------------------------- #
AU = 1.495978707e11           # m
DAY = 86400.0                 # s
MU_SI = 1.32712440018e20      # m^3 s⁻²  (GM☉)

# ----------------------------- #
#  Problem‑specific constants   #
# ----------------------------- #
P = 1.0e9                       # W  (1 GW electric thruster)
M_DRY = 1.0e6                   # kg – dry / payload mass  (1000 kT)
M0 = 3.0e6                      # kg – total initial mass (3000 kT ⇒ 2000 kT propellant)
R0 = AU                         # m  – start at Earth’s orbit
VTHETA0 = np.sqrt(MU_SI / R0)   # circular speed, m s⁻¹
VR0 = 0.0


def ode_system(t, y, mu, power, m_dry, C_m, C_theta):
    """
    The ODE system implements Eq. (24) in the paper:
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
    r, theta, v_r, v_theta, m, lam_r, lam_vr, lam_vtheta = y

    # ---- costate-dependent thrust law
    # This is just a calibration relating input parameter C_m to k_gain. It would be good to clean up the code and
    # throughly test it again, keeping for now. It should not have effect, only shifts the meaning of the input
    # parameter that still needs to be found by the numerical procedure.
    k_gain = C_m - 3.725e-6 - 4.91294688e-06
    a_r = k_gain * lam_vr
    a_th = k_gain * lam_vtheta
    accel_sq = a_r*a_r + a_th*a_th

    # ---- state derivatives ----
    drdt = v_r
    dthetadt = v_theta / r
    dv_rdt = v_theta**2 / r - mu/r**2 + a_r
    dv_thdt = -v_r * v_theta / r + a_th
    dmdt = - m**2 / (2.0 * power) * accel_sq

    # ---- costate derivatives (eq. 4) ----
    dlam_r = (C_theta * v_theta)/r**2 + (lam_vr * v_theta**2)/r**2 \
                    - 2.0 * lam_vr * mu / r**3 - lam_vtheta * v_r * v_theta / r**2
    dlam_vr = -lam_r + lam_vtheta * v_theta / r
    dlam_vtheta = -C_theta / r - 2.0 * lam_vr * v_theta / r + lam_vtheta * v_r / r

    return [drdt, dthetadt, dv_rdt, dv_thdt, dmdt, dlam_r, dlam_vr, dlam_vtheta]


def fuel_out_event(t, y, *args):
    """Event: stop if propellant exhausted (m == M_DRY)"""
    return y[4] - M_DRY       # m(t) - m_dry  — stops when zero
fuel_out_event.terminal  = True
fuel_out_event.direction = -1


def circular_velocity_event(t, y, *args):
    """Circular velocity of planet at given radius reached"""
    v_circ_target = np.sqrt(MU_SI / y[0])
    velocity_err = np.hypot(y[2], y[3] - v_circ_target)
    return velocity_err + (500 if y[0] / AU < 1.5 else 0)
circular_velocity_event.terminal  = True        # was run with a bug, fuel_out_event.terminal defined here
circular_velocity_event.direction = -1          # was run with a bug, fuel_out_event.direction defined here


def integrate_trajectory(params, t_max_days=365*3, max_step_days=0.5, record=True):
    """
    Integrate trajectory (one‑shot)
    params = [λ_r0, λ_vr0, λ_vθ0, C_m, C_theta]
    Returns SciPy solution object from solve_ivp
    """
    lam_r0, lam_vr0, lam_vtheta0, C_m, C_theta = params

    y0 = [R0, 0.0, VR0, VTHETA0, M0, lam_r0, lam_vr0, lam_vtheta0]

    t_span = (0.0, t_max_days * DAY)
    sol = solve_ivp(ode_system, t_span, y0,
                    events=(fuel_out_event,circular_velocity_event),
                    args=(MU_SI, P, M_DRY, C_m, C_theta),
                    rtol=1e-8, atol=1e-9,
                    max_step=max_step_days * DAY,
                    dense_output=False)

    if record:
        print(f"Integration finished at t = {sol.t[-1]/DAY:.2f} days "
              f"with m = {sol.y[4, -1] / 1e6:.1f} kT, theta = {np.rad2deg(sol.y[1, -1]):.3f} deg "
              f"and r = {sol.y[0, -1] / AU:.3f} AU")
    return sol


# saving some solutions that worked, as starting points
SOLUTION0 = (-9.39556446e-05, -2.30484959e+02, -2.10191027e+03, 0.00000000e+00, -2.75116990e+07)
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


def objective_scaled(z, rt, tht):
    return objective(unpack(z), rt, tht)


def refine_theta_forward(theta_T_old, r_T_old, params):
    """
    The method that starts from working solution and gradually sets the target by small steps towards new
    target position of the planet
    :param theta_T_old: Initial solution true anomaly
    :param r_T_old: Initial solution radius
    :param params: Initial parameters of the working solution, will be re-written locally by each new solution found
    """
    theta_T_new = theta_T_old   # start from the original true anomaly
    r_target = r_T_old          # and original radius
    th_target = theta_T_new     # set target to new
    while theta_T_new < 2 * np.pi * 51 / 50:    # increase true anomaly until desired range
        res = minimize(objective_scaled, pack(params), args=(r_target, th_target),  # try to find a solution
                       method="Powell", options={"maxiter": 300, 'disp': False})

        if res.fun < 1e-12:     # solution found, print it and move to new target
            sol0 = integrate_trajectory(unpack(res.x))
            make_plots(sol0, unpack(res.x))
            print(f"{res.fun}, {res.x})")

            params = unpack(res.x)      # feed best guess into next stage
            theta_T_new = th_target     # try to achieve target again
            th_target = theta_T_new + np.pi / 45    # move the target by a step for next iteration
        else:                   # convergence failed, try to move target half distance to working solution
            th_target = (th_target - theta_T_new) / 2 + theta_T_new
            print('             /2')
        if not res.success:     # not only is not the target function stuck in local minimum, but failed to converge,
                                # abort
            print(f"Failure", res.message)
            return


def refine_theta_backward(theta_T_old, r_T_old, params):
    """See implementation of refine_theta_forward, this just walks towards smaller true anomalies, not larger"""
    theta_T_new = theta_T_old
    r_target = r_T_old
    th_target = theta_T_new
    while theta_T_new > -np.pi * 100 / 180:
        res = minimize(objective_scaled, pack(params), args=(r_target, th_target),
                       method="Powell", options={"maxiter": 300, 'disp': False})

        if res.fun < 1e-12:
            sol0 = integrate_trajectory(unpack(res.x))
            make_plots(sol0, unpack(res.x))
            print(f"{res.fun}, {res.x})")

            params = unpack(res.x)      # feed best guess into next stage
            theta_T_new = th_target
            th_target = theta_T_new - np.pi / 45
        else:
            th_target = (th_target - theta_T_new) / 2 + theta_T_new
            print('             /2')
        if not res.success:
            print(f"Failure", res.message)
            return


def refine_r(theta_T_old, r_T_old, params, r_T_goal):
    """
    Called prior to the theta loop, this function moves the target RADIUS from the working solution to the desired
    radius. The algorithm is the same as in refine_theta_forward() otherwise, just moving r, not theta.
    """
    r_target = r_T_goal
    th_target = theta_T_old
    r_T_new = r_T_old
    while True:
        res = minimize(objective_scaled, pack(params), args=(r_target, th_target),
                       method="Powell", options={"maxiter": 300, 'disp': False})

        if res.fun < 1e-12:
            # sol0 = integrate_trajectory(unpack(res.x))
            # make_plots(sol0, unpack(res.x))
            print(f"{res.fun}, {r_target}, {res.x})")

            params = unpack(res.x)      # feed best guess into next stage

            if np.abs(r_T_goal-r_target) < 1e-6:
                print('Finished setting R')
                return params

            r_T_new = r_target
            r_target = r_T_goal
        else:
            r_target = (r_target - r_T_new) / 2 + r_T_new
            print('             /2')
        if not res.success:
            print(f"Failure", res.message)


def refine_r0(th_target, r_T_old, params, r_T_goal):
    """
    Called prior to the theta loop, this function moves the starting planet RADIUS from the working solution to the
    desired radius. The algorithm is the same as in refine_theta_forward() otherwise, just moving r, not theta.
    This function is used for tables calculating paths from planets that are not Earth to Earth, hence the
    initial radius needs to be adapted, not the target radius.
    """
    global R0, VTHETA0      # initial radius not in params, it is a global variable

    R0 = r_T_goal*AU
    VTHETA0 = np.sqrt(MU_SI / R0)

    r_T_new = 1.0
    while True:
        res = minimize(objective_scaled, pack(params), args=(r_T_old, th_target),
                       method="Powell", options={"maxiter": 300, 'disp': False})

        if res.fun < 1e-12:
            # sol0 = integrate_trajectory(unpack(res.x))
            # make_plots(sol0, unpack(res.x))
            print(f"{res.fun}, {R0/AU}, {VTHETA0}, {res.x})")

            params = unpack(res.x)      # feed best guess into next stage

            if np.abs(r_T_goal-R0/AU) < 1e-6:
                print('Finished setting R0')
                return params

            r_T_new = R0/AU
            R0 = r_T_goal*AU
            VTHETA0 = np.sqrt(MU_SI / R0)
        else:
            R0 = ((R0/AU - r_T_new) / 2 + r_T_new)*AU
            VTHETA0 = np.sqrt(MU_SI / R0)
            print('             /2')
        if not res.success:
            print(f"Failure", res.message)


def refine_params(prev_params=SOLUTION0):
    """
    Start with existing solution and by small steps, lead the solver to the desired solution.
    If it does not converge, half the step from the already working solution. This way, a sufficiently
    small step that will converge is usually found eventually.
    """
    sol = integrate_trajectory(prev_params)
    r_T_old = sol.y[0, -1] / AU
    theta_T_old = sol.y[1, -1]
    params = np.asarray(prev_params, dtype=float).copy()

    for r00 in [9.58]:
        new_params_r0 = refine_r0(theta_T_old, r_T_old, params, r00)
        for r in [1]:   # 1.52, 5.20, 9.58, 19.22, 30.05 , 0.72, 0.39,  #  1.643, 3, 5, 10, 15, 20, 0.75,
            new_params = refine_r(theta_T_old, r_T_old, new_params_r0, r)       # params

            # refine_theta_forward(theta_T_old, r, new_params)
            refine_theta_backward(theta_T_old, r, new_params)


def objective(params, r_target=None, th_target=None):
    """
    Objective mirrors the two-stage strategy from the paper (§ Numerical Procedure):
     - If r_target / th_target are None ⇒ global "any circular orbit far enough" search:
         • penalize fuel exhaustion, encourage r ≥ 1.5 AU, minimize circular-velocity mismatch
     - If targets are set ⇒ local refinement to a precise (r_target, θ_target):
         • add quadratic penalties on r and θ errors

    The velocity error is ||(v_r, v_θ) − (0, sqrt(μ/r))|| at the final state,
    i.e., distance to the local circular velocity vector (Eq. (28) definition).
    """
    sol = integrate_trajectory(params, record=False)
    t_end = sol.t[-1]
    r_end = sol.y[0, -1] / AU        # AU
    th_end = sol.y[1, -1]
    v_r_end = sol.y[2, -1]
    v_th_end = sol.y[3, -1]
    m_end = sol.y[4, -1]

    # ---- penalty if fuel exhausted early ----
    fuel_frac = (m_end - M_DRY) / (M0 - M_DRY)
    # fuel_frac*100 is a shaping term so near-feasible runs still prefer saving fuel.
    penalty_fuel = 1e6 if fuel_frac < 0 else fuel_frac*100

    # ---- radial penalty (needs to reach >1.5 AU) ----
    if r_end < 1.5 and (th_target is None):
        penalty_r = (1.5 - r_end)**2 * 1e3
    else:
        penalty_r = 0.0

    # ---- “reward” for matching any outer planet circular orbit ----
    v_circ_target = np.sqrt(MU_SI / (r_end * AU))
    velocity_err = np.hypot(v_r_end, v_th_end - v_circ_target)

    if (r_target is None) or (th_target is None):
        reward = (velocity_err / 1e4)**2   #   + np.abs(r_end-3)**2      # any theta is possible, for easiser solution finding
    else:
        reward = (velocity_err / 1e4) ** 2 + np.abs(r_end - r_target) ** 2 + np.abs(th_end-th_target)**2*10

    return penalty_fuel + penalty_r + reward


def make_plots(sol, params):
    t_days = sol.t / DAY
    r = sol.y[0]
    theta = sol.y[1]
    m = sol.y[4]
    lam_vr = sol.y[6]
    lam_vth = sol.y[7]

    # Recover lam_m and thrust → exhaust velocity
    C_m = params[-2]
    # lam_m = C_m_guess - 2.0 * np.log(m)
    # k_gain = P / (lam_m * m * m)
    k_gain = C_m - 3.725e-6 - 4.91294688e-06        # must match the one in ode_system, would be better to get rid of

    a_mag = np.abs(k_gain) * np.sqrt(lam_vr**2 + lam_vth**2)
    v_e = 2.0 * P / (m * a_mag)      # m/s
    v_e_kms = v_e / 1e3              # km/s

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
    ax_traj.set_title(f'{R0/AU:.1f} AU → {np.round(r[-1]/AU, 1):.1f} AU, {np.round(t_days[-1], 0):.0f} days, '
                      f'{np.round(np.rad2deg(theta[-1]), 1):.1f} deg')
    # ax_traj.legend()

    # --- propellant mass
    ax_fuel.plot(t_days, (m - M_DRY)/1e6)
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
    # plt.show()
    plt.savefig(r'c:\target-directory' +
                # f'{np.round(r[-1]/AU, 1):.1f}-{np.round(np.rad2deg(theta[-1]), 1):.1f}.png', dpi=600)
                f'{np.round(R0/AU, 1):.1f}-{np.round(np.rad2deg(theta[-1]), 1):.1f}.png', dpi=300)
    plt.close()


def main():
    print("\n--- Constant‑power electric transfer demo ---")
    print("Initial conditions: Earth orbit, m₀ = 3000 kT (payload 1000 kT + prop 2000 kT)")
    print("Thruster power     : 1 GW\n")

    search_for_new_solution = False

    if search_for_new_solution:
        bounds = [(-1e-4, 1e-4),  # λ_r0
                  (-3000, 0.0),   # λ_vr0
                  (-6000, 0.0),   # λ_vθ0
                  (-1, 0),        # C_m
                  (-1e9, 0)]      # C_theta

        result = differential_evolution(objective, bounds, maxiter=400, popsize=100,
                                        polish=True, tol=1e-4, workers=4, disp=True)
        print("\nBest parameters:", result.x)
        sol_opt = integrate_trajectory(result.x)
        make_plots(sol_opt, result.x)
    else:
        sol = [-9.26130852, -112.96960747, -0.13010519, 0.24847801]
        # sol = [-8.33529969, -99.6312038,    0.43134401,   0.66967974]  # for Mercury -> Earth
        refine_params(prev_params=unpack(sol))


if __name__ == "__main__":
    main()
