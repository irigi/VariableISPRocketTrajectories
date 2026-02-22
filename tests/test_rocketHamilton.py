import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import rocketHamilton as rh


def test_pack_unpack_roundtrip_keeps_physical_values():
    physical = np.array([-9.0e-5, -22.0, -2800.0, 0.0, -1.5e8])
    packed = rh.pack(physical)
    unpacked = rh.unpack(packed)

    np.testing.assert_allclose(unpacked, physical, rtol=0, atol=1e-12)
    assert unpacked[-2] == 0.0  # fixed gauge parameter remains fixed


def test_angle_wrap_bounds_and_known_values():
    values = np.array([-4 * np.pi, -3 * np.pi / 2, 0.0, 3 * np.pi / 2, 4 * np.pi])
    wrapped = rh.angle_wrap(values)

    assert np.all(wrapped >= -np.pi)
    assert np.all(wrapped <= np.pi)
    np.testing.assert_allclose(wrapped, np.array([0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0]), atol=1e-12)


def test_ode_system_matches_manual_derivatives():
    y = [
        1.2 * rh.AU,
        0.2,
        10.0,
        np.sqrt(rh.MU_SI / (1.2 * rh.AU)) * 0.9,
        2.5e6,
        -1.0e-4,
        -20.0,
        -2000.0,
    ]

    out = rh.ode_system(0.0, y, rh.MU_SI, rh.P, rh.M_DRY, 0.0, -1.0e8)

    r, _, v_r, v_theta, m, lam_r, lam_vr, lam_vtheta = y
    k = rh.K_GAIN_FIXED
    a_r = k * lam_vr
    a_theta = k * lam_vtheta
    accel_sq = a_r**2 + a_theta**2

    expected = np.array([
        v_r,
        v_theta / r,
        v_theta**2 / r - rh.MU_SI / r**2 + a_r,
        -v_r * v_theta / r + a_theta,
        -(m**2) / (2.0 * rh.P) * accel_sq,
        (-1.0e8 * v_theta) / r**2 + (lam_vr * v_theta**2) / r**2 - 2.0 * lam_vr * rh.MU_SI / r**3 - lam_vtheta * v_r * v_theta / r**2,
        -lam_r + lam_vtheta * v_theta / r,
        -(-1.0e8) / r - 2.0 * lam_vr * v_theta / r + lam_vtheta * v_r / r,
    ])

    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


def test_integrate_fixed_time_runs_and_mass_decreases():
    params = np.array(rh.SOLUTION0, dtype=float).reshape(-1)
    sol = rh.integrate_fixed_time(params, t_days=1.0, max_step_days=0.25, rtol=1e-7, atol=1e-8)

    assert sol.success
    assert sol.t[-1] > 0
    assert sol.y[4, -1] <= sol.y[4, 0]


def test_boundary_residual_penalizes_dry_mass_violation(monkeypatch):
    class DummySol:
        def __init__(self, y):
            self.y = y

    y = np.zeros((8, 1), dtype=float)
    y[0, -1] = 1.0 * rh.AU
    y[1, -1] = 0.0
    y[2, -1] = 0.0
    y[3, -1] = np.sqrt(rh.MU_SI / y[0, -1])
    y[4, -1] = rh.M_DRY - 1000.0

    monkeypatch.setattr(rh, "integrate_fixed_time", lambda *args, **kwargs: DummySol(y))

    z = np.zeros(5)
    residual = rh.boundary_residual(z, r_target=1.0, theta_target=0.0)

    assert residual.shape == (5,)
    assert residual[-1] > 0


def test_solve_arbitrary_transfer_can_be_tested_with_fast_stubs(monkeypatch):
    calls = {"steps": 0}

    class DummySol:
        def __init__(self, r_au, theta, tf_days):
            self.y = np.zeros((8, 2), dtype=float)
            self.y[0, -1] = r_au * rh.AU
            self.y[1, -1] = theta
            self.t = np.array([0.0, tf_days * rh.DAY])

    monkeypatch.setattr(rh, "integrate_trajectory", lambda params, record=False: DummySol(1.2, 0.3, 50.0))

    def fake_solve_target_fast(r_target, theta_target, params, t_guess_days=0.0, **kwargs):
        calls["steps"] += 1
        return np.asarray(params, dtype=float), t_guess_days + 1.0, type("Info", (), {"fun": np.zeros(5), "nfev": 1})()

    monkeypatch.setattr(rh, "solve_target_fast", fake_solve_target_fast)
    monkeypatch.setattr(rh, "integrate_fixed_time", lambda params, t_days: DummySol(1.0, -0.2, t_days))

    params_opt, t_opt_days, sol = rh.solve_arbitrary_transfer(
        r0_au=1.0,
        r_target_au=1.5,
        theta_target_rad=0.5,
        seed_params=np.array(rh.SOLUTION0, dtype=float).reshape(-1),
        n_homotopy_steps=3,
    )

    assert calls["steps"] == 3
    assert t_opt_days == 53.0
    assert np.isfinite(sol.y[0, -1])
    assert np.all(np.isfinite(params_opt))
