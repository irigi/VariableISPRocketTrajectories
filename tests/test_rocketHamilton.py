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

    out = rh.ode_system(0.0, y, rh.DEFAULT_CONFIG, 0.0, -1.0e8)

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

    monkeypatch.setattr(rh, "integrate_trajectory", lambda params, record=False, config=None: DummySol(1.2, 0.3, 50.0))

    def fake_solve_target_fast(r_target, theta_target, params, t_guess_days=0.0, **kwargs):
        calls["steps"] += 1
        return np.asarray(params, dtype=float), t_guess_days + 1.0, type("Info", (), {"fun": np.zeros(5), "nfev": 1})()

    monkeypatch.setattr(rh, "solve_target_fast", fake_solve_target_fast)
    monkeypatch.setattr(rh, "integrate_fixed_time", lambda params, t_days, config=None: DummySol(1.0, -0.2, t_days))

    params_opt, t_opt_days, sol, used_config = rh.solve_arbitrary_transfer(
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
    assert np.isclose(used_config.r0, 1.0 * rh.AU)


def test_integrate_fixed_time_respects_custom_mass_and_power():
    params = np.array(rh.SOLUTION0, dtype=float).reshape(-1)
    cfg = rh.TrajectoryConfig(power=5.0e8, m0=4.0e6, m_dry=1.2e6)

    sol = rh.integrate_fixed_time(params, t_days=0.5, config=cfg)

    assert sol.success
    assert np.isclose(sol.y[4, 0], cfg.m0)
    assert sol.y[4, -1] < cfg.m0


def test_solve_arbitrary_transfer_does_not_mutate_module_initial_conditions(monkeypatch):
    initial_r0 = rh.R0
    initial_vtheta0 = rh.VTHETA0

    class DummySol:
        def __init__(self, r_au, theta, tf_days):
            self.y = np.zeros((8, 2), dtype=float)
            self.y[0, -1] = r_au * rh.AU
            self.y[1, -1] = theta
            self.t = np.array([0.0, tf_days * rh.DAY])

    monkeypatch.setattr(rh, "integrate_trajectory", lambda params, record=False, config=None: DummySol(1.2, 0.3, 10.0))
    monkeypatch.setattr(rh, "solve_target_fast", lambda *args, **kwargs: (np.asarray(args[2]), kwargs.get("t_guess_days", 10.0), type("Info", (), {"fun": np.zeros(5), "nfev": 1})()))
    monkeypatch.setattr(rh, "integrate_fixed_time", lambda params, t_days, config=None: DummySol(1.0, 0.0, t_days))

    rh.solve_arbitrary_transfer(
        r0_au=2.0,
        r_target_au=3.0,
        theta_target_rad=0.1,
        seed_params=np.array(rh.SOLUTION0, dtype=float).reshape(-1),
        n_homotopy_steps=1,
    )

    assert rh.R0 == initial_r0
    assert rh.VTHETA0 == initial_vtheta0


def test_scaled_power_for_mass_scaling_is_linear_in_mass_scale():
    scaled = rh.scaled_power_for_mass_scaling(base_power=1.0e9, base_m0=3.0e6, new_m0=6.0e6)
    assert np.isclose(scaled, 2.0e9)


def test_build_trajectory_cache_npz_writes_expected_arrays(tmp_path, monkeypatch):
    def fake_estimate(*args, **kwargs):
        return 0.5, -0.5

    class DummyInfo:
        fun = np.zeros(5)

    def fake_solve_target_fast(*args, **kwargs):
        return np.array(rh.SOLUTION0, dtype=float).reshape(-1), 42.0, DummyInfo()

    monkeypatch.setattr(rh, "estimate_reachable_theta_bounds", fake_estimate)
    monkeypatch.setattr(rh, "solve_target_fast", fake_solve_target_fast)

    spec = rh.CacheSpec(
        r0_grid_au=(1.0,),
        rf_grid_au=(1.5,),
        theta_samples=3,
        max_workers=1,
        m0_scales=(1.0,),
        dry_mass_fractions=(1.0 / 3.0,),
        power_factors=(1.0,),
    )

    out = tmp_path / "cache.npz"
    rh.build_trajectory_cache_npz(out, spec)

    data = np.load(out)
    assert data["time_days"].shape == (1, 1, 1, 3)
    assert data["params"].shape == (1, 1, 1, 3, 5)
    assert data["success"].dtype == np.bool_
