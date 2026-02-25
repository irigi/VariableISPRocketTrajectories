import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import numpy as np

import generate_atlas as ga


def test_build_axes_shapes_and_monotonicity():
    spec = ga.AtlasGridSpec(rho_points=4, kappa_points=3, theta_points=5)
    rho, kappa, theta = ga.build_axes(spec)

    assert rho.shape == (4,)
    assert kappa.shape == (3,)
    assert theta.shape == (5,)
    assert np.all(np.diff(rho) > 0)
    assert np.all(np.diff(kappa) > 0)
    assert np.all(np.diff(theta) > 0)


def test_init_atlas_tensor_default_values():
    spec = ga.AtlasGridSpec(rho_points=2, kappa_points=3, theta_points=4)
    values, status, seed_source, residual_norm = ga.init_atlas_tensor(spec)

    assert values.shape == (2, 3, 4, ga.ATLAS_VECTOR_SIZE)
    assert status.shape == (2, 3, 4)
    assert seed_source.shape == (2, 3, 4)
    assert residual_norm.shape == (2, 3, 4)
    assert np.isnan(values).all()
    assert np.all(status == ga.AtlasStatus.EMPTY)


def test_select_anchor_indices_centered():
    spec = ga.AtlasGridSpec(rho_points=7, kappa_points=9)
    i, j = ga.select_anchor_indices(spec)
    assert i == 3
    assert j == 4


def test_run_anchor_thread_populates_status_with_stubbed_solver(monkeypatch):
    spec = ga.AtlasGridSpec(rho_points=3, kappa_points=3, theta_points=4)
    rho, kappa, theta = ga.build_axes(spec)
    values, status, seed_source, residual_norm = ga.init_atlas_tensor(spec)
    ai, aj = ga.select_anchor_indices(spec)

    def fake_solve_target_fast(r_target, theta_target, seed_params, t_guess_days=0.0, config=None):
        del r_target, theta_target, config
        return np.asarray(seed_params, dtype=float), t_guess_days + 1.0, type("Info", (), {"fun": np.zeros(5)})()

    class DummySol:
        def __init__(self):
            self.y = np.zeros((8, 2), dtype=float)
            self.y[4, -1] = ga.DEFAULT_CONFIG.m_dry + 1.0

    monkeypatch.setattr(ga, "solve_target_fast", fake_solve_target_fast)
    monkeypatch.setattr(ga, "integrate_fixed_time", lambda *args, **kwargs: DummySol())

    ga.run_anchor_thread(values, status, seed_source, residual_norm, rho, kappa, theta, ai, aj, t_guess_days=10.0)

    assert np.all(status[ai, aj, :] == ga.AtlasStatus.SOLVED)
    assert np.all(np.isfinite(values[ai, aj, :, :]))
    assert np.all(np.isfinite(residual_norm[ai, aj, :]))


def test_run_wavefront_propagation_fills_grid_with_stubbed_solver(monkeypatch):
    spec = ga.AtlasGridSpec(rho_points=3, kappa_points=3, theta_points=3)
    rho, kappa, theta = ga.build_axes(spec)
    values, status, seed_source, residual_norm = ga.init_atlas_tensor(spec)
    ai, aj = ga.select_anchor_indices(spec)

    def fake_solve_target_fast(r_target, theta_target, seed_params, t_guess_days=0.0, config=None):
        del r_target, theta_target, config
        out = np.asarray(seed_params, dtype=float) + 0.1
        return out, t_guess_days + 1.0, type("Info", (), {"fun": np.array([0.1])})()

    class DummySol:
        def __init__(self):
            self.y = np.zeros((8, 2), dtype=float)
            self.y[4, -1] = ga.DEFAULT_CONFIG.m_dry + 1.0

    monkeypatch.setattr(ga, "solve_target_fast", fake_solve_target_fast)
    monkeypatch.setattr(ga, "integrate_fixed_time", lambda *args, **kwargs: DummySol())

    ga.run_anchor_thread(values, status, seed_source, residual_norm, rho, kappa, theta, ai, aj)
    ga.run_wavefront_propagation(values, status, seed_source, residual_norm, rho, kappa, theta, ai, aj)

    assert np.all(status != ga.AtlasStatus.EMPTY)
    assert np.all(np.isfinite(values[status == ga.AtlasStatus.SOLVED]))


def test_save_atlas_writes_expected_payload(tmp_path):
    spec = ga.AtlasGridSpec(rho_points=2, kappa_points=2, theta_points=2)
    rho, kappa, theta = ga.build_axes(spec)
    values, status, seed_source, residual_norm = ga.init_atlas_tensor(spec)
    meta = ga.AtlasMeta(
        version="vtest",
        anchor_i=0,
        anchor_j=0,
        anchor_completed=False,
        wavefront_completed=False,
        config_power_w=1.0,
        config_m0_kg=2.0,
        config_m_dry_kg=1.0,
        config_mu_si=3.0,
        notes="test",
    )
    out = tmp_path / "atlas.npz"
    ga.save_atlas(out, spec, rho, kappa, theta, values, status, seed_source, residual_norm, meta)

    data = np.load(out, allow_pickle=False)
    assert data["values"].shape == (2, 2, 2, ga.ATLAS_VECTOR_SIZE)
    assert data["status"].shape == (2, 2, 2)
    assert data["seed_source"].shape == (2, 2, 2)
    assert data["residual_norm"].shape == (2, 2, 2)
    assert json.loads(str(data["meta_json"]))["version"] == "vtest"
