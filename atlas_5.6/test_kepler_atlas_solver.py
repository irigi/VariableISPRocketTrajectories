import math
from pathlib import Path

import numpy as np

import kepler_atlas_solver as kas


def test_free_space_seed_satisfies_terminal_problem():
    cfg = kas.SolverConfig(gravity_steps=4)
    p = np.array([math.log(1.5), 0.6, math.log(7.5)])
    x = kas.free_space_seed(p)
    residual, *_ = kas.residual_and_jacobian(x, p, 0.0, cfg)
    assert np.linalg.norm(residual) < 1e-8


def test_analytic_shooting_jacobian_matches_finite_difference():
    cfg = kas.SolverConfig(gravity_steps=5, min_radius=0.02)
    p = np.array([math.log(1.5), 0.6, math.log(7.5)])
    solution = kas.gravity_homotopy_seed(p, cfg)
    assert solution.success
    _, analytic, _, _, _ = kas.residual_and_jacobian(solution.x, p, 1.0, cfg)

    finite = np.zeros_like(analytic)
    for column in range(5):
        h = 1e-6 * max(abs(solution.x[column]), 1.0)
        xp = solution.x.copy()
        xm = solution.x.copy()
        xp[column] += h
        xm[column] -= h
        fp = kas.residual_only(xp, p, 1.0, cfg)[0]
        fm = kas.residual_only(xm, p, 1.0, cfg)[0]
        finite[:, column] = (fp - fm) / (2.0 * h)

    relative_error = np.linalg.norm(analytic - finite) / np.linalg.norm(finite)
    assert relative_error < 2e-7


def test_small_atlas_round_trip(tmp_path: Path):
    output = tmp_path / "atlas.npz"
    cfg = kas.SolverConfig(
        gravity_steps=4,
        checkpoint_every=2,
        progress_every=100,
        newton_tol=7e-8,
    )
    builder = kas.AtlasBuilder(
        np.geomspace(1.25, 1.75, 2),
        np.linspace(0.4, 0.8, 2),
        np.geomspace(5.5, 9.0, 2),
        cfg,
        output,
    )
    builder.build(anchor=(1.45, 0.55, 7.0), resume=False)
    with np.load(output, allow_pickle=False) as atlas:
        assert atlas["valid"].shape == (2, 2, 2, 1)
        assert np.all(np.any(atlas["valid"], axis=-1))
        assert np.nanmax(atlas["residual_norm"][atlas["valid"]]) < 1e-6
