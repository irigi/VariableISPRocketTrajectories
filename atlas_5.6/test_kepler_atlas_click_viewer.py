import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

import kepler_atlas_click_viewer as viewer


ROOT = Path(__file__).resolve().parent
ATLAS = ROOT / "kepler_atlas_smoke.npz"
SOLVER = ROOT / "kepler_atlas_solver.py"


def test_load_and_metrics():
    atlas = viewer.load_atlas(ATLAS)
    branch_map = viewer.choose_branch_map(atlas, "shortest")
    assert atlas.shape == (3, 4, 3)
    assert np.all(branch_map >= 0)
    tau = viewer.metric_cube(atlas, branch_map, "tau")
    assert tau.shape == atlas.shape
    assert np.all(np.isfinite(tau))


def test_replay_first_solution():
    atlas = viewer.load_atlas(ATLAS)
    solver = viewer.load_module_from_path(SOLVER)
    config = viewer.solver_config_from_metadata(solver, atlas)
    branch_map = viewer.choose_branch_map(atlas, "shortest")
    i, j, k = map(int, np.argwhere(np.any(atlas.valid, axis=-1))[0])
    branch = int(branch_map[i, j, k])
    replay = viewer.integrate_replay(
        solver,
        config,
        atlas.shooting[i, j, k, branch],
        float(atlas.rho[i]),
        float(atlas.theta[j]),
        float(atlas.kappa[k]),
        branch,
        240,
    )
    assert replay.state.shape == (10, 240)
    assert np.linalg.norm(replay.residual) < 1.0e-6


def test_catalogue_figure_builds():
    atlas = viewer.load_atlas(ATLAS)
    solver = viewer.load_module_from_path(SOLVER)
    config = viewer.solver_config_from_metadata(solver, atlas)
    branch_map = viewer.choose_branch_map(atlas, "shortest")
    figure = viewer.make_catalogue(
        atlas,
        solver,
        config,
        branch_map,
        metric="tau",
        nrows=2,
        ncols=2,
        samples=120,
        figsize=(8.0, 6.0),
    )
    assert figure is not None
    plt.close(figure)
