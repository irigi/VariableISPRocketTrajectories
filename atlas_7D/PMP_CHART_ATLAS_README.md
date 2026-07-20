# Validated local-inverse PMP chart atlas

This package replaces the raw seven-dimensional point cloud with an overlapping
cover of branch-aware local inverse charts. The old `pmp_extremal_atlas.py`
file remains the dynamics and exact shooting backend; it is not used as the
atlas representation.

## Representation

Each chart stores:

- an accurate canonical endpoint center
  `[u0,w0,log_rho,theta,ur_f,ut_f,log_kappa]`;
- its exact feasible launch `[u0,w0,Ar0,At0,Jr0,ell,tau]`;
- a normalized affine inverse map from endpoint displacement to
  `[Ar0,At0,Jr0,ell,log_tau]`;
- a low-rank quadratic inverse correction;
- an empirically validated scalar radius in the chart's anisotropic prediction
  metric;
- winding/radial-turn diagnostics, Jacobian conditioning, latency and
  success/failure counts.

A target is *certified covered* only when it lies inside a chart's validated
trust region. The production bounded Newton corrector still verifies every
returned trajectory.

## Long generation

```bash
python pmp_chart_atlas.py generate interplanetary_chart_atlas.npz \
  --config chart_atlas_config_long.json \
  --solver pmp_extremal_atlas.py
```

The supplied long configuration permits up to 100 million bulk launch attempts
and 500,000 charts, but stops early after the persistent and fresh feasible
validation suites satisfy both success and p95-latency criteria for five
consecutive rounds.

Circular trajectories are never inserted. They remain a thin independent
coverage probe in the viewer.

## Continuous checkpoints

The output NPZ is created before the first integration. It is atomically
replaced:

- after feasible candidate selection;
- after every configured chart-build batch;
- after every validation round;
- after normal completion;
- when Ctrl+C is caught.

A write goes to a temporary file and is `fsync`ed before replacement, so the
last complete checkpoint remains valid if a write is interrupted.

A deterministic Sobol cursor is advanced only after a complete forward batch.
If integration is interrupted, restart repeats that unfinished batch rather
than skipping it.

## Restart

Use the exact same JSON file:

```bash
python pmp_chart_atlas.py generate interplanetary_chart_atlas.npz \
  --config chart_atlas_config_long.json \
  --solver pmp_extremal_atlas.py \
  --resume
```

The NPZ restores charts, the rolling local-fit pool, feasible validation probes,
pending chart centers, Sobol cursor, coverage history and patience counter.
Configuration changes are rejected because they would invalidate the restart
state.

## Checkpoint compression and final repacking

The long-run configuration uses uncompressed NPZ checkpoints because rewriting
a growing compressed ZIP archive after every chart batch can dominate runtime.
The chart representation is still substantially smaller than the raw point
cloud. After a run or interruption, create a compressed delivery copy with:

```bash
python pmp_chart_atlas.py repack \
  interplanetary_chart_atlas.npz \
  interplanetary_chart_atlas_compressed.npz
```

The original restartable checkpoint is unchanged.

## Inspect and validate

```bash
python pmp_chart_atlas.py inspect interplanetary_chart_atlas.npz
```

```bash
python pmp_chart_atlas.py validate interplanetary_chart_atlas.npz \
  --solver pmp_extremal_atlas.py \
  --launches 8192 --rows 1024
```

Validation generates a fresh feasible full-arc set and uses the same chart
router and bounded exact corrector as production queries.

## Circular-section viewer

```bash
python pmp_chart_circular_viewer.py interplanetary_chart_atlas.npz \
  --chart-module pmp_chart_atlas.py \
  --solver pmp_extremal_atlas.py
```

The background is `log10(best trust ratio)`:

- ratio <= 1: inside at least one validated chart;
- ratio > 1: unsupported by the current certified cover.

The viewer calculates only displayed angle slices in batched KD-tree calls and
caches them beside the atlas. Clicking a pixel runs the production bounded
corrector. A successful query opens the corrected canonical trajectory; a
failed or unsupported query is marked on the panel.

## Programmatic use

```python
from pmp_chart_atlas import LocalInverseChartAtlas

atlas = LocalInverseChartAtlas(
    "interplanetary_chart_atlas.npz",
    "pmp_extremal_atlas.py",
)

# Canonical endpoint query.
launch, chart_index, ratio, reason, seconds = atlas.correct_target(
    target7,
    return_reason=True,
)
```

For dimensional boundary states, use `atlas.solve(initial, final, capability,
revolutions=N)`. The state and capability classes are provided by
`pmp_extremal_atlas.py`.

## Important behavior

- Bulk forward rows are temporary chart-discovery and regression data. They are
  not treated as solved coverage by themselves.
- Chart centers and validation targets are recomputed with the accurate
  production integrator before certification, even when bulk discovery uses
  the faster Numba RK4 backend.
- Failed audits inside a claimed trust region shrink it. Successful audits just
  outside a region may enlarge it, bounded by the configured maximum.
- Sparse early charts may begin as exact affine point charts. As the bulk fit
  pool becomes denser, low-radius or failure-prone charts are periodically
  rebuilt from richer same-branch neighbourhoods and replaced only when their
  validated radius or failure behavior improves.
- `coverage_converged=false` does not make a checkpoint unusable. It means only
  that the configured global success/latency certificate has not yet passed.
