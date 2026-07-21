# Controlled-patch PMP chart atlas

This revision replaces neighbour-cloud chart fitting with a connected local
experimental design around each exact chart center.

## Chart construction

For each failed feasible probe selected as a chart center, the generator:

1. Reintegrates the center at production tolerances and computes the exact
   five-by-five endpoint Jacobian.
2. Normalizes the Jacobian using `feature_scale` and `q_scale`.
3. Perturbs the launch along its five right singular vectors, along the `u0`
   and `w0` axes, and along deterministic mixed/random local directions.
4. Reintegrates every perturbation accurately and retains only points on the
   same winding/radial-turn branch.
5. Holds out part of this connected patch from fitting.
6. Fits the exact affine inverse plus a low-rank quadratic correction.
7. Benchmarks several bounded correctors on held-out patch targets.
8. Stores the corrector with the largest independently validated trust radius.
9. Rejects the chart unless it covers a nontrivial neighbourhood beyond its
   center.

The available bounded profiles are:

- `newton_balanced`
- `newton_aggressive`
- `newton_damped`
- `trf_fast`

Each chart stores a stable canonical method ID, so changing the enabled profile
order does not reinterpret existing charts.

## Generate

```bash
python pmp_chart_atlas.py generate interplanetary_chart_atlas_v3.npz \
  --config chart_atlas_config_controlled_patch.json \
  --solver pmp_extremal_atlas.py
```

## Resume

For a checkpoint created by this revision:

```bash
python pmp_chart_atlas.py generate interplanetary_chart_atlas_v3.npz \
  --config chart_atlas_config_controlled_patch.json \
  --solver pmp_extremal_atlas.py \
  --resume
```

For a compatible older chart checkpoint:

```bash
python pmp_chart_atlas.py generate interplanetary_chart_atlas.npz \
  --config chart_atlas_config_controlled_patch.json \
  --solver pmp_extremal_atlas.py \
  --resume --resume-policy-change
```

Old charts remain queryable and default to `newton_balanced`. New and refined
charts use controlled patches. Starting a new atlas is recommended when the old
bank consists mostly of minimum-radius point charts.

## New format-v3 arrays

- `chart_solver_method`: canonical bounded-corrector profile ID.
- `chart_patch_points`: number of accepted same-branch controlled perturbations.

The progress log reports the method distribution and median controlled-patch
size in addition to trust-radius quantiles.

## Healthy early-run indicators

A useful run should show:

- trust-radius median above `minimum_trust_radius`;
- a spread of trust radii rather than one fixed minimum;
- multiple corrector profiles being selected where appropriate;
- center self-test and local-inside success near 100%;
- fresh coverage increasing as failed probes become new charts.

If the chart acceptance rate is too low, first reduce `patch_axis_steps` or
`minimum_local_training_success`. Do not enlarge trust regions manually without
exact held-out validation.
