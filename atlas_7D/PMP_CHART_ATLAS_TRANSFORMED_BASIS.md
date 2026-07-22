# PMP chart atlas — analytically transformed endpoint basis

The chart atlas now uses the same default analytical transformation that made the swarm pullback substantially more regular.

## External and internal coordinates

The external canonical target remains

```text
(u0, w0, log_rho, theta_unwrapped, ur_final, ut_final, log_kappa)
```

For a chart centered at canonical endpoint `p_c`, the local model and its trust ellipsoid use

```text
(u0, w0, x_rot, y_rot, vx_rot, vy_rot, log_kappa)
```

where

```text
delta = theta - theta_c
x_rot  = rho*cos(delta)
y_rot  = rho*sin(delta)
vx_rot = ur*cos(delta) - ut*sin(delta)
vy_rot = ur*sin(delta) + ut*cos(delta)
```

This is the same `custom` transform used by `pmp_local_swarm_dashboard.py` with `asinh_scale=0`.

Winding remains discrete branch metadata. It is not inferred from the Cartesian coordinates.

## Query flow

1. A user or viewer supplies a raw canonical target.
2. Candidate routing converts it to an inertial Cartesian search coordinate:

   ```text
   (u0,w0,x,y,vx,vy,log_kappa)
   ```

3. For each candidate chart, the target is transformed again into that chart's center-rotated basis.
4. The stored local polynomial and ellipsoidal trust test are evaluated in that basis.
5. The predicted launch is passed to the exact corrector using the original raw canonical target.

The module also contains the analytical inverse transformation for diagnostics and later transformed-target APIs. Exact shooting never relies on an approximate inverse transformation.

## Per-chart scaling

When `dynamic_transform_scale=true`, each chart stores a local transformed scale derived analytically from its center and the original canonical feature scales. In particular, angular scales become physical tangential position and velocity scales. This prevents one fixed Cartesian scale from being inappropriate at very different radii.

The NPZ format is now version 5 and stores:

```text
chart_transform_mode
chart_transform_scale
```

## Fresh run

```bash
python pmp_chart_atlas.py generate \
  interplanetary_chart_atlas_transformed.npz \
  --config chart_atlas_config_transformed.json \
  --solver pmp_extremal_atlas.py
```

## Resume

```bash
python pmp_chart_atlas.py generate \
  interplanetary_chart_atlas_transformed.npz \
  --config chart_atlas_config_transformed.json \
  --solver pmp_extremal_atlas.py \
  --resume
```

## Migrate a version-4 checkpoint

```bash
python pmp_chart_atlas.py generate \
  interplanetary_chart_atlas.npz \
  --config chart_atlas_config_transformed.json \
  --solver pmp_extremal_atlas.py \
  --resume \
  --resume-policy-change
```

Existing version-4 charts remain tagged as `raw`, because their coefficients and ellipsoids were fitted in canonical polar coordinates. Newly generated and refined charts use `custom_rotated_cartesian`. A fresh run is preferable for a homogeneous transformed atlas.

## Circular viewer

```bash
python pmp_chart_circular_viewer.py \
  interplanetary_chart_atlas_transformed.npz \
  --chart-module pmp_chart_atlas.py \
  --solver pmp_extremal_atlas.py
```

The background is now the best transformed-space ellipsoidal trust ratio. Clicked solutions report the chart type, selected corrector, patch scale, transform mode, and trust ratio.

## Tuning the transformation

The corresponding transform code is in the marked analytical-transform section near the top of `pmp_chart_atlas.py` and in `custom_analytical_transform` in the swarm dashboard. When the analytical map is changed, both the forward map and `_local_output_jacobian5` must be changed consistently. Start a fresh atlas whenever the transform itself changes; old coefficients cannot be reinterpreted in a different nonlinear coordinate system.
