# PMP local swarm linearization dashboard

`pmp_local_swarm_dashboard.py` perturbs a selected launch on a normalized 7D sphere or ball, integrates the entire swarm, applies an analytical endpoint transformation, and plots all 21 pairwise projections in 4x3 dashboard pages.

The purpose is to tune a transformation that makes the endpoint map as close to affine as possible before using it in a future atlas.

## Three views

For a selected transformation the script writes:

1. **Raw endpoint dashboard**
2. **Analytically transformed endpoint dashboard**
3. **Local affine pullback dashboard**

The pullback first fits the best local affine map from normalized launch perturbations to transformed outputs and then maps the transformed outputs back into launch perturbation coordinates. If the analytical transformation is effective, the pullback should look close to the original nested sphere or ball. Curvature, folds, and outliers remain visible as deviations from that shape.

## Endpoint and launch coordinates

Raw endpoint:

```text
(u0, w0, log_rho, theta_unwrapped, ur_final, ut_final, log_kappa)
```

Launch perturbation:

```text
(u0, w0, Ar0, At0, Jr0, ell, log_tau)
```

`tau` is perturbed multiplicatively through `log_tau`.

## User-tunable function

The script contains an isolated section headed:

```text
USER-TUNABLE ANALYTICAL OUTPUT TRANSFORM
```

Edit:

```python
custom_analytical_transform(raw_endpoints, center_raw_endpoint, params)
```

The function receives an `N x 7` endpoint array and must return another `N x 7` array plus seven axis labels.

The current custom candidate uses final Cartesian position and velocity in a frame rotated by the nominal trajectory's final angle:

```text
(u0, w0, x_rot, y_rot, vx_rot, vy_rot, log_kappa)
```

This removes the direct polar radius-angle coupling. Winding can later remain separate discrete branch metadata.

An optional smooth compression can be tested without editing code:

```bash
--transform custom --transform-param asinh_scale=1.0
```

## Built-in transformations

```text
raw
cartesian
cartesian_rotated
equinoctial
custom
```

`equinoctial` uses a planar modified-equinoctial-like state:

```text
(u0, w0, log_p, ecc_x, ecc_y, delta_theta, log_kappa)
```

## Typical run

```bash
python pmp_local_swarm_dashboard.py \
  --solver pmp_extremal_atlas.py \
  --config chart_atlas_config_long.json \
  --atlas-row interplanetary_chart_atlas_v4.npz 100 \
  --radius 0.03 \
  --points 4096 \
  --shells 4 \
  --transform custom \
  --workers 8 \
  --output-prefix local_linearization \
  --save-npz
```

Or use an explicit launch center:

```bash
python pmp_local_swarm_dashboard.py \
  --solver pmp_extremal_atlas.py \
  --center 0 1 0.1 0 0 0 2 \
  --radius 0.01 \
  --points 4096 \
  --shells 4 \
  --transform cartesian_rotated \
  --workers 8 \
  --output-prefix local_linearization
```

## Outputs

For prefix `local_linearization` and transform `custom`:

```text
local_linearization_raw_dashboard_01.png
local_linearization_raw_dashboard_02.png
local_linearization_raw.pdf

local_linearization_custom_dashboard_01.png
local_linearization_custom_dashboard_02.png
local_linearization_custom.pdf

local_linearization_custom_pullback_dashboard_01.png
local_linearization_custom_pullback_dashboard_02.png
local_linearization_custom_pullback.pdf

local_linearization_linearity_metrics.json
local_linearization.npz                 # with --save-npz
```

## Quantitative comparison

The JSON report includes, for raw and transformed coordinates:

- relative RMS nonlinear residual after the best affine fit;
- p50 and p95 output residual norms;
- p50 and p95 pullback errors in normalized launch space;
- affine Jacobian singular values and condition number;
- per-coordinate R-squared and relative RMS residual;
- transformation improvement factor.

The main initial ranking metric is:

```text
relative_rms_nonlinearity
```

Smaller is better. The pullback plots remain important because one scalar metric can hide folds or strongly directional errors.

## Comparing candidates

Run the same center, radius, point count, shells, and random seed with different transforms:

```bash
--transform raw
--transform cartesian_rotated
--transform equinoctial
--transform custom
```

Do not compare transformations from different swarms or radii because the nonlinear residual depends strongly on the sampled neighborhood.

## Practical notes

- Start with an atlas center known to integrate successfully.
- If many points are `non-normal` or hit guard events, reduce `--radius`.
- Use at least 4096 points for a visually dense 7D swarm; 8192 is useful for detailed tuning.
- Nested shells make radial deformation easier to see. Use `--mode ball` to inspect filled-volume folds.
- The analytical transform is evaluated before the local affine pullback. The affine pullback is diagnostic and is not itself proposed as a global atlas coordinate transformation.
