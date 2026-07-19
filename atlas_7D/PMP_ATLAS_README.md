# Planar Kepler PMP extremal atlas

`pmp_extremal_atlas.py` builds a compact `.npz` atlas of normal PMP extremals
for the uncapped, constant-useful-power, variable-Isp model.

## Chosen representation

The atlas stores a **forward extremal map**, not a grid of full trajectories:

- launch data: `(u0, w0, Ar0, At0, Jr0, ell, tau)`;
- endpoint data: `(u0, w0, log(rho), Theta, ur_f, ut_f, log(kappa))`;
- branch/quality diagnostics.

Each integrated extremal contributes several canonically rescaled subarcs. A
query finds nearby atlas rows, builds a local inverse prediction, and then
solves the exact five-equation shooting problem with integrated variational
equations. The returned trajectory is therefore re-integrated and corrected;
it is not merely interpolated from stored pixels.

The solver returns the shortest **converged normal PMP extremal found among the
retrieved seeds**. This is not by itself a proof of global or local optimality.

## Install

```bash
pip install numpy scipy
```

## Generate an atlas

```bash
python pmp_extremal_atlas.py generate interplanetary_extremals.npz \
  --config atlas_config_example.json
```

Inspect it with:

```bash
python pmp_extremal_atlas.py inspect interplanetary_extremals.npz
```

## Query

See `pmp_atlas_example.py`. The core call is:

```python
atlas = ExtremalAtlas("interplanetary_extremals.npz")
solution = atlas.solve(initial_state, final_state, rocket, revolutions=0)
trajectory = solution.trajectory
```

The trajectory dictionary contains:

- `time`
- `position`
- `velocity`
- `acceleration`
- `jerk`
- `mass`
- `thrust`
- `exhaust_speed`
- `theta_unwrapped`
- `resource_fraction`

## Atlas bounds matter

There is no finite universal atlas. Set the launch and endpoint bounds to the
mission family you expect. Query failures report the distance to the nearest
stored endpoint. Increase the sample count, widen the relevant bounds, or try
a different winding number when the query lies outside good coverage.

A production atlas should normally be built in layers: a broad coarse atlas,
then additional focused atlases for important initial-state, radius, resource,
and winding ranges.
