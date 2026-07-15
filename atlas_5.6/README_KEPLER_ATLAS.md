# Kepler extremal atlas solver

`kepler_atlas_solver.py` constructs a dimensionless atlas for the uncapped,
constant-useful-power, variable-exhaust-speed model in the planar Kepler field.
It stores the result as a compressed NumPy `.npz` file.

## Important scope

The mathematical parameter space `(rho, Theta, kappa)` is unbounded, so a
literal finite "whole atlas" does not exist. The script maps a configurable
rectangular domain. Its default domain is broad:

- `rho`: 0.5 to 8, logarithmically spaced;
- unwrapped `Theta`: 0.2 to 2 pi;
- `kappa`: 0.75 to 24, logarithmically spaced;
- default grid: 13 x 25 x 13 = 4,225 target points.

Every stored solution is a PMP extremal candidate. It is not a proof of global
or even strict local optimality. Disconnected solution sheets require multiple
branch seeds; use `--max-branches` together with `--anchor-perturbations`.

## Why it is faster than independent per-pixel searches

The solver:

1. constructs an exact gravity-free seed;
2. deforms it to the Kepler problem with a gravity homotopy;
3. continues neighboring solutions through `(log rho, Theta, log kappa)`;
4. integrates exact first-variation equations, obtaining the complete 5 x 5
   shooting Jacobian in one ODE integration;
5. uses predictor-corrector continuation rather than a global optimizer at each
   target;
6. fails fast on pathological close-Sun trial trajectories;
7. checkpoints automatically and resumes from the NPZ file.

## Requirements

- Python 3.10+
- NumPy
- SciPy

## Quick validation

```bash
python kepler_atlas_solver.py smoke --output smoke_atlas.npz
python kepler_atlas_solver.py inspect smoke_atlas.npz
pytest -q test_kepler_atlas_solver.py
```

## Default atlas

```bash
python kepler_atlas_solver.py build --output kepler_atlas.npz
```

The default is configured as a practical single-laptop run, not as a guarantee
of a particular wall time. Runtime depends strongly on CPU, requested domain,
minimum perihelion, tolerance, and how many disconnected branches are sought.

## A safer first production run

```bash
python kepler_atlas_solver.py build \
  --output kepler_atlas.npz \
  --rho-min 0.7 --rho-max 6 --n-rho 13 \
  --theta-min 0.2 --theta-max 6.283185307179586 --n-theta 25 \
  --kappa-min 1.5 --kappa-max 20 --n-kappa 13 \
  --min-radius 0.02
```

`min-radius` is dimensionless relative to the departure radius. Setting it very
small makes close-Sun branches increasingly stiff and can dominate runtime.

## Multiple candidate branches

```bash
python kepler_atlas_solver.py build \
  --output kepler_atlas_multibranch.npz \
  --max-branches 3 \
  --anchor-perturbations 24 \
  --perturbation-scale 0.35
```

This is more expensive and does not guarantee that every disconnected sheet is
found. It retains distinct converged branches sorted by flight time.

## Resume after interruption

Run the same command again. The script validates the axes and resumes from the
existing output file. Use `--no-resume` to overwrite the calculation from
scratch.

## NPZ contents

- `rho`, `theta`, `kappa`: one-dimensional axes;
- `shooting[..., branch, 5]`: `(A_r0, A_t0, J_r0, rotational_invariant, tau_f)`;
- `dxdp[..., branch, 5, 3]`: local derivative of shooting parameters with
  respect to `(log rho, Theta, log kappa)`;
- `valid`: converged-entry mask;
- `residual_norm`: terminal residual norm;
- `condition`: shooting-Jacobian condition number;
- `iterations`, `nfev`: solver diagnostics;
- `status`: per-grid-point status;
- `metadata_json`: configuration and format metadata.

## Dimensionless equations represented by the shooting data

```text
R' = V
V' = -R/|R|^3 + A
A' = J
J' = -|R|^-3 (I - 3 Rhat Rhat^T) A
q' = |A|^2
theta' = cross(R,V)/|R|^2
```

Initial state:

```text
R=(1,0), V=(0,1)
A=(A_r0,A_t0)
J=(J_r0,-rotational_invariant-A_r0)
q=0, theta=0
```

The terminal conditions enforce radius `rho`, unwrapped angle `Theta`, zero
radial velocity, circular tangential velocity `rho^-1/2`, and `q=kappa`.
