# PMP Extremal Atlas — format v5

## Files

- `pmp_extremal_atlas.py`: generator, validator, and exact query solver.
- `pmp_circular_subspace_viewer.py`: click viewer for the circular endpoint slice.
- `atlas_config_stratified_7d.json`: balanced production configuration.
- `PMP_ATLAS_POINT_SELECTION_IMPLEMENTED.md`: detailed selection design.

## Generate

```bash
python pmp_extremal_atlas.py generate interplanetary_extremals_v5.npz \
  --config atlas_config_stratified_7d.json
```

The output is saved atomically before generation, after every bootstrap launch
chunk, after initialization, and after each completed coverage round. The
bootstrap chunk size is controlled by `coverage.checkpoint_bootstrap_launches`.

## Resume

```bash
python pmp_extremal_atlas.py generate interplanetary_extremals_v5.npz \
  --config atlas_config_stratified_7d.json \
  --resume
```

The JSON configuration must match the checkpoint exactly.

## Inspect

```bash
python pmp_extremal_atlas.py inspect interplanetary_extremals_v5.npz
```

## Validate independently

```bash
python pmp_extremal_atlas.py validate interplanetary_extremals_v5.npz \
  --launches 8192 --rows 1024 --workers 8
```

## View circular-to-circular queries

```bash
python pmp_circular_subspace_viewer.py interplanetary_extremals_v5.npz \
  --solver pmp_extremal_atlas.py
```

Every viewer cell is an exact query, including cells without raw circular
samples.

## Format-v5 additions

```text
endpoint_jacobian       (n,5,5), float32
jacobian_condition      (n,)
jacobian_sigma_min      (n,)
persistent_probe_tested (m,)
circular_probe_status   (c,)
```

The atlas remains backward-readable: older NPZ files receive empty Jacobian
arrays and use global endpoint distance plus regression/direct seeds.

## Interpretation of coverage

Coverage means that known-feasible validation endpoints can be recovered by the
same exact shooting correction used for user queries. It is not a mathematical
proof that every point of the enclosing rectangular seven-dimensional domain is
feasible.
