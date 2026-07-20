# PMP Atlas: Implemented Adaptive Point Selection

This revision implements the point-selection plan in `pmp_extremal_atlas.py`.
The adaptive atlas format is now version 5.

## Implemented mechanisms

### 1. Cumulative feasible validation reservoir

The generator keeps a bounded, spatially diverse reservoir of known-feasible
forward extremals. Fresh independent audit rows are added every round. The log
reports reservoir and fresh-audit statistics separately.

Important controls:

```json
{
  "validation_rows": 1024,
  "validation_reservoir_rows": 8192,
  "validation_retest_rows_per_round": 2048,
  "validation_reservoir_cell_size": 0.10,
  "fresh_target_success": 0.90,
  "fresh_maximum_p95_distance": 3.0
}
```

The stopping criterion requires both cumulative-reservoir and fresh-audit
coverage, so memorizing a fixed holdout set cannot certify the atlas.

### 2. Acquisition-scored insertion

Candidate rows are scored by:

- distance from the existing atlas;
- distance relative to the nearest seed's empirical convergence radius;
- exact-correction failure;
- adaptive-cell coverage deficit;
- winding/radial-turn branch novelty;
- nearest-chart Jacobian conditioning.

High-scoring rows are selected with branch-aware mutual separation.

### 3. Exact endpoint Jacobians

The NPZ stores lazy exact 5x5 Jacobians

```text
d(log rho, theta, ur_f, ut_f, log kappa)
------------------------------------------------
d(Ar0, At0, Jr0, ell, log tau)
```

in:

```text
endpoint_jacobian
jacobian_condition
jacobian_sigma_min
```

Jacobians are calculated only for important chart seeds. They are used for:

- nearest-chart ranking;
- local inverse launch prediction;
- convergence-cell distances;
- frontier expansion;
- acquisition scoring;
- fold protection during pruning;
- normal user queries.

### 4. Frontier expansion

The generator identifies seeds facing failed validation probes and proposes
endpoint targets just beyond their current empirical cells. The local inverse
Jacobian predicts the new launch parameters, followed by exact shooting
correction.

### 5. Adaptive endpoint partition

A recursive endpoint partition is rebuilt from the validation reservoir. Cells
with failures or large p95 chart distances are split. Their deficit scores feed
the acquisition function, directing samples toward unresolved regions rather
than uniformly spending the launch budget.

### 6. Coverage-dominance pruning

Periodic and final pruning are conservative. A seed is removed only when:

- a nearby seed belongs to the same branch family;
- the seed is not the only representative of a rare branch;
- it is not protected as an ill-conditioned/fold-adjacent chart;
- its own endpoint remains exactly recoverable without it;
- every successful validation probe assigned to it remains exactly recoverable.

### 7. Target-family classification

Circular probes are stored with status codes:

```text
0 = untested
1 = covered
2 = tested but unresolved
```

in `circular_probe_status`. An unresolved direct target is not automatically
classified as physically infeasible.

## Atomic continuous saving

A resumable NPZ is atomically saved:

- before the first launch batch;
- after every bootstrap launch chunk during initial-seed and validation-reservoir construction;
- after the initial Jacobian batch;
- after every configured number of coverage rounds;
- on graceful Ctrl+C where possible;
- after final pruning and audit.

Once at least one seed row has survived, the incomplete NPZ is also queryable.
The maximum launch work at risk is controlled by
`checkpoint_bootstrap_launches` during bootstrap and by `batch_launches` during
the adaptive coverage loop.

Resume with:

```bash
python pmp_extremal_atlas.py generate atlas.npz \
  --config atlas_config_stratified_7d.json \
  --resume
```

Format-v5 checkpoints include the validation reservoir, Jacobians, circular
probe states, fresh audit, Sobol cursor, RNG state, and coverage history.

## Generation

```bash
python pmp_extremal_atlas.py generate interplanetary_extremals_v5.npz \
  --config atlas_config_stratified_7d.json
```

## Independent audit

```bash
python pmp_extremal_atlas.py validate interplanetary_extremals_v5.npz \
  --launches 8192 --rows 1024 --workers 8
```

## Notes on runtime

The largest added costs are exact shooting of the cumulative reservoir and
exact endpoint-Jacobian integration. The supplied configuration uses bounded
per-round budgets. Increase the following only after measuring runtime:

```text
validation_retest_rows_per_round
jacobian_compute_per_round
frontier_seeds_per_round
frontier_insertions_per_round
prune_max_checks
```

The generator remains embarrassingly parallel for forward launch integration
and Jacobian calculation.
