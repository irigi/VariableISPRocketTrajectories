# Fast production query solver

This revision makes atlas coverage mean: **the same bounded numerical method used by the viewer converges within a small query budget**.

## Main conclusions from the previous run

- Fresh feasible probes converged at roughly 96–98%, but the exact circular family stalled near 23%. The forward atlas covers common arbitrary endpoints much better than the circular-to-circular slice.
- The cumulative reservoir success drifted downward because the reservoir kept adding deliberately difficult and distant probes. This is expected for a growing hard-probe bank.
- p95/p99 geometric distances increased even while correction success stayed high. Global feature-space Euclidean distance is therefore not a reliable hard stopping criterion. It is now diagnostic by default.
- The acquisition budget was saturated every round (`inserted=1536`), so the launch budget ended before the selected domain was covered under the requested special circular family.

## Fast method

The default query corrector is `fast_newton`:

1. Rank nearby atlas charts, including stored endpoint-Jacobian predictions.
2. Try at most a configured number of seeds.
3. Use a damped, regularized Newton step with exact variational Jacobian.
4. Use state-only integrations for the short line search.
5. Enforce a wall-clock deadline inside the ODE RHS.
6. Mark the query failed when the budget is exhausted.

Robust trust-region least squares and endpoint continuation are optional and disabled by default.

## Generate

```bash
python pmp_extremal_atlas.py generate interplanetary_extremals_fast.npz \
  --config atlas_config_fast_query.json
```

The generator prints the query policy used by validation. Coverage success is evaluated with the same bounded method as the viewer.

## Continue an old checkpoint using the new query policy

```bash
python pmp_extremal_atlas.py generate interplanetary_extremals_v3.npz \
  --config atlas_config_fast_query.json \
  --resume --resume-policy-change
```

`--resume-policy-change` permits changes only to the adaptive/query policy. Launch bounds, seed, integration guards, feature scaling and sample budget must still match.

## Independent validation

```bash
python pmp_extremal_atlas.py validate interplanetary_extremals_fast.npz \
  --launches 8192 --rows 1024 --workers 8 \
  --query-method fast_newton \
  --query-seconds 2 \
  --query-iterations 5 \
  --query-seeds 3
```

The report records the exact query policy.

## Viewer

```bash
python pmp_circular_subspace_viewer.py interplanetary_extremals_fast.npz \
  --solver pmp_extremal_atlas.py \
  --query-method fast_newton \
  --query-seconds 2 \
  --query-iterations 5 \
  --query-seeds 3
```

Failed cells are marked with a red `x` and cached. Successful cells are marked with a circle and their trajectory is cached, so repeated clicks do not solve again.

For diagnostic comparison only:

```bash
--query-method robust_least_squares --robust-fallback --allow-continuation
```

This can be much slower and should not be used to certify a quick-response atlas.

## Recommended coverage semantics

Keep these false for the production certificate:

```json
"distance_criteria_enabled": false,
"circular_distance_criteria_enabled": false
```

Nearest distances remain useful diagnostics, but success under the bounded production solver is the natural convergence-cell definition.

The circular slice should still be validated separately. If circular coverage remains poor, generate a dedicated circular-start/circular-end shard or use expensive offline continuation only to insert additional seeds; do not count those expensive solves as production-query successes.
