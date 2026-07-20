# Bulk 7D PMP atlas, format 6

This revision builds only bulk seven-dimensional forward-extremal atlases. Exact circular-to-circular states are retained as validation probes, but successful circular corrections are never inserted as a separate shard or as targeted seeds.

## Main changes

### Mixed bulk launch shards

`AtlasConfig.bulk_shards` partitions the Sobol launch budget among several full seven-dimensional launch distributions. The supplied configuration includes:

- short, near-ballistic trajectories;
- medium-duration, low-control trajectories;
- medium-duration, high-control trajectories;
- long-duration, low-control trajectories;
- a broad Cartesian-uniform shard preserving box corners and linear sampling near zero.

All accepted rows are merged into one NPZ and queried through one atlas. These are sampling shards, not mission-family atlases.

The log-polar shards sample acceleration magnitude logarithmically and direction uniformly. Radial jerk and the rotational invariant are signed-log sampled, allocating points both near zero and near the configured extremes. Flight time is log-uniform inside each shard.

### Faster forward generation

- `generation_backend: "numba_rk4"` uses a compiled fixed-step RK4 integrator for offline forward trajectories.
- Query correction and validation still use the adaptive DOP853 variational integrator.
- Worker processes persist across bootstrap and coverage batches instead of being recreated for every chunk.
- `worker_batch_size` controls task granularity.
- Checkpoints can store Jacobians sparsely while the atlas is incomplete.

The compiled generation path is intended to produce approximate seed endpoints quickly. Exact trajectory queries are still corrected with the accurate solver. Set `generation_backend` to `dop853` when conservative offline integration accuracy is more important than throughput.

### High-throughput cell filling

Each adaptive round first inserts known-feasible forward rows occupying previously empty branch-aware endpoint cells. This stage requires no shooting solve. A smaller acquisition-scored stage then adds unresolved, distant, branch-novel, or poorly conditioned candidates.

The supplied configuration permits up to:

- 32,768 new-cell rows per round;
- 16,384 acquisition rows per round.

Pruning and frontier continuation are disabled in the production configuration because earlier runs showed negligible removal/expansion relative to their cost.

### Circular probes are validation-only

Circular targets are tested by the same bounded production solver used by fresh general validation and by the viewer:

- no robust fallback;
- no continuation;
- no circular insertion;
- the same iteration, seed, and wall-clock limits.

Thus circular success remains a difficult diagnostic section of the bulk 7D atlas rather than a separately trained family.

### Faster viewer startup

For `nearest_distance`, `validated_radius`, and `coverage_ratio`, the viewer now:

1. determines the angle panels that will actually be displayed;
2. constructs only those target slices;
3. submits one batched `cKDTree.query(..., workers=-1)` call;
4. caches the result in `<atlas-stem>.viewer_distance_cache.npz`.

With the default 3×4 catalogue, only 12 angle slices are calculated instead of all 120 slices. Reopening the same atlas/grid normally loads the cache.

## Generate a new atlas

The sampling distribution differs from earlier versions, so use a new output file rather than resuming an old format-5 cloud:

```bash
python pmp_extremal_atlas.py generate interplanetary_extremals_bulk_v6.npz \
  --config atlas_config_bulk_fast.json
```

Checkpoints are written atomically after each bootstrap chunk and coverage round. Resume with:

```bash
python pmp_extremal_atlas.py generate interplanetary_extremals_bulk_v6.npz \
  --config atlas_config_bulk_fast.json \
  --resume
```

The supplied production configuration uses:

- 524,288 launch attempts;
- 65% for the initial bulk cloud;
- 32,768 persistent validation launches;
- 32,768 launches per adaptive batch;
- a 0.5-second production-query budget;
- two seed attempts and three Newton iterations;
- uncompressed atomic checkpoints with sparse Jacobian storage.

## Validate with the production policy

```bash
python pmp_extremal_atlas.py validate interplanetary_extremals_bulk_v6.npz \
  --launches 8192 \
  --rows 1024 \
  --workers 8 \
  --query-method fast_newton \
  --query-seconds 0.5 \
  --query-iterations 3 \
  --query-seeds 2
```

A point is covered only when it converges under that bounded query policy.

## Viewer

```bash
python pmp_circular_subspace_viewer.py interplanetary_extremals_bulk_v6.npz \
  --solver pmp_extremal_atlas.py \
  --metric nearest_distance \
  --distance-workers -1 \
  --query-seconds 0.5 \
  --query-iterations 3 \
  --query-seeds 2
```

The first run creates the sidecar distance cache. Disable it with `--no-distance-cache` or select a path with `--distance-cache`.

## Accuracy note

The fixed-step backend was regression-tested against high-accuracy DOP853 on a small representative sample. It produced endpoint errors far below the atlas feature scales in that test, but long-duration/extreme-control production runs should still be independently audited. Reducing `max_step`, or switching generation back to `dop853`, is the conservative response if discrepancies appear.
