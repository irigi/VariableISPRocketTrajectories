# Implemented bulk-atlas changes

- Added `BulkShardConfig` and deterministic Sobol mixture sampling.
- Added log-polar acceleration sampling and signed-log jerk/invariant sampling.
- Added optional Numba RK4 offline generation backend.
- Reused a persistent initialized process pool across all forward batches.
- Added cheap branch-aware new-cell insertion before acquisition scoring.
- Increased configurable bulk and acquisition insertion capacities.
- Removed all targeted circular-row insertion paths.
- Forced circular generation and standalone validation to use direct bounded production queries with zero continuation.
- Added sparse Jacobian checkpoint representation and compatible resume/loading.
- Retained full Jacobian arrays in completed atlases for query compatibility.
- Changed format metadata to version 6.
- Made viewer coverage maps displayed-slice-only, parallel, and cached.
- Preserved atomic checkpointing and resume behavior.
