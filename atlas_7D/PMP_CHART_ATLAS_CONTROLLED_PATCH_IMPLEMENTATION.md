# Implementation summary

- Format version increased to 3.
- Controlled same-center perturbation patches replace unrelated-neighbour
  fitting by default.
- Patch axes use normalized Jacobian right singular vectors plus `u0`, `w0`,
  mixed and random directions.
- Every patch point is accurately reintegrated and branch-filtered.
- Fits and validations use disjoint patch subsets.
- A pilot tournament evaluates all enabled bounded correctors; the two best
  profiles complete the held-out benchmark.
- The selected profile and validated radius are stored per chart.
- Exact validation is the trust certificate; interpolation error is only a
  stability guard.
- Old format-v2 checkpoints load with default method IDs and can be resumed
  under compatible policy changes.
- The viewer automatically uses each chart's stored corrector and displays the
  selected method on trajectory plots.
