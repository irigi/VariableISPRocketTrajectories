# Implementation summary

The chart generator is intentionally independent of the previous point-cloud
generation loop.

1. A chunked mixed-shard Sobol sampler generates full seven-dimensional launch
   attempts.
2. The PMP backend integrates them cheaply and harvests feasible subarcs.
3. A spatially thinned rolling pool supplies same-branch local regression data.
4. Existing chart trust regions classify covered and uncovered candidates.
5. Boundary audits use the production exact solver; false chart claims shrink
   and uncovered feasible extremals become chart centers.
6. At each center an accurate variational integration supplies the exact
   fixed-initial-velocity inverse Jacobian.
7. Nearby feasible rows estimate initial-velocity dependence and a regularized
   low-rank quadratic residual.
8. Known-launch prediction tests plus bounded exact local tests initialize the
   trust radius. Later global audits update it empirically.
9. Low-radius and failure-prone charts are periodically rebuilt after the
   rolling fit pool becomes denser.
10. Independent feasible probes measure production query success and latency.
11. Atomic NPZ checkpoints contain all state required to continue.

The final query router searches only nearby chart centers, evaluates their
nonlinear trust score, rejects unsupported targets, predicts at most a few
launches and certifies them with the bounded exact Newton solver.
