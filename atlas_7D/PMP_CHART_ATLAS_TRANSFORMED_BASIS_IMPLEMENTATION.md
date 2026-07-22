# Transformed-basis implementation notes

Implemented changes:

- NPZ format increased from 4 to 5.
- New charts fit their affine and low-rank quadratic inverse maps in center-rotated Cartesian final-state coordinates.
- The exact variational endpoint Jacobian is analytically pushed through the coordinate transformation before inversion.
- Controlled local-patch SVD directions are computed from the transformed normalized Jacobian.
- Trust ellipsoids are fitted and evaluated in transformed coordinates.
- Every chart stores its own seven transformed-coordinate scales.
- KD-tree routing uses a global inertial Cartesian transform; chart scoring then uses the chart-local rotated transform.
- Raw canonical endpoints remain stored for branch labels, exact correction, public query APIs, and witness targets.
- Version-4 charts load as raw-coordinate charts; version-5 charts can mix raw and transformed modes.
- The circular viewer uses transformed-space coverage scores and displays the selected transform for clicked trajectories.

Smoke tests completed:

- compilation of generator, viewer, and swarm dashboard;
- fresh transformed chart generation;
- atomic checkpoint writing and query-only loading;
- storage and reload of transform modes and per-chart scales;
- exact equality with the swarm dashboard's custom transform;
- analytical inverse round trip to numerical precision;
- migration of a version-4 checkpoint to version 5;
- independent validation command;
- static circular-viewer coverage rendering.
