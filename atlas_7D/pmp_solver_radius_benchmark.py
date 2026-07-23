#!/usr/bin/env python3
"""Benchmark PMP boundary-value solver convergence radius.

For each random feasible launch center, this script creates exact spherical
launch-space swarms at a sequence of normalized radii.  Each swarm launch is
integrated forward to obtain a known-feasible endpoint.  The boundary-value
problem for that endpoint is then solved from the *central* launch seed.

The experiment measures recovery probability versus initial launch-space
radius and estimates, for every center and nonlinear solver, the radius R90 at
which 90% of known-feasible boundary-value problems remain recoverable.
Requested 5/25/50/75/95 percentiles of R90 are written to summary files.

All exact shooting evaluations use the existing solver's adaptive DOP853
trajectory propagation.  The methods compared here are nonlinear BVP
correctors (Newton, trust-region least squares, LM/hybrid root, and optional
continuation), not alternative ODE propagators.

Results are stored continuously in SQLite and can be resumed safely.

With ``--nonlinear-basis``, the target population is unchanged, but each BVP
corrector works in the same rotated-Cartesian analytical output coordinates
used by ``pmp_local_swarm_dashboard.py``.  An exact local affine pullback,
formed from the variational Jacobian at the central seed, maps transformed
endpoint residuals into normalized launch-coordinate residuals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, root
from scipy.special import expit, logit


ALL_METHODS = (
    "newton_balanced",
    "newton_aggressive",
    "newton_damped",
    "robust_trf_dop853",
    "robust_dogbox_dop853",
    "lm_dop853",
    "hybr_dop853",
    "hybrid_newton_trf",
    "continuation_trf_4",
)
DEFAULT_METHODS = (
    "newton_balanced",
    "newton_aggressive",
    "newton_damped",
    "robust_trf_dop853",
    "robust_dogbox_dop853",
    "lm_dop853",
    "hybrid_newton_trf",
    "continuation_trf_4",
)
DEFAULT_RADII = (0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.18, 0.25)

_WORKER_SOLVER = None
_WORKER_BACKEND = None
_WORKER_QUERY = None


class MethodTimeout(RuntimeError):
    pass


def _load_module(path: str):
    name = f"pmp_radius_solver_{abs(hash(os.path.abspath(path))) & 0xFFFFFFFF:x}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load solver module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _read_config_payload(path: str) -> tuple[dict, Optional[dict]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "backend_config" in payload:
        return dict(payload["backend_config"]), dict(payload.get("query_config", {}))
    return dict(payload), None


def _build_configs(solver, config_path: str):
    backend_payload, query_payload = _read_config_payload(config_path)
    backend_payload["n_samples"] = max(1, int(backend_payload.get("n_samples", 1)))
    backend = solver._atlas_config_from_dict(backend_payload)
    backend.validate()
    if query_payload:
        query = solver.QueryConfig(**query_payload)
    else:
        query = solver._coverage_query_config(backend)
    # Always use the accurate query-side integration guards/tolerances.
    query.r_collision = backend.r_collision
    query.r_escape = backend.r_escape
    query.max_acceleration = backend.max_acceleration
    return backend, query


def _worker_init(solver_path: str, backend_dict: dict, query_dict: dict):
    global _WORKER_SOLVER, _WORKER_BACKEND, _WORKER_QUERY
    _WORKER_SOLVER = _load_module(solver_path)
    _WORKER_BACKEND = _WORKER_SOLVER._atlas_config_from_dict(backend_dict)
    _WORKER_QUERY = _WORKER_SOLVER.QueryConfig(**query_dict)


def _canonical_endpoint(solver, launch: np.ndarray, query) -> tuple[bool, str, np.ndarray]:
    sol, normal_constant, status = solver._integrate_launch(
        np.asarray(launch, dtype=float),
        rtol=query.rtol,
        atol=query.atol,
        max_step=query.max_step,
        r_collision=query.r_collision,
        r_escape=query.r_escape,
        max_acceleration=query.max_acceleration,
        dense_output=False,
    )
    if sol is None or status != "ok" or normal_constant <= 0.0:
        return False, status, np.full(7, np.nan)
    radius, theta, ur, ut, kappa = solver._endpoint_from_state(sol.y[:, -1])
    if radius <= 0.0 or kappa <= 0.0:
        return False, "invalid-endpoint", np.full(7, np.nan)
    endpoint = np.array(
        [launch[0], launch[1], math.log(radius), theta, ur, ut, math.log(kappa)],
        dtype=float,
    )
    if not np.all(np.isfinite(endpoint)):
        return False, "non-finite-endpoint", np.full(7, np.nan)
    return True, "ok", endpoint


def _endpoint_in_domain(endpoint: np.ndarray, backend) -> bool:
    rho = math.exp(float(endpoint[2]))
    kappa = math.exp(float(endpoint[6]))
    speed = math.hypot(float(endpoint[4]), float(endpoint[5]))
    return bool(
        backend.rho_bounds[0] <= rho <= backend.rho_bounds[1]
        and backend.theta_bounds[0] <= endpoint[3] <= backend.theta_bounds[1]
        and backend.kappa_bounds[0] <= kappa <= backend.kappa_bounds[1]
        and speed <= backend.final_speed_max
    )


def _worker_forward(task: tuple[int, list[float]]):
    idx, launch_list = task
    launch = np.asarray(launch_list, dtype=float)
    ok, reason, endpoint = _canonical_endpoint(_WORKER_SOLVER, launch, _WORKER_QUERY)
    if ok and not _endpoint_in_domain(endpoint, _WORKER_BACKEND):
        ok, reason = False, "outside-endpoint-domain"
    return idx, ok, reason, endpoint.tolist()


def _q5_from_launch(launch: np.ndarray) -> np.ndarray:
    return np.array([launch[2], launch[3], launch[4], launch[5], math.log(launch[6])], dtype=float)


def _launch_from_q5(u0: float, w0: float, q: np.ndarray) -> np.ndarray:
    return np.array([u0, w0, q[0], q[1], q[2], q[3], math.exp(float(q[4]))], dtype=float)


def _q7_from_launch(launch: np.ndarray) -> np.ndarray:
    return np.array([*launch[:6], math.log(float(launch[6]))], dtype=float)


def _launch_from_q7(q: np.ndarray) -> np.ndarray:
    return np.array([*q[:6], math.exp(float(q[6]))], dtype=float)


def _normalized_launch_error(recovered: np.ndarray, truth: np.ndarray, scales7: np.ndarray) -> float:
    rq = _q7_from_launch(recovered)
    tq = _q7_from_launch(truth)
    return float(np.linalg.norm((rq - tq) / scales7))


def _rotated_cartesian5(
    raw5: np.ndarray, theta_ref: float, asinh_scale: float = 0.0,
    center_state: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Transform canonical final coordinates to the swarm-dashboard basis.

    ``raw5`` is ``(log_rho, theta, ur, ut, log_kappa)``.  The returned
    coordinates are ``(x_rot, y_rot, vx_rot, vy_rot, log_kappa)`` in a frame
    whose orientation is fixed by ``theta_ref``.
    """
    raw5 = np.asarray(raw5, dtype=float)
    rho = math.exp(float(raw5[0]))
    delta = float(raw5[1]) - float(theta_ref)
    c, s = math.cos(delta), math.sin(delta)
    ur, ut = float(raw5[2]), float(raw5[3])
    state = np.array(
        [rho * c, rho * s, ur * c - ut * s, ur * s + ut * c],
        dtype=float,
    )
    if asinh_scale > 0.0:
        if center_state is None:
            raise ValueError("asinh transformed basis requires center_state")
        state = np.arcsinh((state - np.asarray(center_state, dtype=float)) / float(asinh_scale))
    return np.array([*state, float(raw5[4])], dtype=float)


def _rotated_cartesian5_jacobian(
    raw5: np.ndarray, theta_ref: float, asinh_scale: float = 0.0,
    center_state: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Analytical derivative of :func:`_rotated_cartesian5`."""
    raw5 = np.asarray(raw5, dtype=float)
    rho = math.exp(float(raw5[0]))
    delta = float(raw5[1]) - float(theta_ref)
    c, s = math.cos(delta), math.sin(delta)
    ur, ut = float(raw5[2]), float(raw5[3])
    x, y = rho * c, rho * s
    vx, vy = ur * c - ut * s, ur * s + ut * c
    jac = np.array(
        [
            [x, -y, 0.0, 0.0, 0.0],
            [y,  x, 0.0, 0.0, 0.0],
            [0.0, -vy, c, -s, 0.0],
            [0.0,  vx, s,  c, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    if asinh_scale > 0.0:
        if center_state is None:
            raise ValueError("asinh transformed basis requires center_state")
        scale = float(asinh_scale)
        state = np.array([x, y, vx, vy], dtype=float)
        shifted = state - np.asarray(center_state, dtype=float)
        derivative = 1.0 / (scale * np.sqrt(1.0 + (shifted / scale) ** 2))
        jac[:4, :] *= derivative[:, None]
    return jac


def _raw5_from_state(solver, yf: np.ndarray) -> Optional[np.ndarray]:
    radius, theta, ur, ut, kappa = solver._endpoint_from_state(yf)
    if radius <= 0.0 or kappa <= 0.0:
        return None
    raw5 = np.array([math.log(radius), theta, ur, ut, math.log(kappa)], dtype=float)
    return raw5 if np.all(np.isfinite(raw5)) else None


def _build_pullback_basis(
    solver, raw_details, config, q_scale5: np.ndarray, asinh_scale: float
) -> dict[str, Any]:
    """Build the exact local affine pullback at one BVP seed.

    If ``z = F(q)`` denotes the transformed endpoint and
    ``xi = (q-q0)/q_scale``, then ``dz ~= A dxi``.  The stored matrix is
    ``A^+`` and the nonlinear residual supplied to the optimizer is
    ``A^+ (z(q)-z_target)``.
    """
    _scaled, jac_scaled, yf, normal_constant, status, _launch = raw_details
    if yf is None or status != "ok" or normal_constant <= 0.0:
        raise ValueError(f"cannot construct nonlinear pullback from status={status}")
    seed_raw5 = _raw5_from_state(solver, yf)
    if seed_raw5 is None:
        raise ValueError("invalid seed endpoint for nonlinear pullback")
    theta_ref = float(seed_raw5[1])
    rho0 = math.exp(float(seed_raw5[0]))
    center_state = np.array([rho0, 0.0, float(seed_raw5[2]), float(seed_raw5[3])], dtype=float)
    raw_jac = np.asarray(jac_scaled, dtype=float) * np.asarray(config.residual_scale, dtype=float)[:, None]
    transform_jac = _rotated_cartesian5_jacobian(
        seed_raw5, theta_ref, asinh_scale, center_state
    )
    transformed_jac = transform_jac @ raw_jac
    q_scale5 = np.asarray(q_scale5, dtype=float)
    affine = transformed_jac @ np.diag(q_scale5)
    if not np.all(np.isfinite(affine)):
        raise ValueError("non-finite affine map in nonlinear pullback")
    singular = np.linalg.svd(affine, compute_uv=False)
    if len(singular) != 5 or singular[0] <= 0.0:
        raise ValueError("singular affine map in nonlinear pullback")
    # A mild pseudoinverse cutoff prevents an almost-null direction from
    # turning into an enormous residual multiplier.
    pullback = np.linalg.pinv(affine, rcond=1.0e-10)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0.0 else math.inf
    return {
        "theta_ref": theta_ref,
        "pullback": pullback,
        "condition": condition,
        "singular_values": singular,
        "asinh_scale": float(asinh_scale),
        "center_state": center_state,
    }


def _apply_pullback_to_details(
    solver, raw_details, target5: np.ndarray, config, basis: dict[str, Any]
):
    """Replace canonical scaled residual/Jacobian by pullback coordinates."""
    _scaled, jac_scaled, yf, normal_constant, status, launch = raw_details
    if yf is None or status != "ok" or normal_constant <= 0.0:
        penalty = np.full(5, 1.0e3, dtype=float)
        return penalty, np.eye(5), yf, normal_constant, status, launch
    current_raw5 = _raw5_from_state(solver, yf)
    if current_raw5 is None:
        penalty = np.full(5, 1.0e3, dtype=float)
        return penalty, np.eye(5), yf, normal_constant, "invalid-endpoint", launch
    theta_ref = float(basis["theta_ref"])
    asinh_scale = float(basis["asinh_scale"])
    center_state = np.asarray(basis["center_state"], dtype=float)
    current_z = _rotated_cartesian5(current_raw5, theta_ref, asinh_scale, center_state)
    target_z = _rotated_cartesian5(
        np.asarray(target5, dtype=float), theta_ref, asinh_scale, center_state
    )
    raw_jac = np.asarray(jac_scaled, dtype=float) * np.asarray(config.residual_scale, dtype=float)[:, None]
    transformed_jac = _rotated_cartesian5_jacobian(
        current_raw5, theta_ref, asinh_scale, center_state
    ) @ raw_jac
    pullback = np.asarray(basis["pullback"], dtype=float)
    residual = pullback @ (current_z - target_z)
    jacobian = pullback @ transformed_jac
    if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(jacobian)):
        penalty = np.full(5, 1.0e3, dtype=float)
        return penalty, np.eye(5), yf, normal_constant, "non-finite-pullback", launch
    return residual, jacobian, yf, normal_constant, status, launch


def _profile_newton(base, name: str, timeout: float):
    if name == "newton_balanced":
        return replace(
            base, method="fast_newton", max_iterations=7, line_search_steps=4,
            step_limit=0.35, regularization=1.0e-8, wall_time_seconds=timeout,
            robust_fallback=False,
        )
    if name == "newton_aggressive":
        return replace(
            base, method="fast_newton", max_iterations=9, line_search_steps=3,
            step_limit=0.75, regularization=1.0e-10, wall_time_seconds=timeout,
            robust_fallback=False,
        )
    if name == "newton_damped":
        return replace(
            base, method="fast_newton", max_iterations=11, line_search_steps=7,
            step_limit=0.16, regularization=1.0e-6, wall_time_seconds=timeout,
            robust_fallback=False,
        )
    raise KeyError(name)


class ExactEvaluator:
    def __init__(
        self,
        solver,
        u0: float,
        w0: float,
        target5: np.ndarray,
        config,
        deadline: float,
        *,
        seed: Optional[np.ndarray] = None,
        nonlinear_basis: bool = False,
        q_scale5: Optional[np.ndarray] = None,
        nonlinear_asinh_scale: float = 0.0,
    ):
        self.solver = solver
        self.u0 = float(u0)
        self.w0 = float(w0)
        self.target5 = np.asarray(target5, dtype=float)
        self.config = config
        self.deadline = float(deadline)
        self.cache_x: Optional[np.ndarray] = None
        self.cache = None
        self.ode_evaluations = 0
        self.nonlinear_basis = bool(nonlinear_basis)
        self.basis: Optional[dict[str, Any]] = None

        if self.nonlinear_basis:
            if seed is None or q_scale5 is None:
                raise ValueError("nonlinear basis requires seed and q_scale5")
            seed = np.asarray(seed, dtype=float)
            raw = self._evaluate_raw(seed)
            self.basis = _build_pullback_basis(
                self.solver, raw, self.config, np.asarray(q_scale5, dtype=float),
                float(nonlinear_asinh_scale),
            )
            self.cache_x = seed.copy()
            self.cache = _apply_pullback_to_details(
                self.solver, raw, self.target5, self.config, self.basis
            )

    def _evaluate_raw(self, x: np.ndarray):
        if time.monotonic() >= self.deadline:
            raise MethodTimeout
        details = self.solver.ExtremalAtlas._shooting_residual_and_jacobian(
            np.asarray(x, dtype=float), self.u0, self.w0, self.target5,
            self.config, deadline=self.deadline
        )
        self.ode_evaluations += 1
        if details[4] == "timeout":
            raise MethodTimeout
        return details

    def evaluate(self, x: np.ndarray):
        if time.monotonic() >= self.deadline:
            raise MethodTimeout
        x = np.asarray(x, dtype=float)
        if self.cache_x is None or not np.array_equal(x, self.cache_x):
            raw = self._evaluate_raw(x)
            self.cache_x = x.copy()
            if self.nonlinear_basis:
                assert self.basis is not None
                self.cache = _apply_pullback_to_details(
                    self.solver, raw, self.target5, self.config, self.basis
                )
            else:
                self.cache = raw
        return self.cache

    def residual(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.evaluate(x)[0], dtype=float)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.evaluate(x)[1], dtype=float)

    def residual_details(self, x: np.ndarray):
        details = self.evaluate(x)
        scaled, _jac, yf, normal_constant, status, launch = details
        if yf is None or status != "ok" or normal_constant <= 0.0:
            raw = np.full(5, 1.0e3, dtype=float)
        else:
            current_raw5 = _raw5_from_state(self.solver, yf)
            raw = (current_raw5 - self.target5) if current_raw5 is not None else np.full(5, 1.0e3)
        return np.asarray(scaled, dtype=float), yf, normal_constant, status, raw, launch

    def accepted(self, x: np.ndarray):
        details = self.evaluate(x)
        return self.solver._accepted_correction_details(details, self.target5, self.config)



def _least_squares_direct(
    solver,
    seed: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    target: np.ndarray,
    config,
    *,
    method: str,
    timeout: float,
    max_nfev: int,
    nonlinear_basis: bool = False,
    q_scale5: Optional[np.ndarray] = None,
    nonlinear_asinh_scale: float = 0.0,
):
    deadline = time.monotonic() + timeout
    ev = ExactEvaluator(
        solver, target[0], target[1], target[2:7], config, deadline,
        seed=seed, nonlinear_basis=nonlinear_basis, q_scale5=q_scale5,
        nonlinear_asinh_scale=nonlinear_asinh_scale,
    )
    try:
        result = least_squares(
            ev.residual,
            np.clip(seed, lo + 1e-12, hi - 1e-12),
            jac=ev.jacobian,
            bounds=(lo, hi),
            method=method,
            x_scale="jac",
            max_nfev=max_nfev,
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
        )
        accepted, norm, launch = ev.accepted(result.x)
        reason = "converged" if accepted else f"{method}-failed"
        return launch if accepted else None, ev.ode_evaluations, reason, norm
    except MethodTimeout:
        return None, ev.ode_evaluations, "timeout", math.inf
    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        return None, ev.ode_evaluations, f"exception:{type(exc).__name__}", math.inf


def _bounded_root_method(
    solver,
    seed: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    target: np.ndarray,
    config,
    *,
    method: str,
    timeout: float,
    max_nfev: int,
    nonlinear_basis: bool = False,
    q_scale5: Optional[np.ndarray] = None,
    nonlinear_asinh_scale: float = 0.0,
):
    deadline = time.monotonic() + timeout
    ev = ExactEvaluator(
        solver, target[0], target[1], target[2:7], config, deadline,
        seed=seed, nonlinear_basis=nonlinear_basis, q_scale5=q_scale5,
        nonlinear_asinh_scale=nonlinear_asinh_scale,
    )
    span = hi - lo
    frac = np.clip((seed - lo) / span, 1e-9, 1.0 - 1e-9)
    y0 = logit(frac)

    def unpack(y):
        s = expit(np.asarray(y, dtype=float))
        return lo + span * s

    def fun(y):
        return ev.residual(unpack(y))

    def jac(y):
        y = np.asarray(y, dtype=float)
        s = expit(y)
        dxdy = span * s * (1.0 - s)
        return ev.jacobian(unpack(y)) * dxdy[None, :]

    try:
        if method == "lm":
            result = root(fun, y0, jac=jac, method="lm", options={"maxiter": max_nfev, "ftol": 1e-11, "xtol": 1e-11})
        else:
            result = root(fun, y0, jac=jac, method="hybr", options={"maxfev": max_nfev, "xtol": 1e-10})
        x = unpack(result.x)
        accepted, norm, launch = ev.accepted(x)
        reason = "converged" if accepted else f"{method}-failed"
        return launch if accepted else None, ev.ode_evaluations, reason, norm
    except MethodTimeout:
        return None, ev.ode_evaluations, "timeout", math.inf
    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        return None, ev.ode_evaluations, f"exception:{type(exc).__name__}", math.inf


def _run_fast_newton(
    solver, seed, lo, hi, target, base_config, name, timeout,
    *, nonlinear_basis: bool = False, q_scale5: Optional[np.ndarray] = None,
    nonlinear_asinh_scale: float = 0.0,
):
    cfg = _profile_newton(base_config, name, timeout)
    deadline = time.monotonic() + timeout
    if not nonlinear_basis:
        launch, nfev, reason = solver._correct_one_seed(
            seed, lo, hi, float(target[0]), float(target[1]), target[2:7], cfg, deadline=deadline
        )
        norm = math.inf
        if launch is not None:
            q = _q5_from_launch(launch)
            details = solver.ExtremalAtlas._shooting_residual_and_jacobian(
                q, float(target[0]), float(target[1]), target[2:7], cfg, deadline=deadline
            )
            accepted, norm, checked = solver._accepted_correction_details(details, target[2:7], cfg)
            launch = checked if accepted else None
            if launch is None:
                reason = "acceptance-recheck-failed"
            nfev += 1
        return launch, nfev, reason, norm

    ev = ExactEvaluator(
        solver, target[0], target[1], target[2:7], cfg, deadline,
        seed=seed, nonlinear_basis=True, q_scale5=q_scale5,
        nonlinear_asinh_scale=nonlinear_asinh_scale,
    )
    launch, _reported, reason = solver._bounded_fast_newton(
        seed, lo, hi, target[2:7], cfg, ev.evaluate, ev.residual_details, deadline=deadline
    )
    norm = math.inf
    if launch is not None:
        accepted, norm, checked = ev.accepted(_q5_from_launch(launch))
        launch = checked if accepted else None
        if launch is None:
            reason = "acceptance-recheck-failed"
    return launch, ev.ode_evaluations, reason, norm



def _run_continuation_trf(
    solver,
    seed: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    center_endpoint: np.ndarray,
    target_endpoint: np.ndarray,
    config,
    timeout: float,
    max_nfev: int,
    steps: int,
    *,
    nonlinear_basis: bool = False,
    q_scale5: Optional[np.ndarray] = None,
    nonlinear_asinh_scale: float = 0.0,
):
    start = time.monotonic()
    q = seed.copy()
    total_eval = 0
    last_launch = None
    last_norm = math.inf
    for step in range(1, steps + 1):
        remaining = timeout - (time.monotonic() - start)
        if remaining <= 0.0:
            return None, total_eval, "timeout", math.inf
        alpha = step / steps
        target = center_endpoint + alpha * (target_endpoint - center_endpoint)
        launch, nfev, reason, norm = _least_squares_direct(
            solver, q, lo, hi, target, config,
            method="trf", timeout=remaining, max_nfev=max_nfev,
            nonlinear_basis=nonlinear_basis, q_scale5=q_scale5,
            nonlinear_asinh_scale=nonlinear_asinh_scale,
        )
        total_eval += nfev
        if launch is None:
            return None, total_eval, f"continuation-step-{step}:{reason}", norm
        q = _q5_from_launch(launch)
        last_launch, last_norm = launch, norm
    return last_launch, total_eval, "converged", last_norm


def _worker_solve(task: dict):
    solver = _WORKER_SOLVER
    base = _WORKER_QUERY
    method = task["method"]
    center_launch = np.asarray(task["center_launch"], dtype=float)
    center_endpoint = np.asarray(task["center_endpoint"], dtype=float)
    target = np.asarray(task["target_endpoint"], dtype=float)
    truth = np.asarray(task["true_launch"], dtype=float)
    lo = np.asarray(task["lo"], dtype=float)
    hi = np.asarray(task["hi"], dtype=float)
    scales7 = np.asarray(task["scales7"], dtype=float)
    seed = _q5_from_launch(center_launch)
    fast_timeout = float(task["fast_timeout"])
    robust_timeout = float(task["robust_timeout"])
    max_nfev = int(task["robust_max_nfev"])
    nonlinear_basis = bool(task.get("nonlinear_basis", False))
    nonlinear_asinh_scale = float(task.get("nonlinear_asinh_scale", 0.0))
    q_scale5 = scales7[2:]
    t0 = time.monotonic()

    try:
        if method in {"newton_balanced", "newton_aggressive", "newton_damped"}:
            recovered, nfev, reason, norm = _run_fast_newton(
                solver, seed, lo, hi, target, base, method, fast_timeout,
                nonlinear_basis=nonlinear_basis, q_scale5=q_scale5,
                nonlinear_asinh_scale=nonlinear_asinh_scale,
            )
        elif method == "robust_trf_dop853":
            recovered, nfev, reason, norm = _least_squares_direct(
                solver, seed, lo, hi, target, base,
                method="trf", timeout=robust_timeout, max_nfev=max_nfev,
                nonlinear_basis=nonlinear_basis, q_scale5=q_scale5,
                nonlinear_asinh_scale=nonlinear_asinh_scale,
            )
        elif method == "robust_dogbox_dop853":
            recovered, nfev, reason, norm = _least_squares_direct(
                solver, seed, lo, hi, target, base,
                method="dogbox", timeout=robust_timeout, max_nfev=max_nfev,
                nonlinear_basis=nonlinear_basis, q_scale5=q_scale5,
                nonlinear_asinh_scale=nonlinear_asinh_scale,
            )
        elif method == "lm_dop853":
            recovered, nfev, reason, norm = _bounded_root_method(
                solver, seed, lo, hi, target, base,
                method="lm", timeout=robust_timeout, max_nfev=max_nfev,
                nonlinear_basis=nonlinear_basis, q_scale5=q_scale5,
                nonlinear_asinh_scale=nonlinear_asinh_scale,
            )
        elif method == "hybr_dop853":
            recovered, nfev, reason, norm = _bounded_root_method(
                solver, seed, lo, hi, target, base,
                method="hybr", timeout=robust_timeout, max_nfev=max_nfev,
                nonlinear_basis=nonlinear_basis, q_scale5=q_scale5,
                nonlinear_asinh_scale=nonlinear_asinh_scale,
            )
        elif method == "hybrid_newton_trf":
            recovered, n1, reason1, norm = _run_fast_newton(
                solver, seed, lo, hi, target, base, "newton_balanced", fast_timeout,
                nonlinear_basis=nonlinear_basis, q_scale5=q_scale5,
                nonlinear_asinh_scale=nonlinear_asinh_scale,
            )
            nfev = n1
            reason = reason1
            if recovered is None:
                elapsed = time.monotonic() - t0
                remaining = max(1e-3, robust_timeout - elapsed)
                recovered, n2, reason2, norm = _least_squares_direct(
                    solver, seed, lo, hi, target, base,
                    method="trf", timeout=remaining, max_nfev=max_nfev,
                    nonlinear_basis=nonlinear_basis, q_scale5=q_scale5,
                    nonlinear_asinh_scale=nonlinear_asinh_scale,
                )
                nfev += n2
                reason = f"newton:{reason1};trf:{reason2}"
        elif method == "continuation_trf_4":
            recovered, nfev, reason, norm = _run_continuation_trf(
                solver, seed, lo, hi, center_endpoint, target, base,
                robust_timeout, max_nfev, 4,
                nonlinear_basis=nonlinear_basis, q_scale5=q_scale5,
                nonlinear_asinh_scale=nonlinear_asinh_scale,
            )
        else:
            raise KeyError(method)
    except Exception as exc:  # Worker must return a row rather than killing the experiment.
        recovered, nfev, norm = None, 0, math.inf
        reason = f"unhandled:{type(exc).__name__}:{exc}"

    elapsed = time.monotonic() - t0
    success = recovered is not None
    launch_error = _normalized_launch_error(recovered, truth, scales7) if success else math.inf
    return {
        "success": bool(success),
        "elapsed": float(elapsed),
        "nfev": int(nfev),
        "reason": str(reason),
        "residual_norm": float(norm),
        "launch_error": float(launch_error),
        "recovered_launch": recovered.tolist() if success else None,
    }


def _launch_scales7(backend) -> np.ndarray:
    return np.array(
        [
            0.5 * (backend.u0_bounds[1] - backend.u0_bounds[0]),
            0.5 * (backend.w0_bounds[1] - backend.w0_bounds[0]),
            0.5 * (backend.ar0_bounds[1] - backend.ar0_bounds[0]),
            0.5 * (backend.at0_bounds[1] - backend.at0_bounds[0]),
            0.5 * (backend.jr0_bounds[1] - backend.jr0_bounds[0]),
            0.5 * (backend.ell_bounds[1] - backend.ell_bounds[0]),
            0.5 * math.log(backend.tau_bounds[1] / backend.tau_bounds[0]),
        ],
        dtype=float,
    )


def _launch_bounds7(backend) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array(
        [backend.u0_bounds[0], backend.w0_bounds[0], backend.ar0_bounds[0], backend.at0_bounds[0],
         backend.jr0_bounds[0], backend.ell_bounds[0], math.log(backend.tau_bounds[0])], dtype=float
    )
    hi = np.array(
        [backend.u0_bounds[1], backend.w0_bounds[1], backend.ar0_bounds[1], backend.at0_bounds[1],
         backend.jr0_bounds[1], backend.ell_bounds[1], math.log(backend.tau_bounds[1])], dtype=float
    )
    return lo, hi


def _random_directions(rng: np.random.Generator, n: int) -> np.ndarray:
    x = rng.normal(size=(n, 7))
    x /= np.linalg.norm(x, axis=1)[:, None]
    return x


def _open_database(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=FULL")
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS centers (
            center_id INTEGER PRIMARY KEY,
            launch_json TEXT NOT NULL,
            endpoint_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS targets (
            center_id INTEGER NOT NULL,
            radius_index INTEGER NOT NULL,
            radius REAL NOT NULL,
            target_index INTEGER NOT NULL,
            direction_json TEXT NOT NULL,
            launch_json TEXT NOT NULL,
            endpoint_json TEXT NOT NULL,
            PRIMARY KEY(center_id, radius_index, target_index)
        );
        CREATE TABLE IF NOT EXISTS runs (
            center_id INTEGER NOT NULL,
            radius_index INTEGER NOT NULL,
            target_index INTEGER NOT NULL,
            method TEXT NOT NULL,
            success INTEGER NOT NULL,
            elapsed REAL NOT NULL,
            nfev INTEGER NOT NULL,
            reason TEXT NOT NULL,
            residual_norm REAL NOT NULL,
            launch_error REAL NOT NULL,
            recovered_launch_json TEXT,
            PRIMARY KEY(center_id, radius_index, target_index, method)
        );
        """
    )
    con.commit()
    return con


def _meta_get(con: sqlite3.Connection, key: str) -> Optional[str]:
    row = con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def _meta_set(con: sqlite3.Connection, key: str, value: str):
    con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)", (key, value))
    con.commit()


def _fingerprint(args, backend_dict: dict, query_dict: dict) -> str:
    payload = {
        "solver": str(Path(args.solver).resolve()),
        "solver_size": os.path.getsize(args.solver),
        "config": backend_dict,
        "query": query_dict,
        "radii": list(args.radii),
        "points_per_radius": args.points_per_radius,
        "methods": list(args.methods),
        "threshold": args.threshold,
        "seed": args.seed,
    }
    # Keep the raw-mode fingerprint identical to the original script so an
    # existing benchmark can still be resumed without the new option.
    if args.nonlinear_basis:
        payload["nonlinear_basis"] = {
            "mode": "custom_rotated_cartesian_pullback",
            "asinh_scale": float(args.nonlinear_asinh_scale),
        }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _store_center(con, center_id: int, launch: np.ndarray, endpoint: np.ndarray):
    con.execute(
        "INSERT OR REPLACE INTO centers(center_id,launch_json,endpoint_json) VALUES (?,?,?)",
        (center_id, json.dumps(launch.tolist()), json.dumps(endpoint.tolist())),
    )
    con.commit()


def _load_center(con, center_id: int):
    row = con.execute("SELECT launch_json,endpoint_json FROM centers WHERE center_id=?", (center_id,)).fetchone()
    if not row:
        return None
    return np.asarray(json.loads(row[0]), dtype=float), np.asarray(json.loads(row[1]), dtype=float)


def _sample_center(solver, backend, query, center_id: int, seed: int, max_radius: float, scales7: np.ndarray):
    lo7, hi7 = _launch_bounds7(backend)
    margin = max_radius * scales7 * 1.02
    for batch in range(100):
        cfg = replace(backend, n_samples=256, seed=seed + center_id * 10007 + batch * 97)
        candidates = solver._sample_launches(cfg)
        for launch in candidates:
            q7 = _q7_from_launch(launch)
            if np.any(q7 <= lo7 + margin) or np.any(q7 >= hi7 - margin):
                continue
            ok, _reason, endpoint = _canonical_endpoint(solver, launch, query)
            if ok and _endpoint_in_domain(endpoint, backend):
                return np.asarray(launch, dtype=float), endpoint
    raise RuntimeError(f"Could not sample a valid interior center {center_id}")


def _target_count(con, center_id: int, radius_index: int) -> int:
    return int(con.execute(
        "SELECT COUNT(*) FROM targets WHERE center_id=? AND radius_index=?",
        (center_id, radius_index),
    ).fetchone()[0])


def _load_targets(con, center_id: int, radius_index: int):
    rows = con.execute(
        "SELECT target_index,direction_json,launch_json,endpoint_json FROM targets "
        "WHERE center_id=? AND radius_index=? ORDER BY target_index",
        (center_id, radius_index),
    ).fetchall()
    return [
        (int(r[0]), np.asarray(json.loads(r[1]), dtype=float),
         np.asarray(json.loads(r[2]), dtype=float), np.asarray(json.loads(r[3]), dtype=float))
        for r in rows
    ]


def _generate_targets(
    con,
    pool,
    center_id: int,
    radius_index: int,
    radius: float,
    center_launch: np.ndarray,
    scales7: np.ndarray,
    count: int,
    seed: int,
):
    existing = _target_count(con, center_id, radius_index)
    if existing >= count:
        return _load_targets(con, center_id, radius_index)[:count]

    rng = np.random.default_rng(seed + center_id * 1_000_003 + radius_index * 10_007)
    center_q = _q7_from_launch(center_launch)
    accepted: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    attempts = 0
    while len(accepted) < count and attempts < count * 30:
        batch_n = min(max(32, 2 * (count - len(accepted))), count * 4)
        directions = _random_directions(rng, batch_n)
        launches = [_launch_from_q7(center_q + radius * d * scales7) for d in directions]
        futures = {
            pool.submit(_worker_forward, (i, launch.tolist())): (directions[i], launch)
            for i, launch in enumerate(launches)
        }
        for fut in as_completed(futures):
            direction, launch = futures[fut]
            _idx, ok, _reason, endpoint_list = fut.result()
            attempts += 1
            if ok:
                accepted.append((direction, launch, np.asarray(endpoint_list, dtype=float)))
                if len(accepted) >= count:
                    break
    if len(accepted) < count:
        print(f"    warning: only {len(accepted)}/{count} valid forward targets at radius {radius:g}")

    # Replace target rows for deterministic clean restart if a partial generation existed.
    con.execute("DELETE FROM targets WHERE center_id=? AND radius_index=?", (center_id, radius_index))
    for i, (direction, launch, endpoint) in enumerate(accepted):
        con.execute(
            "INSERT INTO targets(center_id,radius_index,radius,target_index,direction_json,launch_json,endpoint_json) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                center_id, radius_index, radius, i,
                json.dumps(direction.tolist()), json.dumps(launch.tolist()), json.dumps(endpoint.tolist()),
            ),
        )
    con.commit()
    return _load_targets(con, center_id, radius_index)


def _method_complete(con, center_id: int, radius_index: int, method: str, target_count: int) -> bool:
    n = con.execute(
        "SELECT COUNT(*) FROM runs WHERE center_id=? AND radius_index=? AND method=?",
        (center_id, radius_index, method),
    ).fetchone()[0]
    return int(n) >= target_count and target_count > 0


def _method_rate(con, center_id: int, radius_index: int, method: str) -> tuple[int, int, float, float, float]:
    row = con.execute(
        "SELECT COUNT(*),COALESCE(SUM(success),0),AVG(elapsed) FROM runs "
        "WHERE center_id=? AND radius_index=? AND method=?",
        (center_id, radius_index, method),
    ).fetchone()
    n, successes, mean_time = int(row[0]), int(row[1]), float(row[2] or 0.0)
    times = [r[0] for r in con.execute(
        "SELECT elapsed FROM runs WHERE center_id=? AND radius_index=? AND method=? ORDER BY elapsed",
        (center_id, radius_index, method),
    ).fetchall()]
    p95 = float(np.percentile(times, 95)) if times else math.nan
    return successes, n, successes / n if n else math.nan, mean_time, p95


def _run_method_radius(
    con,
    pool,
    center_id: int,
    radius_index: int,
    method: str,
    center_launch: np.ndarray,
    center_endpoint: np.ndarray,
    targets,
    lo5: np.ndarray,
    hi5: np.ndarray,
    scales7: np.ndarray,
    args,
):
    missing = []
    for target_index, _direction, true_launch, target_endpoint in targets:
        exists = con.execute(
            "SELECT 1 FROM runs WHERE center_id=? AND radius_index=? AND target_index=? AND method=?",
            (center_id, radius_index, target_index, method),
        ).fetchone()
        if exists:
            continue
        task = {
            "method": method,
            "center_launch": center_launch.tolist(),
            "center_endpoint": center_endpoint.tolist(),
            "target_endpoint": target_endpoint.tolist(),
            "true_launch": true_launch.tolist(),
            "lo": lo5.tolist(),
            "hi": hi5.tolist(),
            "scales7": scales7.tolist(),
            "fast_timeout": args.fast_timeout,
            "robust_timeout": args.robust_timeout,
            "robust_max_nfev": args.robust_max_nfev,
            "nonlinear_basis": bool(args.nonlinear_basis),
            "nonlinear_asinh_scale": float(args.nonlinear_asinh_scale),
        }
        missing.append((target_index, pool.submit(_worker_solve, task)))

    completed = 0
    for target_index, fut in missing:
        result = fut.result()
        con.execute(
            "INSERT OR REPLACE INTO runs(center_id,radius_index,target_index,method,success,elapsed,nfev,reason,residual_norm,launch_error,recovered_launch_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                center_id, radius_index, target_index, method, int(result["success"]),
                result["elapsed"], result["nfev"], result["reason"], result["residual_norm"],
                result["launch_error"], json.dumps(result["recovered_launch"]) if result["recovered_launch"] is not None else None,
            ),
        )
        con.commit()  # Intentional continuous storage after every expensive solve.
        completed += 1
        if completed % max(1, len(missing) // 4) == 0:
            s, n, rate, mean_t, _p95 = _method_rate(con, center_id, radius_index, method)
            print(f"      {method}: {completed}/{len(missing)} new, cumulative {s}/{n} ({100*rate:.1f}%), mean={mean_t:.2f}s")


def _estimate_r90(radii: list[float], rates: list[float], threshold: float):
    valid = [(r, p) for r, p in zip(radii, rates) if np.isfinite(p)]
    if not valid:
        return math.nan, "missing"
    r = np.array([x[0] for x in valid], dtype=float)
    p = np.array([x[1] for x in valid], dtype=float)
    p_mono = np.minimum.accumulate(p)
    if p_mono[0] < threshold:
        return float(r[0]), "below-minimum"
    below = np.flatnonzero(p_mono < threshold)
    if len(below) == 0:
        return float(r[-1]), "above-maximum"
    j = int(below[0])
    i = j - 1
    # Interpolate in log-radius because the experimental grid is logarithmic.
    if p_mono[i] == p_mono[j]:
        return float(math.sqrt(r[i] * r[j])), "interpolated"
    frac = (threshold - p_mono[i]) / (p_mono[j] - p_mono[i])
    log_r = math.log(r[i]) + frac * (math.log(r[j]) - math.log(r[i]))
    return float(math.exp(log_r)), "interpolated"


def _write_summaries(con: sqlite3.Connection, out_dir: Path, methods: Iterable[str], radii: list[float], threshold: float):
    curve_rows = []
    center_rows = []
    center_ids = [int(r[0]) for r in con.execute("SELECT center_id FROM centers ORDER BY center_id")]
    for center_id in center_ids:
        for method in methods:
            rates = []
            for ri, radius in enumerate(radii):
                s, n, rate, mean_t, p95_t = _method_rate(con, center_id, ri, method)
                rates.append(rate)
                if n:
                    curve_rows.append({
                        "center_id": center_id, "method": method, "radius": radius,
                        "successes": s, "tested": n, "recovery_rate": rate,
                        "mean_seconds": mean_t, "p95_seconds": p95_t,
                    })
            r90, censoring = _estimate_r90(radii, rates, threshold)
            center_rows.append({"center_id": center_id, "method": method, "r90": r90, "censoring": censoring})

    with open(out_dir / "recovery_curves.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["center_id", "method", "radius", "successes", "tested", "recovery_rate", "mean_seconds", "p95_seconds"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(curve_rows)
    with open(out_dir / "center_r90.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["center_id", "method", "r90", "censoring"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(center_rows)

    summary = {"threshold": threshold, "methods": {}}
    for method in methods:
        vals = np.array([row["r90"] for row in center_rows if row["method"] == method and np.isfinite(row["r90"])], dtype=float)
        censor = [row["censoring"] for row in center_rows if row["method"] == method]
        method_curves = [row for row in curve_rows if row["method"] == method]
        all_success = sum(row["successes"] for row in method_curves)
        all_tested = sum(row["tested"] for row in method_curves)
        summary["methods"][method] = {
            "centers": int(len(vals)),
            "r90_percentiles": {
                "p05": float(np.percentile(vals, 5)) if len(vals) else math.nan,
                "p25": float(np.percentile(vals, 25)) if len(vals) else math.nan,
                "p50": float(np.percentile(vals, 50)) if len(vals) else math.nan,
                "p75": float(np.percentile(vals, 75)) if len(vals) else math.nan,
                "p95": float(np.percentile(vals, 95)) if len(vals) else math.nan,
            },
            "below_minimum_centers": int(sum(x == "below-minimum" for x in censor)),
            "above_maximum_centers": int(sum(x == "above-maximum" for x in censor)),
            "overall_recovery_rate": all_success / all_tested if all_tested else math.nan,
        }

    tmp = out_dir / "summary.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    os.replace(tmp, out_dir / "summary.json")

    # Recovery curve plot: median and interquartile range across centers.
    fig, ax = plt.subplots(figsize=(10, 7))
    for method in methods:
        med, lo, hi, xs = [], [], [], []
        for radius in radii:
            values = [row["recovery_rate"] for row in curve_rows if row["method"] == method and row["radius"] == radius]
            if values:
                xs.append(radius); med.append(np.median(values)); lo.append(np.percentile(values, 25)); hi.append(np.percentile(values, 75))
        if xs:
            line, = ax.plot(xs, med, marker="o", label=method)
            ax.fill_between(xs, lo, hi, alpha=0.12, color=line.get_color())
    ax.axhline(threshold, linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Normalized initial launch-space radius R")
    ax.set_ylabel("Recovered boundary-value problems")
    ax.set_title("PMP solver recovery versus initial radius")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "recovery_curves.png", dpi=220)
    plt.close(fig)

    # R90 quantile plot.
    fig, ax = plt.subplots(figsize=(10, 6))
    labels, medians, lower, upper = [], [], [], []
    for method in methods:
        p = summary["methods"][method]["r90_percentiles"]
        if np.isfinite(p["p50"]):
            labels.append(method); medians.append(p["p50"]); lower.append(p["p50"] - p["p25"]); upper.append(p["p75"] - p["p50"])
    y = np.arange(len(labels))
    if labels:
        ax.errorbar(medians, y, xerr=np.vstack([lower, upper]), fmt="o", capsize=4)
        ax.set_yticks(y, labels)
        ax.set_xscale("log")
    ax.set_xlabel("R90 (median with 25–75% interval across centers)")
    ax.set_title("Typical 90%-recovery convergence radius")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "r90_summary.png", dpi=220)
    plt.close(fig)
    return summary


def _should_skip_higher(con, center_id: int, method: str, current_ri: int, threshold: float) -> bool:
    if current_ri < 1:
        return False
    rates = []
    for ri in (current_ri - 1, current_ri):
        _s, n, rate, _mt, _p95 = _method_rate(con, center_id, ri, method)
        if n == 0:
            return False
        rates.append(rate)
    return bool(rates[0] < threshold and rates[1] < threshold)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver", required=True, help="Path to pmp_extremal_atlas.py")
    p.add_argument("--config", required=True, help="Atlas config JSON or chart config containing backend_config/query_config")
    p.add_argument("--output-dir", default="solver_radius_benchmark")
    p.add_argument("--centers", type=int, default=24, help="Random feasible centers (default: 24)")
    p.add_argument("--points-per-radius", type=int, default=48, help="Known-feasible targets per radius and center (default: 48)")
    p.add_argument("--radii", type=float, nargs="+", default=list(DEFAULT_RADII))
    p.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=list(DEFAULT_METHODS))
    p.add_argument("--threshold", type=float, default=0.90)
    p.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    p.add_argument("--fast-timeout", type=float, default=8.0, help="Wall-time budget per Newton solve")
    p.add_argument("--robust-timeout", type=float, default=120.0, help="Wall-time budget per robust solve")
    p.add_argument("--robust-max-nfev", type=int, default=160)
    p.add_argument("--seed", type=int, default=20260722)
    p.add_argument(
        "--nonlinear-basis", action="store_true",
        help=(
            "Use the swarm-dashboard rotated-Cartesian output transform and "
            "an exact local affine pullback for every nonlinear BVP solve. "
            "The sampled launch-space spheres and reported radii remain unchanged."
        ),
    )
    p.add_argument(
        "--nonlinear-asinh-scale", type=float, default=0.0,
        help=(
            "Optional asinh compression scale for the four transformed Cartesian "
            "coordinates; 0 reproduces the default custom dashboard transform."
        ),
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-adaptive-stop", action="store_true", help="Test all radii even after two consecutive rates below threshold")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.radii = sorted(set(float(x) for x in args.radii))
    if not args.radii or args.radii[0] <= 0.0:
        raise ValueError("Radii must be positive")
    if not (0.0 < args.threshold < 1.0):
        raise ValueError("Threshold must lie in (0,1)")
    if args.nonlinear_asinh_scale < 0.0:
        raise ValueError("--nonlinear-asinh-scale must be nonnegative")
    if args.nonlinear_asinh_scale > 0.0 and not args.nonlinear_basis:
        raise ValueError("--nonlinear-asinh-scale requires --nonlinear-basis")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    solver = _load_module(args.solver)
    backend, query = _build_configs(solver, args.config)
    backend_dict = asdict(backend)
    query_dict = asdict(query)
    scales7 = _launch_scales7(backend)
    lo5, hi5 = solver._configured_correction_bounds(backend)

    con = _open_database(out_dir / "benchmark.sqlite")
    fp = _fingerprint(args, backend_dict, query_dict)
    old_fp = _meta_get(con, "fingerprint")
    if old_fp is not None and old_fp != fp:
        raise RuntimeError("Existing benchmark has a different configuration. Use a new output directory.")
    if old_fp is None:
        _meta_set(con, "fingerprint", fp)
        _meta_set(con, "arguments", json.dumps(vars(args), sort_keys=True))

    print("All BVP methods use exact DOP853 propagation; only the nonlinear correction strategy differs.")
    if args.nonlinear_basis:
        print(
            "Correction basis=custom rotated Cartesian + exact local affine pullback "
            f"(asinh_scale={args.nonlinear_asinh_scale:g}); test radii remain launch-space radii."
        )
    else:
        print("Correction basis=original canonical scaled endpoint residuals.")
    print(f"Centers={args.centers}, points/radius={args.points_per_radius}, radii={args.radii}")
    print(f"Methods={args.methods}")

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(args.solver, backend_dict, query_dict),
    ) as pool:
        try:
            for center_id in range(args.centers):
                loaded = _load_center(con, center_id)
                if loaded is None:
                    center_launch, center_endpoint = _sample_center(
                        solver, backend, query, center_id, args.seed, max(args.radii), scales7
                    )
                    _store_center(con, center_id, center_launch, center_endpoint)
                else:
                    center_launch, center_endpoint = loaded
                print(f"\nCenter {center_id + 1}/{args.centers}: launch={np.array2string(center_launch, precision=3)}")

                stopped = {m: False for m in args.methods}
                for radius_index, radius in enumerate(args.radii):
                    if all(stopped.values()):
                        break
                    targets = _generate_targets(
                        con, pool, center_id, radius_index, radius, center_launch,
                        scales7, args.points_per_radius, args.seed,
                    )
                    print(f"  Radius {radius:g}: {len(targets)} valid known-feasible targets")
                    for method in args.methods:
                        if stopped[method]:
                            continue
                        if not _method_complete(con, center_id, radius_index, method, len(targets)):
                            _run_method_radius(
                                con, pool, center_id, radius_index, method, center_launch,
                                center_endpoint, targets, lo5, hi5, scales7, args,
                            )
                        s, n, rate, mean_t, p95_t = _method_rate(con, center_id, radius_index, method)
                        print(f"    {method}: {s}/{n} = {100*rate:.1f}%, mean={mean_t:.2f}s, p95={p95_t:.2f}s")
                        if not args.no_adaptive_stop and _should_skip_higher(
                            con, center_id, method, radius_index, args.threshold
                        ):
                            stopped[method] = True
                            print(f"      adaptive stop for {method}: two consecutive radii below {100*args.threshold:.0f}%")
                    _write_summaries(con, out_dir, args.methods, args.radii, args.threshold)
        except KeyboardInterrupt:
            print("\nInterrupted. Completed solves are already committed; resume with --resume.")
        finally:
            summary = _write_summaries(con, out_dir, args.methods, args.radii, args.threshold)
            print("\nCurrent R90 percentiles:")
            for method, item in summary["methods"].items():
                p = item["r90_percentiles"]
                print(
                    f"  {method}: p05={p['p05']:.4g}, p25={p['p25']:.4g}, "
                    f"p50={p['p50']:.4g}, p75={p['p75']:.4g}, p95={p['p95']:.4g}; "
                    f"censored low/high={item['below_minimum_centers']}/{item['above_maximum_centers']}"
                )
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
