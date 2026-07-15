#!/usr/bin/env python3
"""Fast continuation atlas for uncapped constant-power transfers in a planar Kepler field.

The solver works entirely in the dimensionless variables used in the manuscript.
It does NOT run a global optimizer independently at every target point.  Instead it:

1. obtains one analytic free-space seed and deforms it to the Kepler problem;
2. continues that solution through a structured (log rho, theta, log kappa) grid;
3. uses analytic variational equations for Newton predictors/correctors;
4. optionally tries perturbed and neighbouring seeds to retain multiple branches;
5. checkpoints and stores the atlas in compressed NumPy NPZ format.

Dimensionless extremal equations
--------------------------------
    R' = V
    V' = -gamma R / |R|^3 + A
    A' = J
    J' = -gamma |R|^-3 (I - 3 Rhat Rhat^T) A
    q' = |A|^2
    theta' = cross(R,V) / |R|^2

The shooting vector is
    x = (A_r0, A_t0, J_r0, rotational_invariant, tau_f)
with J_t0 = -rotational_invariant - A_r0.

For target p=(log rho, Theta, log kappa), the five terminal residuals are
    log |R_f| - log rho
    theta_f - Theta
    radial velocity
    tangential velocity - rho^-1/2
    log q_f - log kappa

This code constructs PMP extremals.  It does not prove local or global optimality,
and no finite rectangular atlas can literally cover the unbounded parameter space.
"""

from __future__ import annotations

import argparse
import dataclasses
import heapq
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

STATE_SIZE = 10
SHOOT_SIZE = 5
PARAM_SIZE = 3


class IntegrationFailure(RuntimeError):
    pass


@dataclasses.dataclass(slots=True)
class SolverConfig:
    # Integration and Newton settings.
    rtol: float = 2.0e-9
    atol: float = 2.0e-11
    max_step: float = math.inf
    min_radius: float = 2.0e-2
    max_radius: float = 100.0
    max_time: float = 80.0
    max_integration_nfev: int = 12000
    max_correction_nfev: int = 30000
    max_continuation_nfev: int = 60000
    max_node_nfev: int = 45000
    max_integration_seconds: float = 3.0
    max_continuation_seconds: float = 20.0
    max_node_seconds: float = 12.0
    newton_tol: float = 2.0e-8
    max_newton_iterations: int = 11
    max_line_search: int = 9
    condition_limit: float = 1.0e14
    # Continuation settings.
    gravity_steps: int = 8
    max_parameter_step: float = 0.38
    max_path_subdivisions: int = 5
    # Branch enrichment.
    max_branches: int = 1
    anchor_perturbations: int = 0
    perturbation_scale: float = 0.20
    random_seed: int = 12345
    # Persistence/progress.
    checkpoint_every: int = 100
    progress_every: int = 25


_DEFAULT_CONFIG = SolverConfig()


@dataclasses.dataclass(slots=True)
class SolveResult:
    success: bool
    x: NDArray[np.float64]
    residual_norm: float = math.inf
    condition: float = math.inf
    iterations: int = 0
    nfev: int = 0
    dxdp: Optional[NDArray[np.float64]] = None
    message: str = ""


def _cross2(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def initial_state(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Initial ten-component physical/extremal state."""
    ar0, at0, jr0, angular_constant = map(float, z)
    y = np.zeros(STATE_SIZE, dtype=np.float64)
    y[0:2] = (1.0, 0.0)
    y[2:4] = (0.0, 1.0)
    y[4:6] = (ar0, at0)
    y[6:8] = (jr0, -angular_constant - ar0)
    return y


def initial_augmented_state(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Initial state plus sensitivities wrt the four non-time shooting variables."""
    y = initial_state(z)

    sensitivity = np.zeros((STATE_SIZE, 4), dtype=np.float64)
    sensitivity[4, 0] = 1.0
    sensitivity[7, 0] = -1.0
    sensitivity[5, 1] = 1.0
    sensitivity[6, 2] = 1.0
    sensitivity[7, 3] = -1.0
    return np.concatenate((y, sensitivity.ravel()))


def augmented_rhs(
    _tau: float,
    augmented: NDArray[np.float64],
    gravity: float,
) -> NDArray[np.float64]:
    """State and exact first-variation equations."""
    y = augmented[:STATE_SIZE]
    R = y[0:2]
    V = y[2:4]
    A = y[4:6]
    J = y[6:8]

    r2 = float(np.dot(R, R))
    if not np.isfinite(r2) or r2 <= 0.0:
        raise IntegrationFailure("invalid radius")
    r = math.sqrt(r2)
    inv_r3 = 1.0 / (r2 * r)
    unit = R / r

    tidal = -gravity * inv_r3 * (np.eye(2) - 3.0 * np.outer(unit, unit))

    derivative = np.empty(STATE_SIZE, dtype=np.float64)
    derivative[0:2] = V
    derivative[2:4] = -gravity * R * inv_r3 + A
    derivative[4:6] = J
    derivative[6:8] = tidal @ A
    derivative[8] = float(np.dot(A, A))
    angular_momentum = _cross2(R, V)
    derivative[9] = angular_momentum / r2

    if augmented.size == STATE_SIZE:
        return derivative

    S = augmented[STATE_SIZE:].reshape(STATE_SIZE, 4)
    jac = np.zeros((STATE_SIZE, STATE_SIZE), dtype=np.float64)
    jac[0:2, 2:4] = np.eye(2)
    jac[2:4, 0:2] = tidal
    jac[2:4, 4:6] = np.eye(2)
    jac[4:6, 6:8] = np.eye(2)

    radial_accel_projection = float(np.dot(R, A))
    # Derivative wrt R of tidal(R) @ A.
    jerk_R = 3.0 * gravity / (r**5) * (
        np.outer(A, R)
        + radial_accel_projection * np.eye(2)
        + np.outer(R, A)
        - 5.0 * radial_accel_projection * np.outer(R, R) / r2
    )
    jac[6:8, 0:2] = jerk_R
    jac[6:8, 4:6] = tidal
    jac[8, 4:6] = 2.0 * A

    jac[9, 0:2] = (
        np.array((V[1], -V[0]), dtype=np.float64) / r2
        - 2.0 * angular_momentum * R / (r2 * r2)
    )
    jac[9, 2:4] = np.array((-R[1], R[0]), dtype=np.float64) / r2

    return np.concatenate((derivative, (jac @ S).ravel()))


def _radius_event(min_radius: float):
    def event(_t: float, augmented: NDArray[np.float64]) -> float:
        return float(np.linalg.norm(augmented[0:2]) - min_radius)

    event.terminal = True  # type: ignore[attr-defined]
    event.direction = -1  # type: ignore[attr-defined]
    return event


def _outer_radius_event(max_radius: float):
    def event(_t: float, augmented: NDArray[np.float64]) -> float:
        return float(max_radius - np.linalg.norm(augmented[0:2]))

    event.terminal = True  # type: ignore[attr-defined]
    event.direction = -1  # type: ignore[attr-defined]
    return event


def integrate_terminal_state(
    x: NDArray[np.float64],
    gravity: float,
    config: SolverConfig,
) -> tuple[NDArray[np.float64], int]:
    """Cheaper terminal integration used during Newton line searches."""
    z = np.asarray(x[:4], dtype=np.float64)
    final_time = float(x[4])
    if not np.isfinite(final_time) or final_time <= 1.0e-6 or final_time > config.max_time:
        raise IntegrationFailure("flight time outside configured bounds")

    initial = initial_state(z)
    rhs_calls = 0
    integration_started = time.monotonic()

    def counted_rhs(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        nonlocal rhs_calls
        rhs_calls += 1
        if (rhs_calls & 63) == 0 and time.monotonic() - integration_started > config.max_integration_seconds:
            raise IntegrationFailure("integration wall-time limit exceeded")
        if rhs_calls > config.max_integration_nfev:
            raise IntegrationFailure("integration RHS-evaluation limit exceeded")
        return augmented_rhs(t, y, gravity)

    try:
        sol = solve_ivp(
            counted_rhs,
            (0.0, final_time),
            initial,
            method="DOP853",
            rtol=config.rtol,
            atol=config.atol,
            max_step=config.max_step,
            events=(
                _radius_event(config.min_radius),
                _outer_radius_event(config.max_radius),
            ),
        )
    except (FloatingPointError, ValueError, IntegrationFailure) as exc:
        raise IntegrationFailure(str(exc)) from exc
    if not sol.success:
        raise IntegrationFailure(sol.message)
    if sol.t[-1] < final_time * (1.0 - 1.0e-9):
        raise IntegrationFailure("trajectory crossed a configured radius boundary")
    return sol.y[:, -1], int(sol.nfev)


def terminal_residual_from_state(
    y: NDArray[np.float64],
    p: NDArray[np.float64],
    config: SolverConfig,
) -> NDArray[np.float64]:
    log_rho, target_theta, log_kappa = map(float, p)
    rho = math.exp(log_rho)
    R = y[0:2]
    V = y[2:4]
    r = float(np.linalg.norm(R))
    q = float(y[8])
    if not np.isfinite(r) or r <= config.min_radius or not np.isfinite(q) or q <= 0.0:
        raise IntegrationFailure("invalid terminal state")
    dot_rv = float(np.dot(R, V))
    cross_rv = _cross2(R, V)
    return np.array((
        math.log(r) - log_rho,
        y[9] - target_theta,
        dot_rv / r,
        cross_rv / r - rho ** -0.5,
        math.log(q) - log_kappa,
    ), dtype=np.float64)


def residual_only(
    x: NDArray[np.float64],
    p: NDArray[np.float64],
    gravity: float,
    config: SolverConfig,
) -> tuple[NDArray[np.float64], int]:
    y, nfev = integrate_terminal_state(x, gravity, config)
    return terminal_residual_from_state(y, p, config), nfev


def integrate_terminal(
    x: NDArray[np.float64],
    gravity: float,
    config: SolverConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int]:
    z = np.asarray(x[:4], dtype=np.float64)
    final_time = float(x[4])
    if not np.isfinite(final_time) or final_time <= 1.0e-6 or final_time > config.max_time:
        raise IntegrationFailure("flight time outside configured bounds")

    initial = initial_augmented_state(z)
    rhs_calls = 0
    integration_started = time.monotonic()

    def counted_rhs(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        nonlocal rhs_calls
        rhs_calls += 1
        if (rhs_calls & 63) == 0 and time.monotonic() - integration_started > config.max_integration_seconds:
            raise IntegrationFailure("integration wall-time limit exceeded")
        if rhs_calls > config.max_integration_nfev:
            raise IntegrationFailure("integration RHS-evaluation limit exceeded")
        return augmented_rhs(t, y, gravity)

    try:
        sol = solve_ivp(
            counted_rhs,
            (0.0, final_time),
            initial,
            method="DOP853",
            rtol=config.rtol,
            atol=config.atol,
            max_step=config.max_step,
            events=(
                _radius_event(config.min_radius),
                _outer_radius_event(config.max_radius),
            ),
        )
    except (FloatingPointError, ValueError, IntegrationFailure) as exc:
        raise IntegrationFailure(str(exc)) from exc

    if not sol.success:
        raise IntegrationFailure(sol.message)
    if sol.t[-1] < final_time * (1.0 - 1.0e-9):
        raise IntegrationFailure("trajectory crossed a configured radius boundary")

    terminal_augmented = sol.y[:, -1]
    terminal_state = terminal_augmented[:STATE_SIZE]
    sensitivity = terminal_augmented[STATE_SIZE:].reshape(STATE_SIZE, 4)
    terminal_rhs = augmented_rhs(final_time, terminal_state, gravity)
    return terminal_state, sensitivity, terminal_rhs, int(sol.nfev)


def residual_and_jacobian(
    x: NDArray[np.float64],
    p: NDArray[np.float64],
    gravity: float,
    config: SolverConfig,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    int,
    NDArray[np.float64],
]:
    """Return terminal residual, shooting Jacobian, target Jacobian, nfev, state."""
    log_rho, target_theta, log_kappa = map(float, p)
    rho = math.exp(log_rho)

    y, sensitivity, terminal_rhs, nfev = integrate_terminal(x, gravity, config)
    R = y[0:2]
    V = y[2:4]
    r = float(np.linalg.norm(R))
    q = float(y[8])
    if not np.isfinite(r) or r <= config.min_radius or not np.isfinite(q) or q <= 0.0:
        raise IntegrationFailure("invalid terminal state")

    dot_rv = float(np.dot(R, V))
    cross_rv = _cross2(R, V)
    radial_velocity = dot_rv / r
    tangential_velocity = cross_rv / r
    circular_speed = rho ** -0.5

    residual = np.array(
        (
            math.log(r) - log_rho,
            y[9] - target_theta,
            radial_velocity,
            tangential_velocity - circular_speed,
            math.log(q) - log_kappa,
        ),
        dtype=np.float64,
    )

    endpoint_gradient = np.zeros((SHOOT_SIZE, STATE_SIZE), dtype=np.float64)
    endpoint_gradient[0, 0:2] = R / (r * r)
    endpoint_gradient[1, 9] = 1.0
    endpoint_gradient[2, 0:2] = V / r - dot_rv * R / (r**3)
    endpoint_gradient[2, 2:4] = R / r
    endpoint_gradient[3, 0:2] = (
        np.array((V[1], -V[0]), dtype=np.float64) / r
        - cross_rv * R / (r**3)
    )
    endpoint_gradient[3, 2:4] = np.array((-R[1], R[0]), dtype=np.float64) / r
    endpoint_gradient[4, 8] = 1.0 / q

    shooting_jacobian = np.empty((SHOOT_SIZE, SHOOT_SIZE), dtype=np.float64)
    shooting_jacobian[:, :4] = endpoint_gradient @ sensitivity
    shooting_jacobian[:, 4] = endpoint_gradient @ terminal_rhs

    # Derivative of residual wrt p=(log rho, Theta, log kappa).
    target_jacobian = np.zeros((SHOOT_SIZE, PARAM_SIZE), dtype=np.float64)
    target_jacobian[0, 0] = -1.0
    target_jacobian[1, 1] = -1.0
    target_jacobian[3, 0] = 0.5 * circular_speed
    target_jacobian[4, 2] = -1.0

    return residual, shooting_jacobian, target_jacobian, nfev, y


def solve_fixed_target(
    seed: NDArray[np.float64],
    p: NDArray[np.float64],
    gravity: float,
    config: SolverConfig,
) -> SolveResult:
    """Damped Newton correction at a fixed target."""
    x = np.asarray(seed, dtype=np.float64).copy()
    total_nfev = 0
    last_condition = math.inf
    last_norm = math.inf

    for iteration in range(1, config.max_newton_iterations + 1):
        try:
            residual, jac, target_jac, nfev, _ = residual_and_jacobian(
                x, p, gravity, config
            )
        except IntegrationFailure as exc:
            return SolveResult(False, x, message=f"integration failed: {exc}")
        total_nfev += nfev
        if total_nfev > config.max_correction_nfev:
            return SolveResult(False, x, residual_norm=last_norm, condition=last_condition,
                               iterations=iteration, nfev=total_nfev,
                               message="correction work budget exceeded")
        last_norm = float(np.linalg.norm(residual))
        try:
            last_condition = float(np.linalg.cond(jac))
        except np.linalg.LinAlgError:
            last_condition = math.inf

        if last_norm <= config.newton_tol:
            if last_condition <= config.condition_limit:
                try:
                    dxdp = -np.linalg.solve(jac, target_jac)
                except np.linalg.LinAlgError:
                    dxdp = -np.linalg.lstsq(jac, target_jac, rcond=None)[0]
            else:
                # The solution itself may be valid at a fold, but its local
                # parameter-space predictor is unreliable.
                dxdp = None
            return SolveResult(
                True,
                x,
                residual_norm=last_norm,
                condition=last_condition,
                iterations=iteration,
                nfev=total_nfev,
                dxdp=dxdp,
            )

        try:
            step = np.linalg.solve(jac, -residual)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(jac, -residual, rcond=None)[0]

        if not np.all(np.isfinite(step)):
            return SolveResult(
                False,
                x,
                residual_norm=last_norm,
                condition=last_condition,
                iterations=iteration,
                nfev=total_nfev,
                message="non-finite Newton step",
            )

        # Keep one wild ill-conditioned step from destroying a good continuation seed.
        relative_scale = np.maximum(np.abs(x), np.array((1.0, 1.0, 1.0, 1.0, 0.2)))
        scaled_norm = float(np.max(np.abs(step) / relative_scale))
        if scaled_norm > 5.0:
            step *= 5.0 / scaled_norm

        accepted = False
        alpha = 1.0
        for _ in range(config.max_line_search):
            trial = x + alpha * step
            if trial[4] <= 1.0e-6 or trial[4] > config.max_time:
                alpha *= 0.5
                continue
            try:
                trial_residual, nfev = residual_only(trial, p, gravity, config)
                total_nfev += nfev
                if total_nfev > config.max_correction_nfev:
                    return SolveResult(False, x, residual_norm=last_norm, condition=last_condition,
                                       iterations=iteration, nfev=total_nfev,
                                       message="correction work budget exceeded")
                trial_norm = float(np.linalg.norm(trial_residual))
            except IntegrationFailure:
                alpha *= 0.5
                continue
            if trial_norm < last_norm * (1.0 - 1.0e-4 * alpha):
                x = trial
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            return SolveResult(
                False,
                x,
                residual_norm=last_norm,
                condition=last_condition,
                iterations=iteration,
                nfev=total_nfev,
                message="line search failed",
            )

    return SolveResult(
        False,
        x,
        residual_norm=last_norm,
        condition=last_condition,
        iterations=config.max_newton_iterations,
        nfev=total_nfev,
        message="maximum Newton iterations reached",
    )


def free_space_seed(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """Analytic gravity-free extremal used as a homotopy seed.

    This naturally seeds the direct geometric branch. Multi-revolution branches are
    obtained by continuation in unwrapped theta after gravity has been restored.
    """
    log_rho, theta, log_kappa = map(float, p)
    rho = math.exp(log_rho)
    kappa = math.exp(log_kappa)
    # The direct branch uses the principal endpoint direction.
    endpoint_angle = (theta + math.pi) % (2.0 * math.pi) - math.pi

    R0 = np.array((1.0, 0.0))
    V0 = np.array((0.0, 1.0))
    Rf = rho * np.array((math.cos(endpoint_angle), math.sin(endpoint_angle)))
    Vf = rho ** -0.5 * np.array(
        (-math.sin(endpoint_angle), math.cos(endpoint_angle))
    )
    displacement = Rf - R0
    velocity_change = Vf - V0

    def coefficients(final_time: float):
        acceleration0 = (
            6.0 * (displacement - V0 * final_time) / final_time**2
            - 2.0 * velocity_change / final_time
        )
        jerk = (
            6.0 * velocity_change / final_time**2
            - 12.0 * (displacement - V0 * final_time) / final_time**3
        )
        expenditure = (
            float(np.dot(acceleration0, acceleration0)) * final_time
            + float(np.dot(acceleration0, jerk)) * final_time**2
            + float(np.dot(jerk, jerk)) * final_time**3 / 3.0
        )
        return acceleration0, jerk, expenditure

    lower = 1.0e-5
    upper = 0.5
    while coefficients(upper)[2] > kappa and upper < 1.0e5:
        upper *= 2.0
    if upper >= 1.0e5:
        raise RuntimeError("could not bracket free-space flight time")

    final_time = brentq(lambda t: coefficients(t)[2] - kappa, lower, upper)
    acceleration0, jerk, _ = coefficients(final_time)
    angular_constant = -jerk[1] - acceleration0[0]
    return np.array(
        (
            acceleration0[0],
            acceleration0[1],
            jerk[0],
            angular_constant,
            final_time,
        ),
        dtype=np.float64,
    )


def gravity_homotopy_seed(
    p: NDArray[np.float64],
    config: SolverConfig,
) -> SolveResult:
    """Deform an analytic gamma=0 direct solution to gamma=1."""
    try:
        x = free_space_seed(p)
    except Exception as exc:
        return SolveResult(False, np.full(5, np.nan), message=str(exc))

    # First correct the exact algebraic seed. This also rejects a requested
    # unwrapped angle incompatible with the direct gravity-free cubic.
    initial = solve_fixed_target(x, p, 0.0, config)
    if not initial.success:
        return initial
    x = initial.x
    total_nfev = initial.nfev
    total_iterations = initial.iterations

    gravity = 0.0
    nominal_step = 1.0 / max(config.gravity_steps, 1)
    step = nominal_step
    last_result = initial
    while gravity < 1.0 - 1.0e-14:
        next_gravity = min(1.0, gravity + step)
        result = solve_fixed_target(x, p, next_gravity, config)
        total_nfev += result.nfev
        total_iterations += result.iterations
        if result.success:
            x = result.x
            gravity = next_gravity
            last_result = result
            step = min(nominal_step, step * 1.4)
        else:
            step *= 0.5
            if step < 1.0 / (max(config.gravity_steps, 1) * 64.0):
                result.nfev = total_nfev
                result.iterations = total_iterations
                result.message = f"gravity homotopy stopped at gamma={gravity:.6g}: {result.message}"
                return result

    last_result.nfev = total_nfev
    last_result.iterations = total_iterations
    return last_result


def continue_between_targets(
    seed_result: SolveResult,
    p0: NDArray[np.float64],
    p1: NDArray[np.float64],
    config: SolverConfig,
) -> SolveResult:
    """Predictor-corrector continuation along a straight path in parameter space."""
    continuation_started = time.monotonic()
    delta = np.asarray(p1 - p0, dtype=np.float64)
    # Theta is naturally of order unity; logarithmic coordinates likewise.
    distance = float(np.linalg.norm(delta))
    segments = max(1, int(math.ceil(distance / config.max_parameter_step)))
    if segments > 2**config.max_path_subdivisions:
        return SolveResult(False, seed_result.x.copy(), message="continuation path too long")

    current_p = np.asarray(p0, dtype=np.float64).copy()
    current = seed_result
    total_nfev = 0
    total_iterations = 0
    index = 0
    while index < segments:
        target_fraction = (index + 1) / segments
        next_p = p0 + target_fraction * delta
        dp = next_p - current_p
        predicted = current.x.copy()
        if current.dxdp is not None and np.all(np.isfinite(current.dxdp)):
            predicted += current.dxdp @ dp

        if time.monotonic() - continuation_started > config.max_continuation_seconds:
            return SolveResult(False, current.x.copy(), iterations=total_iterations, nfev=total_nfev,
                               message="continuation wall-time budget exceeded")
        corrected = solve_fixed_target(predicted, next_p, 1.0, config)
        total_nfev += corrected.nfev
        total_iterations += corrected.iterations
        if total_nfev > config.max_continuation_nfev:
            return SolveResult(False, current.x.copy(), iterations=total_iterations, nfev=total_nfev,
                               message="continuation work budget exceeded")
        if corrected.success:
            current = corrected
            current_p = next_p
            index += 1
            continue

        # Restart the path with twice as many segments. This is simple and robust.
        if segments >= 2**config.max_path_subdivisions:
            corrected.nfev = total_nfev
            corrected.iterations = total_iterations
            return corrected
        segments *= 2
        index = 0
        current_p = np.asarray(p0, dtype=np.float64).copy()
        current = seed_result

    current.nfev = total_nfev
    current.iterations = total_iterations
    return current


def _axis(minimum: float, maximum: float, count: int, logarithmic: bool) -> NDArray[np.float64]:
    if count < 1:
        raise ValueError("axis count must be positive")
    if count == 1:
        value = math.sqrt(minimum * maximum) if logarithmic else 0.5 * (minimum + maximum)
        return np.array((value,), dtype=np.float64)
    if logarithmic:
        if minimum <= 0.0:
            raise ValueError("logarithmic axis minimum must be positive")
        return np.geomspace(minimum, maximum, count)
    return np.linspace(minimum, maximum, count)


def _parameter_grid(
    rho_axis: NDArray[np.float64],
    theta_axis: NDArray[np.float64],
    kappa_axis: NDArray[np.float64],
) -> NDArray[np.float64]:
    grid = np.empty((rho_axis.size, theta_axis.size, kappa_axis.size, 3), dtype=np.float64)
    grid[..., 0] = np.log(rho_axis)[:, None, None]
    grid[..., 1] = theta_axis[None, :, None]
    grid[..., 2] = np.log(kappa_axis)[None, None, :]
    return grid


def _neighbors(index: tuple[int, int, int], shape: tuple[int, int, int]):
    i, j, k = index
    for di, dj, dk in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)):
        ni, nj, nk = i + di, j + dj, k + dk
        if 0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]:
            yield (ni, nj, nk)


def _all_near_neighbors(index: tuple[int, int, int], shape: tuple[int, int, int]):
    i, j, k = index
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                if di == dj == dk == 0:
                    continue
                ni, nj, nk = i + di, j + dj, k + dk
                if 0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]:
                    yield (ni, nj, nk)


def _deduplicate(candidates: list[SolveResult], max_branches: int) -> list[SolveResult]:
    candidates = [c for c in candidates if c.success and np.all(np.isfinite(c.x))]
    candidates.sort(key=lambda c: c.x[4])
    accepted: list[SolveResult] = []
    for candidate in candidates:
        duplicate = False
        for previous in accepted:
            time_relative = abs(candidate.x[4] - previous.x[4]) / max(previous.x[4], 1.0e-6)
            vector_relative = np.linalg.norm(candidate.x[:4] - previous.x[:4]) / max(
                np.linalg.norm(previous.x[:4]), 1.0
            )
            if time_relative < 2.0e-4 and vector_relative < 2.0e-3:
                duplicate = True
                break
        if not duplicate:
            accepted.append(candidate)
        if len(accepted) >= max_branches:
            break
    return accepted


class AtlasBuilder:
    def __init__(
        self,
        rho_axis: NDArray[np.float64],
        theta_axis: NDArray[np.float64],
        kappa_axis: NDArray[np.float64],
        config: SolverConfig,
        output: Path,
    ) -> None:
        self.rho = rho_axis
        self.theta = theta_axis
        self.kappa = kappa_axis
        self.params = _parameter_grid(rho_axis, theta_axis, kappa_axis)
        self.shape = (rho_axis.size, theta_axis.size, kappa_axis.size)
        self.config = config
        self.output = output
        B = config.max_branches
        full = self.shape + (B,)
        self.shooting = np.full(full + (SHOOT_SIZE,), np.nan, dtype=np.float64)
        self.dxdp = np.full(full + (SHOOT_SIZE, PARAM_SIZE), np.nan, dtype=np.float64)
        self.valid = np.zeros(full, dtype=bool)
        self.residual_norm = np.full(full, np.nan, dtype=np.float64)
        self.condition = np.full(full, np.nan, dtype=np.float64)
        self.iterations = np.zeros(full, dtype=np.int16)
        self.nfev = np.zeros(full, dtype=np.int32)
        self.status = np.zeros(self.shape, dtype=np.int8)  # 0 unknown, 1 solved, -1 attempted failure
        self.total_attempts = 0
        self.total_nfev = 0
        self.start_time = time.monotonic()
        self._stop_requested = False

    def request_stop(self, *_args) -> None:
        self._stop_requested = True

    def _store(self, index: tuple[int, int, int], results: Sequence[SolveResult]) -> None:
        for branch, result in enumerate(results[: self.config.max_branches]):
            key = index + (branch,)
            self.valid[key] = result.success
            self.shooting[key] = result.x
            if result.dxdp is not None:
                self.dxdp[key] = result.dxdp
            self.residual_norm[key] = result.residual_norm
            self.condition[key] = result.condition
            self.iterations[key] = result.iterations
            self.nfev[key] = result.nfev
        self.status[index] = 1 if results else -1

    def _result_at(self, index: tuple[int, int, int], branch: int = 0) -> Optional[SolveResult]:
        key = index + (branch,)
        if not self.valid[key]:
            return None
        return SolveResult(
            True,
            self.shooting[key].copy(),
            residual_norm=float(self.residual_norm[key]),
            condition=float(self.condition[key]),
            iterations=int(self.iterations[key]),
            nfev=int(self.nfev[key]),
            dxdp=self.dxdp[key].copy(),
        )

    def _anchor_index(self, anchor: tuple[float, float, float]) -> tuple[int, int, int]:
        arho, atheta, akappa = anchor
        return (
            int(np.argmin(np.abs(np.log(self.rho) - math.log(arho)))),
            int(np.argmin(np.abs(self.theta - atheta))),
            int(np.argmin(np.abs(np.log(self.kappa) - math.log(akappa)))),
        )

    def solve_anchor(self, anchor: tuple[float, float, float]) -> tuple[int, int, int]:
        index = self._anchor_index(anchor)
        p = self.params[index]
        print(
            f"Solving anchor index={index}, rho={math.exp(p[0]):.6g}, "
            f"theta={p[1]:.6g}, kappa={math.exp(p[2]):.6g}",
            flush=True,
        )
        primary = gravity_homotopy_seed(p, self.config)
        self.total_attempts += 1
        self.total_nfev += primary.nfev
        candidates = [primary] if primary.success else []

        if primary.success and self.config.anchor_perturbations > 0:
            rng = np.random.default_rng(self.config.random_seed)
            scales = np.maximum(np.abs(primary.x), np.array((1.0, 1.0, 1.0, 1.0, 0.2)))
            for _ in range(self.config.anchor_perturbations):
                perturb = rng.normal(size=SHOOT_SIZE) * scales * self.config.perturbation_scale
                perturb[4] *= 0.5
                seed = primary.x + perturb
                seed[4] = max(0.02, min(self.config.max_time * 0.9, seed[4]))
                result = solve_fixed_target(seed, p, 1.0, self.config)
                self.total_attempts += 1
                self.total_nfev += result.nfev
                if result.success:
                    candidates.append(result)

        accepted = _deduplicate(candidates, self.config.max_branches)
        if not accepted:
            raise RuntimeError(f"anchor solution failed: {primary.message}")
        self._store(index, accepted)
        return index

    def _candidate_seeds(self, index: tuple[int, int, int]) -> list[tuple[SolveResult, NDArray[np.float64]]]:
        candidates: list[tuple[float, SolveResult, NDArray[np.float64]]] = []
        p = self.params[index]
        for neighbor in _all_near_neighbors(index, self.shape):
            p0 = self.params[neighbor]
            distance = float(np.linalg.norm(p - p0))
            for branch in range(self.config.max_branches):
                result = self._result_at(neighbor, branch)
                if result is not None:
                    candidates.append((distance, result, p0))
        candidates.sort(key=lambda item: (item[0], item[1].x[4]))
        return [(result, p0) for _, result, p0 in candidates]

    def solve_node(self, index: tuple[int, int, int]) -> list[SolveResult]:
        p = self.params[index]
        candidate_results: list[SolveResult] = []
        seeds = self._candidate_seeds(index)
        node_nfev_start = self.total_nfev
        node_started = time.monotonic()

        # Try the cheapest local predictor-correctors first.
        for seed_result, p0 in seeds[: max(2, 2 * self.config.max_branches)]:
            predicted = seed_result.x.copy()
            if seed_result.dxdp is not None and np.all(np.isfinite(seed_result.dxdp)):
                predicted += seed_result.dxdp @ (p - p0)
            result = solve_fixed_target(predicted, p, 1.0, self.config)
            self.total_attempts += 1
            self.total_nfev += result.nfev
            if result.success:
                candidate_results.append(result)
                if self.config.max_branches == 1:
                    break
            if (self.total_nfev - node_nfev_start >= self.config.max_node_nfev
                    or time.monotonic() - node_started >= self.config.max_node_seconds):
                break

        # A subdivided continuation path is slower but rescues difficult cells.
        if (not candidate_results and seeds
                and self.total_nfev - node_nfev_start < self.config.max_node_nfev
                and time.monotonic() - node_started < self.config.max_node_seconds):
            seed_result, p0 = seeds[0]
            result = continue_between_targets(seed_result, p0, p, self.config)
            self.total_attempts += 1
            self.total_nfev += result.nfev
            if result.success:
                candidate_results.append(result)

        # Isolated failures get one direct homotopy attempt when the requested
        # angle is on the principal geometric branch.
        principal = (p[1] + math.pi) % (2.0 * math.pi) - math.pi
        if (not candidate_results and abs(p[1] - principal) < 1.0e-7
                and self.total_nfev - node_nfev_start < self.config.max_node_nfev
                and time.monotonic() - node_started < self.config.max_node_seconds):
            result = gravity_homotopy_seed(p, self.config)
            self.total_attempts += 1
            self.total_nfev += result.nfev
            if result.success:
                candidate_results.append(result)

        return _deduplicate(candidate_results, self.config.max_branches)

    def _metadata(self, complete: bool) -> dict:
        return {
            "format": "kepler-extremal-atlas-v1",
            "complete": bool(complete),
            "created_unix": time.time(),
            "elapsed_seconds": time.monotonic() - self.start_time,
            "solver_config": dataclasses.asdict(self.config),
            "parameter_coordinates": ["rho", "unwrapped_theta", "kappa"],
            "shooting_coordinates": [
                "A_r0",
                "A_t0",
                "J_r0",
                "rotational_invariant",
                "tau_f",
            ],
            "notes": (
                "Each valid entry is a PMP extremal candidate, not a proof of global optimality. "
                "Branch 0 is the shortest candidate found by the configured continuation seeds."
            ),
            "total_attempts": self.total_attempts,
            "total_nfev": self.total_nfev,
        }

    def save(self, complete: bool = False) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output.with_suffix(self.output.suffix + ".tmp.npz")
        np.savez_compressed(
            temporary,
            rho=self.rho,
            theta=self.theta,
            kappa=self.kappa,
            shooting=self.shooting,
            dxdp=self.dxdp,
            valid=self.valid,
            residual_norm=self.residual_norm,
            condition=self.condition,
            iterations=self.iterations,
            nfev=self.nfev,
            status=self.status,
            metadata_json=np.array(json.dumps(self._metadata(complete), sort_keys=True)),
        )
        os.replace(temporary, self.output)

    def load(self) -> bool:
        if not self.output.exists():
            return False
        with np.load(self.output, allow_pickle=False) as atlas:
            for name, expected in (
                ("rho", self.rho),
                ("theta", self.theta),
                ("kappa", self.kappa),
            ):
                if name not in atlas or not np.allclose(atlas[name], expected, rtol=0.0, atol=1e-14):
                    raise ValueError(f"checkpoint axis {name} does not match requested atlas")
            self.shooting[...] = atlas["shooting"]
            self.dxdp[...] = atlas["dxdp"]
            self.valid[...] = atlas["valid"]
            self.residual_norm[...] = atlas["residual_norm"]
            self.condition[...] = atlas["condition"]
            self.iterations[...] = atlas["iterations"]
            self.nfev[...] = atlas["nfev"]
            self.status[...] = atlas["status"]
        return True

    def build(self, anchor: tuple[float, float, float], resume: bool = True) -> None:
        if resume and self.load():
            print(f"Resumed checkpoint {self.output}", flush=True)

        if not np.any(self.valid):
            anchor_index = self.solve_anchor(anchor)
            self.save(complete=False)
        else:
            solved_indices = np.argwhere(np.any(self.valid, axis=-1))
            anchor_index = tuple(map(int, solved_indices[0]))

        queue: list[tuple[int, tuple[int, int, int]]] = []
        queued: set[tuple[int, int, int]] = set()
        for index_array in np.argwhere(np.any(self.valid, axis=-1)):
            index = tuple(map(int, index_array))
            for neighbor in _neighbors(index, self.shape):
                if self.status[neighbor] == 0 and neighbor not in queued:
                    distance = sum(abs(a - b) for a, b in zip(neighbor, anchor_index))
                    heapq.heappush(queue, (distance, neighbor))
                    queued.add(neighbor)

        solved_before = int(np.count_nonzero(np.any(self.valid, axis=-1)))
        solved = solved_before
        total_nodes = int(np.prod(self.shape))
        last_checkpoint = solved

        while queue and not self._stop_requested:
            _, index = heapq.heappop(queue)
            queued.discard(index)
            if self.status[index] == 1:
                continue
            results = self.solve_node(index)
            self._store(index, results)
            if results:
                solved += 1
                for neighbor in _neighbors(index, self.shape):
                    if self.status[neighbor] == 0 and neighbor not in queued:
                        distance = sum(abs(a - b) for a, b in zip(neighbor, anchor_index))
                        heapq.heappush(queue, (distance, neighbor))
                        queued.add(neighbor)

            if solved % self.config.progress_every == 0 and solved != solved_before:
                elapsed = time.monotonic() - self.start_time
                rate = max((solved - solved_before) / max(elapsed, 1.0e-9), 1.0e-12)
                remaining = total_nodes - solved
                print(
                    f"solved {solved}/{total_nodes} ({100*solved/total_nodes:.1f}%), "
                    f"{rate:.2f} nodes/s, projected {remaining/rate/60:.1f} min, "
                    f"nfev={self.total_nfev}",
                    flush=True,
                )

            if solved - last_checkpoint >= self.config.checkpoint_every:
                self.save(complete=False)
                last_checkpoint = solved

        # Retry holes using the nearest solved grid point. This is intentionally
        # conservative: it does not fabricate values where continuation fails.
        holes = [tuple(map(int, idx)) for idx in np.argwhere(self.status != 1)]
        if holes and not self._stop_requested:
            solved_points = [tuple(map(int, idx)) for idx in np.argwhere(np.any(self.valid, axis=-1))]
            print(f"Retrying {len(holes)} unsolved grid points", flush=True)
            for hole_number, index in enumerate(holes, start=1):
                p = self.params[index]
                nearest = min(
                    solved_points,
                    key=lambda candidate: float(np.linalg.norm(self.params[candidate] - p)),
                )
                seed = self._result_at(nearest, 0)
                if seed is None:
                    continue
                result = continue_between_targets(seed, self.params[nearest], p, self.config)
                self.total_attempts += 1
                self.total_nfev += result.nfev
                if result.success:
                    self._store(index, [result])
                    solved_points.append(index)
                    solved += 1
                if hole_number % self.config.progress_every == 0:
                    print(f"hole retry {hole_number}/{len(holes)}, solved={solved}", flush=True)
                if solved - last_checkpoint >= self.config.checkpoint_every:
                    self.save(complete=False)
                    last_checkpoint = solved
                if self._stop_requested:
                    break

        complete = bool(np.all(self.status == 1)) and not self._stop_requested
        self.save(complete=complete)
        valid_count = int(np.count_nonzero(np.any(self.valid, axis=-1)))
        elapsed = time.monotonic() - self.start_time
        print(
            f"Atlas saved to {self.output}: {valid_count}/{total_nodes} grid points, "
            f"elapsed {elapsed/60:.2f} min.",
            flush=True,
        )


def inspect_atlas(path: Path) -> None:
    with np.load(path, allow_pickle=False) as atlas:
        metadata = json.loads(str(atlas["metadata_json"]))
        valid = atlas["valid"]
        residual = atlas["residual_norm"]
        shooting = atlas["shooting"]
        point_valid = np.any(valid, axis=-1)
        count = int(np.count_nonzero(point_valid))
        total = int(np.prod(point_valid.shape))
        finite_residual = residual[valid]
        times = shooting[..., 4][valid]
        print(json.dumps(metadata, indent=2, sort_keys=True))
        print(f"grid shape: {point_valid.shape}")
        print(f"valid grid points: {count}/{total} ({100*count/total:.2f}%)")
        if finite_residual.size:
            print(f"max residual norm: {np.nanmax(finite_residual):.6e}")
            print(f"median residual norm: {np.nanmedian(finite_residual):.6e}")
            print(f"tau range: {np.nanmin(times):.6g} .. {np.nanmax(times):.6g}")


def smoke_test(output: Path) -> None:
    config = SolverConfig(
        gravity_steps=5,
        max_branches=1,
        checkpoint_every=5,
        progress_every=2,
        newton_tol=5e-8,
    )
    builder = AtlasBuilder(
        rho_axis=np.geomspace(1.2, 2.0, 3),
        theta_axis=np.linspace(0.35, 1.05, 4),
        kappa_axis=np.geomspace(4.5, 10.0, 3),
        config=config,
        output=output,
    )
    builder.build(anchor=(1.5, 0.58, 7.0), resume=False)
    inspect_atlas(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build or resume an atlas")
    build.add_argument("--output", type=Path, default=Path("kepler_atlas.npz"))
    build.add_argument("--rho-min", type=float, default=0.5)
    build.add_argument("--rho-max", type=float, default=8.0)
    build.add_argument("--n-rho", type=int, default=13)
    build.add_argument("--theta-min", type=float, default=0.20)
    build.add_argument("--theta-max", type=float, default=2.0 * math.pi)
    build.add_argument("--n-theta", type=int, default=25)
    build.add_argument("--kappa-min", type=float, default=0.75)
    build.add_argument("--kappa-max", type=float, default=24.0)
    build.add_argument("--n-kappa", type=int, default=13)
    build.add_argument("--rho-linear", action="store_true")
    build.add_argument("--kappa-linear", action="store_true")
    build.add_argument("--anchor-rho", type=float, default=1.5)
    build.add_argument("--anchor-theta", type=float, default=0.60)
    build.add_argument("--anchor-kappa", type=float, default=7.5)
    build.add_argument("--no-resume", action="store_true")

    build.add_argument("--rtol", type=float, default=_DEFAULT_CONFIG.rtol)
    build.add_argument("--atol", type=float, default=_DEFAULT_CONFIG.atol)
    build.add_argument("--max-step", type=float, default=_DEFAULT_CONFIG.max_step)
    build.add_argument("--min-radius", type=float, default=_DEFAULT_CONFIG.min_radius)
    build.add_argument("--max-radius", type=float, default=_DEFAULT_CONFIG.max_radius)
    build.add_argument("--max-time", type=float, default=_DEFAULT_CONFIG.max_time)
    build.add_argument("--max-integration-nfev", type=int, default=_DEFAULT_CONFIG.max_integration_nfev)
    build.add_argument("--max-correction-nfev", type=int, default=_DEFAULT_CONFIG.max_correction_nfev)
    build.add_argument("--max-continuation-nfev", type=int, default=_DEFAULT_CONFIG.max_continuation_nfev)
    build.add_argument("--max-node-nfev", type=int, default=_DEFAULT_CONFIG.max_node_nfev)
    build.add_argument("--max-integration-seconds", type=float, default=_DEFAULT_CONFIG.max_integration_seconds)
    build.add_argument("--max-continuation-seconds", type=float, default=_DEFAULT_CONFIG.max_continuation_seconds)
    build.add_argument("--max-node-seconds", type=float, default=_DEFAULT_CONFIG.max_node_seconds)
    build.add_argument("--newton-tol", type=float, default=_DEFAULT_CONFIG.newton_tol)
    build.add_argument("--max-newton-iterations", type=int, default=_DEFAULT_CONFIG.max_newton_iterations)
    build.add_argument("--gravity-steps", type=int, default=_DEFAULT_CONFIG.gravity_steps)
    build.add_argument("--max-parameter-step", type=float, default=_DEFAULT_CONFIG.max_parameter_step)
    build.add_argument("--max-path-subdivisions", type=int, default=_DEFAULT_CONFIG.max_path_subdivisions)
    build.add_argument("--max-branches", type=int, default=_DEFAULT_CONFIG.max_branches)
    build.add_argument("--anchor-perturbations", type=int, default=_DEFAULT_CONFIG.anchor_perturbations)
    build.add_argument("--perturbation-scale", type=float, default=_DEFAULT_CONFIG.perturbation_scale)
    build.add_argument("--checkpoint-every", type=int, default=_DEFAULT_CONFIG.checkpoint_every)
    build.add_argument("--progress-every", type=int, default=_DEFAULT_CONFIG.progress_every)
    build.add_argument("--random-seed", type=int, default=_DEFAULT_CONFIG.random_seed)

    inspect = subparsers.add_parser("inspect", help="print atlas summary")
    inspect.add_argument("path", type=Path)

    smoke = subparsers.add_parser("smoke", help="run a small end-to-end test")
    smoke.add_argument("--output", type=Path, default=Path("smoke_atlas.npz"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        inspect_atlas(args.path)
        return 0
    if args.command == "smoke":
        smoke_test(args.output)
        return 0

    rho = _axis(args.rho_min, args.rho_max, args.n_rho, not args.rho_linear)
    theta = _axis(args.theta_min, args.theta_max, args.n_theta, False)
    kappa = _axis(args.kappa_min, args.kappa_max, args.n_kappa, not args.kappa_linear)
    config = SolverConfig(
        rtol=args.rtol,
        atol=args.atol,
        max_step=args.max_step,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        max_time=args.max_time,
        max_integration_nfev=args.max_integration_nfev,
        max_correction_nfev=args.max_correction_nfev,
        max_continuation_nfev=args.max_continuation_nfev,
        max_node_nfev=args.max_node_nfev,
        max_integration_seconds=args.max_integration_seconds,
        max_continuation_seconds=args.max_continuation_seconds,
        max_node_seconds=args.max_node_seconds,
        newton_tol=args.newton_tol,
        max_newton_iterations=args.max_newton_iterations,
        gravity_steps=args.gravity_steps,
        max_parameter_step=args.max_parameter_step,
        max_path_subdivisions=args.max_path_subdivisions,
        max_branches=args.max_branches,
        anchor_perturbations=args.anchor_perturbations,
        perturbation_scale=args.perturbation_scale,
        random_seed=args.random_seed,
        checkpoint_every=args.checkpoint_every,
        progress_every=args.progress_every,
    )
    builder = AtlasBuilder(rho, theta, kappa, config, args.output)
    signal.signal(signal.SIGINT, builder.request_stop)
    signal.signal(signal.SIGTERM, builder.request_stop)
    builder.build(
        anchor=(args.anchor_rho, args.anchor_theta, args.anchor_kappa),
        resume=not args.no_resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
