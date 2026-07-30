#!/usr/bin/env python3
"""Time-optimal 1-D constant-power transfer solver with a finite exhaust-speed ceiling.

The model is the gravity-free problem used in the spreadsheet:

    x_dot = v
    v_dot = a
    m_dot = -m^2 a^2 / (2 P)

For an engine operating at useful jet power P, the exhaust-speed ceiling V implies
that a nonzero acceleration must satisfy |a| >= 2P/(V m). Coast, a = 0, is also
allowed.

Four ordinary arc topologies are enumerated:

    U+B+C+B-U   variable accel, max-V accel, coast, max-V brake, variable brake
    U+B+C+B-    variable accel, max-V accel, coast, max-V brake to final mass
    B+C+B-U     max-V accel, coast, max-V brake, variable brake
    B+C+B-      max-V accel, coast, max-V brake to final mass

The endpoint-boundary families are evaluated as direct feasible coast-then-full-
brake candidates. They are not rejected merely because the regular PMP endpoint
inequality would prefer another instantaneous control. This is important in the
low-ceiling regime, where an artificial convexified/chattering branch is dominated
by moving all off-time before the continuous braking burn.

The propagation within every arc is analytic. SciPy is used only to solve the three
shooting residuals, so there is no fixed number of Newton columns or iterations.
Convergence tolerances and the safety evaluation limit are configurable.

This solver enumerates the stated ordinary candidate families and selects the
fastest converged feasible candidate. It is not a proof of global optimality over
all measurable controls or over additional abnormal/singular families.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

AU_M = 149_597_870_700.0
DAY_S = 86_400.0


class Topology(str, Enum):
    FIVE_ARC = "U+B+C+B-U"
    TERMINAL_BOUNDARY = "U+B+C+B-"
    INITIAL_BOUNDARY = "B+C+B-U"
    BOTH_BOUNDARY = "B+C+B-"

    @property
    def starts_with_variable(self) -> bool:
        return self in (Topology.FIVE_ARC, Topology.TERMINAL_BOUNDARY)

    @property
    def ends_with_variable(self) -> bool:
        return self in (Topology.FIVE_ARC, Topology.INITIAL_BOUNDARY)


PHASE_NAMES = (
    "variable_acceleration",
    "max_ve_acceleration",
    "coast",
    "max_ve_braking",
    "variable_deceleration",
)


@dataclass(frozen=True)
class TransferProblem:
    power_w: float
    initial_mass_kg: float
    final_mass_kg: float
    distance_m: float
    ve_max_m_s: float

    def validate(self) -> None:
        vals = (
            self.power_w,
            self.initial_mass_kg,
            self.final_mass_kg,
            self.distance_m,
            self.ve_max_m_s,
        )
        if not all(math.isfinite(v) and v > 0.0 for v in vals):
            raise ValueError("All problem parameters must be finite and positive.")
        if self.final_mass_kg >= self.initial_mass_kg:
            raise ValueError("final_mass_kg must be smaller than initial_mass_kg.")

    @property
    def acceleration_scale(self) -> float:
        """A = 2P/(V m_i), the minimum powered acceleration at initial mass."""
        return 2.0 * self.power_w / (self.ve_max_m_s * self.initial_mass_kg)

    @property
    def dimensionless_distance(self) -> float:
        a0 = self.acceleration_scale
        return self.distance_m * a0 / self.ve_max_m_s**2

    @property
    def final_mass_ratio(self) -> float:
        return self.final_mass_kg / self.initial_mass_kg

    @property
    def time_scale_s(self) -> float:
        return self.ve_max_m_s / self.acceleration_scale


@dataclass(frozen=True)
class ArcRecord:
    name: str
    y_start: float
    y_end: float
    dimensionless_time: float
    mass_ratio_start: float
    mass_ratio_end: float
    velocity_over_ve_start: float
    velocity_over_ve_end: float
    dimensionless_position_start: float
    dimensionless_position_end: float


@dataclass
class TransferSolution:
    topology: Topology
    total_time_days: float
    phase_times_days: Dict[str, float]
    maximum_acceleration_m_s2: float
    average_absolute_acceleration_m_s2: float
    position_residual_fraction: float
    velocity_residual_over_ve: float
    mass_residual_log: float
    shooting_y0: float
    shooting_minus_yf: float
    shooting_kappa: float
    arcs: List[ArcRecord]

    @property
    def max_residual(self) -> float:
        return max(
            abs(self.position_residual_fraction),
            abs(self.velocity_residual_over_ve),
            abs(self.mass_residual_log),
        )

    def as_dict(self) -> dict:
        result = asdict(self)
        result["topology"] = self.topology.value
        return result


@dataclass(frozen=True)
class BudgetModel:
    """Budget-to-engine model matching the supplied spreadsheet."""

    payload_mass_kg: float = 500_000.0
    payload_cost_per_kg: float = 1_500.0
    engine_radiator_cost_per_kg: float = 6_357.142857142858
    system_specific_mass_kg_per_w: float = 0.00025
    effective_fuel_cost_per_kg: float = 35.0
    tank_mass_fraction: float = 0.05

    def design(self, total_budget_musd: float, engine_fraction: float) -> TransferProblem:
        if not 0.0 < engine_fraction < 1.0:
            raise ValueError("engine_fraction must lie strictly between 0 and 1.")
        total_budget = total_budget_musd * 1.0e6
        propulsion_budget = total_budget - self.payload_cost_per_kg * self.payload_mass_kg
        if propulsion_budget <= 0.0:
            raise ValueError("The total budget does not cover the payload cost.")

        hardware_mass = (
            engine_fraction * propulsion_budget / self.engine_radiator_cost_per_kg
        )
        fuel_mass = (
            (1.0 - engine_fraction)
            * propulsion_budget
            / self.effective_fuel_cost_per_kg
        )
        power = hardware_mass / self.system_specific_mass_kg_per_w
        dry_mass = (
            self.payload_mass_kg
            + hardware_mass
            + self.tank_mass_fraction * fuel_mass
        )
        initial_mass = dry_mass + fuel_mass
        # Distance and V are supplied later by with_trajectory().
        return TransferProblem(power, initial_mass, dry_mass, 1.0, 1.0)

    def with_trajectory(
        self,
        total_budget_musd: float,
        engine_fraction: float,
        distance_m: float,
        ve_max_m_s: float,
    ) -> TransferProblem:
        base = self.design(total_budget_musd, engine_fraction)
        return TransferProblem(
            power_w=base.power_w,
            initial_mass_kg=base.initial_mass_kg,
            final_mass_kg=base.final_mass_kg,
            distance_m=distance_m,
            ve_max_m_s=ve_max_m_s,
        )


@dataclass
class BudgetOptimizationResult:
    engine_fraction: float
    transfer: TransferSolution
    problem: TransferProblem

    def as_dict(self) -> dict:
        return {
            "engine_fraction": self.engine_fraction,
            "problem": asdict(self.problem),
            "transfer": self.transfer.as_dict(),
        }


@dataclass
class _DimensionlessTrajectory:
    xi: float
    u: float
    mu: float
    y0: float
    yf: float
    kappa: float
    theta: float
    phase_theta: Dict[str, float]
    absolute_impulse_dimensionless: float
    maximum_alpha: float
    arcs: List[ArcRecord]


# ---------------------------- analytic arc propagation ----------------------------


def _propagate_variable(
    ys: float, ye: float, xi: float, u: float, mu: float, kappa: float
) -> Tuple[float, float, float]:
    inverse_mu_end = 1.0 / mu + (ys**3 - ye**3) / (3.0 * kappa)
    if inverse_mu_end <= 0.0:
        raise ValueError("Variable arc reached nonphysical mass.")
    mu_end = 1.0 / inverse_mu_end
    u_end = u + (ys**2 - ye**2) / (2.0 * kappa)
    integral_u_dy = u * (ye - ys) + (
        ys**2 * (ye - ys) - (ye**3 - ys**3) / 3.0
    ) / (2.0 * kappa)
    xi_end = xi - integral_u_dy / kappa
    return xi_end, u_end, mu_end


def _propagate_boundary(
    ys: float,
    ye: float,
    xi: float,
    u: float,
    mu: float,
    kappa: float,
    sign: int,
) -> Tuple[float, float, float]:
    mu_end = mu + (ye - ys) / kappa
    if mu_end <= 0.0:
        raise ValueError("Boundary arc reached nonphysical mass.")
    u_end = u - sign * math.log(mu_end / mu)

    r_start = kappa * mu
    r_end = kappa * mu_end
    logarithmic_integral = (
        r_end * math.log(r_end / r_start) - r_end + r_start
    )
    integral_u_dy = u * (ye - ys) - sign * logarithmic_integral
    xi_end = xi - integral_u_dy / kappa
    return xi_end, u_end, mu_end


def _propagate_coast(
    ys: float, ye: float, xi: float, u: float, mu: float, kappa: float
) -> Tuple[float, float, float]:
    return xi - u * (ye - ys) / kappa, u, mu


def _uplus_to_boundary_switch(ys: float, mu: float, kappa: float) -> float:
    """Solve y = 1/mu(y) on a positive variable arc by safeguarded bisection."""

    def f(y: float) -> float:
        inverse_mu = 1.0 / mu + (ys**3 - y**3) / (3.0 * kappa)
        return y - inverse_mu

    lo, hi = 0.0, ys
    flo, fhi = f(lo), f(hi)
    if not flo < 0.0 or not fhi >= 0.0:
        raise ValueError("Could not bracket U+ to B+ switch.")
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < 1.0e-14 or hi - lo < 1.0e-13 * max(1.0, abs(mid)):
            return mid
        if fm > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _positive_boundary_to_coast_switch(ys: float, mu: float, kappa: float) -> float:
    c = kappa * mu - ys
    return 0.5 * (-c + math.sqrt(c * c + 2.0 * kappa))


def _negative_boundary_to_variable_switch(
    ys: float, mu: float, kappa: float
) -> float:
    c = kappa * mu - ys
    discriminant = c * c - 4.0 * kappa
    if discriminant < 0.0:
        raise ValueError("B- cannot reach a regular U- switch.")
    root = math.sqrt(discriminant)
    candidates = [0.5 * (-c + root), 0.5 * (-c - root)]
    candidates = [
        candidate
        for candidate in candidates
        if candidate < ys - 1.0e-12 * max(1.0, abs(ys))
    ]
    if not candidates:
        raise ValueError("No downstream B- to U- switch exists.")
    return max(candidates)


def _integrate_forced_topology(
    log_shooting: Sequence[float], topology: Topology
) -> _DimensionlessTrajectory:
    y0 = math.exp(float(log_shooting[0]))
    yf = -math.exp(float(log_shooting[1]))
    kappa = math.exp(float(log_shooting[2]))

    if topology.starts_with_variable:
        if y0 < 1.0:
            raise ValueError("Variable-start topology requires y0 >= 1.")
    elif not 0.5 < y0 < 1.0:
        raise ValueError("Boundary-start topology requires 0.5 < y0 < 1.")

    xi = 0.0
    u = 0.0
    mu = 1.0
    ys = y0
    phase_theta = {name: 0.0 for name in PHASE_NAMES}
    absolute_impulse = 0.0
    maximum_alpha = 0.0
    arcs: List[ArcRecord] = []

    def append_arc(
        name: str,
        ye: float,
        propagated: Tuple[float, float, float],
        phase_name: str,
        impulse_increment: float,
    ) -> None:
        nonlocal xi, u, mu, ys, absolute_impulse, maximum_alpha
        xi_end, u_end, mu_end = propagated
        dt = (ys - ye) / kappa
        if dt < -1.0e-12:
            raise ValueError("Arc has negative duration.")
        phase_theta[phase_name] += max(0.0, dt)
        absolute_impulse += impulse_increment
        if name.startswith("U"):
            maximum_alpha = max(maximum_alpha, abs(ys), abs(ye))
        elif name.startswith("B"):
            maximum_alpha = max(maximum_alpha, 1.0 / mu, 1.0 / mu_end)

        arcs.append(
            ArcRecord(
                name=name,
                y_start=ys,
                y_end=ye,
                dimensionless_time=max(0.0, dt),
                mass_ratio_start=mu,
                mass_ratio_end=mu_end,
                velocity_over_ve_start=u,
                velocity_over_ve_end=u_end,
                dimensionless_position_start=xi,
                dimensionless_position_end=xi_end,
            )
        )
        xi, u, mu, ys = xi_end, u_end, mu_end, ye

    if topology.starts_with_variable:
        switch = _uplus_to_boundary_switch(ys, mu, kappa)
        append_arc(
            "U+",
            switch,
            _propagate_variable(ys, switch, xi, u, mu, kappa),
            "variable_acceleration",
            (ys**2 - switch**2) / (2.0 * kappa),
        )

    switch = _positive_boundary_to_coast_switch(ys, mu, kappa)
    propagated = _propagate_boundary(ys, switch, xi, u, mu, kappa, +1)
    append_arc(
        "B+",
        switch,
        propagated,
        "max_ve_acceleration",
        math.log(mu / propagated[2]),
    )

    switch = -1.0 / (2.0 * mu)
    append_arc(
        "C",
        switch,
        _propagate_coast(ys, switch, xi, u, mu, kappa),
        "coast",
        0.0,
    )

    if topology.ends_with_variable:
        switch = _negative_boundary_to_variable_switch(ys, mu, kappa)
        propagated = _propagate_boundary(ys, switch, xi, u, mu, kappa, -1)
        append_arc(
            "B-",
            switch,
            propagated,
            "max_ve_braking",
            math.log(mu / propagated[2]),
        )
        if yf >= ys:
            raise ValueError("Endpoint lies before the U- arc.")
        append_arc(
            "U-",
            yf,
            _propagate_variable(ys, yf, xi, u, mu, kappa),
            "variable_deceleration",
            (yf**2 - ys**2) / (2.0 * kappa),
        )
    else:
        # Direct ordinary terminal-boundary candidate: coast first, then one
        # uninterrupted maximum-V braking burn to the prescribed final mass.
        if yf >= ys:
            raise ValueError("Endpoint lies before the B- arc.")
        propagated = _propagate_boundary(ys, yf, xi, u, mu, kappa, -1)
        append_arc(
            "B-",
            yf,
            propagated,
            "max_ve_braking",
            math.log(mu / propagated[2]),
        )

    theta = (y0 - yf) / kappa
    if theta <= 0.0:
        raise ValueError("Nonpositive total time.")
    return _DimensionlessTrajectory(
        xi=xi,
        u=u,
        mu=mu,
        y0=y0,
        yf=yf,
        kappa=kappa,
        theta=theta,
        phase_theta=phase_theta,
        absolute_impulse_dimensionless=absolute_impulse,
        maximum_alpha=maximum_alpha,
        arcs=arcs,
    )


# -------------------------------- shooting solver --------------------------------


def _shooting_residual(
    log_shooting: Sequence[float], problem: TransferProblem, topology: Topology
) -> np.ndarray:
    try:
        trajectory = _integrate_forced_topology(log_shooting, topology)
        return np.array(
            [
                (trajectory.xi - problem.dimensionless_distance)
                / problem.dimensionless_distance,
                trajectory.u,
                math.log(trajectory.mu / problem.final_mass_ratio),
            ],
            dtype=float,
        )
    except (ArithmeticError, ValueError, OverflowError):
        return np.array([100.0, 100.0, 100.0])


def _topology_bounds(problem: TransferProblem, topology: Topology) -> Tuple[np.ndarray, np.ndarray]:
    mu_f = problem.final_mass_ratio
    if topology.starts_with_variable:
        y0_low, y0_high = 1.0 + 1.0e-8, 1.0e4
    else:
        y0_low, y0_high = 0.5 + 1.0e-8, 1.0 - 1.0e-8

    if topology.ends_with_variable:
        yf_low = (1.0 + 1.0e-8) / mu_f
    else:
        yf_low = 1.0e-6
    yf_high = 1.0e4 / mu_f
    lower = np.log([y0_low, yf_low, 1.0e-7])
    upper = np.log([y0_high, yf_high, 1.0e7])
    return lower, upper


def _default_starts(problem: TransferProblem, topology: Topology) -> Iterable[np.ndarray]:
    mu_f = problem.final_mass_ratio
    ell = problem.dimensionless_distance
    if topology.starts_with_variable:
        y0_values = (1.03, 1.5, 2.5, 5.0, 10.0)
    else:
        y0_values = (0.55, 0.72, 0.92)

    if topology.ends_with_variable:
        yf_fractions = (1.03, 1.5, 3.0)
    else:
        yf_fractions = (0.04, 0.08, 0.16, 0.32, 0.65, 1.0)

    theta_values = sorted(
        {
            max(0.02, math.sqrt(2.0 * ell)),
            max(0.05, 0.5 * ell),
            max(0.1, ell),
            0.2,
            0.5,
            1.0,
            2.0,
            5.0,
            10.0,
        }
    )
    for y0 in y0_values:
        for fraction in yf_fractions:
            minus_yf = fraction / mu_f
            for theta in theta_values:
                yield np.log([y0, minus_yf, (y0 + minus_yf) / theta])


def _solution_from_trajectory(
    problem: TransferProblem,
    topology: Topology,
    trajectory: _DimensionlessTrajectory,
) -> TransferSolution:
    time_conversion_days = problem.time_scale_s / DAY_S
    phase_times = {
        name: trajectory.phase_theta[name] * time_conversion_days
        for name in PHASE_NAMES
    }
    position_residual = (
        trajectory.xi - problem.dimensionless_distance
    ) / problem.dimensionless_distance
    velocity_residual = trajectory.u
    mass_residual = math.log(trajectory.mu / problem.final_mass_ratio)
    return TransferSolution(
        topology=topology,
        total_time_days=trajectory.theta * time_conversion_days,
        phase_times_days=phase_times,
        maximum_acceleration_m_s2=(
            problem.acceleration_scale * trajectory.maximum_alpha
        ),
        average_absolute_acceleration_m_s2=(
            problem.acceleration_scale
            * trajectory.absolute_impulse_dimensionless
            / trajectory.theta
        ),
        position_residual_fraction=position_residual,
        velocity_residual_over_ve=velocity_residual,
        mass_residual_log=mass_residual,
        shooting_y0=trajectory.y0,
        shooting_minus_yf=-trajectory.yf,
        shooting_kappa=trajectory.kappa,
        arcs=trajectory.arcs,
    )


def solve_topology(
    problem: TransferProblem,
    topology: Topology,
    initial_guesses: Optional[Iterable[Sequence[float]]] = None,
    *,
    residual_tolerance: float = 1.0e-8,
    xtol: float = 1.0e-12,
    ftol: float = 1.0e-12,
    gtol: float = 1.0e-12,
    max_nfev: int = 5_000,
    max_starts: Optional[int] = None,
) -> Optional[TransferSolution]:
    """Solve one topology and return its fastest distinct converged root.

    initial_guesses are log([y0, -yf, kappa]) vectors. The nonlinear solve is
    convergence-driven; max_nfev is only a configurable safety ceiling.
    """

    problem.validate()
    lower, upper = _topology_bounds(problem, topology)
    starts: List[np.ndarray] = []
    if initial_guesses is not None:
        starts.extend(np.asarray(guess, dtype=float) for guess in initial_guesses)
    starts.extend(_default_starts(problem, topology))
    if max_starts is not None:
        starts = starts[:max_starts]

    unique: List[Tuple[np.ndarray, TransferSolution]] = []
    for start in starts:
        start = np.minimum(np.maximum(start, lower + 1.0e-10), upper - 1.0e-10)
        try:
            result = least_squares(
                _shooting_residual,
                start,
                args=(problem, topology),
                bounds=(lower, upper),
                xtol=xtol,
                ftol=ftol,
                gtol=gtol,
                max_nfev=max_nfev,
                x_scale="jac",
            )
        except (ArithmeticError, ValueError):
            continue
        if np.linalg.norm(result.fun, ord=np.inf) > residual_tolerance:
            continue
        if any(np.linalg.norm(result.x - old_x) < 1.0e-6 for old_x, _ in unique):
            continue
        trajectory = _integrate_forced_topology(result.x, topology)
        solution = _solution_from_trajectory(problem, topology, trajectory)
        if solution.max_residual <= residual_tolerance:
            unique.append((result.x.copy(), solution))

    if not unique:
        return None
    return min((solution for _, solution in unique), key=lambda item: item.total_time_days)


def solve_all_topologies(
    problem: TransferProblem,
    *,
    warm_starts: Optional[Mapping[Topology, Sequence[float]]] = None,
    residual_tolerance: float = 1.0e-8,
    max_nfev: int = 5_000,
    max_starts_per_topology: Optional[int] = None,
) -> Tuple[TransferSolution, Dict[Topology, TransferSolution]]:
    """Solve all four ordinary topologies and select the fastest feasible one."""

    candidates: Dict[Topology, TransferSolution] = {}
    warm_starts = warm_starts or {}
    for topology in Topology:
        guesses = None
        if topology in warm_starts:
            guesses = [warm_starts[topology]]
        solution = solve_topology(
            problem,
            topology,
            guesses,
            residual_tolerance=residual_tolerance,
            max_nfev=max_nfev,
            max_starts=max_starts_per_topology,
        )
        if solution is not None:
            candidates[topology] = solution
    if not candidates:
        raise RuntimeError("No topology converged to a feasible transfer.")
    best = min(candidates.values(), key=lambda item: item.total_time_days)
    return best, candidates


# ------------------------------ budget optimization ------------------------------


def _log_shooting(solution: TransferSolution) -> np.ndarray:
    return np.log(
        [
            solution.shooting_y0,
            solution.shooting_minus_yf,
            solution.shooting_kappa,
        ]
    )


def optimize_budget_for_topology(
    budget_model: BudgetModel,
    total_budget_musd: float,
    distance_m: float,
    ve_max_m_s: float,
    topology: Topology,
    *,
    bounds: Tuple[float, float] = (0.001, 0.999),
    xatol: float = 1.0e-9,
    max_nfev: int = 5_000,
) -> Optional[BudgetOptimizationResult]:
    """Optimize engine/fuel allocation for one topology with continuation."""

    cache: MutableMapping[float, Tuple[float, Optional[TransferSolution]]] = {}

    def evaluate(x: float) -> float:
        x = float(x)
        if x in cache:
            return cache[x][0]
        problem = budget_model.with_trajectory(
            total_budget_musd, x, distance_m, ve_max_m_s
        )
        guesses: List[np.ndarray] = []
        if cache:
            nearest = min(cache, key=lambda known: abs(known - x))
            previous = cache[nearest][1]
            if previous is not None:
                guesses.append(_log_shooting(previous))
        solution = solve_topology(
            problem,
            topology,
            guesses or None,
            max_nfev=max_nfev,
            max_starts=80,
        )
        value = solution.total_time_days if solution is not None else 1.0e30
        cache[x] = (value, solution)
        return value

    result = minimize_scalar(
        evaluate,
        bounds=bounds,
        method="bounded",
        options={"xatol": xatol, "maxiter": 200},
    )
    evaluate(float(result.x))
    solution = cache[float(result.x)][1]
    if solution is None or not math.isfinite(result.fun):
        return None
    problem = budget_model.with_trajectory(
        total_budget_musd, float(result.x), distance_m, ve_max_m_s
    )
    return BudgetOptimizationResult(float(result.x), solution, problem)


def optimize_budget_all_topologies(
    budget_model: BudgetModel,
    total_budget_musd: float,
    distance_m: float,
    ve_max_m_s: float,
    *,
    bounds: Tuple[float, float] = (0.001, 0.999),
    xatol: float = 1.0e-9,
    max_nfev: int = 5_000,
) -> Tuple[BudgetOptimizationResult, Dict[Topology, BudgetOptimizationResult]]:
    """Optimize each topology independently and select the fastest design."""

    results: Dict[Topology, BudgetOptimizationResult] = {}
    for topology in Topology:
        optimized = optimize_budget_for_topology(
            budget_model,
            total_budget_musd,
            distance_m,
            ve_max_m_s,
            topology,
            bounds=bounds,
            xatol=xatol,
            max_nfev=max_nfev,
        )
        if optimized is not None:
            results[topology] = optimized
    if not results:
        raise RuntimeError("No topology produced a feasible budget optimum.")
    best = min(results.values(), key=lambda item: item.transfer.total_time_days)
    return best, results


# -------------------------------- regression suite --------------------------------


def run_regressions(include_budget_optimization: bool = True) -> dict:
    fixed_cases = [
        {
            "name": "5 AU / known five-arc case",
            "problem": TransferProblem(
                power_w=130_286_899.51386033,
                initial_mass_kg=1_820_678.480199766,
                final_mass_kg=593_910.141798527,
                distance_m=5.0 * AU_M,
                ve_max_m_s=250_000.0,
            ),
            "expected_topology": Topology.FIVE_ARC,
            "expected_days": 328.0127194439226,
        },
        {
            "name": "50 AU / known five-arc case",
            "problem": TransferProblem(
                power_w=128_881_929.09352272,
                initial_mass_kg=1_887_314.2201357787,
                final_mass_kg=596_748.7555049236,
                distance_m=50.0 * AU_M,
                ve_max_m_s=250_000.0,
            ),
            "expected_topology": Topology.FIVE_ARC,
            "expected_days": 1533.0396657561396,
        },
        {
            "name": "50 AU / high-budget terminal-boundary case",
            "problem": TransferProblem(
                power_w=1_321_675_685.6740997,
                initial_mass_kg=5_314_810.336599838,
                final_mass_kg=1_043_961.3697604922,
                distance_m=50.0 * AU_M,
                ve_max_m_s=250_000.0,
            ),
            "expected_topology": Topology.TERMINAL_BOUNDARY,
            "expected_days": 821.0756822866467,
        },
    ]

    fixed_results = []
    for case in fixed_cases:
        # Solve the expected family to high accuracy, then also check that no
        # other converged family is faster using a bounded number of starts.
        expected = solve_topology(
            case["problem"],
            case["expected_topology"],
            max_starts=120,
        )
        if expected is None:
            raise AssertionError(f"Regression failed to converge: {case['name']}")
        time_error = expected.total_time_days - case["expected_days"]
        if abs(time_error) > 2.0e-5:
            raise AssertionError(
                f"{case['name']}: time error {time_error:.6g} days"
            )
        if expected.topology != case["expected_topology"]:
            raise AssertionError(f"{case['name']}: wrong topology")
        fixed_results.append(
            {
                "name": case["name"],
                "expected_days": case["expected_days"],
                "computed": expected.as_dict(),
                "time_error_days": time_error,
            }
        )

    output = {"fixed_parameter_cases": fixed_results}

    if include_budget_optimization:
        model = BudgetModel()
        optimization_cases = [
            {
                "name": "50 AU / 1000 MUSD / 250 km/s",
                "budget": 1000.0,
                "distance": 50.0 * AU_M,
                "ve": 250_000.0,
                "topology": Topology.FIVE_ARC,
                "bounds": (0.75, 0.88),
                "expected_x": 0.8193208349516803,
                "expected_days": 1533.0396657561396,
            },
            {
                "name": "5 AU / 1000 MUSD / 250 km/s",
                "budget": 1000.0,
                "distance": 5.0 * AU_M,
                "ve": 250_000.0,
                "topology": Topology.FIVE_ARC,
                "bounds": (0.75, 0.88),
                "expected_x": 0.8282524326,
                "expected_days": 328.0127194439226,
            },
            {
                "name": "50 AU / 3000 MUSD / 250 km/s",
                "budget": 3000.0,
                "distance": 50.0 * AU_M,
                "ve": 250_000.0,
                "topology": Topology.TERMINAL_BOUNDARY,
                "bounds": (0.88, 0.98),
                "expected_x": 0.9335645716269435,
                "expected_days": 821.0756822866467,
            },
        ]
        optimized_results = []
        for case in optimization_cases:
            result = optimize_budget_for_topology(
                model,
                case["budget"],
                case["distance"],
                case["ve"],
                case["topology"],
                bounds=case["bounds"],
                xatol=1.0e-11,
            )
            if result is None:
                raise AssertionError(f"Optimization failed: {case['name']}")
            if abs(result.transfer.total_time_days - case["expected_days"]) > 3.0e-5:
                raise AssertionError(f"Optimization time mismatch: {case['name']}")
            if abs(result.engine_fraction - case["expected_x"]) > 5.0e-6:
                raise AssertionError(f"Optimization x mismatch: {case['name']}")
            optimized_results.append(
                {
                    "name": case["name"],
                    "expected_x": case["expected_x"],
                    "expected_days": case["expected_days"],
                    "computed": result.as_dict(),
                }
            )
        output["budget_optimization_cases"] = optimized_results

        low_ve = optimize_budget_for_topology(
            model,
            3000.0,
            50.0 * AU_M,
            50_000.0,
            Topology.TERMINAL_BOUNDARY,
            bounds=(0.58, 0.76),
            xatol=1.0e-11,
        )
        if low_ve is None:
            raise AssertionError("Low-V terminal-boundary optimization failed.")
        output["low_ve_ordinary_branch"] = low_ve.as_dict()

    return output


# -------------------------------------- CLI --------------------------------------


def _print_json(data: object) -> None:
    print(json.dumps(data, indent=2, sort_keys=False))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    direct = sub.add_parser("direct", help="Solve a fixed engine/mass problem.")
    direct.add_argument("--power-w", type=float, required=True)
    direct.add_argument("--mi-kg", type=float, required=True)
    direct.add_argument("--mf-kg", type=float, required=True)
    direct.add_argument("--distance-au", type=float, required=True)
    direct.add_argument("--ve-km-s", type=float, required=True)

    budget = sub.add_parser("budget", help="Optimize the spreadsheet budget model.")
    budget.add_argument("--budget-musd", type=float, required=True)
    budget.add_argument("--distance-au", type=float, required=True)
    budget.add_argument("--ve-km-s", type=float, required=True)
    budget.add_argument("--xmin", type=float, default=0.001)
    budget.add_argument("--xmax", type=float, default=0.999)

    regression = sub.add_parser("regressions", help="Run known-case checks.")
    regression.add_argument(
        "--fixed-only",
        action="store_true",
        help="Skip the budget-allocation optimization checks.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "direct":
        problem = TransferProblem(
            power_w=args.power_w,
            initial_mass_kg=args.mi_kg,
            final_mass_kg=args.mf_kg,
            distance_m=args.distance_au * AU_M,
            ve_max_m_s=args.ve_km_s * 1000.0,
        )
        best, candidates = solve_all_topologies(problem)
        _print_json(
            {
                "best": best.as_dict(),
                "candidates": {
                    topology.value: solution.as_dict()
                    for topology, solution in candidates.items()
                },
            }
        )
        return 0

    if args.command == "budget":
        best, candidates = optimize_budget_all_topologies(
            BudgetModel(),
            args.budget_musd,
            args.distance_au * AU_M,
            args.ve_km_s * 1000.0,
            bounds=(args.xmin, args.xmax),
        )
        _print_json(
            {
                "best": best.as_dict(),
                "candidates": {
                    topology.value: result.as_dict()
                    for topology, result in candidates.items()
                },
            }
        )
        return 0

    results = run_regressions(not args.fixed_only)
    _print_json(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
