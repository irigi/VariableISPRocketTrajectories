#!/usr/bin/env python3
"""Nonlinear passenger-payload and ticket-cost optimizer.

This companion module leaves ``time_optimal_transfer_solver.py`` unchanged and
uses its analytic arc propagator for every ordinary topology.

Payload model
-------------

For N passengers and transfer time T in days,

    payload_mass_kg = constant_kg
                    + per_passenger_kg * N
                    + per_day_kg * T
                    + per_passenger_day_kg * N * T.

The journey time therefore changes the payload, while the payload changes the
ship masses and journey time.  The circular dependence is solved directly as a
constrained nonlinear program rather than by a fixed count of Newton iterations.

Modes
-----

``ship``
    Fixed total budget.  Minimize self-consistent transfer time over engine/fuel
    allocation and all requested transfer topologies.

``ticket``
    Minimize total project budget per passenger.  The passenger count is fixed,
    so this is equivalent to minimizing total budget.  After the minimum feasible
    budget is found, the ship is re-optimized for minimum self-consistent transfer
    time at that budget.

The decision variables contain the three transfer shooting variables, engine
fraction, self-consistent journey time, and (in ticket mode) total budget.  Four
constraints enforce final distance, final velocity, final mass, and equality of
assumed and dynamically calculated journey time.

The result is a candidate optimum over the ordinary topologies implemented by the
base solver.  Ticket cost here means project budget divided by passengers; finance,
operations, profit, insurance, and return-trip costs are outside this model.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from time_optimal_transfer_solver import (
    AU_M,
    BudgetModel,
    DAY_S,
    Topology,
    TransferProblem,
    TransferSolution,
    _integrate_forced_topology,
    _solution_from_trajectory,
    solve_topology,
)


@dataclass(frozen=True)
class LinearPassengerPayload:
    constant_kg: float
    per_passenger_kg: float
    per_day_kg: float
    per_passenger_day_kg: float

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")

    def mass_kg(self, passengers: int, transfer_days: float) -> float:
        self.validate()
        if passengers <= 0:
            raise ValueError("passengers must be a positive integer.")
        if not math.isfinite(transfer_days) or transfer_days < 0.0:
            raise ValueError("transfer_days must be finite and nonnegative.")
        return (
            self.constant_kg
            + self.per_passenger_kg * passengers
            + self.per_day_kg * transfer_days
            + self.per_passenger_day_kg * passengers * transfer_days
        )


@dataclass
class PassengerShipResult:
    passengers: int
    total_budget_musd: float
    ticket_cost_usd_per_passenger: float
    engine_fraction: float
    payload_mass_kg: float
    assumed_transfer_days: float
    fixed_point_residual_days: float
    problem: TransferProblem
    transfer: TransferSolution
    constraint_residuals: List[float]
    optimizer_success: bool
    optimizer_message: str

    @property
    def topology(self) -> Topology:
        return self.transfer.topology

    def as_dict(self) -> dict:
        return {
            "passengers": self.passengers,
            "total_budget_musd": self.total_budget_musd,
            "ticket_cost_usd_per_passenger": self.ticket_cost_usd_per_passenger,
            "engine_fraction": self.engine_fraction,
            "payload_mass_kg": self.payload_mass_kg,
            "assumed_transfer_days": self.assumed_transfer_days,
            "fixed_point_residual_days": self.fixed_point_residual_days,
            "problem": asdict(self.problem),
            "transfer": self.transfer.as_dict(),
            "constraint_residuals": self.constraint_residuals,
            "optimizer_success": self.optimizer_success,
            "optimizer_message": self.optimizer_message,
        }


@dataclass
class PassengerOptimizationResult:
    best: PassengerShipResult
    topology_results: Dict[Topology, PassengerShipResult]
    mode: str

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "best": self.best.as_dict(),
            "topology_results": {
                topology.value: result.as_dict()
                for topology, result in self.topology_results.items()
            },
        }


def _log_shooting(solution: TransferSolution) -> np.ndarray:
    return np.log(
        [
            solution.shooting_y0,
            solution.shooting_minus_yf,
            solution.shooting_kappa,
        ]
    )


def _budget_problem(
    budget_model: BudgetModel,
    payload_mass_kg: float,
    total_budget_musd: float,
    engine_fraction: float,
    distance_m: float,
    ve_max_m_s: float,
) -> TransferProblem:
    dynamic_model = BudgetModel(
        payload_mass_kg=payload_mass_kg,
        payload_cost_per_kg=budget_model.payload_cost_per_kg,
        engine_radiator_cost_per_kg=budget_model.engine_radiator_cost_per_kg,
        system_specific_mass_kg_per_w=budget_model.system_specific_mass_kg_per_w,
        effective_fuel_cost_per_kg=budget_model.effective_fuel_cost_per_kg,
        tank_mass_fraction=budget_model.tank_mass_fraction,
    )
    return dynamic_model.with_trajectory(
        total_budget_musd,
        engine_fraction,
        distance_m,
        ve_max_m_s,
    )


def _shooting_bounds(topology: Topology) -> List[Tuple[float, float]]:
    if topology.starts_with_variable:
        y0 = (math.log(1.0 + 1.0e-8), math.log(1.0e4))
    else:
        y0 = (math.log(0.5 + 1.0e-8), math.log(1.0 - 1.0e-8))
    if topology.ends_with_variable:
        minus_yf = (math.log(1.0 + 1.0e-8), math.log(1.0e8))
    else:
        minus_yf = (math.log(1.0e-6), math.log(1.0e8))
    return [y0, minus_yf, (math.log(1.0e-7), math.log(1.0e7))]


@dataclass(frozen=True)
class _Context:
    payload: LinearPassengerPayload
    passengers: int
    budget_model: BudgetModel
    distance_m: float
    ve_max_m_s: float
    topology: Topology
    fixed_budget_musd: Optional[float]
    budget_scale_musd: float


def _decode(z: Sequence[float], context: _Context) -> Tuple[np.ndarray, float, float, float]:
    shooting = np.asarray(z[:3], dtype=float)
    if context.fixed_budget_musd is None:
        total_budget_musd = math.exp(float(z[3]))
        engine_fraction = float(z[4])
        transfer_days = math.exp(float(z[5]))
    else:
        total_budget_musd = context.fixed_budget_musd
        engine_fraction = float(z[3])
        transfer_days = math.exp(float(z[4]))
    return shooting, total_budget_musd, engine_fraction, transfer_days


def _evaluate_state(
    z: Sequence[float], context: _Context
) -> Tuple[TransferProblem, object, float, float, float]:
    shooting, budget_musd, x, assumed_days = _decode(z, context)
    payload_mass = context.payload.mass_kg(context.passengers, assumed_days)
    problem = _budget_problem(
        context.budget_model,
        payload_mass,
        budget_musd,
        x,
        context.distance_m,
        context.ve_max_m_s,
    )
    trajectory = _integrate_forced_topology(shooting, context.topology)
    calculated_days = trajectory.theta * problem.time_scale_s / DAY_S
    return problem, trajectory, payload_mass, assumed_days, calculated_days


def _constraints(z: Sequence[float], context: _Context) -> np.ndarray:
    try:
        problem, trajectory, _, assumed_days, calculated_days = _evaluate_state(z, context)
        return np.array(
            [
                (trajectory.xi - problem.dimensionless_distance)
                / problem.dimensionless_distance,
                trajectory.u,
                math.log(trajectory.mu / problem.final_mass_ratio),
                math.log(calculated_days / assumed_days),
            ],
            dtype=float,
        )
    except (ArithmeticError, OverflowError, ValueError):
        return np.array([100.0, 100.0, 100.0, 100.0], dtype=float)


def _seed_shooting(
    payload: LinearPassengerPayload,
    passengers: int,
    budget_model: BudgetModel,
    total_budget_musd: float,
    engine_fraction: float,
    distance_m: float,
    ve_max_m_s: float,
    topology: Topology,
    initial_days: float,
    *,
    max_nfev: int,
    max_starts: int,
) -> Optional[Tuple[np.ndarray, float]]:
    """Continuation seed using the unchanged transfer solver."""

    days = float(initial_days)
    shooting_guess: Optional[np.ndarray] = None
    solution: Optional[TransferSolution] = None
    for iteration in range(6):
        try:
            payload_mass = payload.mass_kg(passengers, days)
            problem = _budget_problem(
                budget_model,
                payload_mass,
                total_budget_musd,
                engine_fraction,
                distance_m,
                ve_max_m_s,
            )
        except ValueError:
            return None
        guesses = [shooting_guess] if shooting_guess is not None else None
        solution = solve_topology(
            problem,
            topology,
            guesses,
            max_nfev=max_nfev,
            max_starts=(5 if guesses else max_starts),
        )
        if solution is None:
            return None
        shooting_guess = _log_shooting(solution)
        new_days = solution.total_time_days
        if abs(new_days - days) < 1.0e-4:
            days = new_days
            break
        days = 0.25 * days + 0.75 * new_days
    if solution is None or shooting_guess is None:
        return None
    return shooting_guess, max(days, 1.0e-8)


def _result_from_z(
    z: Sequence[float],
    context: _Context,
    *,
    optimizer_success: bool,
    optimizer_message: str,
) -> Optional[PassengerShipResult]:
    residuals = _constraints(z, context)
    if not np.all(np.isfinite(residuals)) or np.linalg.norm(residuals, ord=np.inf) > 2.0e-6:
        return None
    try:
        problem, trajectory, payload_mass, assumed_days, calculated_days = _evaluate_state(z, context)
        transfer = _solution_from_trajectory(problem, context.topology, trajectory)
        _, budget_musd, x, _ = _decode(z, context)
    except (ArithmeticError, OverflowError, ValueError):
        return None
    return PassengerShipResult(
        passengers=context.passengers,
        total_budget_musd=budget_musd,
        ticket_cost_usd_per_passenger=budget_musd * 1.0e6 / context.passengers,
        engine_fraction=x,
        payload_mass_kg=payload_mass,
        assumed_transfer_days=assumed_days,
        fixed_point_residual_days=calculated_days - assumed_days,
        problem=problem,
        transfer=transfer,
        constraint_residuals=[float(value) for value in residuals],
        optimizer_success=optimizer_success,
        optimizer_message=optimizer_message,
    )


def _fixed_budget_topology(
    payload: LinearPassengerPayload,
    passengers: int,
    budget_model: BudgetModel,
    total_budget_musd: float,
    distance_m: float,
    ve_max_m_s: float,
    topology: Topology,
    *,
    engine_fraction_bounds: Tuple[float, float],
    time_bounds_days: Tuple[float, float],
    engine_seed_points: int,
    maxiter: int,
    max_nfev: int,
    max_starts: int,
    extra_seeds: Optional[Iterable[PassengerShipResult]] = None,
) -> Optional[PassengerShipResult]:
    context = _Context(
        payload,
        passengers,
        budget_model,
        distance_m,
        ve_max_m_s,
        topology,
        total_budget_musd,
        total_budget_musd,
    )
    xmin, xmax = engine_fraction_bounds
    tmin, tmax = time_bounds_days
    bounds = _shooting_bounds(topology) + [(xmin, xmax), (math.log(tmin), math.log(tmax))]

    seeds: List[np.ndarray] = []
    if extra_seeds:
        for result in extra_seeds:
            if result.topology == topology:
                seeds.append(
                    np.concatenate(
                        [
                            _log_shooting(result.transfer),
                            [result.engine_fraction, math.log(result.assumed_transfer_days)],
                        ]
                    )
                )

    x_values = np.linspace(xmin, xmax, max(3, int(engine_seed_points)))
    initial_days = math.sqrt(tmin * tmax)
    for x in x_values:
        seeded = _seed_shooting(
            payload,
            passengers,
            budget_model,
            total_budget_musd,
            float(x),
            distance_m,
            ve_max_m_s,
            topology,
            initial_days,
            max_nfev=max_nfev,
            max_starts=max_starts,
        )
        if seeded is not None:
            shooting, days = seeded
            seeds.append(np.concatenate([shooting, [float(x), math.log(days)]]))

    candidates: List[PassengerShipResult] = []
    for seed in seeds:
        result = minimize(
            lambda z: math.exp(float(z[4])) / max(1.0, initial_days),
            seed,
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "eq", "fun": lambda z: _constraints(z, context)},
            options={"ftol": 1.0e-11, "maxiter": maxiter, "disp": False},
        )
        candidate = _result_from_z(
            result.x,
            context,
            optimizer_success=bool(result.success),
            optimizer_message=str(result.message),
        )
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        return None
    return min(candidates, key=lambda item: item.transfer.total_time_days)


def optimize_passenger_ship(
    payload: LinearPassengerPayload,
    passengers: int,
    budget_model: BudgetModel,
    total_budget_musd: float,
    distance_m: float,
    ve_max_m_s: float,
    *,
    topologies: Optional[Iterable[Topology]] = None,
    engine_fraction_bounds: Tuple[float, float] = (0.02, 0.98),
    time_bounds_days: Tuple[float, float] = (0.05, 50_000.0),
    engine_seed_points: int = 5,
    maxiter: int = 600,
    max_nfev: int = 5_000,
    max_starts: int = 30,
    extra_seeds: Optional[Mapping[Topology, PassengerShipResult]] = None,
) -> PassengerOptimizationResult:
    """Minimize self-consistent journey time at a fixed total budget."""

    payload.validate()
    selected = tuple(topologies) if topologies is not None else tuple(Topology)
    results: Dict[Topology, PassengerShipResult] = {}
    for topology in selected:
        extra = None
        if extra_seeds and topology in extra_seeds:
            extra = [extra_seeds[topology]]
        result = _fixed_budget_topology(
            payload,
            passengers,
            budget_model,
            total_budget_musd,
            distance_m,
            ve_max_m_s,
            topology,
            engine_fraction_bounds=engine_fraction_bounds,
            time_bounds_days=time_bounds_days,
            engine_seed_points=engine_seed_points,
            maxiter=maxiter,
            max_nfev=max_nfev,
            max_starts=max_starts,
            extra_seeds=extra,
        )
        if result is not None:
            results[topology] = result
    if not results:
        raise RuntimeError("No self-consistent passenger ship converged.")
    best = min(results.values(), key=lambda item: item.transfer.total_time_days)
    return PassengerOptimizationResult(best=best, topology_results=results, mode="ship")


def _ticket_topology(
    payload: LinearPassengerPayload,
    passengers: int,
    budget_model: BudgetModel,
    distance_m: float,
    ve_max_m_s: float,
    topology: Topology,
    initial: PassengerShipResult,
    *,
    budget_bounds_musd: Tuple[float, float],
    engine_fraction_bounds: Tuple[float, float],
    time_bounds_days: Tuple[float, float],
    maxiter: int,
) -> Optional[PassengerShipResult]:
    bmin, bmax = budget_bounds_musd
    context = _Context(
        payload,
        passengers,
        budget_model,
        distance_m,
        ve_max_m_s,
        topology,
        None,
        initial.total_budget_musd,
    )
    bounds = (
        _shooting_bounds(topology)
        + [(math.log(bmin), math.log(bmax))]
        + [engine_fraction_bounds, (math.log(time_bounds_days[0]), math.log(time_bounds_days[1]))]
    )
    seed = np.concatenate(
        [
            _log_shooting(initial.transfer),
            [
                math.log(initial.total_budget_musd),
                initial.engine_fraction,
                math.log(initial.assumed_transfer_days),
            ],
        ]
    )
    result = minimize(
        lambda z: math.exp(float(z[3])) / context.budget_scale_musd,
        seed,
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "eq", "fun": lambda z: _constraints(z, context)},
        options={"ftol": 1.0e-11, "maxiter": maxiter, "disp": False},
    )
    return _result_from_z(
        result.x,
        context,
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
    )


def optimize_ticket_cost(
    payload: LinearPassengerPayload,
    passengers: int,
    budget_model: BudgetModel,
    distance_m: float,
    ve_max_m_s: float,
    *,
    budget_bounds_musd: Tuple[float, float],
    topologies: Optional[Iterable[Topology]] = None,
    engine_fraction_bounds: Tuple[float, float] = (0.02, 0.98),
    time_bounds_days: Tuple[float, float] = (0.05, 50_000.0),
    engine_seed_points: int = 5,
    maxiter: int = 800,
    max_nfev: int = 5_000,
    max_starts: int = 30,
) -> PassengerOptimizationResult:
    """Minimize project budget per passenger, then time-optimize that ship."""

    bmin, bmax = budget_bounds_musd
    if not 0.0 < bmin < bmax:
        raise ValueError("Invalid budget_bounds_musd.")
    selected = tuple(topologies) if topologies is not None else tuple(Topology)

    # Obtain feasible high-budget seeds with the original transfer solver.
    high = optimize_passenger_ship(
        payload,
        passengers,
        budget_model,
        bmax,
        distance_m,
        ve_max_m_s,
        topologies=selected,
        engine_fraction_bounds=engine_fraction_bounds,
        time_bounds_days=time_bounds_days,
        engine_seed_points=engine_seed_points,
        maxiter=maxiter,
        max_nfev=max_nfev,
        max_starts=max_starts,
    )

    budget_candidates: Dict[Topology, PassengerShipResult] = {}
    for topology, initial in high.topology_results.items():
        candidate = _ticket_topology(
            payload,
            passengers,
            budget_model,
            distance_m,
            ve_max_m_s,
            topology,
            initial,
            budget_bounds_musd=budget_bounds_musd,
            engine_fraction_bounds=engine_fraction_bounds,
            time_bounds_days=time_bounds_days,
            maxiter=maxiter,
        )
        if candidate is not None:
            budget_candidates[topology] = candidate
    if not budget_candidates:
        raise RuntimeError("Ticket optimization did not produce a feasible candidate.")

    minimum_candidate = min(
        budget_candidates.values(), key=lambda item: item.total_budget_musd
    )

    # At the minimum feasible budget, re-optimize the ship for shortest time.
    # Ticket candidates are supplied as additional warm starts.
    try:
        final_ship = optimize_passenger_ship(
            payload,
            passengers,
            budget_model,
            minimum_candidate.total_budget_musd,
            distance_m,
            ve_max_m_s,
            topologies=selected,
            engine_fraction_bounds=engine_fraction_bounds,
            time_bounds_days=time_bounds_days,
            engine_seed_points=max(3, engine_seed_points),
            maxiter=maxiter,
            max_nfev=max_nfev,
            max_starts=max_starts,
            extra_seeds=budget_candidates,
        )
        final_ship.mode = "ticket"
        return final_ship
    except RuntimeError:
        # The constrained minimum itself is still a valid design; this fallback
        # can occur at a numerically singular feasibility boundary.
        return PassengerOptimizationResult(
            best=minimum_candidate,
            topology_results=budget_candidates,
            mode="ticket",
        )


def run_regressions() -> dict:
    """Check old-solver compatibility and nonlinear fixed-point closure."""

    fixed_payload = LinearPassengerPayload(
        constant_kg=350_000.0,
        per_passenger_kg=1_500.0,
        per_day_kg=0.0,
        per_passenger_day_kg=0.0,
    )
    compatibility = optimize_passenger_ship(
        fixed_payload,
        100,
        BudgetModel(),
        1000.0,
        5.0 * AU_M,
        250_000.0,
        topologies=(Topology.FIVE_ARC,),
        engine_fraction_bounds=(0.79, 0.86),
        time_bounds_days=(250.0, 450.0),
        engine_seed_points=3,
        max_starts=30,
    )
    if abs(compatibility.best.transfer.total_time_days - 328.0127194439226) > 5.0e-5:
        raise AssertionError("Fixed-payload compatibility time mismatch.")
    if abs(compatibility.best.engine_fraction - 0.8282524326) > 2.0e-5:
        raise AssertionError("Fixed-payload compatibility allocation mismatch.")

    nonlinear_payload = LinearPassengerPayload(
        constant_kg=220_000.0,
        per_passenger_kg=1_600.0,
        per_day_kg=40.0,
        per_passenger_day_kg=1.2,
    )
    nonlinear = optimize_passenger_ship(
        nonlinear_payload,
        120,
        BudgetModel(),
        1300.0,
        5.0 * AU_M,
        250_000.0,
        topologies=(Topology.FIVE_ARC, Topology.TERMINAL_BOUNDARY),
        engine_fraction_bounds=(0.72, 0.92),
        time_bounds_days=(150.0, 500.0),
        engine_seed_points=3,
        max_starts=24,
    )
    expected_mass = nonlinear_payload.mass_kg(
        120, nonlinear.best.transfer.total_time_days
    )
    if abs(expected_mass - nonlinear.best.payload_mass_kg) > 0.01:
        raise AssertionError("Nonlinear payload closure failed.")
    if abs(nonlinear.best.fixed_point_residual_days) > 1.0e-4:
        raise AssertionError("Nonlinear time closure failed.")

    return {
        "fixed_payload_compatibility": compatibility.as_dict(),
        "nonlinear_payload_ship": nonlinear.as_dict(),
    }


def _add_payload_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--passengers", type=int, required=True)
    parser.add_argument("--payload-constant-kg", type=float, required=True)
    parser.add_argument("--payload-per-passenger-kg", type=float, required=True)
    parser.add_argument("--payload-per-day-kg", type=float, required=True)
    parser.add_argument("--payload-per-passenger-day-kg", type=float, required=True)
    parser.add_argument("--distance-au", type=float, required=True)
    parser.add_argument("--ve-km-s", type=float, required=True)
    parser.add_argument("--xmin", type=float, default=0.02)
    parser.add_argument("--xmax", type=float, default=0.98)
    parser.add_argument("--time-min-days", type=float, default=0.05)
    parser.add_argument("--time-max-days", type=float, default=50_000.0)
    parser.add_argument("--engine-seed-points", type=int, default=5)
    parser.add_argument("--maxiter", type=int, default=800)
    parser.add_argument("--max-starts", type=int, default=30)


def _payload_from_args(args: argparse.Namespace) -> LinearPassengerPayload:
    return LinearPassengerPayload(
        args.payload_constant_kg,
        args.payload_per_passenger_kg,
        args.payload_per_day_kg,
        args.payload_per_passenger_day_kg,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ship = sub.add_parser("ship", help="Optimize a fixed-budget passenger ship.")
    _add_payload_arguments(ship)
    ship.add_argument("--budget-musd", type=float, required=True)

    ticket = sub.add_parser("ticket", help="Minimize project budget per passenger.")
    _add_payload_arguments(ticket)
    ticket.add_argument("--budget-min-musd", type=float, required=True)
    ticket.add_argument("--budget-max-musd", type=float, required=True)

    sub.add_parser("regressions", help="Run compatibility and nonlinear checks.")
    return parser


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=False))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "regressions":
        _print_json(run_regressions())
        return 0

    payload = _payload_from_args(args)
    common = dict(
        engine_fraction_bounds=(args.xmin, args.xmax),
        time_bounds_days=(args.time_min_days, args.time_max_days),
        engine_seed_points=args.engine_seed_points,
        maxiter=args.maxiter,
        max_starts=args.max_starts,
    )
    if args.command == "ship":
        result = optimize_passenger_ship(
            payload,
            args.passengers,
            BudgetModel(),
            args.budget_musd,
            args.distance_au * AU_M,
            args.ve_km_s * 1000.0,
            **common,
        )
    else:
        result = optimize_ticket_cost(
            payload,
            args.passengers,
            BudgetModel(),
            args.distance_au * AU_M,
            args.ve_km_s * 1000.0,
            budget_bounds_musd=(args.budget_min_musd, args.budget_max_musd),
            **common,
        )
    _print_json(result.as_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
