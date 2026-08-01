#!/usr/bin/env python3
"""Lifecycle fare optimizer for a fusion passenger ship.

The public command is ``fare``. It minimizes ticket cost per paying passenger
using lifecycle accounting only. The reusable ship capital is amortized over
lifetime trips; recurring payload, propellant, and operations are charged per
trip.

Payload model
-------------
For ``N`` passengers and self-consistent transfer time ``T`` in days::

    payload_mass_kg = constant_kg
                    + per_passenger_kg * N
                    + per_day_kg * T
                    + per_passenger_day_kg * N * T

The transfer equations and the payload/time fixed point are solved together.
The internal fixed-budget ship solver remains available because lifecycle fare
optimization uses it to generate feasible continuation seeds.
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
    SpreadsheetEngineeringEconomics,
    Topology,
    TransferProblem,
    TransferSolution,
    _integrate_forced_topology,
    _solution_from_trajectory,
    add_engineering_economics_arguments,
    engineering_economics_from_args,
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

    def components_kg(self, passengers: int, transfer_days: float) -> Dict[str, float]:
        """Return the four additive mass components separately."""

        self.mass_kg(passengers, transfer_days)  # validates all inputs
        return {
            "constant": self.constant_kg,
            "per_passenger": self.per_passenger_kg * passengers,
            "per_day": self.per_day_kg * transfer_days,
            "per_passenger_day": (
                self.per_passenger_day_kg * passengers * transfer_days
            ),
        }


@dataclass(frozen=True)
class LifecycleTicketEconomics:
    """Amortization and operating assumptions for a passenger ticket.

    With utilization=1, cycle_time_multiplier=1, turnaround=0, load_factor=1,
    no recurring costs, and no markup, this reduces exactly to

        SHIP_PRICE / (SHIP_LIFETIME / TRIP_DURATION) / PASSENGERS.

    The reusable fractions classify the four payload-model terms.  Defaults
    treat constant and per-passenger equipment as reusable ship hardware and
    the time-dependent terms as recurring trip consumables.
    """

    ship_lifetime_days: float
    utilization_fraction: float = 1.0
    cycle_time_multiplier: float = 1.0
    turnaround_days: float = 0.0
    load_factor: float = 1.0
    operations_cost_per_trip_usd: float = 0.0
    operations_cost_per_day_usd: float = 0.0
    ticket_markup_fraction: float = 0.0
    reusable_constant_fraction: float = 1.0
    reusable_per_passenger_fraction: float = 1.0
    reusable_per_day_fraction: float = 0.0
    reusable_per_passenger_day_fraction: float = 0.0

    def validate(self) -> None:
        if not math.isfinite(self.ship_lifetime_days) or self.ship_lifetime_days <= 0.0:
            raise ValueError("ship_lifetime_days must be finite and positive.")
        if not math.isfinite(self.utilization_fraction) or not 0.0 < self.utilization_fraction <= 1.0:
            raise ValueError("utilization_fraction must satisfy 0 < value <= 1.")
        if not math.isfinite(self.cycle_time_multiplier) or self.cycle_time_multiplier <= 0.0:
            raise ValueError("cycle_time_multiplier must be finite and positive.")
        if not math.isfinite(self.turnaround_days) or self.turnaround_days < 0.0:
            raise ValueError("turnaround_days must be finite and nonnegative.")
        if not math.isfinite(self.load_factor) or not 0.0 < self.load_factor <= 1.0:
            raise ValueError("load_factor must satisfy 0 < value <= 1.")
        for name in (
            "operations_cost_per_trip_usd",
            "operations_cost_per_day_usd",
            "ticket_markup_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        for name in (
            "reusable_constant_fraction",
            "reusable_per_passenger_fraction",
            "reusable_per_day_fraction",
            "reusable_per_passenger_day_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must satisfy 0 <= value <= 1.")

    def cycle_time_days(self, transfer_days: float) -> float:
        self.validate()
        if not math.isfinite(transfer_days) or transfer_days <= 0.0:
            raise ValueError("transfer_days must be finite and positive.")
        return self.cycle_time_multiplier * transfer_days + self.turnaround_days

    def lifetime_trips(self, transfer_days: float) -> float:
        return (
            self.ship_lifetime_days * self.utilization_fraction
            / self.cycle_time_days(transfer_days)
        )


@dataclass(frozen=True)
class TicketCostBreakdown:
    ship_capital_cost_usd: float
    recurring_cost_per_trip_usd: float
    physical_ship_capital_cost_usd: float
    physical_recurring_cost_per_trip_usd: float
    operations_cost_per_trip_usd: float
    cycle_time_days: float
    lifetime_trips: float
    capital_cost_per_trip_usd: float
    paying_passengers_per_trip: float
    ticket_cost_before_markup_usd: float
    ticket_cost_usd_per_passenger: float
    reusable_payload_mass_kg: float
    recurring_payload_mass_kg: float
    fuel_mass_kg: float
    engine_radiator_capital_cost_usd: float
    tank_capital_cost_usd: float
    propellant_recurring_cost_usd: float


def calculate_lifecycle_ticket_cost(
    payload: LinearPassengerPayload,
    passengers: int,
    transfer_days: float,
    total_budget_musd: float,
    engine_fraction: float,
    engineering: SpreadsheetEngineeringEconomics,
    lifecycle: LifecycleTicketEconomics,
) -> TicketCostBreakdown:
    """Split the Excel budget into reusable capital and per-trip costs."""

    engineering.validate()
    lifecycle.validate()
    if passengers <= 0:
        raise ValueError("passengers must be positive.")
    if not 0.0 < engine_fraction < 1.0:
        raise ValueError("engine_fraction must lie strictly between 0 and 1.")

    components = payload.components_kg(passengers, transfer_days)
    reusable_fractions = {
        "constant": lifecycle.reusable_constant_fraction,
        "per_passenger": lifecycle.reusable_per_passenger_fraction,
        "per_day": lifecycle.reusable_per_day_fraction,
        "per_passenger_day": lifecycle.reusable_per_passenger_day_fraction,
    }
    reusable_payload_mass = sum(
        components[name] * reusable_fractions[name] for name in components
    )
    total_payload_mass = sum(components.values())
    recurring_payload_mass = total_payload_mass - reusable_payload_mass

    total_budget_usd = total_budget_musd * 1.0e6
    payload_cost_usd = total_payload_mass * engineering.payload_cost_usd_per_kg
    propulsion_budget_usd = total_budget_usd - payload_cost_usd
    if propulsion_budget_usd <= 0.0:
        raise ValueError("Budget does not cover payload cost.")

    engine_radiator_capital = engine_fraction * propulsion_budget_usd
    fuel_budget_usd = (1.0 - engine_fraction) * propulsion_budget_usd
    fuel_mass_kg = fuel_budget_usd / engineering.effective_fuel_cost_usd_per_kg
    tank_capital = (
        engineering.tank_mass_fraction
        * fuel_mass_kg
        * engineering.tank_cost_usd_per_kg
    )
    propellant_recurring = (
        fuel_mass_kg * engineering.propellant_cost_usd_per_kg
    )

    reusable_payload_cost = (
        reusable_payload_mass * engineering.payload_cost_usd_per_kg
    )
    recurring_payload_cost = (
        recurring_payload_mass * engineering.payload_cost_usd_per_kg
    )
    physical_ship_capital_cost = (
        reusable_payload_cost + engine_radiator_capital + tank_capital
    )
    physical_recurring_cost = recurring_payload_cost + propellant_recurring

    # This identity is a useful accounting check: the Excel total budget buys
    # reusable hardware plus one trip's recurring payload and propellant.
    if abs(
        (physical_ship_capital_cost + physical_recurring_cost) - total_budget_usd
    ) > max(
        1.0, 1.0e-9 * total_budget_usd
    ):
        raise ArithmeticError("Capital/recurring cost split does not close.")

    ship_capital_cost = physical_ship_capital_cost
    recurring_cost = physical_recurring_cost

    cycle_days = lifecycle.cycle_time_days(transfer_days)
    lifetime_trips = lifecycle.lifetime_trips(transfer_days)
    capital_per_trip = ship_capital_cost / lifetime_trips
    operations = (
        lifecycle.operations_cost_per_trip_usd
        + lifecycle.operations_cost_per_day_usd * cycle_days
    )
    paying_passengers = passengers * lifecycle.load_factor
    before_markup = (capital_per_trip + recurring_cost + operations) / paying_passengers
    fare = before_markup * (1.0 + lifecycle.ticket_markup_fraction)
    return TicketCostBreakdown(
        ship_capital_cost_usd=ship_capital_cost,
        recurring_cost_per_trip_usd=recurring_cost,
        physical_ship_capital_cost_usd=physical_ship_capital_cost,
        physical_recurring_cost_per_trip_usd=physical_recurring_cost,
        operations_cost_per_trip_usd=operations,
        cycle_time_days=cycle_days,
        lifetime_trips=lifetime_trips,
        capital_cost_per_trip_usd=capital_per_trip,
        paying_passengers_per_trip=paying_passengers,
        ticket_cost_before_markup_usd=before_markup,
        ticket_cost_usd_per_passenger=fare,
        reusable_payload_mass_kg=reusable_payload_mass,
        recurring_payload_mass_kg=recurring_payload_mass,
        fuel_mass_kg=fuel_mass_kg,
        engine_radiator_capital_cost_usd=engine_radiator_capital,
        tank_capital_cost_usd=tank_capital,
        propellant_recurring_cost_usd=propellant_recurring,
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
    ticket_cost_mode: str = "fixed_budget_ship"
    ticket_cost_breakdown: Optional[TicketCostBreakdown] = None

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
            "ticket_cost_mode": self.ticket_cost_mode,
            "ticket_cost_breakdown": (
                asdict(self.ticket_cost_breakdown)
                if self.ticket_cost_breakdown is not None
                else None
            ),
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
    engineering: Optional[SpreadsheetEngineeringEconomics] = None
    lifecycle: Optional[LifecycleTicketEconomics] = None


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
    breakdown: Optional[TicketCostBreakdown] = None
    ticket_mode = "fixed_budget_ship"
    ticket_cost = budget_musd * 1.0e6 / context.passengers
    if context.engineering is not None and context.lifecycle is not None:
        try:
            breakdown = calculate_lifecycle_ticket_cost(
                context.payload,
                context.passengers,
                assumed_days,
                budget_musd,
                x,
                context.engineering,
                context.lifecycle,
            )
        except (ArithmeticError, OverflowError, ValueError):
            return None
        ticket_mode = "lifecycle_amortized"
        ticket_cost = breakdown.ticket_cost_usd_per_passenger
    return PassengerShipResult(
        passengers=context.passengers,
        total_budget_musd=budget_musd,
        ticket_cost_usd_per_passenger=ticket_cost,
        engine_fraction=x,
        payload_mass_kg=payload_mass,
        assumed_transfer_days=assumed_days,
        fixed_point_residual_days=calculated_days - assumed_days,
        problem=problem,
        transfer=transfer,
        constraint_residuals=[float(value) for value in residuals],
        optimizer_success=optimizer_success,
        optimizer_message=optimizer_message,
        ticket_cost_mode=ticket_mode,
        ticket_cost_breakdown=breakdown,
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


def _fare_topology(
    payload: LinearPassengerPayload,
    passengers: int,
    budget_model: BudgetModel,
    engineering: SpreadsheetEngineeringEconomics,
    lifecycle: LifecycleTicketEconomics,
    distance_m: float,
    ve_max_m_s: float,
    topology: Topology,
    initials: Sequence[PassengerShipResult],
    *,
    budget_bounds_musd: Tuple[float, float],
    engine_fraction_bounds: Tuple[float, float],
    time_bounds_days: Tuple[float, float],
    maxiter: int,
) -> Optional[PassengerShipResult]:
    """Optimize amortized lifecycle fare for one topology."""

    bmin, bmax = budget_bounds_musd
    context = _Context(
        payload,
        passengers,
        budget_model,
        distance_m,
        ve_max_m_s,
        topology,
        None,
        max(initial.total_budget_musd for initial in initials),
        engineering,
        lifecycle,
    )
    bounds = (
        _shooting_bounds(topology)
        + [(math.log(bmin), math.log(bmax))]
        + [engine_fraction_bounds, (math.log(time_bounds_days[0]), math.log(time_bounds_days[1]))]
    )

    def objective(z: Sequence[float]) -> float:
        try:
            _, budget_musd, x, assumed_days = _decode(z, context)
            breakdown = calculate_lifecycle_ticket_cost(
                payload,
                passengers,
                assumed_days,
                budget_musd,
                x,
                engineering,
                lifecycle,
            )
            return breakdown.ticket_cost_usd_per_passenger / 1.0e6
        except (ArithmeticError, OverflowError, ValueError):
            return 1.0e30

    candidates: List[PassengerShipResult] = []
    for initial in initials:
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
            objective,
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
    return min(candidates, key=lambda item: item.ticket_cost_usd_per_passenger)


def optimize_lifecycle_ticket_cost(
    payload: LinearPassengerPayload,
    passengers: int,
    engineering: SpreadsheetEngineeringEconomics,
    lifecycle: LifecycleTicketEconomics,
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
    warm_start: Optional[PassengerOptimizationResult] = None,
    continuation_seed_budgets: Optional[Sequence[float]] = None,
) -> PassengerOptimizationResult:
    """Minimize amortized lifecycle fare per paying passenger.

    The objective includes reusable ship-capital amortization, recurring payload
    and propellant costs, optional operations, occupancy, cycle time, and markup.
    The transfer and nonlinear payload fixed point are solved simultaneously.
    """

    engineering.validate()
    lifecycle.validate()
    bmin, bmax = budget_bounds_musd
    if not 0.0 < bmin < bmax:
        raise ValueError("Invalid budget_bounds_musd.")
    selected = tuple(topologies) if topologies is not None else tuple(Topology)
    budget_model = engineering.budget_model(payload_mass_kg=1.0)

    # Generate feasible continuation seeds. Sensitivity sweeps may pass a
    # converged nearby result, avoiding a full multibudget restart at every point.
    seed_results: Dict[Topology, List[PassengerShipResult]] = {
        topology: [] for topology in selected
    }
    previous: Optional[Mapping[Topology, PassengerShipResult]] = None
    if warm_start is not None:
        previous = warm_start.topology_results
        for topology, result in warm_start.topology_results.items():
            if topology in seed_results:
                seed_results[topology].append(result)
    if continuation_seed_budgets is None:
        seed_budgets = [bmax, math.sqrt(bmin * bmax), 0.5 * (bmin + bmax)]
    else:
        seed_budgets = list(continuation_seed_budgets)
    for budget in seed_budgets:
        try:
            optimized = optimize_passenger_ship(
                payload,
                passengers,
                budget_model,
                budget,
                distance_m,
                ve_max_m_s,
                topologies=selected,
                engine_fraction_bounds=engine_fraction_bounds,
                time_bounds_days=time_bounds_days,
                engine_seed_points=engine_seed_points,
                maxiter=maxiter,
                max_nfev=max_nfev,
                max_starts=max_starts,
                extra_seeds=previous,
            )
        except RuntimeError:
            continue
        previous = optimized.topology_results
        for topology, result in optimized.topology_results.items():
            seed_results[topology].append(result)

    candidates: Dict[Topology, PassengerShipResult] = {}
    for topology in selected:
        initials = seed_results.get(topology, [])
        if not initials:
            continue
        candidate = _fare_topology(
            payload,
            passengers,
            budget_model,
            engineering,
            lifecycle,
            distance_m,
            ve_max_m_s,
            topology,
            initials,
            budget_bounds_musd=budget_bounds_musd,
            engine_fraction_bounds=engine_fraction_bounds,
            time_bounds_days=time_bounds_days,
            maxiter=maxiter,
        )
        if candidate is not None:
            candidates[topology] = candidate
    if not candidates:
        raise RuntimeError("Lifecycle fare optimization produced no feasible candidate.")
    best = min(candidates.values(), key=lambda item: item.ticket_cost_usd_per_passenger)
    return PassengerOptimizationResult(best=best, topology_results=candidates, mode="fare")


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


DEFAULT_FARE_VALUES = {
    "ship_lifetime_years": 25.0,
    "utilization_fraction": 1.0,
    "cycle_time_multiplier": 1.0,
    "turnaround_days": 0.0,
    "load_factor": 1.0,
    "ticket_markup_fraction": 0.0,
    "budget_min_musd": 150.0,
    "budget_max_musd": 80000.0,
    "passengers": 600,
    "payload_constant_kg": 300_000.0,
    "payload_per_passenger_kg": 2_500.0,
    "payload_per_day_kg": 100.0,
    "payload_per_passenger_day_kg": 2.0,
    "distance_au": 1.5,
    "ve_km_s": 250.0,
    "alpha_eng_w_per_kg": 20_000.0,
    "phi_heat_to_total": 0.15,
    "rho_rad_w_per_kg": 10_000.0,
    "payload_cost_usd_per_kg": 1_500.0,
    "propellant_cost_usd_per_kg": 20.0,
    "engine_core_cost_usd_per_kg": 10_000.0,
    "radiator_cost_usd_per_kg": 1_500.0,
    "tank_mass_fraction": 0.05,
    "tank_cost_usd_per_kg": 300.0,
}


def add_fare_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the complete lifecycle-fare input surface to ``parser``."""

    parser.add_argument("--passengers", type=int, default=DEFAULT_FARE_VALUES["passengers"])
    parser.add_argument("--payload-constant-kg", type=float, default=DEFAULT_FARE_VALUES["payload_constant_kg"])
    parser.add_argument("--payload-per-passenger-kg", type=float, default=DEFAULT_FARE_VALUES["payload_per_passenger_kg"])
    parser.add_argument("--payload-per-day-kg", type=float, default=DEFAULT_FARE_VALUES["payload_per_day_kg"])
    parser.add_argument("--payload-per-passenger-day-kg", type=float, default=DEFAULT_FARE_VALUES["payload_per_passenger_day_kg"])
    parser.add_argument("--distance-au", type=float, default=DEFAULT_FARE_VALUES["distance_au"])
    parser.add_argument("--ve-km-s", type=float, default=DEFAULT_FARE_VALUES["ve_km_s"])
    parser.add_argument("--budget-min-musd", type=float, default=DEFAULT_FARE_VALUES["budget_min_musd"])
    parser.add_argument("--budget-max-musd", type=float, default=DEFAULT_FARE_VALUES["budget_max_musd"])

    lifetime = parser.add_mutually_exclusive_group(required=False)
    lifetime.add_argument("--ship-lifetime-days", type=float, default=None)
    lifetime.add_argument("--ship-lifetime-years", type=float, default=DEFAULT_FARE_VALUES["ship_lifetime_years"])
    parser.add_argument("--utilization-fraction", type=float, default=DEFAULT_FARE_VALUES["utilization_fraction"])
    parser.add_argument("--cycle-time-multiplier", type=float, default=DEFAULT_FARE_VALUES["cycle_time_multiplier"])
    parser.add_argument("--turnaround-days", type=float, default=DEFAULT_FARE_VALUES["turnaround_days"])
    parser.add_argument("--load-factor", type=float, default=DEFAULT_FARE_VALUES["load_factor"])
    parser.add_argument("--operations-cost-per-trip-usd", type=float, default=0.0)
    parser.add_argument("--operations-cost-per-day-usd", type=float, default=0.0)
    parser.add_argument("--ticket-markup-fraction", type=float, default=DEFAULT_FARE_VALUES["ticket_markup_fraction"])
    parser.add_argument("--reusable-constant-fraction", type=float, default=1.0)
    parser.add_argument("--reusable-per-passenger-fraction", type=float, default=1.0)
    parser.add_argument("--reusable-per-day-fraction", type=float, default=0.0)
    parser.add_argument("--reusable-per-passenger-day-fraction", type=float, default=0.0)

    parser.add_argument("--xmin", type=float, default=0.01)
    parser.add_argument("--xmax", type=float, default=0.99)
    parser.add_argument("--time-min-days", type=float, default=0.05)
    parser.add_argument("--time-max-days", type=float, default=50_000.0)
    parser.add_argument("--engine-seed-points", type=int, default=5)
    parser.add_argument("--maxiter", type=int, default=800)
    parser.add_argument("--max-starts", type=int, default=30)

    add_engineering_economics_arguments(parser, include_payload_mass=False)
    parser.set_defaults(
        alpha_eng_w_per_kg=DEFAULT_FARE_VALUES["alpha_eng_w_per_kg"],
        phi_heat_to_total=DEFAULT_FARE_VALUES["phi_heat_to_total"],
        rho_rad_w_per_kg=DEFAULT_FARE_VALUES["rho_rad_w_per_kg"],
        payload_cost_usd_per_kg=DEFAULT_FARE_VALUES["payload_cost_usd_per_kg"],
        propellant_cost_usd_per_kg=DEFAULT_FARE_VALUES["propellant_cost_usd_per_kg"],
        engine_core_cost_usd_per_kg=DEFAULT_FARE_VALUES["engine_core_cost_usd_per_kg"],
        radiator_cost_usd_per_kg=DEFAULT_FARE_VALUES["radiator_cost_usd_per_kg"],
        tank_mass_fraction=DEFAULT_FARE_VALUES["tank_mass_fraction"],
        tank_cost_usd_per_kg=DEFAULT_FARE_VALUES["tank_cost_usd_per_kg"],
    )


def payload_from_args(args: argparse.Namespace) -> LinearPassengerPayload:
    return LinearPassengerPayload(
        args.payload_constant_kg,
        args.payload_per_passenger_kg,
        args.payload_per_day_kg,
        args.payload_per_passenger_day_kg,
    )


def lifecycle_from_args(args: argparse.Namespace) -> LifecycleTicketEconomics:
    lifetime_days = (
        args.ship_lifetime_days
        if args.ship_lifetime_days is not None
        else args.ship_lifetime_years * 365.25
    )
    return LifecycleTicketEconomics(
        ship_lifetime_days=lifetime_days,
        utilization_fraction=args.utilization_fraction,
        cycle_time_multiplier=args.cycle_time_multiplier,
        turnaround_days=args.turnaround_days,
        load_factor=args.load_factor,
        operations_cost_per_trip_usd=args.operations_cost_per_trip_usd,
        operations_cost_per_day_usd=args.operations_cost_per_day_usd,
        ticket_markup_fraction=args.ticket_markup_fraction,
        reusable_constant_fraction=args.reusable_constant_fraction,
        reusable_per_passenger_fraction=args.reusable_per_passenger_fraction,
        reusable_per_day_fraction=args.reusable_per_day_fraction,
        reusable_per_passenger_day_fraction=args.reusable_per_passenger_day_fraction,
    )


def optimize_fare_from_args(
    args: argparse.Namespace,
    *,
    warm_start: Optional[PassengerOptimizationResult] = None,
    continuation_seed_budgets: Optional[Sequence[float]] = None,
) -> PassengerOptimizationResult:
    payload = payload_from_args(args)
    engineering = engineering_economics_from_args(args)
    return optimize_lifecycle_ticket_cost(
        payload,
        args.passengers,
        engineering,
        lifecycle_from_args(args),
        args.distance_au * AU_M,
        args.ve_km_s * 1000.0,
        budget_bounds_musd=(args.budget_min_musd, args.budget_max_musd),
        engine_fraction_bounds=(args.xmin, args.xmax),
        time_bounds_days=(args.time_min_days, args.time_max_days),
        engine_seed_points=args.engine_seed_points,
        maxiter=args.maxiter,
        max_starts=args.max_starts,
        warm_start=warm_start,
        continuation_seed_budgets=continuation_seed_budgets,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fare = sub.add_parser(
        "fare",
        help="Minimize lifecycle-amortized fare per paying passenger.",
    )
    add_fare_arguments(fare)
    sub.add_parser("regressions", help="Run transfer and nonlinear-payload checks.")
    return parser


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=False))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "regressions":
        _print_json(run_regressions())
        return 0
    result = optimize_fare_from_args(args)
    _print_json(result.as_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
