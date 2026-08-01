#!/usr/bin/env python3
"""Sensitivity tables and fare-dependence plots for the passenger ship model.

The script accepts the same lifecycle-fare inputs as
``passenger_ticket_optimizer.py fare``. It performs:

1. A local -10%/+10% sensitivity analysis for physical, economic, payload, and
   operating parameters where the perturbation is meaningful and valid.
2. Fare sweeps for distance, passenger count, and payload mass per passenger.

Outputs are written as CSV, JSON, and PNG files. Numerical optimizer controls
and budget search bounds are intentionally excluded from economic sensitivity:
they are solution settings, not modeled physical parameters.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from passenger_ticket_optimizer import (
    PassengerOptimizationResult,
    add_fare_arguments,
    optimize_fare_from_args,
)


@dataclass(frozen=True)
class SensitivitySpec:
    attribute: str
    label: str
    minimum: Optional[float] = 0.0
    maximum: Optional[float] = None
    integer: bool = False


SENSITIVITY_SPECS: Tuple[SensitivitySpec, ...] = (
    SensitivitySpec("passengers", "Passengers", 1.0, None, True),
    SensitivitySpec("payload_constant_kg", "Payload constant", 0.0),
    SensitivitySpec("payload_per_passenger_kg", "Payload per passenger", 0.0),
    SensitivitySpec("payload_per_day_kg", "Payload per day", 0.0),
    SensitivitySpec("payload_per_passenger_day_kg", "Payload per passenger-day", 0.0),
    SensitivitySpec("distance_au", "Distance", 1.0e-9),
    SensitivitySpec("ve_km_s", "Maximum exhaust velocity", 1.0e-9),
    SensitivitySpec("ship_lifetime_years", "Ship lifetime", 1.0e-9),
    SensitivitySpec("utilization_fraction", "Utilization fraction", 1.0e-9, 1.0),
    SensitivitySpec("cycle_time_multiplier", "Cycle-time multiplier", 1.0e-9),
    SensitivitySpec("turnaround_days", "Turnaround time", 0.0),
    SensitivitySpec("load_factor", "Load factor", 1.0e-9, 1.0),
    SensitivitySpec("operations_cost_per_trip_usd", "Operations cost per trip", 0.0),
    SensitivitySpec("operations_cost_per_day_usd", "Operations cost per day", 0.0),
    SensitivitySpec("ticket_markup_fraction", "Ticket markup", 0.0),
    SensitivitySpec("reusable_constant_fraction", "Reusable constant-payload fraction", 0.0, 1.0),
    SensitivitySpec("reusable_per_passenger_fraction", "Reusable per-passenger fraction", 0.0, 1.0),
    SensitivitySpec("reusable_per_day_fraction", "Reusable per-day fraction", 0.0, 1.0),
    SensitivitySpec("reusable_per_passenger_day_fraction", "Reusable passenger-day fraction", 0.0, 1.0),
    SensitivitySpec("alpha_eng_w_per_kg", "Engine specific power", 1.0e-9),
    SensitivitySpec("phi_heat_to_total", "Waste-heat fraction", 0.0, 1.0 - 1.0e-9),
    SensitivitySpec("rho_rad_w_per_kg", "Radiator specific rejection", 1.0e-9),
    SensitivitySpec("payload_cost_usd_per_kg", "Payload hardware cost", 0.0),
    SensitivitySpec("propellant_cost_usd_per_kg", "Propellant cost", 0.0),
    SensitivitySpec("engine_core_cost_usd_per_kg", "Engine core cost", 0.0),
    SensitivitySpec("radiator_cost_usd_per_kg", "Radiator cost", 0.0),
    SensitivitySpec("tank_mass_fraction", "Tank mass fraction", 0.0),
    SensitivitySpec("tank_cost_usd_per_kg", "Tank cost", 0.0),
)


def _namespace_copy(args: argparse.Namespace, **changes: object) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(changes)
    return argparse.Namespace(**values)


def _fare(result: PassengerOptimizationResult) -> float:
    return result.best.ticket_cost_usd_per_passenger


def _solve(
    args: argparse.Namespace,
    warm_start: Optional[PassengerOptimizationResult],
) -> PassengerOptimizationResult:
    """Use a nearby solution first; retry with full continuation if needed."""

    if warm_start is not None:
        try:
            return optimize_fare_from_args(
                args,
                warm_start=warm_start,
                continuation_seed_budgets=(),
            )
        except RuntimeError:
            pass
    return optimize_fare_from_args(args)


def _valid_perturbation(value: float, spec: SensitivitySpec) -> bool:
    if spec.minimum is not None and value < spec.minimum:
        return False
    if spec.maximum is not None and value > spec.maximum:
        return False
    return math.isfinite(value)


def local_sensitivity(
    args: argparse.Namespace,
    baseline: PassengerOptimizationResult,
    change_fraction: float,
    *,
    solve_fn: Callable[[argparse.Namespace, Optional[PassengerOptimizationResult]], PassengerOptimizationResult] = _solve,
) -> List[Dict[str, object]]:
    if not 0.0 < change_fraction < 1.0:
        raise ValueError("change_fraction must satisfy 0 < value < 1.")

    baseline_fare = _fare(baseline)
    rows: List[Dict[str, object]] = []
    for spec in SENSITIVITY_SPECS:
        # If lifetime is expressed in days, sensitivity belongs to that active input.
        if spec.attribute == "ship_lifetime_years" and args.ship_lifetime_days is not None:
            active = SensitivitySpec("ship_lifetime_days", "Ship lifetime", 1.0e-9)
        else:
            active = spec
        value = getattr(args, active.attribute)
        if value is None or not math.isfinite(float(value)) or float(value) == 0.0:
            rows.append({
                "parameter": active.label,
                "attribute": active.attribute,
                "baseline_value": value,
                "minus_10_percent_fare_change_percent": None,
                "plus_10_percent_fare_change_percent": None,
                "status": "skipped: zero or inactive baseline",
            })
            continue

        output: Dict[str, object] = {
            "parameter": active.label,
            "attribute": active.attribute,
            "baseline_value": value,
            "minus_10_percent_fare_change_percent": None,
            "plus_10_percent_fare_change_percent": None,
            "status": "ok",
        }
        solved_directions = 0
        for direction, key in ((-1.0, "minus_10_percent_fare_change_percent"), (1.0, "plus_10_percent_fare_change_percent")):
            new_value = float(value) * (1.0 + direction * change_fraction)
            if active.integer:
                new_value = float(max(1, int(round(new_value))))
            if not _valid_perturbation(new_value, active) or new_value == float(value):
                continue
            perturbed = _namespace_copy(args, **{active.attribute: int(new_value) if active.integer else new_value})
            try:
                result = solve_fn(perturbed, baseline)
            except (RuntimeError, ValueError) as exc:
                output[key] = None
                output["status"] = f"partial: {direction:+.0%} solve failed: {exc}"
                continue
            output[key] = 100.0 * (_fare(result) / baseline_fare - 1.0)
            output[f"{key}_fare_usd"] = _fare(result)
            output[f"{key}_trip_days"] = result.best.transfer.total_time_days
            solved_directions += 1
        if solved_directions == 0 and output["status"] == "ok":
            output["status"] = "skipped: both perturbations violate parameter bounds"
        elif solved_directions == 1 and output["status"] == "ok":
            output["status"] = "one-sided: baseline lies on a parameter bound"
        rows.append(output)
    return rows


def _result_row(variable: str, value: float, result: PassengerOptimizationResult) -> Dict[str, object]:
    best = result.best
    return {
        variable: value,
        "ticket_cost_usd_per_passenger": best.ticket_cost_usd_per_passenger,
        "transfer_days": best.transfer.total_time_days,
        "budget_musd": best.total_budget_musd,
        "engine_fraction": best.engine_fraction,
        "payload_mass_kg": best.payload_mass_kg,
        "topology": best.topology.value,
        "max_constraint_residual": max(abs(x) for x in best.constraint_residuals),
    }


def run_sweep(
    args: argparse.Namespace,
    baseline: PassengerOptimizationResult,
    attribute: str,
    values: Iterable[float],
    *,
    integer: bool = False,
    solve_fn: Callable[[argparse.Namespace, Optional[PassengerOptimizationResult]], PassengerOptimizationResult] = _solve,
) -> List[Dict[str, object]]:
    base_value = float(getattr(args, attribute))
    unique_values = sorted({int(round(v)) if integer else float(v) for v in values})
    # Solve nearest-to-baseline points first for continuation, then sort output.
    solve_order = sorted(unique_values, key=lambda v: abs(float(v) - base_value))
    warm = baseline
    rows: List[Dict[str, object]] = []
    for value in solve_order:
        modified = _namespace_copy(args, **{attribute: int(value) if integer else value})
        result = solve_fn(modified, warm)
        warm = result
        rows.append(_result_row(attribute, value, result))
    return sorted(rows, key=lambda row: float(row[attribute]))


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: Sequence[Dict[str, object]], x_key: str, xlabel: str, path: Path) -> None:
    x = [float(row[x_key]) for row in rows]
    y = [float(row["ticket_cost_usd_per_passenger"]) / 1.0e6 for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(x, y, marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Lifecycle fare [million USD/passenger]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_fare_arguments(parser)
    parser.add_argument("--sensitivity-change-fraction", type=float, default=0.10)
    parser.add_argument("--sweep-points", type=int, default=15)
    parser.add_argument("--distance-min-au", type=float, default=1.0)
    parser.add_argument("--distance-max-au", type=float, default=10.0)
    parser.add_argument("--passengers-min", type=int, default=50)
    parser.add_argument("--passengers-max", type=int, default=1500)
    parser.add_argument("--payload-per-passenger-min-kg", type=float, default=None)
    parser.add_argument("--payload-per-passenger-max-kg", type=float, default=None)
    parser.add_argument("--output-dir", default="fare_analysis")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.sweep_points < 2:
        raise ValueError("--sweep-points must be at least 2.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Solving baseline fare...", file=sys.stderr)
    baseline = optimize_fare_from_args(args)
    sensitivity = local_sensitivity(args, baseline, args.sensitivity_change_fraction)

    payload_min = (
        args.payload_per_passenger_min_kg
        if args.payload_per_passenger_min_kg is not None
        else 0.5 * args.payload_per_passenger_kg
    )
    payload_max = (
        args.payload_per_passenger_max_kg
        if args.payload_per_passenger_max_kg is not None
        else 1.5 * args.payload_per_passenger_kg
    )

    print("Sweeping distance...", file=sys.stderr)
    distance_rows = run_sweep(
        args,
        baseline,
        "distance_au",
        np.linspace(args.distance_min_au, args.distance_max_au, args.sweep_points),
    )
    print("Sweeping passenger count...", file=sys.stderr)
    passenger_rows = run_sweep(
        args,
        baseline,
        "passengers",
        np.linspace(args.passengers_min, args.passengers_max, args.sweep_points),
        integer=True,
    )
    print("Sweeping payload per passenger...", file=sys.stderr)
    payload_rows = run_sweep(
        args,
        baseline,
        "payload_per_passenger_kg",
        np.linspace(payload_min, payload_max, args.sweep_points),
    )

    _write_csv(output_dir / "local_sensitivity.csv", sensitivity)
    _write_csv(output_dir / "distance_sweep.csv", distance_rows)
    _write_csv(output_dir / "passenger_sweep.csv", passenger_rows)
    _write_csv(output_dir / "payload_per_passenger_sweep.csv", payload_rows)
    _plot(distance_rows, "distance_au", "Distance [AU]", output_dir / "fare_vs_distance.png")
    _plot(passenger_rows, "passengers", "Passenger capacity", output_dir / "fare_vs_passengers.png")
    _plot(
        payload_rows,
        "payload_per_passenger_kg",
        "Payload per passenger [kg]",
        output_dir / "fare_vs_payload_per_passenger.png",
    )

    summary = {
        "baseline": baseline.as_dict(),
        "sensitivity_change_fraction": args.sensitivity_change_fraction,
        "local_sensitivity": sensitivity,
        "files": {
            "local_sensitivity_csv": "local_sensitivity.csv",
            "distance_sweep_csv": "distance_sweep.csv",
            "passenger_sweep_csv": "passenger_sweep.csv",
            "payload_per_passenger_sweep_csv": "payload_per_passenger_sweep.csv",
            "fare_vs_distance_png": "fare_vs_distance.png",
            "fare_vs_passengers_png": "fare_vs_passengers.png",
            "fare_vs_payload_per_passenger_png": "fare_vs_payload_per_passenger.png",
        },
        "excluded_from_sensitivity": [
            "budget-min/max (search bounds)",
            "xmin/xmax (optimizer bounds)",
            "time-min/max (optimizer bounds)",
            "engine-seed-points, maxiter, max-starts (numerical controls)",
        ],
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "baseline_fare_usd_per_passenger": baseline.best.ticket_cost_usd_per_passenger,
        "baseline_transfer_days": baseline.best.transfer.total_time_days,
        "output_dir": str(output_dir.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
