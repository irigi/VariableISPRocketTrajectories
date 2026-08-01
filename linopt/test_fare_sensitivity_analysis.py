import argparse
from types import SimpleNamespace

from fare_sensitivity_analysis import local_sensitivity, run_sweep
from passenger_ticket_optimizer import _build_parser


def _fake_result(args):
    fare = 1_000_000.0 * (
        args.distance_au
        * args.payload_per_passenger_kg
        / max(args.passengers, 1)
    )
    best = SimpleNamespace(
        ticket_cost_usd_per_passenger=fare,
        transfer=SimpleNamespace(total_time_days=100.0 * args.distance_au),
        total_budget_musd=500.0,
        engine_fraction=0.7,
        payload_mass_kg=1_000_000.0,
        topology=SimpleNamespace(value="test"),
        constraint_residuals=[0.0],
    )
    return SimpleNamespace(best=best, topology_results={})


def _fake_solve(args, warm):
    return _fake_result(args)


def test_sensitivity_reports_expected_distance_elasticity():
    args = _build_parser().parse_args(["fare"])
    baseline = _fake_result(args)
    rows = local_sensitivity(args, baseline, 0.10, solve_fn=_fake_solve)
    distance = next(row for row in rows if row["attribute"] == "distance_au")
    assert abs(distance["plus_10_percent_fare_change_percent"] - 10.0) < 1e-10
    assert abs(distance["minus_10_percent_fare_change_percent"] + 10.0) < 1e-10


def test_sweep_sorts_points_and_changes_passengers_as_integer():
    args = _build_parser().parse_args(["fare"])
    baseline = _fake_result(args)
    rows = run_sweep(
        args,
        baseline,
        "passengers",
        [1500, 50, 600],
        integer=True,
        solve_fn=_fake_solve,
    )
    assert [row["passengers"] for row in rows] == [50, 600, 1500]
