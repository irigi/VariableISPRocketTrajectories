import math

from passenger_ticket_optimizer import (
    DEFAULT_FARE_VALUES,
    LinearPassengerPayload,
    _build_parser,
    optimize_passenger_ship,
)
from time_optimal_transfer_solver import AU_M, BudgetModel, Topology


def test_linear_payload_formula():
    model = LinearPassengerPayload(100.0, 20.0, 3.0, 0.5)
    assert model.mass_kg(10, 4.0) == 100.0 + 20.0 * 10 + 3.0 * 4.0 + 0.5 * 10 * 4.0


def test_zero_time_terms_reproduce_known_5_au_case():
    payload = LinearPassengerPayload(
        constant_kg=350_000.0,
        per_passenger_kg=1_500.0,
        per_day_kg=0.0,
        per_passenger_day_kg=0.0,
    )
    result = optimize_passenger_ship(
        payload,
        passengers=100,
        budget_model=BudgetModel(),
        total_budget_musd=1000.0,
        distance_m=5.0 * AU_M,
        ve_max_m_s=250_000.0,
        topologies=(Topology.FIVE_ARC,),
        engine_fraction_bounds=(0.79, 0.86),
        time_bounds_days=(250.0, 450.0),
        engine_seed_points=3,
        max_starts=30,
    )
    assert abs(result.best.transfer.total_time_days - 328.0127194439226) < 5.0e-5
    assert abs(result.best.engine_fraction - 0.8282524326) < 2.0e-5
    assert max(abs(value) for value in result.best.constraint_residuals) < 1.0e-8


def test_nonlinear_payload_closes_time_and_mass():
    payload = LinearPassengerPayload(220_000.0, 1_600.0, 40.0, 1.2)
    result = optimize_passenger_ship(
        payload,
        passengers=120,
        budget_model=BudgetModel(),
        total_budget_musd=1300.0,
        distance_m=5.0 * AU_M,
        ve_max_m_s=250_000.0,
        topologies=(Topology.FIVE_ARC, Topology.TERMINAL_BOUNDARY),
        engine_fraction_bounds=(0.72, 0.92),
        time_bounds_days=(150.0, 500.0),
        engine_seed_points=3,
        max_starts=24,
    )
    expected_payload = payload.mass_kg(120, result.best.transfer.total_time_days)
    assert abs(expected_payload - result.best.payload_mass_kg) < 0.01
    assert abs(result.best.fixed_point_residual_days) < 1.0e-4
    assert max(abs(value) for value in result.best.constraint_residuals) < 1.0e-8


def test_public_cli_only_exposes_fare_and_uses_requested_defaults():
    args = _build_parser().parse_args(["fare"])
    for key, expected in DEFAULT_FARE_VALUES.items():
        assert getattr(args, key) == expected


def test_ticket_subcommand_is_removed():
    import pytest

    with pytest.raises(SystemExit):
        _build_parser().parse_args(["ticket"])
