import math

from fare_sensitivity_analysis import SENSITIVITY_SPECS
from passenger_ticket_optimizer import (
    LifecycleTicketEconomics,
    LinearPassengerPayload,
    LinearPassengerPayloadCosts,
    _build_parser,
    calculate_lifecycle_ticket_cost,
    payload_costs_from_args,
)
from time_optimal_transfer_solver import SpreadsheetEngineeringEconomics


def test_new_payload_price_defaults():
    args = _build_parser().parse_args(["fare"])
    costs = payload_costs_from_args(args)
    assert costs == LinearPassengerPayloadCosts(1500.0, 1000.0, 100.0, 15.0)


def test_component_prices_are_applied_separately():
    payload = LinearPassengerPayload(300_000.0, 2_500.0, 100.0, 2.0)
    costs = LinearPassengerPayloadCosts(1500.0, 1000.0, 100.0, 15.0)
    component_costs = costs.component_costs_usd(payload, 600, 73.0)
    assert component_costs == {
        "constant": 450_000_000.0,
        "per_passenger": 1_500_000_000.0,
        "per_day": 730_000.0,
        "per_passenger_day": 1_314_000.0,
    }


def test_lifecycle_accounting_closes_with_component_prices():
    payload = LinearPassengerPayload(300_000.0, 2_500.0, 100.0, 2.0)
    costs = LinearPassengerPayloadCosts(1500.0, 1000.0, 100.0, 15.0)
    result = calculate_lifecycle_ticket_cost(
        payload=payload,
        passengers=600,
        transfer_days=73.0,
        total_budget_musd=5000.0,
        engine_fraction=0.8,
        engineering=SpreadsheetEngineeringEconomics(
            alpha_eng_w_per_kg=20_000.0,
            phi_heat_to_total=0.15,
            rho_rad_w_per_kg=10_000.0,
        ),
        lifecycle=LifecycleTicketEconomics(ship_lifetime_days=25.0 * 365.25),
        payload_costs=costs,
    )
    assert math.isclose(
        result.physical_ship_capital_cost_usd
        + result.physical_recurring_cost_per_trip_usd,
        5.0e9,
        rel_tol=0.0,
        abs_tol=1.0e-4,
    )
    assert result.reusable_payload_cost_usd == 1_950_000_000.0
    assert result.recurring_payload_cost_usd == 2_044_000.0


def test_sensitivity_contains_all_component_prices():
    names = {spec.attribute for spec in SENSITIVITY_SPECS}
    assert {
        "payload_constant_cost_usd_per_kg",
        "payload_per_passenger_cost_usd_per_kg",
        "payload_per_day_cost_usd_per_kg",
        "payload_per_passenger_day_cost_usd_per_kg",
    } <= names
