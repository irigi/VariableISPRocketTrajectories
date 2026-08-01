import math

from passenger_ticket_optimizer import (
    LifecycleTicketEconomics,
    LinearPassengerPayload,
    calculate_lifecycle_ticket_cost,
    optimize_lifecycle_ticket_cost,
)
from time_optimal_transfer_solver import (
    AU_M,
    SpreadsheetEngineeringEconomics,
    Topology,
)


def test_excel_primitive_inputs_reproduce_aggregate_cells():
    model = SpreadsheetEngineeringEconomics()
    assert math.isclose(model.beta_kg_per_jet_w, 0.00025, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(
        model.engine_radiator_cost_usd_per_kg,
        6357.142857142858,
        rel_tol=0.0,
        abs_tol=1e-9,
    )
    assert math.isclose(
        model.effective_fuel_cost_usd_per_kg,
        35.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_lifecycle_accounting_closes_to_excel_total_budget():
    payload = LinearPassengerPayload(100_000.0, 1_000.0, 20.0, 0.2)
    result = calculate_lifecycle_ticket_cost(
        payload=payload,
        passengers=50,
        transfer_days=300.0,
        total_budget_musd=400.0,
        engine_fraction=0.7,
        engineering=SpreadsheetEngineeringEconomics(),
        lifecycle=LifecycleTicketEconomics(ship_lifetime_days=20.0 * 365.25),
    )
    assert math.isclose(
        result.physical_ship_capital_cost_usd
        + result.physical_recurring_cost_per_trip_usd,
        400.0e6,
        rel_tol=0.0,
        abs_tol=1e-5,
    )
    assert result.physical_recurring_cost_per_trip_usd > 0.0


def test_lifecycle_fare_optimizer_closes_transfer_constraints():
    payload = LinearPassengerPayload(100_000.0, 1_000.0, 20.0, 0.2)
    result = optimize_lifecycle_ticket_cost(
        payload=payload,
        passengers=50,
        engineering=SpreadsheetEngineeringEconomics(),
        lifecycle=LifecycleTicketEconomics(
            ship_lifetime_days=20.0 * 365.25,
            utilization_fraction=0.8,
            turnaround_days=10.0,
            load_factor=0.9,
        ),
        distance_m=1.0 * AU_M,
        ve_max_m_s=250_000.0,
        budget_bounds_musd=(150.0, 800.0),
        topologies=(Topology.FIVE_ARC,),
        engine_fraction_bounds=(0.55, 0.95),
        time_bounds_days=(20.0, 1_000.0),
        engine_seed_points=3,
        max_starts=15,
        maxiter=400,
    )
    assert result.mode == "fare"
    assert result.best.ticket_cost_breakdown is not None
    assert result.best.ticket_cost_usd_per_passenger > 0.0
    assert max(abs(value) for value in result.best.constraint_residuals) < 1e-8
