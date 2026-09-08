import math

from fare_sensitivity_analysis import (
    format_ticket_price_breakdown,
    ticket_price_breakdown_rows,
)
from passenger_ticket_optimizer import (
    LifecycleTicketEconomics,
    LinearPassengerPayload,
    LinearPassengerPayloadCosts,
    PassengerOptimizationResult,
    PassengerShipResult,
    calculate_lifecycle_ticket_cost,
)
from time_optimal_transfer_solver import (
    SpreadsheetEngineeringEconomics,
    Topology,
    TransferProblem,
    TransferSolution,
)


def _breakdown():
    return calculate_lifecycle_ticket_cost(
        payload=LinearPassengerPayload(300_000.0, 2_500.0, 100.0, 2.0),
        passengers=600,
        transfer_days=73.0,
        total_budget_musd=5_000.0,
        engine_fraction=0.8,
        engineering=SpreadsheetEngineeringEconomics(
            alpha_eng_w_per_kg=20_000.0,
            phi_heat_to_total=0.15,
            rho_rad_w_per_kg=10_000.0,
            propellant_cost_usd_per_kg=20.0,
            engine_core_cost_usd_per_kg=10_000.0,
            radiator_cost_usd_per_kg=1_500.0,
            tank_mass_fraction=0.05,
            tank_cost_usd_per_kg=300.0,
        ),
        lifecycle=LifecycleTicketEconomics(ship_lifetime_days=25.0 * 365.25),
        payload_costs=LinearPassengerPayloadCosts(1500.0, 1000.0, 100.0, 15.0),
    )


def test_engine_and_radiator_split_closes():
    breakdown = _breakdown()
    assert math.isclose(
        breakdown.engine_capital_cost_usd + breakdown.radiator_capital_cost_usd,
        breakdown.engine_radiator_capital_cost_usd,
        rel_tol=0.0,
        abs_tol=1.0e-5,
    )


def test_printed_rows_close_to_final_fare():
    breakdown = _breakdown()
    rows = ticket_price_breakdown_rows(breakdown)
    total = sum(float(row["fare_contribution_usd_per_passenger"]) for row in rows)
    assert math.isclose(
        total,
        breakdown.ticket_cost_usd_per_passenger,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    )
    labels = {row["item"] for row in rows}
    assert {
        "Constant payload amortization",
        "Passenger-dependent payload amortization",
        "Time-dependent shared payload recurring cost",
        "Time- and passenger-dependent payload recurring cost",
        "Engine core amortization",
        "Radiator amortization",
        "Propellant / reaction-mass cost",
    } <= labels


def test_terminal_formatter_contains_requested_categories():
    breakdown = _breakdown()
    transfer = TransferSolution(
        topology=Topology.FIVE_ARC,
        total_time_days=73.0,
        phase_times_days={},
        maximum_acceleration_m_s2=0.0,
        average_absolute_acceleration_m_s2=0.0,
        position_residual_fraction=0.0,
        velocity_residual_over_ve=0.0,
        mass_residual_log=0.0,
        shooting_y0=1.0,
        shooting_minus_yf=1.0,
        shooting_kappa=1.0,
        arcs=[],
    )
    ship = PassengerShipResult(
        passengers=600,
        total_budget_musd=5000.0,
        ticket_cost_usd_per_passenger=breakdown.ticket_cost_usd_per_passenger,
        engine_fraction=0.8,
        payload_mass_kg=0.0,
        assumed_transfer_days=73.0,
        fixed_point_residual_days=0.0,
        problem=TransferProblem(1.0, 2.0, 1.0, 1.0, 1.0),
        transfer=transfer,
        constraint_residuals=[0.0] * 4,
        optimizer_success=True,
        optimizer_message="test",
        ticket_cost_mode="lifecycle_amortized",
        ticket_cost_breakdown=breakdown,
    )
    text = format_ticket_price_breakdown(
        PassengerOptimizationResult(ship, {Topology.FIVE_ARC: ship}, "fare")
    )
    for phrase in (
        "Constant payload amortization",
        "Time-dependent shared payload recurring cost",
        "Engine core amortization",
        "Radiator amortization",
        "Propellant / reaction-mass cost",
        "FINAL TICKET PRICE",
    ):
        assert phrase in text
