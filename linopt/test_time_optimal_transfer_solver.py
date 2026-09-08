import math

from time_optimal_transfer_solver import (
    AU_M,
    BudgetModel,
    Topology,
    TransferProblem,
    optimize_budget_for_topology,
    solve_topology,
)


def test_known_5_au_five_arc():
    problem = TransferProblem(
        130_286_899.51386033,
        1_820_678.480199766,
        593_910.141798527,
        5.0 * AU_M,
        250_000.0,
    )
    solution = solve_topology(problem, Topology.FIVE_ARC, max_starts=120)
    assert solution is not None
    assert abs(solution.total_time_days - 328.0127194439226) < 2.0e-5
    assert solution.max_residual < 1.0e-8


def test_known_50_au_five_arc():
    problem = TransferProblem(
        128_881_929.09352272,
        1_887_314.2201357787,
        596_748.7555049236,
        50.0 * AU_M,
        250_000.0,
    )
    solution = solve_topology(problem, Topology.FIVE_ARC, max_starts=120)
    assert solution is not None
    assert abs(solution.total_time_days - 1533.0396657561396) < 2.0e-5
    assert solution.max_residual < 1.0e-8


def test_known_terminal_boundary():
    problem = TransferProblem(
        1_321_675_685.6740997,
        5_314_810.336599838,
        1_043_961.3697604922,
        50.0 * AU_M,
        250_000.0,
    )
    solution = solve_topology(problem, Topology.TERMINAL_BOUNDARY, max_starts=120)
    assert solution is not None
    assert abs(solution.total_time_days - 821.0756822866467) < 2.0e-5
    assert solution.max_residual < 1.0e-8


def test_low_ve_uses_coast_then_continuous_braking():
    result = optimize_budget_for_topology(
        BudgetModel(),
        3000.0,
        50.0 * AU_M,
        50_000.0,
        Topology.TERMINAL_BOUNDARY,
        bounds=(0.58, 0.76),
        xatol=1.0e-10,
    )
    assert result is not None
    assert abs(result.engine_fraction - 0.680977637) < 1.0e-5
    assert abs(result.transfer.total_time_days - 1633.49775277) < 3.0e-5
    phases = result.transfer.phase_times_days
    assert phases["max_ve_braking"] > 0.0
    assert phases["variable_deceleration"] == 0.0
    assert result.transfer.max_residual < 1.0e-8
