# Python time-optimal transfer solver

`time_optimal_transfer_solver.py` implements the gravity-free, constant-useful-power transfer model with a finite exhaust-speed ceiling.

## Ordinary candidate topologies

The solver enumerates four arc families:

1. `U+B+C+B-U` — variable acceleration, maximum-exhaust-speed acceleration, coast, maximum-exhaust-speed braking, variable deceleration.
2. `U+B+C+B-` — the same positive side, followed by coast and one uninterrupted maximum-exhaust-speed braking burn to final mass.
3. `B+C+B-U` — starts directly on the maximum-exhaust-speed acceleration boundary.
4. `B+C+B-` — both endpoints truncate the variable-power arcs.

The endpoint-boundary families are treated as direct feasible coast-then-full-brake candidates. The code does **not** use the previously introduced relaxed/chattering braking arc.

Propagation within each arc is analytic. SciPy's nonlinear least-squares solver adjusts three shooting parameters until position, velocity, and final-mass residuals converge. The iteration count is not fixed as it was in Excel; tolerances and a safety evaluation ceiling are configurable.

## Dependencies

```bash
python -m pip install numpy scipy
```

Pytest is only needed to run the separate test file.

## Run known-case regressions

```bash
python time_optimal_transfer_solver.py regressions
```

Run only the fixed-engine cases:

```bash
python time_optimal_transfer_solver.py regressions --fixed-only
```

## Solve a fixed engine/mass problem

```bash
python time_optimal_transfer_solver.py direct \
  --power-w 128881929.09352272 \
  --mi-kg 1887314.2201357787 \
  --mf-kg 596748.7555049236 \
  --distance-au 50 \
  --ve-km-s 250
```

The command solves all four topologies and prints the fastest converged candidate plus the other feasible candidates.

## Optimize the spreadsheet budget model

```bash
python time_optimal_transfer_solver.py budget \
  --budget-musd 3000 \
  --distance-au 50 \
  --ve-km-s 50
```

The budget command now accepts the primitive Excel engineering and cost inputs directly. `beta`, the weighted engine/radiator hardware cost, and the effective fuel cost are derived from them.

```bash
python time_optimal_transfer_solver.py economics
```

prints the default derivation. Important arguments include `--alpha-eng-w-per-kg`, `--phi-heat-to-total`, `--rho-rad-w-per-kg`, all four cost-per-kg inputs, `--tank-mass-fraction`, and `--tank-cost-usd-per-kg`. See `README_passenger_ticket_optimizer.md` for the formulas and full mapping.

## Python API

```python
from time_optimal_transfer_solver import (
    AU_M,
    BudgetModel,
    SpreadsheetEngineeringEconomics,
    TransferProblem,
    solve_all_topologies,
    optimize_budget_all_topologies,
)

problem = TransferProblem(
    power_w=128_881_929.09352272,
    initial_mass_kg=1_887_314.2201357787,
    final_mass_kg=596_748.7555049236,
    distance_m=50 * AU_M,
    ve_max_m_s=250_000,
)

best, candidates = solve_all_topologies(problem)
print(best.topology, best.total_time_days)

engineering = SpreadsheetEngineeringEconomics()

optimized, topology_results = optimize_budget_all_topologies(
    engineering.budget_model(payload_mass_kg=500_000),
    total_budget_musd=3000,
    distance_m=50 * AU_M,
    ve_max_m_s=50_000,
    bounds=(0.55, 0.80),
)
```

## Verification summary

The included regression suite reproduces:

- 5 AU, 250 km/s, known five-arc case: `328.0127194439 d`.
- 50 AU, 250 km/s, known five-arc case: `1533.0396657561 d`.
- 50 AU, high-budget 250 km/s terminal-boundary case: `821.0756822867 d`.
- Budget-optimized low-ceiling ordinary branch at 50 AU, 3000 MUSD, 50 km/s: `x = 0.6809776370`, `T = 1633.4977527700 d`, with a long coast followed by continuous maximum-exhaust-speed braking.

The solver enumerates candidate families and selects the fastest converged one. This is not a mathematical proof of global optimality over every possible abnormal or singular control family.
