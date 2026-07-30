# Passenger payload and ticket-cost optimizer

`passenger_ticket_optimizer.py` is a companion to `time_optimal_transfer_solver.py`. The original transfer solver is unchanged.

## Payload model

For `N` passengers and a journey lasting `T` days:

```text
payload_kg = constant_kg
           + per_passenger_kg * N
           + per_day_kg * T
           + per_passenger_day_kg * N * T
```

All four coefficients are nonnegative and use kilograms, days, and passengers as shown.

The time-dependent terms make the design nonlinear: payload changes the dry and initial masses, which changes transfer time, which changes payload again. The new solver handles this with a direct constrained nonlinear program. Its variables include the three transfer shooting parameters, engine/fuel budget fraction, self-consistent transfer time, and—only in ticket mode—total project budget.

Four equality constraints enforce:

1. final distance;
2. zero final velocity;
3. prescribed final mass;
4. equality between assumed journey time and the transfer time calculated by the trajectory solver.

The ordinary topologies are inherited from the base solver and are all attempted by default.

## Meaning of ticket cost

The reported ticket cost is:

```text
total project budget / number of passengers
```

The total project budget already includes payload, engine/radiator hardware, tank mass, and propellant under the base `BudgetModel`. Financing, profit, operations, insurance, launch, return-trip costs, and occupancy risk are not included.

## Dependencies

```bash
python -m pip install numpy scipy
```

Keep these files in the same directory:

```text
time_optimal_transfer_solver.py
passenger_ticket_optimizer.py
```

## Fixed-budget passenger ship

This mode minimizes self-consistent journey time for a specified total budget:

```bash
python passenger_ticket_optimizer.py ship \
  --budget-musd 1300 \
  --passengers 120 \
  --payload-constant-kg 220000 \
  --payload-per-passenger-kg 1600 \
  --payload-per-day-kg 40 \
  --payload-per-passenger-day-kg 1.2 \
  --distance-au 5 \
  --ve-km-s 250
```

## Minimum ticket cost

This mode minimizes project budget per passenger and then re-optimizes the ship for shortest self-consistent journey time at that budget:

```bash
python passenger_ticket_optimizer.py ticket \
  --budget-min-musd 150 \
  --budget-max-musd 800 \
  --passengers 50 \
  --payload-constant-kg 100000 \
  --payload-per-passenger-kg 1000 \
  --payload-per-day-kg 20 \
  --payload-per-passenger-day-kg 0.2 \
  --distance-au 1 \
  --ve-km-s 250
```

The budget bounds are search bounds. If the optimum lands on either bound, widen the range before interpreting it as an unconstrained optimum.

## Python API

```python
from passenger_ticket_optimizer import (
    LinearPassengerPayload,
    optimize_passenger_ship,
    optimize_ticket_cost,
)
from time_optimal_transfer_solver import AU_M, BudgetModel

payload = LinearPassengerPayload(
    constant_kg=100_000,
    per_passenger_kg=1_000,
    per_day_kg=20,
    per_passenger_day_kg=0.2,
)

result = optimize_ticket_cost(
    payload,
    passengers=50,
    budget_model=BudgetModel(),
    distance_m=1 * AU_M,
    ve_max_m_s=250_000,
    budget_bounds_musd=(150, 800),
)

print(result.best.ticket_cost_usd_per_passenger)
print(result.best.transfer.total_time_days)
```

## Verification

Run the new checks:

```bash
pytest -q test_passenger_ticket_optimizer.py
```

Run the unchanged base-solver checks:

```bash
pytest -q test_time_optimal_transfer_solver.py
```

The new regression suite verifies that zero time-dependent payload terms reproduce the known 5 AU result, a nonlinear payload closes both the mass and time equations, and a synthetic ticket case produces an interior minimum budget.
