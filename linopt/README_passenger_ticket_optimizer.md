# Passenger ship and lifecycle ticket optimizer

`passenger_ticket_optimizer.py` is a companion to `time_optimal_transfer_solver.py`.
The original trajectory API remains available. The passenger script adds a nonlinear
payload model and two ticket-cost modes.

## 1. Payload model

For `N` passengers and transfer time `T` in days:

```text
payload_kg = constant_kg
           + per_passenger_kg * N
           + per_day_kg * T
           + per_passenger_day_kg * N * T
```

Set the coefficients with:

```text
--payload-constant-kg
--payload-per-passenger-kg
--payload-per-day-kg
--payload-per-passenger-day-kg
```

The solver enforces the fixed point between payload and trip time as a nonlinear
constraint; it does not use a fixed number of Excel-style Newton columns.

## 2. Excel engineering inputs

The following command-line arguments map directly to the workbook inputs:

| Excel quantity | Python argument | Units |
|---|---|---|
| `alpha_eng` | `--alpha-eng-w-per-kg` | W/kg |
| `phi = P_heat/P_total` | `--phi-heat-to-total` | fraction |
| `rho_rad` | `--rho-rad-w-per-kg` | W/kg |
| Payload Cost | `--payload-cost-usd-per-kg` | USD/kg |
| Propellant Cost | `--propellant-cost-usd-per-kg` | USD/kg |
| Engine Core Cost | `--engine-core-cost-usd-per-kg` | USD/kg |
| Radiator Cost | `--radiator-cost-usd-per-kg` | USD/kg |
| Tank Mass Fraction | `--tank-mass-fraction` | kg tank/kg propellant |
| Tank Cost | `--tank-cost-usd-per-kg` | USD/kg tank |

`beta` is derived, as in the Excel workbook:

```text
beta = 1 / ((1 - phi) * alpha_eng)
     + phi / ((1 - phi) * rho_rad)
```

The engine/radiator hardware cost per kg is the mass-weighted average:

```text
engine_specific_mass   = 1 / ((1 - phi) * alpha_eng)
radiator_specific_mass = phi / ((1 - phi) * rho_rad)

engine_radiator_cost_per_kg =
    (engine_core_cost * engine_specific_mass
   + radiator_cost * radiator_specific_mass) / beta
```

The effective cost of one kilogram of propellant allocation is:

```text
effective_fuel_cost = propellant_cost
                    + tank_mass_fraction * tank_cost
```

With the spreadsheet defaults, the Python model reproduces:

```text
beta                              = 0.00025 kg/W
engine_radiator_cost_per_kg       = 6357.142857142858 USD/kg
effective_fuel_cost_per_kg        = 35 USD/kg
```

Print the derived values without running a trajectory:

```bash
python time_optimal_transfer_solver.py economics \
  --alpha-eng-w-per-kg 10000 \
  --phi-heat-to-total 0.3 \
  --rho-rad-w-per-kg 4000 \
  --payload-cost-usd-per-kg 1500 \
  --propellant-cost-usd-per-kg 20 \
  --engine-core-cost-usd-per-kg 10000 \
  --radiator-cost-usd-per-kg 1500 \
  --tank-mass-fraction 0.05 \
  --tank-cost-usd-per-kg 300
```

## 3. Ship optimization

This mode minimizes self-consistent transfer time at a fixed total budget:

```bash
python passenger_ticket_optimizer.py ship \
  --budget-musd 1300 \
  --passengers 120 \
  --payload-constant-kg 220000 \
  --payload-per-passenger-kg 1600 \
  --payload-per-day-kg 40 \
  --payload-per-passenger-day-kg 1.2 \
  --distance-au 5 \
  --ve-km-s 250 \
  --alpha-eng-w-per-kg 10000 \
  --phi-heat-to-total 0.3 \
  --rho-rad-w-per-kg 4000
```

All unspecified engineering and cost arguments use the Excel defaults shown above.

## 4. Legacy `ticket` mode

The legacy mode is retained for compatibility:

```text
ticket_cost = total_project_budget / passengers
```

It finds the minimum feasible one-time project budget. It does **not** amortize the
ship over multiple trips.

## 5. Lifecycle `fare` mode

### Simple accounting

`--fare-accounting simple` implements the literal formula:

```text
lifetime_trips = ship_lifetime_days * utilization_fraction
               / (cycle_time_multiplier * trip_duration_days + turnaround_days)

paying_passengers = passengers * load_factor

ticket_cost = total_project_budget / lifetime_trips / paying_passengers
            * (1 + ticket_markup_fraction)
```

With `utilization_fraction=1`, `cycle_time_multiplier=1`, `turnaround_days=0`,
`load_factor=1`, and zero markup, this is exactly:

```text
SHIP_PRICE / (LIFETIME_OF_THE_SHIP / TRIP_DURATION) / NUMBER_OF_PASSENGERS
```

Example:

```bash
python passenger_ticket_optimizer.py fare \
  --fare-accounting simple \
  --ship-lifetime-years 20 \
  --budget-min-musd 150 \
  --budget-max-musd 1000 \
  --passengers 50 \
  --payload-constant-kg 100000 \
  --payload-per-passenger-kg 1000 \
  --payload-per-day-kg 20 \
  --payload-per-passenger-day-kg 0.2 \
  --distance-au 1 \
  --ve-km-s 250
```

### Lifecycle accounting

`--fare-accounting lifecycle` is the more physical default. It separates the
one-time Excel budget into reusable ship capital and recurring trip cost:

```text
ship_capital = reusable_payload_cost
             + engine_and_radiator_cost
             + tank_cost

recurring_trip_cost = recurring_payload_cost
                    + propellant_cost

fare = (ship_capital / lifetime_trips
      + recurring_trip_cost
      + operations_cost_per_trip)
      / paying_passengers
      * (1 + markup)
```

By default:

- constant payload mass is reusable;
- per-passenger payload mass is reusable;
- per-day payload mass is recurring;
- per-passenger-day payload mass is recurring.

Override those assumptions with:

```text
--reusable-constant-fraction
--reusable-per-passenger-fraction
--reusable-per-day-fraction
--reusable-per-passenger-day-fraction
```

Each is between 0 and 1. For example, `0.25` means 25% of that payload term is
reusable capital and 75% is repurchased each trip.

Additional lifecycle parameters:

```text
--ship-lifetime-days or --ship-lifetime-years
--utilization-fraction
--cycle-time-multiplier
--turnaround-days
--load-factor
--operations-cost-per-trip-usd
--operations-cost-per-day-usd
--ticket-markup-fraction
```

Set `--cycle-time-multiplier 2` if one revenue cycle requires an outbound and a
similar-duration return transfer. Add servicing time with `--turnaround-days`.

## 6. Python API

```python
from passenger_ticket_optimizer import (
    LifecycleTicketEconomics,
    LinearPassengerPayload,
    optimize_lifecycle_ticket_cost,
)
from time_optimal_transfer_solver import AU_M, SpreadsheetEngineeringEconomics

payload = LinearPassengerPayload(
    constant_kg=100_000,
    per_passenger_kg=1_000,
    per_day_kg=20,
    per_passenger_day_kg=0.2,
)

engineering = SpreadsheetEngineeringEconomics(
    alpha_eng_w_per_kg=10_000,
    phi_heat_to_total=0.30,
    rho_rad_w_per_kg=4_000,
    payload_cost_usd_per_kg=1_500,
    propellant_cost_usd_per_kg=20,
    engine_core_cost_usd_per_kg=10_000,
    radiator_cost_usd_per_kg=1_500,
    tank_mass_fraction=0.05,
    tank_cost_usd_per_kg=300,
)

lifecycle = LifecycleTicketEconomics(
    ship_lifetime_days=20 * 365.25,
    accounting_mode="lifecycle",  # or "simple"
    utilization_fraction=0.8,
    cycle_time_multiplier=2.0,
    turnaround_days=10,
    load_factor=0.9,
)

result = optimize_lifecycle_ticket_cost(
    payload,
    passengers=50,
    engineering=engineering,
    lifecycle=lifecycle,
    distance_m=1 * AU_M,
    ve_max_m_s=250_000,
    budget_bounds_musd=(150, 1000),
)

print(result.best.ticket_cost_usd_per_passenger)
print(result.best.ticket_cost_breakdown)
```

## 7. Verification

```bash
pytest -q test_time_optimal_transfer_solver.py \
          test_passenger_ticket_optimizer.py \
          test_economics_and_fare.py
```

The current suite has 12 tests covering the original transfer regressions,
nonlinear payload closure, Excel aggregate formulas, literal fare amortization,
lifecycle cost closure, and a lifecycle fare optimization.
