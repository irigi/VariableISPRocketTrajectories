# Fusion passenger-ship lifecycle fare optimizer

`passenger_ticket_optimizer.py` now has one public optimization mode: `fare`.
Lifecycle accounting is always used. The removed legacy `ticket` command and
`--fare-accounting simple` option are no longer accepted.

## Quick start

All requested scenario values are defaults, so this is sufficient:

```bash
python passenger_ticket_optimizer.py fare
```

The default model is equivalent to:

```bash
python passenger_ticket_optimizer.py fare \
  --ship-lifetime-years 25 \
  --utilization-fraction 1 \
  --cycle-time-multiplier 1 \
  --turnaround-days 0 \
  --load-factor 1 \
  --ticket-markup-fraction 0 \
  --budget-min-musd 150 \
  --budget-max-musd 8000 \
  --passengers 600 \
  --payload-constant-kg 300000 \
  --payload-per-passenger-kg 2500 \
  --payload-per-day-kg 100 \
  --payload-per-passenger-day-kg 2 \
  --distance-au 1.5 \
  --ve-km-s 250 \
  --alpha-eng-w-per-kg 20000 \
  --phi-heat-to-total 0.15 \
  --rho-rad-w-per-kg 10000 \
  --payload-cost-usd-per-kg 1500 \
  --propellant-cost-usd-per-kg 20 \
  --engine-core-cost-usd-per-kg 10000 \
  --radiator-cost-usd-per-kg 1500 \
  --tank-mass-fraction 0.05 \
  --tank-cost-usd-per-kg 300
```

## Payload/time coupling

For passenger count `N` and transfer time `T` in days:

```text
payload_kg = payload_constant_kg
           + payload_per_passenger_kg * N
           + payload_per_day_kg * T
           + payload_per_passenger_day_kg * N * T
```

The optimizer solves the transfer and this payload/time fixed point together.

## Lifecycle fare

```text
cycle_days = cycle_time_multiplier * transfer_days + turnaround_days
lifetime_trips = ship_lifetime_days * utilization_fraction / cycle_days
paying_passengers = passengers * load_factor

fare = (reusable_capital / lifetime_trips
      + recurring_payload
      + propellant
      + operations_per_trip
      + operations_per_day * cycle_days)
      / paying_passengers
      * (1 + ticket_markup_fraction)
```

By default, constant and per-passenger payload are reusable; per-day and
passenger-day payload are recurring. These classifications can be changed with:

```text
--reusable-constant-fraction
--reusable-per-passenger-fraction
--reusable-per-day-fraction
--reusable-per-passenger-day-fraction
```

## Engineering reduction

The Excel primitive inputs are accepted directly. The aggregate engine/radiator
specific mass is derived as:

```text
beta = 1 / ((1 - phi) * alpha_eng)
     + phi / ((1 - phi) * rho_rad)
```

The effective propellant allocation cost is:

```text
propellant_cost + tank_mass_fraction * tank_cost
```

## Sensitivity and plots

`fare_sensitivity_analysis.py` accepts the same fare inputs. With defaults:

```bash
python fare_sensitivity_analysis.py
```

It calculates both -10% and +10% fare changes where valid, and writes:

```text
fare_analysis/local_sensitivity.csv
fare_analysis/distance_sweep.csv
fare_analysis/passenger_sweep.csv
fare_analysis/payload_per_passenger_sweep.csv
fare_analysis/fare_vs_distance.png
fare_analysis/fare_vs_passengers.png
fare_analysis/fare_vs_payload_per_passenger.png
fare_analysis/analysis_summary.json
```

Default graph ranges:

- distance: 1–10 AU;
- passengers: 50–1500;
- payload per passenger: 50%–150% of the input baseline.

Change plot resolution and output location with:

```bash
python fare_sensitivity_analysis.py \
  --sweep-points 21 \
  --output-dir my_analysis
```

Budget bounds and numerical optimizer controls are not included in the 10%
sensitivity table because they are search settings rather than physical or
economic model parameters.

## Tests

```bash
pytest -q \
  test_time_optimal_transfer_solver.py \
  test_passenger_ticket_optimizer.py \
  test_economics_and_fare.py \
  test_fare_sensitivity_analysis.py
```
