# Fusion passenger-ship lifecycle fare optimizer

`passenger_ticket_optimizer.py` exposes one public mode:

```bash
python passenger_ticket_optimizer.py fare
```

Lifecycle accounting is always used.

## Payload mass and price models

For passenger count `N` and transfer time `T` in days:

```text
payload_kg = payload_constant_kg
           + payload_per_passenger_kg * N
           + payload_per_day_kg * T
           + payload_per_passenger_day_kg * N * T
```

Each term now has its own price:

| Payload term | CLI price argument | Default | Interpretation |
|---|---|---:|---|
| Constant | `--payload-constant-cost-usd-per-kg` | 1500 USD/kg | Shared structure, controls, common life support |
| Per passenger | `--payload-per-passenger-cost-usd-per-kg` | 1000 USD/kg | Cabins, seats, personal life-support and safety hardware |
| Per day | `--payload-per-day-cost-usd-per-kg` | 100 USD/kg | Shared spares, medical stores and trip consumables |
| Per passenger-day | `--payload-per-passenger-day-cost-usd-per-kg` | 15 USD/kg | Food, water make-up, hygiene supplies and packaging |

The previous single `--payload-cost-usd-per-kg` fare argument has been removed.
The trajectory budget subtracts the actual sum of the four payload component
costs before purchasing engine/radiator hardware and propellant.

By default, constant and per-passenger mass are reusable ship capital, while
per-day and passenger-day mass are recurring trip consumables. These fractions
remain adjustable with:

```text
--reusable-constant-fraction
--reusable-per-passenger-fraction
--reusable-per-day-fraction
--reusable-per-passenger-day-fraction
```

## Default scenario

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
  --payload-constant-cost-usd-per-kg 1500 \
  --payload-per-passenger-cost-usd-per-kg 1000 \
  --payload-per-day-cost-usd-per-kg 100 \
  --payload-per-passenger-day-cost-usd-per-kg 15 \
  --distance-au 1.5 \
  --ve-km-s 250 \
  --alpha-eng-w-per-kg 20000 \
  --phi-heat-to-total 0.15 \
  --rho-rad-w-per-kg 10000 \
  --propellant-cost-usd-per-kg 20 \
  --engine-core-cost-usd-per-kg 10000 \
  --radiator-cost-usd-per-kg 1500 \
  --tank-mass-fraction 0.05 \
  --tank-cost-usd-per-kg 300
```

All values shown above are already defaults.

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

The JSON output includes `payload_component_costs_usd`,
`reusable_payload_cost_usd`, and `recurring_payload_cost_usd`.

## Sensitivity and plots

`fare_sensitivity_analysis.py` accepts the same fare inputs and includes all
four payload prices in the local sensitivity table:

```bash
python fare_sensitivity_analysis.py
```

It writes CSV, JSON and PNG outputs under `fare_analysis/` by default.

## Baseline ticket-price breakdown

`fare_sensitivity_analysis.py` prints a lifecycle fare table immediately after
solving the baseline. It separates:

- amortized constant and passenger-dependent payload hardware;
- recurring per-day and passenger-day payload consumables;
- engine-core, radiator, and propellant-tank amortization;
- recurring propellant and operations costs;
- markup and the final fare per paying passenger.

The same component rows are saved to `ticket_price_breakdown.csv` and included
in `analysis_summary.json`. The component contributions are checked to sum to
the fare before markup, and the markup line closes to the final ticket price.
