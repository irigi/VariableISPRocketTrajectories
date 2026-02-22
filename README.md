# VariableISP Rocket Trajectories

Numerical experiments for **minimum-time heliocentric transfers** with a constant-power electric propulsion system and **variable specific impulse** (`Isp`).

The model and methods implemented here follow the manuscript in this repository (`manuscript.tex`):

- Constant thruster power, with thrust and mass flow coupled through propulsion physics.
- Time-optimal control solved with an indirect (Pontryagin-style) formulation.
- Planar heliocentric dynamics, with examples of fast interplanetary transfers.

## Repository contents

- `rocketHamilton.py` – main simulation/optimization script.
- `manuscript.tex` – article describing derivations, assumptions, and numerical strategy.
- `requirements.txt` – minimal Python dependencies needed to run the code.

## Problem setup (high level)

The script models a spacecraft in polar heliocentric coordinates `(r, theta)` with states:

- radius and angle,
- radial and tangential velocities,
- spacecraft mass.

The engine is constrained by constant electrical power `P`, and the control law determines acceleration direction/magnitude through costates. Mass depletion is tied to acceleration squared under the constant-power relation, matching the manuscript formulation.

Default values in the script correspond to a high-power example (gigawatt class, large initial mass) and include routines for:

- one-shot trajectory integration with terminal events,
- fixed-time integration for smooth boundary residuals,
- least-squares based target solving for arbitrary transfer endpoints,
- plotting trajectory and onboard propellant history.

## Quick start

### 1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run

```bash
python rocketHamilton.py
```

> In headless environments, if plotting causes display issues, run with:
>
> ```bash
> MPLBACKEND=Agg python rocketHamilton.py
> ```

## Notes

- The default `main()` branch runs an optimization/integration scenario (`solve_arbitrary = True`) and may take time.
- To quickly verify imports and setup only, you can run:

```bash
python -c "import rocketHamilton; print('imports ok')"
```

## License

See `LICENSE`.
