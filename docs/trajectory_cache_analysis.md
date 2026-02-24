This plan outlines the architecture for a "Time-Optimal Trajectory Atlas." By leveraging the physics of the constant-power problem, we can reduce the parameter space from six dimensions to three. This makes pre-computing a comprehensive solution database feasible, enabling instant "warm starts" for your solver.

### 1. Physics-Informed Dimensional Reduction

To avoid simulating every combination of Power ($P$), Mass ($m_0, m_{dry}$), and Radius ($r_0$), we must non-dimensionalize the system. This reveals that all ships with the same "Power-to-Weight-to-Gravity" ratio fly identical normalized trajectories.

#### A. Spatial Normalization

We normalize all distances by the departure radius $R_0$.

* **Radius Ratio ($\rho$):** The primary geometric parameter.

$$\rho = \frac{R_{target}}{R_{departure}}$$


* $\rho > 1$: Outward transfer (e.g., Earth to Mars).
* $\rho < 1$: Inward transfer (e.g., Mars to Earth).
* Since the user specifies a range of $[0.05, 100]$ AU, $\rho$ will vary from $\approx 0.0005$ to $2000$.



#### B. The "Ship Capability" Parameter ($\tilde{\kappa}$)

The trajectory shape is determined by the competition between the engine's ability to impart energy ($\int a^2 dt$) and the gravitational well's depth ($\mu/r$).

We define a single dimensionless parameter, $\tilde{\kappa}$, which encapsulates the ship's power, mass ratios, and the starting depth in the gravity well.

**Derivation:**

1. **Fuel Constraint:** From your manuscript Eq. (65), the constraint on the trajectory "effort" $J = \int a^2 dt$ is:

$$J = 2P \left( \frac{1}{m_{dry}} - \frac{1}{m_0} \right)$$


2. **Normalization:** Using canonical units where distance unit $DU = R_0$ and time unit $TU = \sqrt{R_0^3 / \mu}$, acceleration scales as $DU/TU^2$.
3. **The Master Parameter:**

$$\tilde{\kappa} = \left[ 2P \left( \frac{1}{m_{dry}} - \frac{1}{m_0} \right) \right] \cdot \frac{R_0^{2.5}}{\mu^{1.5}}$$



**Implication:** A 1 GW ship at 1 AU behaves dynamically *identical* to a stronger ship at 0.5 AU (where gravity is stronger), provided their $\tilde{\kappa}$ values match.

---

### 2. The Solution Manifold (The "Atlas")

We will pre-compute a database of optimal costates. Instead of storing full trajectories, we only need to store the **initial guess vector** required to converge the solver.

**The State Space Grid (3 Dimensions):**

1. **Radius Ratio ($\rho$):** Logarithmic grid (e.g., 50 points from $10^{-3}$ to $10^3$).
2. **Ship Capability ($\tilde{\kappa}$):** Logarithmic grid (e.g., 20 points).
* *Range:* Determine bounds by calculating $\tilde{\kappa}$ for "Fast NEP" (high power) vs. "Slow Solar Electric" (low power) scenarios across the Solar System.


3. **Transfer Angle ($\Delta \theta$):** Linear grid (e.g., every $10^\circ$).
* This is the "length" of the spiral. The maximum achievable $\Delta \theta$ varies; higher $\tilde{\kappa}$ (more power) allows for more windings before optimal control breaks down or hits the dry mass limit.



**Data Point Structure (What we store):**
For every grid point $(\rho_i, \tilde{\kappa}_j, \Delta\theta_k)$, we store a vector $S$:


$$S = [\lambda_{r0}, \lambda_{vr0}, \lambda_{v\theta0}, C_m, C_\theta, t_{flight}]$$


*Note: These values must be stored in their normalized (dimensionless) forms.*

---

**Algorithm:**

1. **Grid Initialization:**
* Define the axes:
* $\rho$: Log-space from 0.05 to 100.
* $\tilde{\kappa}$: Log-space covering the power range.
* $\theta$: Linear space (e.g., 0 to $8\pi$).


* Initialize the tensor $T$ with `NaN`.


2. **The "Anchor" Thread (1D Homotopy):**
* Select a "safe" central configuration (e.g., $\rho=1.0$, nominal $\tilde{\kappa}$).
* Solve this single column along the $\theta$ axis using the classic small-step method (0 to $\theta_{max}$). This creates a "spine" of valid solutions in the center of the grid.


3. **Wavefront Propagation (3D Homotopy):**
* Iterate outwards from the "Anchor" indices to fill the rest of the $\rho$ and $\tilde{\kappa}$ dimensions.
* **The Smart Guess (Predictor):** For any empty grid point $(i, j, k)$, generate an initial guess using *parameter continuation*:
* Check neighbors in the physical dimensions first: $T[i-1, j, k]$ (neighboring radius) or $T[i, j-1, k]$ (neighboring power).
* **Reasoning:** Adjusting a known spiral to a slightly different radius is computationally cheaper (often ~1-3 Newton steps) than growing a new spiral from scratch (hundreds of steps).


* **The Fallback (Corrector):**
* If the parameter neighbor guess fails to converge (e.g., crossing a stability boundary), fall back to the angular neighbor $T[i, j, k-1]$.
* This ensures that if the physics changes too abruptly, we safely "regrow" the spiral for that specific configuration.


4. **Persist:**
* Save the completed tensor $T$ as a compressed `.npz` file. This file now acts as a lookup table for the entire physically possible state space.

---

### 4. Runtime Execution (The "Instant Solver")

When the user requests a trajectory, the system performs a "warm start" optimization.

**Workflow:**

1. **Input Normalization:**
* Calculate $\rho = r_{target} / r_{start}$.
* Calculate $\tilde{\kappa}$ using the scaling law derived in Section 1.
* Calculate desired $\Delta \theta$ (taking into account optimal phasing if initial angle is free, or just user input).


2. **Atlas Retrieval:**
* Load the index.
* Find the $k$-nearest neighbors in the $(\rho, \tilde{\kappa}, \Delta \theta)$ space.
* Interpolate the vector $S$ (initial costates and time) from these neighbors.


3. **Dimensional Reconstruction:**
* Scale the interpolated costates and time back to physical SI units using $R_{start}$ and $\mu$.


4. **Instant Convergence:**
* Pass this highly accurate guess to `solve_target_fast` (from your existing code).
* Because the guess is physically valid and close to the root, `least_squares` will likely converge in 3-5 iterations (milliseconds).



---

### 5. Implementation Roadmap

**Step 1: Metric Verification (Python)**

* Write a script to compute $\tilde{\kappa}$ for 5-10 distinct test cases (e.g., Earth-Mars 1GW, Earth-Jupiter 1GW, Mercury-Venus low power).
* Verify that if you normalize their inputs, they map to similar $\tilde{\kappa}$ values, ensuring our grid bounds are realistic.

**Step 2: The Generator Script**

* Create `generate_atlas.py`.
* Implement the grid loops and the angle-stepping homotopy.
* Add error handling: If a specific $(\rho, \tilde{\kappa})$ is impossible (e.g., not enough fuel to reach target radius even at 0 angle), flag it as "Infeasible".

**Step 3: The Lookup Class**

* Create `trajectory_loader.py`.
* Implement `scipy.interpolate.RegularGridInterpolator` for fast retrieval.

**Step 4: Integration**

* Modify your main script to check for the `.npz` file. If present, use it for the initial guess instead of the global `differential_evolution` search.

Would you like me to begin by writing the **Generator Script** to build this database?