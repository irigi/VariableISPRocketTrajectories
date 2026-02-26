"""
verify_grid.py
Checks if the chosen Atlas grid covers realistic mission scenarios.
"""
import numpy as np
import atlas_utils as utils

# Define test cases: (Name, Power, m0, m_dry, r_start_au, r_target_au)
missions = [
    # The Paper's Example [cite: 165]
    ("Paper Demo (Earth->Jup)", 1.0e9, 3.0e6, 1.0e6, 1.0, 5.2),

    # Same ship, starting at Mercury (deep gravity well, harder to escape)
    ("Paper Ship @ Mercury", 1.0e9, 3.0e6, 1.0e6, 0.39, 1.0),

    # Same ship, starting at Neptune (weak gravity, very capable)
    ("Paper Ship @ Neptune", 1.0e9, 3.0e6, 1.0e6, 30.0, 1.0),

    # Future High-Power / Light ship (Fast transit)
    ("Fast Courier (Earth->Mars)", 2.0e9, 1.0e5, 0.5e5, 1.0, 1.52),

    # Current Tech (SEP Cargo) - Low power, low thrust
    ("SEP Cargo (Earth->Mars)", 50e3, 2000, 1000, 1.0, 1.52),
]

print(f"{'Mission Name':<25} | {'Rho':<6} | {'Kappa':<10} | {'Status'}")
print("-" * 60)

kappa_min_observed = float('inf')
kappa_max_observed = float('-inf')

for name, P, m0, mdry, r0_au, rt_au in missions:
    r0 = r0_au * utils.AU
    rt = rt_au * utils.AU

    rho, kappa = utils.physical_to_dimensionless(P, m0, mdry, r0, rt)

    # Check boundaries
    in_rho = utils.RHO_MIN <= rho <= utils.RHO_MAX
    in_kappa = utils.KAPPA_MIN <= kappa <= utils.KAPPA_MAX

    status = "OK" if (in_rho and in_kappa) else "OUT OF BOUNDS"

    print(f"{name:<25} | {rho:<6.3f} | {kappa:<10.2f} | {status}")

    kappa_min_observed = min(kappa_min_observed, kappa)
    kappa_max_observed = max(kappa_max_observed, kappa)

print("-" * 60)
print(f"Observed Kappa Range: [{kappa_min_observed:.2f}, {kappa_max_observed:.2f}]")
print(f"Atlas Config Range:   [{utils.KAPPA_MIN:.2f}, {utils.KAPPA_MAX:.2f}]")

# Suggest adjustment if needed
if kappa_min_observed < utils.KAPPA_MIN or kappa_max_observed > utils.KAPPA_MAX:
    print("\n[!] WARNING: Grid boundaries need adjustment in atlas_utils.py!")
else:
    print("\n[+] Grid boundaries are sufficient.")
    