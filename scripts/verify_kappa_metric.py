"""Step-1 metric verification for the trajectory atlas roadmap.

This script performs two checks:
1) Survey realistic mission cases to derive practical kappa bounds.
2) Verify normalization invariance by constructing scenario pairs with
   intentionally matched kappa_tilde and checking relative error.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rocketHamilton import AU, MU_SI
from trajectory_scaling import (
    compute_canonical_units,
    compute_kappa_tilde,
    compute_rho,
    estimate_kappa_bounds,
    solve_power_for_kappa,
)


@dataclass(frozen=True)
class Scenario:
    name: str
    r_departure_au: float
    r_target_au: float
    power_w: float
    m0_kg: float
    m_dry_kg: float


@dataclass(frozen=True)
class PairCase:
    name: str
    r0_au: float
    m0_kg: float
    m_dry_kg: float


SCENARIOS = [
    Scenario("Earth->Mars", 1.00, 1.52, 1.0e9, 3.0e6, 1.0e6),
    Scenario("Earth->Jupiter", 1.00, 5.20, 1.0e9, 3.0e6, 1.0e6),
    Scenario("Mercury->Venus", 0.39, 0.72, 1.2e8, 1.8e6, 7.0e5),
    Scenario("Mars->Earth", 1.52, 1.00, 5.0e8, 2.5e6, 9.0e5),
    Scenario("Saturn->Earth", 9.58, 1.00, 1.0e9, 3.5e6, 1.2e6),
    Scenario("Earth->Neptune", 1.00, 30.1, 2.0e9, 4.0e6, 1.5e6),
    Scenario("Ceres->Earth", 2.77, 1.00, 8.0e7, 1.2e6, 6.0e5),
]


PAIR_BASELINE = PairCase(name="baseline", r0_au=1.0, m0_kg=3.0e6, m_dry_kg=1.0e6)
PAIR_TARGETS = [
    PairCase(name="inner-well", r0_au=0.50, m0_kg=2.2e6, m_dry_kg=8.0e5),
    PairCase(name="outer-well", r0_au=2.00, m0_kg=3.8e6, m_dry_kg=1.4e6),
    PairCase(name="far-outer", r0_au=5.00, m0_kg=4.2e6, m_dry_kg=1.6e6),
]


REL_TOL = 1e-12


def _scenario_rows() -> list[dict]:
    rows = []
    for s in SCENARIOS:
        r0 = s.r_departure_au * AU
        rt = s.r_target_au * AU
        rho = compute_rho(rt, r0)
        kappa = compute_kappa_tilde(
            power=s.power_w,
            m_dry=s.m_dry_kg,
            m0=s.m0_kg,
            r0=r0,
            mu=MU_SI,
        )
        units = compute_canonical_units(r0=r0, mu=MU_SI)
        rows.append(
            {
                "name": s.name,
                "r_departure_au": s.r_departure_au,
                "r_target_au": s.r_target_au,
                "rho": rho,
                "power_w": s.power_w,
                "m0_kg": s.m0_kg,
                "m_dry_kg": s.m_dry_kg,
                "kappa_tilde": kappa,
                "du_m": units.du_m,
                "tu_s": units.tu_s,
            }
        )
    return rows


def _invariance_rows() -> list[dict]:
    baseline_kappa = compute_kappa_tilde(
        power=1.0e9,
        m_dry=PAIR_BASELINE.m_dry_kg,
        m0=PAIR_BASELINE.m0_kg,
        r0=PAIR_BASELINE.r0_au * AU,
        mu=MU_SI,
    )

    rows = []
    for case in PAIR_TARGETS:
        required_power = solve_power_for_kappa(
            target_kappa=baseline_kappa,
            m_dry=case.m_dry_kg,
            m0=case.m0_kg,
            r0=case.r0_au * AU,
            mu=MU_SI,
        )
        recomputed = compute_kappa_tilde(
            power=required_power,
            m_dry=case.m_dry_kg,
            m0=case.m0_kg,
            r0=case.r0_au * AU,
            mu=MU_SI,
        )
        rel_err = abs(recomputed - baseline_kappa) / baseline_kappa
        rows.append(
            {
                "name": case.name,
                "r0_au": case.r0_au,
                "m0_kg": case.m0_kg,
                "m_dry_kg": case.m_dry_kg,
                "required_power_w": required_power,
                "kappa_target": baseline_kappa,
                "kappa_recomputed": recomputed,
                "relative_error": rel_err,
                "passes": rel_err <= REL_TOL,
            }
        )
    return rows


def main() -> None:
    scenario_rows = _scenario_rows()
    kappas = np.array([r["kappa_tilde"] for r in scenario_rows], dtype=float)
    kappa_min, kappa_max = estimate_kappa_bounds(kappas)

    invariance_rows = _invariance_rows()
    invariance_ok = all(r["passes"] for r in invariance_rows)

    print("Step 1: kappa-tilde metric verification")
    print("=" * 72)
    print("Mission scenario scan")
    for r in scenario_rows:
        print(
            f"{r['name']:18s} rho={r['rho']:>8.4f} "
            f"kappa={r['kappa_tilde']:>12.6e} "
            f"DU={r['du_m']:.3e} m TU={r['tu_s']:.3e} s"
        )

    print("\nSuggested atlas kappa bounds (log-padded)")
    print(f"kappa_min={kappa_min:.6e}")
    print(f"kappa_max={kappa_max:.6e}")

    print("\nNormalization invariance check (matched kappa across different r0/masses)")
    for r in invariance_rows:
        verdict = "PASS" if r["passes"] else "FAIL"
        print(
            f"{r['name']:10s} P={r['required_power_w']:.6e} W "
            f"kappa_target={r['kappa_target']:.6e} kappa_recomputed={r['kappa_recomputed']:.6e} "
            f"rel_err={r['relative_error']:.3e} [{verdict}]"
        )

    out_dir = REPO_ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "kappa_scan.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "name",
            "r_departure_au",
            "r_target_au",
            "rho",
            "power_w",
            "m0_kg",
            "m_dry_kg",
            "kappa_tilde",
            "du_m",
            "tu_s",
        ])
        for r in scenario_rows:
            writer.writerow([
                r["name"],
                r["r_departure_au"],
                r["r_target_au"],
                r["rho"],
                r["power_w"],
                r["m0_kg"],
                r["m_dry_kg"],
                r["kappa_tilde"],
                r["du_m"],
                r["tu_s"],
            ])
        writer.writerow([])
        writer.writerow(["kappa_min", kappa_min])
        writer.writerow(["kappa_max", kappa_max])

    report_path = out_dir / "kappa_step1_report.json"
    report = {
        "roadmap_step": "step_1_metric_verification",
        "baseline_pair_case": asdict(PAIR_BASELINE),
        "relative_tolerance": REL_TOL,
        "invariance_ok": invariance_ok,
        "kappa_bounds": {"kappa_min": kappa_min, "kappa_max": kappa_max},
        "scenario_rows": scenario_rows,
        "invariance_rows": invariance_rows,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nWrote {csv_path}")
    print(f"Wrote {report_path}")

    if not invariance_ok:
        raise SystemExit("Step-1 invariance check failed")


if __name__ == "__main__":
    main()
