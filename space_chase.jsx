import { useState, useMemo, useRef, useEffect } from "react";

// ─── Physics Engine (from the paper's free-space brachistochrone) ───
// Key model: constant power P, variable Isp, no gravity
// ṁ = -m²a²/(2P), fuel budget Δ = 1/m_dry - 1/m_i
// Optimal acceleration is LINEAR in time for each spatial component (PMP result)

function fuelIntegral1D_fixedBC(x0, v0, xf, vf, T) {
  if (T <= 1e-12) return Infinity;
  const L = xf - x0;
  const K1 = -12 * (L - 0.5 * (v0 + vf) * T) / (T * T * T);
  const K0 = (vf - v0) / T - 0.5 * K1 * T;
  return K1 * K1 * T * T * T / 3 + K1 * K0 * T * T + K0 * K0 * T;
}

function fuelIntegral1D_freeVf(x0, v0, xf, T) {
  if (T <= 1e-12) return Infinity;
  const L = xf - x0;
  const K = 3 * (L - v0 * T) / (T * T * T);
  return K * K * T * T * T / 3;
}

function fuelCost2D_fixed(P, r0, v0, rf, vf, T) {
  const Ix = fuelIntegral1D_fixedBC(r0[0], v0[0], rf[0], vf[0], T);
  const Iy = fuelIntegral1D_fixedBC(r0[1], v0[1], rf[1], vf[1], T);
  return (Ix + Iy) / (2 * P);
}

function fuelCost2D_freeVf(P, r0, v0, rf, T) {
  const Ix = fuelIntegral1D_freeVf(r0[0], v0[0], rf[0], T);
  const Iy = fuelIntegral1D_freeVf(r0[1], v0[1], rf[1], T);
  return (Ix + Iy) / (2 * P);
}

function findMinTime(P, Delta, r0, v0, rf, vf, fixedVf = true) {
  let lo = 1e-4, hi = 200;
  for (let i = 0; i < 40; i++) {
    const cost = fixedVf ? fuelCost2D_fixed(P, r0, v0, rf, vf, hi)
                         : fuelCost2D_freeVf(P, r0, v0, rf, hi);
    if (cost <= Delta) break;
    hi *= 2;
    if (hi > 1e9) return Infinity;
  }
  for (let i = 0; i < 80; i++) {
    const mid = (lo + hi) / 2;
    const cost = fixedVf ? fuelCost2D_fixed(P, r0, v0, rf, vf, mid)
                         : fuelCost2D_freeVf(P, r0, v0, rf, mid);
    if (cost > Delta) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}

function getTrajectory1D(x0, v0, xf, vf, T, t) {
  const L = xf - x0;
  const K1 = -12 * (L - 0.5 * (v0 + vf) * T) / (T * T * T);
  const K0 = (vf - v0) / T - 0.5 * K1 * T;
  return {
    x: x0 + v0 * t + 0.5 * K0 * t * t + K1 * t * t * t / 6,
    v: v0 + K0 * t + 0.5 * K1 * t * t,
    a: K1 * t + K0,
  };
}

function getTrajectory2D(r0, v0, rf, vf, T, t) {
  const sx = getTrajectory1D(r0[0], v0[0], rf[0], vf[0], T, t);
  const sy = getTrajectory1D(r0[1], v0[1], rf[1], vf[1], T, t);
  return { pos: [sx.x, sy.x], vel: [sx.v, sy.v], acc: [sx.a, sy.a] };
}

// ─── Main solver / scenario planner ───
// We model each ship as a 2D double-integrator with an L2 acceleration budget:
//   ∫ ||a||^2 dt ≤ Λ   where Λ = 2 P Δ,   Δ = 1/m_dry - 1/m_wet
// In this UI we keep using Δ-units (i.e. fuelCost() returns "Δ required") to match the paper.

function clamp01(x) { return Math.max(0, Math.min(1, x)); }

function buildLeg(r0, v0, rf, vf, T) {
  return { r0, v0, rf, vf, T };
}

function legState(leg, t) {
  return getTrajectory2D(leg.r0, leg.v0, leg.rf, leg.vf, leg.T, t);
}

function planState(plan, t) {
  // plan.legs concatenated
  let accT = 0;
  for (const leg of plan.legs) {
    if (t <= accT + leg.T || leg === plan.legs[plan.legs.length - 1]) {
      const localT = Math.max(0, Math.min(leg.T, t - accT));
      const st = legState(leg, localT);
      return { ...st, t, legIndex: plan.legs.indexOf(leg), localT, legStart: accT };
    }
    accT += leg.T;
  }
  // fallback
  const last = plan.legs[plan.legs.length - 1];
  const st = legState(last, last.T);
  return { ...st, t, legIndex: plan.legs.length - 1, localT: last.T, legStart: plan.T - last.T };
}

function planTrajectory(plan, steps = 140) {
  const pts = [];
  for (let i = 0; i <= steps; i++) {
    const t = (i / steps) * plan.T;
    pts.push(planState(plan, t).pos);
  }
  return pts;
}

function deltaFromShip(ship) {
  return 1 / ship.mDry - 1 / ship.mWet;
}

function planDirectToZ(A, baseZ) {
  const DeltaA = deltaFromShip(A);
  const TA = findMinTime(A.P, DeltaA, A.pos, A.vel, baseZ, [0, 0], true);
  if (!isFinite(TA)) return null;
  const fuel = fuelCost2D_fixed(A.P, A.pos, A.vel, baseZ, [0, 0], TA);
  return {
    type: "direct",
    legs: [buildLeg(A.pos, A.vel, baseZ, [0, 0], TA)],
    T: TA,
    fuelUsed: fuel,
    turnTime: null,
  };
}

function planDetour(A, baseZ, B, variant) {
  // Two-leg plan: A -> (Y, u) then (Y, u) -> (Z, 0)
  // Searches over waypoint position, AND intermediate velocity (fly-through, not just stop-turn).
  // Objective: maximize B's minimum-required Δ along the whole A path.
  const DeltaA = deltaFromShip(A);
  const DeltaB = deltaFromShip(B);
  const LZ = Math.hypot(baseZ[0] - A.pos[0], baseZ[1] - A.pos[1]) || 1;

  const norm = (u) => {
    const n = Math.hypot(u[0], u[1]) || 1;
    return [u[0] / n, u[1] / n];
  };
  const rot = (u, ang) => [u[0] * Math.cos(ang) - u[1] * Math.sin(ang), u[0] * Math.sin(ang) + u[1] * Math.cos(ang)];

  // Build direction set
  const dirs = [];
  const awayB = [A.pos[0] - B.pos[0], A.pos[1] - B.pos[1]];
  const awayZ = [A.pos[0] - baseZ[0], A.pos[1] - baseZ[1]];
  const seeds = [norm(awayB), norm(awayZ), norm([awayB[0] + awayZ[0], awayB[1] + awayZ[1]])];
  for (const s of seeds) {
    for (let k = 0; k < 16; k++) {
      const ang = (k / 16) * Math.PI * 2;
      dirs.push(rot(s, ang));
    }
  }
  const radii = [0.4, 0.7, 1.0, 1.5, 2.2].map(m => m * LZ);

  const stepsScan = 260;

  const bCostAt = (t, stA) => {
    if (variant === 1) return fuelCost2D_fixed(B.P, B.pos, B.vel, stA.pos, stA.vel, t);
    return fuelCost2D_freeVf(B.P, B.pos, B.vel, stA.pos, t);
  };

  const scorePlan = (plan) => {
    let minB = Infinity;
    for (let i = 1; i <= stepsScan; i++) {
      const t = (i / stepsScan) * plan.T;
      const stA = planState(plan, t);
      const cB = bCostAt(t, stA);
      if (cB < minB) minB = cB;
    }
    return { minB, margin: minB - DeltaB };
  };

  const tryPlan = (Y, uWay) => {
    // Leg 1: A.pos,A.vel -> Y,uWay
    const t1 = findMinTime(A.P, DeltaA, A.pos, A.vel, Y, uWay, true);
    if (!isFinite(t1)) return null;
    const c1 = fuelCost2D_fixed(A.P, A.pos, A.vel, Y, uWay, t1);
    if (c1 > DeltaA * 0.95) return null;

    // Leg 2: Y,uWay -> Z,[0,0]
    const remaining = DeltaA - c1;
    const t2 = findMinTime(A.P, remaining, Y, uWay, baseZ, [0, 0], true);
    if (!isFinite(t2)) return null;
    const c2 = fuelCost2D_fixed(A.P, Y, uWay, baseZ, [0, 0], t2);
    const cTot = c1 + c2;
    if (cTot > DeltaA) return null;

    return {
      type: "detour",
      legs: [buildLeg(A.pos, A.vel, Y, uWay, t1), buildLeg(Y, uWay, baseZ, [0, 0], t2)],
      T: t1 + t2,
      fuelUsed: cTot,
      turnTime: t1,
      waypoint: Y,
      waypointVel: uWay,
    };
  };

  let best = null;
  const consider = (plan) => {
    if (!plan) return;
    const sc = scorePlan(plan);
    if (!best || sc.margin > best.score.margin) {
      best = { plan, score: sc };
    }
  };

  for (const d of dirs) {
    for (const R of radii) {
      const Y = [A.pos[0] + d[0] * R, A.pos[1] + d[1] * R];

      // Stop-turn (v=0 at Y)
      consider(tryPlan(Y, [0, 0]));

      // Fly-through: velocity pointing back toward Z from Y
      const toZ = norm([baseZ[0] - Y[0], baseZ[1] - Y[1]]);
      const vScale = Math.sqrt(2 * A.P * DeltaA) * 0.15; // characteristic speed scale
      for (const sp of [0.3, 0.7, 1.2]) {
        consider(tryPlan(Y, [toZ[0] * vScale * sp, toZ[1] * vScale * sp]));
      }

      // Fly-through: velocity continuing outward from A (grazing pass)
      for (const sp of [0.3, 0.7]) {
        consider(tryPlan(Y, [d[0] * vScale * sp, d[1] * vScale * sp]));
      }
    }
  }

  return best ? best.plan : null;
}

function planRunAndWait(A, baseZ, B, variant) {
  // Three-leg plan: A burns away from B, coasts, then returns to Z.
  // The idea: A escapes to a safe distance where B cannot reach, coasts to let B
  // exhaust fuel if B chases, then uses reserved fuel to dock at Z.
  //
  // Leg 1: powered escape  A.pos,A.vel -> Y,uCoast  (burn fuel_1)
  // Leg 2: coast at constant velocity uCoast for time tCoast  (zero fuel)
  // Leg 3: powered return  coastEnd,uCoast -> Z,[0,0]  (burn fuel_3)
  //
  // fuel_1 + fuel_3 <= DeltaA
  // We search over escape direction, fuel split, and coast duration.

  const DeltaA = deltaFromShip(A);
  const DeltaB = deltaFromShip(B);
  const LZ = Math.hypot(baseZ[0] - A.pos[0], baseZ[1] - A.pos[1]) || 1;

  const norm = (u) => {
    const n = Math.hypot(u[0], u[1]) || 1;
    return [u[0] / n, u[1] / n];
  };
  const rot = (u, ang) => [u[0] * Math.cos(ang) - u[1] * Math.sin(ang), u[0] * Math.sin(ang) + u[1] * Math.cos(ang)];

  const awayB = [A.pos[0] - B.pos[0], A.pos[1] - B.pos[1]];
  const awayZ = [A.pos[0] - baseZ[0], A.pos[1] - baseZ[1]];
  // Escape directions: away from B, perpendicular, and mixes
  const escapeDirs = [];
  const seedsE = [norm(awayB), norm(awayZ), norm([awayB[0] + awayZ[0], awayB[1] + awayZ[1]])];
  for (const s of seedsE) {
    for (let k = 0; k < 12; k++) {
      escapeDirs.push(rot(s, (k / 12) * Math.PI * 2));
    }
  }

  const stepsScan = 260;
  const bCostAt = (t, stA) => {
    if (variant === 1) return fuelCost2D_fixed(B.P, B.pos, B.vel, stA.pos, stA.vel, t);
    return fuelCost2D_freeVf(B.P, B.pos, B.vel, stA.pos, t);
  };

  const scorePlan = (plan) => {
    let minB = Infinity;
    for (let i = 1; i <= stepsScan; i++) {
      const t = (i / stepsScan) * plan.T;
      const stA = planState(plan, t);
      const cB = bCostAt(t, stA);
      if (cB < minB) minB = cB;
    }
    return { minB, margin: minB - DeltaB };
  };

  let best = null;

  // Fuel split ratios for escape vs return
  const fuelSplits = [0.15, 0.25, 0.35, 0.45];
  // Coast durations as multiples of direct flight time
  const directT = findMinTime(A.P, DeltaA, A.pos, A.vel, baseZ, [0, 0], true);
  const coastMults = [0.5, 1.0, 2.0, 4.0];

  for (const d of escapeDirs) {
    for (const fSplit of fuelSplits) {
      const fuel1Budget = DeltaA * fSplit;
      const fuel3Budget = DeltaA * (1 - fSplit);

      // Escape: burn fuel1Budget in direction d for minimum time
      // We need an escape endpoint. Use the characteristic distance from fuel budget.
      // For a stop-start from rest: Δ = 6L²/(P·T³), so L ~ (P·Δ·T³/6)^(1/2)
      // But we want to END with velocity (coast), not stop.
      // Use free-final-velocity: C_pos = 3(L-v0·T)²/T³ => with L = R·d direction
      // Pick escape distance by scanning a few radii
      const escapeRadii = [0.3, 0.6, 1.0, 1.5].map(m => m * LZ);

      for (const R of escapeRadii) {
        const Y = [A.pos[0] + d[0] * R, A.pos[1] + d[1] * R];

        // Find minimum-time escape to Y with free final velocity
        const t1 = findMinTime(A.P, fuel1Budget, A.pos, A.vel, Y, [0, 0], false);
        if (!isFinite(t1) || t1 < 0.01) continue;

        // Compute the actual final velocity at Y using free-Vf optimal trajectory
        // For free-Vf, the optimal a(t) = K·(T-t) where K = 3(L-v0·T)/T³
        // v(T) = v0 + K·T²/2 - K·T²/2... let me compute properly:
        // a(t) = K*(T-t), v(t) = v0 + K*T*t - K*t²/2, v(T) = v0 + K*T²/2
        const Kx = 3 * (Y[0] - A.pos[0] - A.vel[0] * t1) / (t1 * t1 * t1);
        const Ky = 3 * (Y[1] - A.pos[1] - A.vel[1] * t1) / (t1 * t1 * t1);
        const uCoast = [A.vel[0] + Kx * t1 * t1 / 2, A.vel[1] + Ky * t1 * t1 / 2];

        const c1 = fuelCost2D_freeVf(A.P, A.pos, A.vel, Y, t1);
        if (c1 > fuel1Budget) continue;

        for (const cMult of coastMults) {
          const tCoast = (isFinite(directT) ? directT : LZ) * cMult;
          if (tCoast < 0.1) continue;

          // Coast endpoint
          const coastEnd = [Y[0] + uCoast[0] * tCoast, Y[1] + uCoast[1] * tCoast];

          // Return: coastEnd,uCoast -> Z,[0,0]
          const t3 = findMinTime(A.P, fuel3Budget, coastEnd, uCoast, baseZ, [0, 0], true);
          if (!isFinite(t3)) continue;
          const c3 = fuelCost2D_fixed(A.P, coastEnd, uCoast, baseZ, [0, 0], t3);
          if (c1 + c3 > DeltaA) continue;

          // Build 3-leg plan. Coast leg: position moves linearly, velocity constant.
          // We model coast as a "leg" with rf = coastEnd, vf = uCoast (constant v).
          // But our leg model uses polynomial acceleration — for a coast, a(t)=0,
          // so rf = r0 + v0*T, vf = v0. This is exactly buildLeg(Y, uCoast, coastEnd, uCoast, tCoast).
          const plan = {
            type: "escape",
            legs: [
              buildLeg(A.pos, A.vel, Y, uCoast, t1),
              buildLeg(Y, uCoast, coastEnd, uCoast, tCoast),
              buildLeg(coastEnd, uCoast, baseZ, [0, 0], t3),
            ],
            T: t1 + tCoast + t3,
            fuelUsed: c1 + c3,
            turnTime: t1,
            coastStart: t1,
            coastEnd: t1 + tCoast,
            waypoint: Y,
            waypointVel: uCoast,
          };

          const sc = scorePlan(plan);
          if (!best || sc.margin > best.score.margin) {
            best = { plan, score: sc };
          }
        }
      }
    }
  }

  return best ? best.plan : null;
}

function chooseAPlan(A, B, baseZ, variant) {
  const direct = planDirectToZ(A, baseZ);
  if (!direct) return { plan: null, reason: "A cannot reach Z" };

  const detour = planDetour(A, baseZ, B, variant);
  const escape = planRunAndWait(A, baseZ, B, variant);

  return { direct, detour, escape };
}

function pickBIntercept(A, B, baseZ, variant, aPlan) {
  const DeltaB = deltaFromShip(B);
  const N = 520;

  const bCostAt = (t, stA) => {
    if (variant === 1) return fuelCost2D_fixed(B.P, B.pos, B.vel, stA.pos, stA.vel, t);
    return fuelCost2D_freeVf(B.P, B.pos, B.vel, stA.pos, t);
  };

  let minFuel = Infinity;
  let minFuelT = null;

  let earliest = null;

  // Track whether any capture exists before a detour turn (for labeling block vs chase)
  const turnT = aPlan.turnTime ?? null;
  let hasBeforeTurn = false;

  for (let i = 1; i <= N; i++) {
    const t = (i / N) * aPlan.T;
    const stA = planState(aPlan, t);
    const cB = bCostAt(t, stA);

    if (cB < minFuel) { minFuel = cB; minFuelT = t; }

    if (cB <= DeltaB && earliest === null) {
      earliest = { tInt: t, fuelNeeded: cB, stateA: stA };
    }
    if (turnT != null && t < turnT && cB <= DeltaB) hasBeforeTurn = true;
  }

  const wins = minFuel <= DeltaB;

  // "Block" option: if A has a turn and B cannot intercept before turn, pick a feasible intercept after the turn
  // such that B can plausibly arrive early (min-time ≤ 0.85 * tInt).
  let block = null;
  if (turnT != null && !hasBeforeTurn) {
    for (let i = 1; i <= N; i++) {
      const t = (i / N) * aPlan.T;
      if (t <= turnT * 1.05) continue; // after the turn
      const stA = planState(aPlan, t);
      const cB = bCostAt(t, stA);
      if (cB > DeltaB) continue;

      // Can B get there early?
      const tMin = findMinTime(
        B.P, DeltaB,
        B.pos, B.vel,
        stA.pos, variant === 1 ? stA.vel : [0, 0],
        variant === 1
      );
      if (!isFinite(tMin) || tMin > 0.85 * t) continue;

      // prefer earlier after-turn capture
      block = { tInt: t, fuelNeeded: cB, stateA: stA };
      break;
    }
  }

  let chosen = null;
  let strategy = "none";
  if (wins) {
    if (block) { chosen = block; strategy = "block"; }
    else { chosen = earliest; strategy = "chase"; }
  } else {
    // best attempt for display
    chosen = { tInt: minFuelT, fuelNeeded: minFuel, stateA: planState(aPlan, minFuelT) };
    strategy = "fail";
  }

  return { wins, earliest, minFuelTInt: minFuelT, minFuel, chosen, strategy };
}

function solveScenario(A, B, baseZ, variant, scenarioMode = "auto") {
  const DeltaA = deltaFromShip(A);
  const DeltaB = deltaFromShip(B);

  const { direct, detour, escape } = chooseAPlan(A, B, baseZ, variant);
  if (!direct) return { feasibleA: false, DeltaA, DeltaB, variant };

  // Evaluate safety of a plan: does B have ANY feasible intercept along it?
  const planIsSafe = (plan) => {
    const N = 420;
    for (let i = 1; i <= N; i++) {
      const t = (i / N) * plan.T;
      const stA = planState(plan, t);
      const cB = variant === 1
        ? fuelCost2D_fixed(B.P, B.pos, B.vel, stA.pos, stA.vel, t)
        : fuelCost2D_freeVf(B.P, B.pos, B.vel, stA.pos, t);
      if (cB <= DeltaB) return false;
    }
    return true;
  };

  // Evaluate margin: how far above Δ_B is B's cheapest intercept?
  const evalMargin = (plan) => {
    let minB = Infinity;
    const N = 300;
    for (let i = 1; i <= N; i++) {
      const t = (i / N) * plan.T;
      const stA = planState(plan, t);
      const cB = variant === 1
        ? fuelCost2D_fixed(B.P, B.pos, B.vel, stA.pos, stA.vel, t)
        : fuelCost2D_freeVf(B.P, B.pos, B.vel, stA.pos, t);
      if (cB < minB) minB = cB;
    }
    return minB - DeltaB;
  };

  const safeDirect = planIsSafe(direct);
  const safeDetour = detour ? planIsSafe(detour) : false;
  const safeEscape = escape ? planIsSafe(escape) : false;

  // Collect all candidate plans with their margins
  const candidates = [{ plan: direct, policy: "direct", margin: evalMargin(direct) }];
  if (detour) candidates.push({ plan: detour, policy: "detour", margin: evalMargin(detour) });
  if (escape) candidates.push({ plan: escape, policy: "escape", margin: evalMargin(escape) });

  // Choose A's plan depending on scenario mode
  let aPlan = direct;
  let aPolicy = "direct";
  if (scenarioMode === "auto") {
    // Pick the plan with the best margin (highest min-B cost relative to Δ_B)
    // Among safe plans prefer shortest; if none safe, pick best margin
    const safeCands = candidates.filter(c => c.margin > 0);
    if (safeCands.length > 0) {
      // Among safe plans, prefer shortest total time
      safeCands.sort((a, b) => a.plan.T - b.plan.T);
      aPlan = safeCands[0].plan;
      aPolicy = safeCands[0].policy;
    } else {
      // No safe plan: pick the one that maximizes margin (best evasion attempt)
      candidates.sort((a, b) => b.margin - a.margin);
      aPlan = candidates[0].plan;
      aPolicy = candidates[0].policy;
    }
  } else if (scenarioMode === "force_direct") {
    aPlan = direct; aPolicy = "direct";
  } else if (scenarioMode === "force_detour") {
    aPlan = detour || direct; aPolicy = detour ? "detour" : "direct";
  } else if (scenarioMode === "force_escape") {
    aPlan = escape || detour || direct;
    aPolicy = escape ? "escape" : (detour ? "detour" : "direct");
  }

  const bRes = pickBIntercept(A, B, baseZ, variant, aPlan);

  // Build scan results for the fuel plot: always show both variants' Δ_B(t) curves against THIS A-plan
  const Nplot = 500;
  const scanResults = [];
  for (let i = 1; i <= Nplot; i++) {
    const tInt = (i / Nplot) * aPlan.T;
    const stA = planState(aPlan, tInt);
    const fuelV1 = fuelCost2D_fixed(B.P, B.pos, B.vel, stA.pos, stA.vel, tInt);
    const fuelV2 = fuelCost2D_freeVf(B.P, B.pos, B.vel, stA.pos, tInt);
    scanResults.push({ tInt, fuelV1, fuelV2 });
  }

  // Trajectories for display
  const steps = 140;
  const trajectoryA = planTrajectory(aPlan, steps);

  const buildTrajB_toState = (tInt, stA, fixed) => {
    if (!tInt || !isFinite(tInt)) return [];
    const pts = [];
    for (let i = 0; i <= steps; i++) {
      const t = (i / steps) * tInt;
      if (fixed) {
        const s = getTrajectory2D(B.pos, B.vel, stA.pos, stA.vel, tInt, t);
        pts.push(s.pos);
      } else {
        const Lx = stA.pos[0] - B.pos[0], Ly = stA.pos[1] - B.pos[1];
        const Kx = 3 * (Lx - B.vel[0] * tInt) / (tInt * tInt * tInt);
        const Ky = 3 * (Ly - B.vel[1] * tInt) / (tInt * tInt * tInt);
        pts.push([
          B.pos[0] + B.vel[0] * t + Kx * (tInt * t * t / 2 - t * t * t / 6),
          B.pos[1] + B.vel[1] * t + Ky * (tInt * t * t / 2 - t * t * t / 6),
        ]);
      }
    }
    return pts;
  };

  const chosen = bRes.chosen;
  const trajB = chosen?.stateA
    ? buildTrajB_toState(chosen.tInt, chosen.stateA, variant === 1)
    : [];

  const intercept = chosen?.stateA
    ? { t: chosen.tInt, pos: chosen.stateA.pos, vel: chosen.stateA.vel, fuel: chosen.fuelNeeded }
    : null;

  return {
    feasibleA: true,
    DeltaA, DeltaB,
    variant,
    aPolicy,
    aPlan,
    safeDirect,
    safeDetour,
    safeEscape,
    b: {
      wins: bRes.wins,
      strategy: bRes.strategy,
      minFuelTInt: bRes.minFuelTInt,
      fuelNeeded: bRes.minFuel,
      fuelRatio: bRes.minFuel / DeltaB,
      earliestTInt: bRes.earliest?.tInt ?? null,
      chosenTInt: chosen?.tInt ?? null,
      chosenFuel: chosen?.fuelNeeded ?? null,
    },
    trajectoryA,
    trajectoryB: trajB,
    scanResults,
    intercept,
  };
}

// ─── Presets (tuned so AUTO picks the intended behavior) ───
const PRESETS = {
  // Direct escape: A can dock before B has any feasible intercept window (both variants tend to show escape).
  "A escapes (direct)": {
    A: { x: 0, y: 0, vx: 2.0, vy: 0.3, P: 1.2, mWet: 3.2, mDry: 1.0 },
    B: { x: -45, y: 25, vx: 0, vy: 0, P: 1.0, mWet: 2.8, mDry: 1.0 },
    Z: { x: 70, y: 0 },
  },

  // B catches: B has clear mobility advantage; AUTO will keep A direct unless detour truly helps, but B still intercepts.
  "B catches (chase)": {
    A: { x: 0, y: 0, vx: 0.5, vy: 0, P: 0.7, mWet: 2.2, mDry: 1.0 },
    B: { x: 18, y: 14, vx: 0, vy: 0, P: 2.2, mWet: 4.2, mDry: 1.0 },
    Z: { x: 85, y: 0 },
  },

  // Block-and-wait demo: A has just enough mobility to make a detour worthwhile in V1,
  // and B cannot catch before the turn but can position for the return leg.
  "B blocks return": {
    A: { x: 0, y: 0, vx: 0, vy: 0, P: 1.05, mWet: 3.2, mDry: 1.0 },
    B: { x: 40, y: 14, vx: 0, vy: -0.2, P: 1.1, mWet: 3.0, mDry: 1.0 },
    Z: { x: 110, y: 0 },
  },

  // A runs away, coasts, and returns after B can't chase effectively
  "A escapes (run+wait)": {
    A: { x: 0, y: 0, vx: 0, vy: 0.5, P: 1.2, mWet: 3.5, mDry: 1.0 },
    B: { x: 35, y: 0, vx: -0.3, vy: 0, P: 1.0, mWet: 2.8, mDry: 1.0 },
    Z: { x: 90, y: 0 },
  },

  // A has energy advantage: can outlast B by running away
  "A outlasts B": {
    A: { x: 0, y: 0, vx: 0, vy: 0, P: 1.5, mWet: 4.0, mDry: 1.0 },
    B: { x: 25, y: 15, vx: 0, vy: 0, P: 1.2, mWet: 2.5, mDry: 1.0 },
    Z: { x: 80, y: 0 },
  },

  "Balanced duel": {
    A: { x: 0, y: 0, vx: 0, vy: 0, P: 1, mWet: 3, mDry: 1 },
    B: { x: 30, y: 20, vx: 0, vy: 0, P: 1, mWet: 3, mDry: 1 },
    Z: { x: 100, y: 0 },
  },
  "Pursuer advantage": {
    A: { x: 0, y: 0, vx: 0, vy: 0, P: 0.5, mWet: 2, mDry: 1 },
    B: { x: 20, y: 10, vx: 0, vy: 0, P: 2, mWet: 4, mDry: 1 },
    Z: { x: 80, y: 0 },
  },
  "Head start escape": {
    A: { x: 0, y: 0, vx: 2, vy: 0, P: 1, mWet: 3, mDry: 1 },
    B: { x: -40, y: 30, vx: 0, vy: 0, P: 1.5, mWet: 3, mDry: 1 },
    Z: { x: 60, y: 0 },
  },
  "Blocking position": {
    A: { x: 0, y: 0, vx: 0, vy: 0, P: 1, mWet: 3, mDry: 1 },
    B: { x: 50, y: 5, vx: 0, vy: 0, P: 0.8, mWet: 3, mDry: 1 },
    Z: { x: 100, y: 0 },
  },
};

// ─── Scenario modes ───
// "Auto" makes A choose direct vs detour based on whether B has an intercept window,
// and makes B choose chase vs block (when A detours and B can't catch before the turn).
const SCENARIO_MODES = {
  "Auto (game)": { mode: "auto" },
  "Force: A direct": { mode: "force_direct" },
  "Force: A detour": { mode: "force_detour" },
  "Force: A escape": { mode: "force_escape" },
};

// ─── Small UI pieces ───
function Slider({ label, value, onChange, min, max, step, unit }) {
  return (
    <div style={{ marginBottom: 5 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#8899aa", marginBottom: 1 }}>
        <span>{label}</span>
        <span style={{ color: "#c0d8f0", fontFamily: "monospace" }}>
          {Number.isInteger(step) ? value : value.toFixed(2)}{unit || ""}
        </span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        style={{ width: "100%", height: 3, accentColor: "#4a9eff" }} />
    </div>
  );
}

function ShipPanel({ title, color, params, keys, onChange }) {
  const set = (k, v) => onChange({ ...params, [k]: v });
  return (
    <div style={{
      background: "rgba(10,20,35,0.85)", border: `1px solid ${color}33`,
      borderLeft: `3px solid ${color}`, borderRadius: 6, padding: "8px 10px", marginBottom: 8,
    }}>
      <div style={{ fontSize: 12, fontWeight: 700, color, marginBottom: 6, letterSpacing: 1 }}>{title}</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 10px" }}>
        <Slider label="x₀" value={params.x} onChange={v => set("x", v)} min={-100} max={200} step={1} />
        <Slider label="y₀" value={params.y} onChange={v => set("y", v)} min={-100} max={100} step={1} />
      </div>
      {keys.includes("vx") && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 10px" }}>
          <Slider label="vx₀" value={params.vx} onChange={v => set("vx", v)} min={-5} max={5} step={0.1} />
          <Slider label="vy₀" value={params.vy} onChange={v => set("vy", v)} min={-5} max={5} step={0.1} />
        </div>
      )}
      {keys.includes("P") && (
        <>
          <Slider label="Power P" value={params.P} onChange={v => set("P", v)} min={0.1} max={5} step={0.1} unit=" GW" />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 10px" }}>
            <Slider label="m_wet" value={params.mWet} onChange={v => set("mWet", v)} min={1.1} max={10} step={0.1} unit=" kt" />
            <Slider label="m_dry" value={params.mDry} onChange={v => set("mDry", v)}
              min={0.5} max={Math.min(params.mWet - 0.1, 9)} step={0.1} unit=" kt" />
          </div>
          <div style={{ fontSize: 9, color: "#556677", marginTop: 2 }}>
            Δ = {(1 / params.mDry - 1 / params.mWet).toFixed(4)} · fuel = {((params.mWet - params.mDry) / params.mWet * 100).toFixed(0)}%
          </div>
        </>
      )}
    </div>
  );
}

// ─── Canvas: trajectory map ───
function drawMap(canvas, result, shipA, shipB, baseZ, activeVariant) {
  if (!canvas || !result) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;

  const trajB = result.trajectoryB || [];
  const allPts = [
    [shipA.x, shipA.y], [shipB.x, shipB.y], [baseZ.x, baseZ.y],
    ...(result.trajectoryA || []), ...(trajB || []),
  ].filter(p => isFinite(p[0]) && isFinite(p[1]));

  let mnX = Infinity, mxX = -Infinity, mnY = Infinity, mxY = -Infinity;
  allPts.forEach(([x, y]) => { mnX = Math.min(mnX, x); mxX = Math.max(mxX, x); mnY = Math.min(mnY, y); mxY = Math.max(mxY, y); });
  const pad = Math.max(mxX - mnX, mxY - mnY) * 0.15 + 5;
  mnX -= pad; mxX += pad; mnY -= pad; mxY += pad;
  const rX = mxX - mnX || 1, rY = mxY - mnY || 1;
  const sc = Math.min(W / rX, H / rY);
  const ox = (W - rX * sc) / 2, oy = (H - rY * sc) / 2;
  const tx = x => ox + (x - mnX) * sc;
  const ty = y => H - (oy + (y - mnY) * sc);

  ctx.fillStyle = "#060d18"; ctx.fillRect(0, 0, W, H);

  // Grid
  const gs = Math.pow(10, Math.floor(Math.log10(rX / 5)));
  ctx.strokeStyle = "#0a1828"; ctx.lineWidth = 0.5;
  for (let x = Math.ceil(mnX / gs) * gs; x <= mxX; x += gs) { ctx.beginPath(); ctx.moveTo(tx(x), 0); ctx.lineTo(tx(x), H); ctx.stroke(); }
  for (let y = Math.ceil(mnY / gs) * gs; y <= mxY; y += gs) { ctx.beginPath(); ctx.moveTo(0, ty(y)); ctx.lineTo(W, ty(y)); ctx.stroke(); }

  // A trajectory
  if (result.trajectoryA?.length > 1) {
    ctx.beginPath(); ctx.strokeStyle = "#22dd88"; ctx.lineWidth = 2.5; ctx.setLineDash([]);
    result.trajectoryA.forEach(([x, y], i) => i === 0 ? ctx.moveTo(tx(x), ty(y)) : ctx.lineTo(tx(x), ty(y)));
    ctx.stroke();
  }
  // Waypoint marker (for detour/escape plans)
  if (result.aPlan?.waypoint) {
    const wp = result.aPlan.waypoint;
    const wx = tx(wp[0]), wy = ty(wp[1]);
    ctx.beginPath(); ctx.arc(wx, wy, 5, 0, Math.PI * 2);
    ctx.fillStyle = "#22dd8866"; ctx.fill();
    ctx.strokeStyle = "#22dd88"; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.fillStyle = "#22dd8899"; ctx.font = "9px monospace";
    const label = result.aPlan.type === "escape" ? "escape" : "waypoint";
    ctx.fillText(label, wx + 9, wy + 3);
  }
  // B trajectory
  if (trajB?.length > 1) {
    ctx.beginPath(); ctx.strokeStyle = "#ff4466"; ctx.lineWidth = 2; ctx.setLineDash([7, 5]);
    trajB.forEach(([x, y], i) => i === 0 ? ctx.moveTo(tx(x), ty(y)) : ctx.lineTo(tx(x), ty(y)));
    ctx.stroke(); ctx.setLineDash([]);
  }

  // Interception point (for the currently-selected variant + scenario plan)
  if (result?.intercept?.t && result?.feasibleA) {
    const ix = tx(result.intercept.pos[0]), iy = ty(result.intercept.pos[1]);
    ctx.beginPath(); ctx.arc(ix, iy, 7, 0, Math.PI * 2);
    const bWins = result.b?.wins;
    ctx.strokeStyle = bWins ? "#ff4466" : "#445566"; ctx.lineWidth = 2; ctx.stroke();
    if (bWins) {
      ctx.beginPath(); ctx.arc(ix, iy, 12, 0, Math.PI * 2);
      ctx.strokeStyle = "#ff446644"; ctx.lineWidth = 1; ctx.stroke();
      ctx.fillStyle = "#ff446699"; ctx.font = "bold 10px monospace";
      ctx.fillText("intercept t=" + result.intercept.t.toFixed(1), ix + 16, iy + 3);
    } else {
      ctx.fillStyle = "#44556688"; ctx.font = "9px monospace";
      ctx.fillText("best attempt", ix + 12, iy + 3);
    }
  }

  // Velocity arrows

  const drawArrow = (x, y, vx, vy, col) => {
    if (Math.hypot(vx, vy) < 0.01) return;
    const s2 = sc * 3;
    ctx.beginPath(); ctx.moveTo(tx(x), ty(y));
    ctx.lineTo(tx(x) + vx * s2, ty(y) - vy * s2);
    ctx.strokeStyle = col + "88"; ctx.lineWidth = 2; ctx.stroke();
  };

  // Ship A
  ctx.beginPath(); ctx.arc(tx(shipA.x), ty(shipA.y), 7, 0, Math.PI * 2);
  ctx.fillStyle = "#22dd88"; ctx.fill();
  ctx.font = "bold 13px monospace"; ctx.fillText("A", tx(shipA.x) + 11, ty(shipA.y) + 4);
  drawArrow(shipA.x, shipA.y, shipA.vx, shipA.vy, "#22dd88");

  // Ship B
  ctx.beginPath(); ctx.arc(tx(shipB.x), ty(shipB.y), 7, 0, Math.PI * 2);
  ctx.fillStyle = "#ff4466"; ctx.fill();
  ctx.fillStyle = "#ff4466"; ctx.font = "bold 13px monospace"; ctx.fillText("B", tx(shipB.x) + 11, ty(shipB.y) + 4);
  drawArrow(shipB.x, shipB.y, shipB.vx, shipB.vy, "#ff4466");

  // Base Z hexagon
  const zx = tx(baseZ.x), zy = ty(baseZ.y);
  ctx.save(); ctx.translate(zx, zy); ctx.beginPath();
  for (let i = 0; i < 6; i++) { const a = Math.PI / 3 * i - Math.PI / 6; const r = 11; i === 0 ? ctx.moveTo(Math.cos(a) * r, Math.sin(a) * r) : ctx.lineTo(Math.cos(a) * r, Math.sin(a) * r); }
  ctx.closePath(); ctx.fillStyle = "#4a9eff22"; ctx.fill();
  ctx.strokeStyle = "#4a9eff"; ctx.lineWidth = 2; ctx.stroke(); ctx.restore();
  ctx.fillStyle = "#4a9eff"; ctx.font = "bold 13px monospace"; ctx.fillText("Z", zx + 15, zy + 4);
}

// ─── Canvas: fuel cost plot ───
function drawFuelPlot(canvas, result) {
  if (!canvas || !result?.scanResults) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  const data = result.scanResults;
  const DeltaB = result.DeltaB;

  ctx.fillStyle = "#060d18"; ctx.fillRect(0, 0, W, H);

  const mL = 70, mR = 30, mT = 50, mB = 55;
  const pW = W - mL - mR, pH = H - mT - mB;
  if (pW < 10 || pH < 10) return;

  // Y scale
  const allFuels = data.map(d => Math.min(d.fuelV1, d.fuelV2)).filter(isFinite);
  const allMin = allFuels.length ? Math.min(...allFuels) : DeltaB;
  const yMax = Math.max(DeltaB * 2.5, allMin * 4, DeltaB * 1.3);
  const maxT = result.aPlan.T;

  const px = t => mL + (t / maxT) * pW;
  const py = f => mT + pH - (Math.min(f, yMax) / yMax) * pH;

  // Grid
  ctx.strokeStyle = "#0f1d30"; ctx.lineWidth = 0.5;
  const nGridY = 6, nGridX = 8;
  ctx.font = "10px monospace"; ctx.textAlign = "right"; ctx.fillStyle = "#445566";
  for (let i = 0; i <= nGridY; i++) {
    const y = mT + (i / nGridY) * pH;
    ctx.beginPath(); ctx.moveTo(mL, y); ctx.lineTo(W - mR, y); ctx.stroke();
    ctx.fillText((yMax * (1 - i / nGridY)).toFixed(3), mL - 8, y + 3);
  }
  ctx.textAlign = "center";
  for (let i = 0; i <= nGridX; i++) {
    const x = mL + (i / nGridX) * pW;
    ctx.beginPath(); ctx.moveTo(x, mT); ctx.lineTo(x, mT + pH); ctx.stroke();
    ctx.fillText((maxT * i / nGridX).toFixed(1), x, H - mB + 16);
  }

  // Plot border
  ctx.strokeStyle = "#1a2a40"; ctx.lineWidth = 1;
  ctx.strokeRect(mL, mT, pW, pH);

  // Δ_B line
  if (DeltaB <= yMax) {
    const yLine = py(DeltaB);
    ctx.beginPath(); ctx.moveTo(mL, yLine); ctx.lineTo(W - mR, yLine);
    ctx.strokeStyle = "#ffffff55"; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.5; ctx.stroke(); ctx.setLineDash([]);
    // Shaded region below
    ctx.fillStyle = "#22dd8808";
    ctx.fillRect(mL, yLine, pW, mT + pH - yLine);
    ctx.fillStyle = "#ffffffbb"; ctx.font = "bold 11px monospace"; ctx.textAlign = "left";
    ctx.fillText("Δ_B = " + DeltaB.toFixed(4) + "  (B's fuel budget)", mL + 10, yLine - 8);
    ctx.fillStyle = "#22dd8844"; ctx.font = "9px monospace";
    ctx.fillText("← A escapes (B lacks fuel)", mL + 10, yLine + 14);
  }

  // Turn / coast markers for detour and escape plans
  if (result.aPlan?.turnTime && maxT > 0) {
    const turnX = px(result.aPlan.turnTime);
    ctx.beginPath(); ctx.moveTo(turnX, mT); ctx.lineTo(turnX, mT + pH);
    ctx.strokeStyle = "#22dd8844"; ctx.setLineDash([4, 4]); ctx.lineWidth = 1; ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#22dd8866"; ctx.font = "9px monospace"; ctx.textAlign = "center";
    ctx.fillText("turn", turnX, mT + pH + 12);
  }
  if (result.aPlan?.coastEnd && maxT > 0) {
    const ceX = px(result.aPlan.coastEnd);
    ctx.beginPath(); ctx.moveTo(ceX, mT); ctx.lineTo(ceX, mT + pH);
    ctx.strokeStyle = "#22dd8844"; ctx.setLineDash([4, 4]); ctx.lineWidth = 1; ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#22dd8866"; ctx.font = "9px monospace"; ctx.textAlign = "center";
    ctx.fillText("coast end", ceX, mT + pH + 12);
    // Shade coast region
    const csX = px(result.aPlan.coastStart || result.aPlan.turnTime);
    ctx.fillStyle = "#22dd8806";
    ctx.fillRect(csX, mT, ceX - csX, pH);
  }

  // V1 curve
  ctx.beginPath(); ctx.strokeStyle = "#ffaa22"; ctx.lineWidth = 2.5;
  let started1 = false;
  data.forEach(d => { if (d.fuelV1 <= yMax * 1.5) { if (!started1) { ctx.moveTo(px(d.tInt), py(d.fuelV1)); started1 = true; } else ctx.lineTo(px(d.tInt), py(d.fuelV1)); } });
  ctx.stroke();

  // V2 curve
  ctx.beginPath(); ctx.strokeStyle = "#ff4466"; ctx.lineWidth = 2.5;
  let started2 = false;
  data.forEach(d => { if (d.fuelV2 <= yMax * 1.5) { if (!started2) { ctx.moveTo(px(d.tInt), py(d.fuelV2)); started2 = true; } else ctx.lineTo(px(d.tInt), py(d.fuelV2)); } });
  ctx.stroke();

  // Best intercept dots (computed from scanResults)
  const getMin = (key) => {
    let best = { t: null, f: Infinity };
    data.forEach(d => { if (d[key] < best.f) best = { t: d.tInt, f: d[key] }; });
    return best;
  };
  const minV1 = getMin("fuelV1");
  const minV2 = getMin("fuelV2");

  const drawDot = (best, col, label) => {
    if (!best?.t || !isFinite(best.f)) return;
    if (best.f <= yMax) {
      const cx = px(best.t), cy = py(best.f);
      ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI * 2);
      ctx.fillStyle = col; ctx.fill();
      ctx.strokeStyle = "#060d18"; ctx.lineWidth = 2; ctx.stroke();
      ctx.fillStyle = col + "cc"; ctx.font = "9px monospace"; ctx.textAlign = "left";
      ctx.fillText(`${label} min Δ=${best.f.toFixed(4)}`, cx + 10, cy + 3);
    }
  };
  drawDot(minV1, "#ffaa22", "V1");
  drawDot(minV2, "#ff4466", "V2");

  // Chosen intercept marker for the active variant (if any)
  if (result?.intercept?.t && isFinite(result.intercept.fuel) && result.intercept.fuel <= yMax) {
    const cx = px(result.intercept.t), cy = py(result.intercept.fuel);
    ctx.beginPath(); ctx.arc(cx, cy, 6, 0, Math.PI * 2);
    ctx.strokeStyle = "#e0f0ffcc"; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = "#e0f0ff"; ctx.font = "bold 9px monospace"; ctx.textAlign = "left";
    ctx.fillText("chosen", cx + 10, cy - 8);

    ctx.beginPath(); ctx.moveTo(cx, mT); ctx.lineTo(cx, mT + pH);
    ctx.strokeStyle = "#e0f0ff33"; ctx.setLineDash([3, 3]); ctx.lineWidth = 1; ctx.stroke(); ctx.setLineDash([]);
  }

  // Axis labels

  ctx.fillStyle = "#7788aa"; ctx.font = "12px monospace"; ctx.textAlign = "center";
  ctx.fillText("Interception time (t.u.)", mL + pW / 2, H - 8);
  ctx.save(); ctx.translate(16, mT + pH / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText("Δ required by B", 0, 0); ctx.restore();

  // Legend
  ctx.font = "12px monospace"; ctx.textAlign = "left";
  const ly = mT + 16;
  ctx.fillStyle = "#ffaa22"; ctx.fillRect(mL + 12, ly - 2, 20, 3);
  ctx.fillText("V1: boarding (match pos + vel)", mL + 38, ly + 2);
  ctx.fillStyle = "#ff4466"; ctx.fillRect(mL + 12, ly + 16, 20, 3);
  ctx.fillText("V2: shooting (match pos only)", mL + 38, ly + 20);

  // Title
  ctx.fillStyle = "#99aabb"; ctx.font = "bold 13px monospace"; ctx.textAlign = "left";
  ctx.fillText("B's minimum fuel cost to intercept A at each moment", mL, mT - 18);
  ctx.fillStyle = "#55667788"; ctx.font = "10px monospace";
  ctx.fillText("Where curve dips below Δ_B, pursuer B has enough fuel to intercept", mL, mT - 4);
}

// ─── Main Component ───
export default function SpaceChaseSimulator() {
  const [preset, setPreset] = useState("Balanced duel");
  const [scenario, setScenario] = useState("Auto (game)");
  const [shipA, setShipA] = useState(PRESETS["Balanced duel"].A);
  const [shipB, setShipB] = useState(PRESETS["Balanced duel"].B);
  const [baseZ, setBaseZ] = useState(PRESETS["Balanced duel"].Z);
  const [activeVariant, setActiveVariant] = useState(2);
  const [viewMode, setViewMode] = useState("both");
  const mapRef = useRef(null);
  const fuelRef = useRef(null);

  const applyPreset = (name) => {
    setShipA(PRESETS[name].A); setShipB(PRESETS[name].B); setBaseZ(PRESETS[name].Z); setPreset(name);
  };

  // Explicit primitive dependencies so React always recomputes when any slider changes
  const result = useMemo(() => {
    try {
      const A = { pos: [shipA.x, shipA.y], vel: [shipA.vx, shipA.vy], P: shipA.P, mWet: shipA.mWet, mDry: shipA.mDry };
      const B = { pos: [shipB.x, shipB.y], vel: [shipB.vx, shipB.vy], P: shipB.P, mWet: shipB.mWet, mDry: shipB.mDry };
      const Z = [baseZ.x, baseZ.y];
      const mode = SCENARIO_MODES[scenario]?.mode || "auto";
      return {
        v1: solveScenario(A, B, Z, 1, mode),
        v2: solveScenario(A, B, Z, 2, mode),
      };
    } catch { return null; }
  }, [shipA.x, shipA.y, shipA.vx, shipA.vy, shipA.P, shipA.mWet, shipA.mDry,
      shipB.x, shipB.y, shipB.vx, shipB.vy, shipB.P, shipB.mWet, shipB.mDry,
      baseZ.x, baseZ.y, activeVariant, scenario]);

  useEffect(() => {
    const activeRes = activeVariant === 1 ? result?.v1 : result?.v2;
    if (viewMode !== "fuel") drawMap(mapRef.current, activeRes, shipA, shipB, baseZ, activeVariant);
    if (viewMode !== "map") drawFuelPlot(fuelRef.current, activeRes);
  });

    const activeRes = result ? (activeVariant === 1 ? result.v1 : result.v2) : null;

  return (
    <div style={{
      fontFamily: "'IBM Plex Mono', 'Fira Code', 'Courier New', monospace",
      background: "linear-gradient(145deg, #050c18 0%, #0a1428 100%)",
      color: "#c0d8f0", minHeight: "100vh",
    }}>
      {/* Header */}
      <div style={{
        padding: "10px 16px", borderBottom: "1px solid #1a2a40",
        background: "rgba(5,10,20,0.95)",
        display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap",
      }}>
        <div>
          <div style={{ fontSize: 15, fontWeight: 800, letterSpacing: 2, color: "#e0f0ff" }}>PURSUIT–EVASION</div>
          <div style={{ fontSize: 9, color: "#4a6680", letterSpacing: 1 }}>VAR-Isp · CONST POWER · FREE SPACE</div>
        </div>
        <div style={{ flex: 1 }} />
        {Object.keys(PRESETS).map(n => (
          <button key={n} onClick={() => applyPreset(n)} style={{
            background: preset === n ? "#4a9eff22" : "transparent",
            border: `1px solid ${preset === n ? "#4a9eff" : "#1a2a40"}`,
            color: preset === n ? "#4a9eff" : "#556677",
            borderRadius: 4, padding: "3px 8px", fontSize: 10, cursor: "pointer", fontFamily: "inherit",
          }}>{n}</button>
        ))}
      
        <span style={{ width: 10 }} />
        {Object.keys(SCENARIO_MODES).map(n => (
          <button key={n} onClick={() => setScenario(n)} style={{
            background: scenario === n ? "#22dd8822" : "transparent",
            border: `1px solid ${scenario === n ? "#22dd88" : "#1a2a40"}`,
            color: scenario === n ? "#22dd88" : "#556677",
            borderRadius: 4, padding: "3px 8px", fontSize: 10, cursor: "pointer", fontFamily: "inherit",
          }}>{n}</button>
        ))}
</div>

      <div style={{ display: "flex", height: "calc(100vh - 46px)" }}>
        {/* Left: Controls + Results */}
        <div style={{
          width: 270, minWidth: 270, overflowY: "auto",
          padding: "10px 12px", borderRight: "1px solid #1a2a40",
          background: "rgba(5,10,20,0.5)",
        }}>
          <ShipPanel title="◆ SHIP A — EVADER" color="#22dd88"
            params={shipA} keys={["vx", "P"]} onChange={setShipA} />
          <ShipPanel title="◆ SHIP B — PURSUER" color="#ff4466"
            params={shipB} keys={["vx", "P"]} onChange={setShipB} />
          <ShipPanel title="⬡ BASE Z" color="#4a9eff"
            params={baseZ} keys={[]} onChange={setBaseZ} />

          {/* Variant toggle */}
          <div style={{ display: "flex", gap: 5, marginBottom: 8 }}>
            {[1, 2].map(v => (
              <button key={v} onClick={() => setActiveVariant(v)} style={{
                flex: 1, padding: "5px 0", fontSize: 10,
                background: activeVariant === v ? (v === 1 ? "#ffaa2222" : "#ff446622") : "transparent",
                border: `1px solid ${activeVariant === v ? (v === 1 ? "#ffaa22" : "#ff4466") : "#1a2a40"}`,
                color: activeVariant === v ? (v === 1 ? "#ffaa22" : "#ff4466") : "#556677",
                borderRadius: 4, cursor: "pointer", fontFamily: "inherit",
              }}>V{v}: {v === 1 ? "Board" : "Shoot"}</button>
            ))}
          </div>

          {/* View toggle */}
          <div style={{ display: "flex", gap: 4, marginBottom: 10 }}>
            {[["map", "Map"], ["fuel", "Fuel Plot"], ["both", "Both"]].map(([k, l]) => (
              <button key={k} onClick={() => setViewMode(k)} style={{
                flex: 1, padding: "4px 0", fontSize: 9,
                background: viewMode === k ? "#1a2a4066" : "transparent",
                border: `1px solid ${viewMode === k ? "#3366aa" : "#1a2a40"}`,
                color: viewMode === k ? "#88aacc" : "#445566",
                borderRadius: 3, cursor: "pointer", fontFamily: "inherit",
              }}>{l}</button>
            ))}
          </div>

          {/* Results */}
          {result && (
            <div style={{
              background: "rgba(10,20,35,0.9)", border: "1px solid #1a2a40",
              borderRadius: 6, padding: 10,
            }}>
              <div style={{ fontSize: 10, fontWeight: 700, color: "#6688aa", marginBottom: 6, letterSpacing: 1 }}>RESULTS</div>
              {!activeRes?.feasibleA ? (
                <div style={{ color: "#ff6644", fontSize: 11 }}>
                  Ship A cannot reach base Z.<br />
                  <span style={{ fontSize: 10, color: "#886644" }}>Insufficient fuel (Δ_A = {activeRes.DeltaA.toFixed(4)})</span>
                </div>
              ) : (
                <>
                  <div style={{ fontSize: 11, marginBottom: 2 }}>
                    <span style={{ color: "#22dd88" }}>A</span> flight time: <b style={{ color: "#e0f0ff" }}>{activeRes.aPlan.T.toFixed(2)}</b> t.u.
                  </div>
                  <div style={{ fontSize: 10, color: "#556677", marginBottom: 6 }}>
                    Δ_A = {activeRes.DeltaA.toFixed(4)} · Δ_B = {activeRes.DeltaB.toFixed(4)}
                  
                  <div style={{ fontSize: 10, color: "#556677", marginBottom: 6 }}>
                    A strategy: <span style={{ color: "#c0d8f0" }}>{activeRes.aPolicy}</span>{" "}
                    · B strategy: <span style={{ color: "#c0d8f0" }}>{activeRes.b?.strategy}</span>
                  </div>
</div>

                  {[1, 2].map(v => {
                    const r = v === 1 ? result.v1 : result.v2;
                    if (!r) return null;
                    const active = activeVariant === v;
                    return (
                      <div key={v} style={{
                        marginBottom: 8, padding: "6px 8px",
                        background: active ? "rgba(255,255,255,0.03)" : "transparent",
                        borderRadius: 4, border: active ? "1px solid #1a2a40" : "1px solid transparent",
                        opacity: active ? 1 : 0.5,
                      }}>
                        <div style={{ fontSize: 11, color: v === 1 ? "#ffaa22" : "#ff4466", fontWeight: 700, marginBottom: 3 }}>
                          V{v}: {v === 1 ? "Boarding" : "Shooting"}
                        </div>
                        <div style={{ fontSize: 10, color: "#8899aa", lineHeight: 1.6 }}>
                          Min-fuel t = {r.b.minFuelTInt?.toFixed(2)} · Δ = {r.b.fuelNeeded.toFixed(4)} ({(r.b.fuelRatio * 100).toFixed(1)}% of Δ_B)<br />                          Plan: A {r.aPolicy} · B {r.b?.strategy}<br />

                          {r.b.wins && r.b.earliestTInt != null
                            ? <>Earliest intercept t = {r.b.earliestTInt.toFixed(2)} ({(r.b.earliestTInt / r.aPlan.T * 100).toFixed(0)}% of A's trip)</>
                            : <>No feasible interception window</>}
                        </div>
                        <div style={{ width: "100%", height: 8, background: "#0a1525", borderRadius: 4, marginTop: 4, overflow: "hidden", position: "relative" }}>
                          <div style={{
                            width: `${Math.min(r.b.fuelRatio * 100, 100)}%`, height: "100%",
                            background: r.b.wins
                              ? `linear-gradient(90deg, ${v === 1 ? "#ffaa22" : "#ff4466"}, ${v === 1 ? "#ff8800" : "#cc2244"})`
                              : "#223344",
                            borderRadius: 4, transition: "width 0.3s",
                          }} />
                        </div>
                        <div style={{
                          fontSize: 13, fontWeight: 800, marginTop: 5,
                          color: r.b.wins ? "#ff4466" : "#22dd88",
                        }}>
                          {r.b.wins ? "⚠ B INTERCEPTS" : "✓ A ESCAPES"}
                        </div>
                      </div>
                    );
                  })}
                </>
              )}
            </div>
          )}

          <div style={{ fontSize: 8, color: "#2a3a4a", marginTop: 10, lineHeight: 1.5 }}>
            Model: constant-P, variable-Isp, gravity-free 2D.<br />
            Optimal a(t) linear per axis (PMP).<br />
            Δ = 1/m_dry − 1/m_wet.<br />
            A chooses direct / detour / escape; B chooses chase vs block.
          </div>
        </div>

        {/* Right: Visualizations */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          {viewMode !== "fuel" && (
            <div style={{ flex: viewMode === "both" ? "0 0 45%" : 1, position: "relative", minHeight: 0 }}>
              <canvas ref={mapRef} style={{ width: "100%", height: "100%", display: "block" }} />
            </div>
          )}
          {viewMode === "both" && (
            <div style={{ height: 1, background: "#1a2a40", flexShrink: 0 }} />
          )}
          {viewMode !== "map" && (
            <div style={{ flex: viewMode === "both" ? "0 0 55%" : 1, minHeight: 0 }}>
              <canvas ref={fuelRef} style={{ width: "100%", height: "100%", display: "block" }} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
