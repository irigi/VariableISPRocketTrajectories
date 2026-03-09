(() => {
  // mnt/data/space_chase.jsx
  var { useState, useMemo } = React;
  var EPS = 1e-9;
  var DEFAULT_MODE = "boarding";
  var PRESETS = {
    "Balanced duel": {
      A: { x: 0, y: 0, vx: 0, vy: 0, P: 1, mWet: 3, mDry: 1 },
      B: { x: 30, y: 20, vx: 0, vy: 0, P: 1, mWet: 3, mDry: 1 },
      Z: { x: 100, y: 0 }
    },
    "Pursuer advantage": {
      A: { x: 0, y: 0, vx: 0, vy: 0, P: 0.7, mWet: 2.4, mDry: 1 },
      B: { x: 20, y: 10, vx: 0, vy: 0, P: 1.8, mWet: 3.5, mDry: 1 },
      Z: { x: 80, y: 0 }
    },
    "Head start escape": {
      A: { x: 0, y: 0, vx: 2, vy: 0, P: 1, mWet: 3, mDry: 1 },
      B: { x: -40, y: 30, vx: 0, vy: 0, P: 1.4, mWet: 3, mDry: 1 },
      Z: { x: 60, y: 0 }
    },
    "Blocking position": {
      A: { x: 0, y: 0, vx: 0, vy: 0, P: 1, mWet: 3, mDry: 1 },
      B: { x: 50, y: 5, vx: 0, vy: 0, P: 0.8, mWet: 3, mDry: 1 },
      Z: { x: 100, y: 0 }
    },
    "Forced draw demo": {
      A: { x: 20, y: 0, vx: 0, vy: 0, P: 1, mWet: 3, mDry: 1 },
      B: { x: 30, y: 20, vx: 0, vy: 0, P: 1, mWet: 3, mDry: 1 },
      Z: { x: 100, y: 0 }
    }
  };
  function add(a, b) {
    return [a[0] + b[0], a[1] + b[1]];
  }
  function sub(a, b) {
    return [a[0] - b[0], a[1] - b[1]];
  }
  function mul(a, s) {
    return [a[0] * s, a[1] * s];
  }
  function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1];
  }
  function norm2(a) {
    return dot(a, a);
  }
  function norm(a) {
    return Math.sqrt(norm2(a));
  }
  function normalize(a, fallback = [1, 0]) {
    const n = norm(a);
    return n > 1e-12 ? [a[0] / n, a[1] / n] : fallback;
  }
  function clamp(x, lo, hi) {
    return Math.max(lo, Math.min(hi, x));
  }
  function fmt(x, digits = 3) {
    if (!isFinite(x)) return "\u221E";
    const ax = Math.abs(x);
    if ((ax >= 1e3 || ax > 0 && ax < 0.01) && ax !== 0) return x.toExponential(2);
    return x.toFixed(digits);
  }
  function shipState(ship) {
    return { r: [ship.x, ship.y], v: [ship.vx, ship.vy] };
  }
  function deltaBudget(ship) {
    return 1 / ship.mDry - 1 / ship.mWet;
  }
  function lambdaBudget(ship) {
    return 2 * ship.P * deltaBudget(ship);
  }
  function fixedCost(lambdaUnused, r0, v0, rf, vf, T) {
    if (T <= 0) return Infinity;
    const dv = sub(vf, v0);
    const residual = sub(sub(rf, r0), mul(add(v0, vf), 0.5 * T));
    return norm2(dv) / T + 12 * norm2(residual) / (T * T * T);
  }
  function hitCost(r0, v0, rf, T) {
    if (T <= 0) return Infinity;
    const residual = sub(sub(rf, r0), mul(v0, T));
    return 3 * norm2(residual) / (T * T * T);
  }
  function minTimeToState(lambda, r0, v0, rf, vf, mode = "fixed") {
    const costFn = mode === "fixed" ? (T) => fixedCost(lambda, r0, v0, rf, vf, T) : (T) => hitCost(r0, v0, rf, T);
    let lo = 1e-4;
    let hi = 1;
    while (costFn(hi) > lambda && hi < 1e6) hi *= 2;
    if (hi >= 1e6 && costFn(hi) > lambda) return Infinity;
    for (let i = 0; i < 80; i++) {
      const mid = 0.5 * (lo + hi);
      if (costFn(mid) > lambda) lo = mid;
      else hi = mid;
    }
    return 0.5 * (lo + hi);
  }
  function fixedTrajectory(r0, v0, rf, vf, T) {
    const alpha = mul(sub(mul(add(v0, vf), 0.5 * T), sub(rf, r0)), 12 / (T * T * T));
    const beta = sub(mul(sub(vf, v0), 1 / T), mul(alpha, 0.5 * T));
    return {
      kind: "fixed",
      T,
      r0,
      v0,
      rf,
      vf,
      alpha,
      beta,
      lambdaUsed: fixedCost(0, r0, v0, rf, vf, T),
      stateAt(t) {
        const tt = clamp(t, 0, T);
        const pos = add(add(add(r0, mul(v0, tt)), mul(beta, 0.5 * tt * tt)), mul(alpha, tt * tt * tt / 6));
        const vel = add(add(v0, mul(beta, tt)), mul(alpha, 0.5 * tt * tt));
        const acc = add(beta, mul(alpha, tt));
        return { pos, vel, acc };
      },
      costUntil(t) {
        const tt = clamp(t, 0, T);
        const a2 = norm2(alpha);
        const b2 = norm2(beta);
        const ab = dot(alpha, beta);
        return a2 * tt * tt * tt / 3 + ab * tt * tt + b2 * tt;
      }
    };
  }
  function freePositionTrajectory(r0, v0, rf, T) {
    const K = mul(sub(sub(rf, r0), mul(v0, T)), 3 / (T * T * T));
    const vf = add(v0, mul(K, 0.5 * T * T));
    return {
      kind: "free",
      T,
      r0,
      v0,
      rf,
      vf,
      K,
      lambdaUsed: hitCost(r0, v0, rf, T),
      stateAt(t) {
        const tt = clamp(t, 0, T);
        const pos = add(add(r0, mul(v0, tt)), mul(K, 0.5 * T * tt * tt - tt * tt * tt / 6));
        const vel = add(v0, mul(K, T * tt - 0.5 * tt * tt));
        const acc = mul(K, T - tt);
        return { pos, vel, acc };
      },
      costUntil(t) {
        const tt = clamp(t, 0, T);
        return norm2(K) * (T * T * tt - T * tt * tt + tt * tt * tt / 3);
      }
    };
  }
  function goldenSectionMin(fn, a, b, iterations = 50) {
    const gr = (Math.sqrt(5) - 1) / 2;
    let c = b - gr * (b - a);
    let d = a + gr * (b - a);
    let fc = fn(c);
    let fd = fn(d);
    for (let i = 0; i < iterations; i++) {
      if (fc < fd) {
        b = d;
        d = c;
        fd = fc;
        c = b - gr * (b - a);
        fc = fn(c);
      } else {
        a = c;
        c = d;
        fc = fd;
        d = a + gr * (b - a);
        fd = fn(d);
      }
    }
    const x = fc < fd ? c : d;
    return { x, fx: Math.min(fc, fd) };
  }
  function minimizeOnInterval(fn, tMin, tMax, samples = 600) {
    let bestT = tMin;
    let bestV = fn(tMin);
    const step = (tMax - tMin) / samples;
    for (let i = 1; i <= samples; i++) {
      const t = tMin + step * i;
      const val = fn(t);
      if (val < bestV) {
        bestV = val;
        bestT = t;
      }
    }
    const left = Math.max(tMin, bestT - step);
    const right = Math.min(tMax, bestT + step);
    const refined = goldenSectionMin(fn, left, right, 60);
    return { t: refined.x, value: refined.fx };
  }
  function earliestFeasible(fn, budget, tMin, tMax, samples = 1e3) {
    let prevT = tMin;
    let prevS = fn(prevT) - budget;
    if (prevS <= 0) return tMin;
    for (let i = 1; i <= samples; i++) {
      const t = tMin + (tMax - tMin) * i / samples;
      const s = fn(t) - budget;
      if (s <= 0) {
        let lo = prevT;
        let hi = t;
        for (let k = 0; k < 60; k++) {
          const mid = 0.5 * (lo + hi);
          if (fn(mid) <= budget) hi = mid;
          else lo = mid;
        }
        return hi;
      }
      prevT = t;
      prevS = s;
    }
    return null;
  }
  function buildPursuerTrajectory(B, targetStateFn, tInt, mode) {
    const { r: rB, v: vB } = shipState(B);
    const target = targetStateFn(tInt);
    if (mode === "boarding") return fixedTrajectory(rB, vB, target.pos, target.vel, tInt);
    return freePositionTrajectory(rB, vB, target.pos, tInt);
  }
  function analyzeIntercept(B, targetStateFn, horizon, mode) {
    const lambdaB = lambdaBudget(B);
    const { r: rB, v: vB } = shipState(B);
    const costFn = (t) => {
      const s = targetStateFn(t);
      return mode === "boarding" ? fixedCost(0, rB, vB, s.pos, s.vel, t) : hitCost(rB, vB, s.pos, t);
    };
    const tMin = Math.max(1e-3, horizon * 1e-4);
    const minRes = minimizeOnInterval(costFn, tMin, horizon, 900);
    const tEarliest = earliestFeasible(costFn, lambdaB, tMin, horizon, 1400);
    const tChosen = tEarliest != null ? tEarliest : minRes.t;
    const trajB = buildPursuerTrajectory(B, targetStateFn, tChosen, mode);
    return {
      wins: tEarliest != null,
      earliestT: tEarliest,
      bestT: minRes.t,
      minCost: minRes.value,
      margin: minRes.value / Math.max(lambdaB, EPS) - 1,
      costFn,
      trajectory: trajB
    };
  }
  function sampleTrajectory(traj, samples = 220) {
    const pts = [];
    for (let i = 0; i <= samples; i++) {
      const t = traj.T * i / samples;
      pts.push(traj.stateAt(t).pos);
    }
    return pts;
  }
  function chooseEscapeEndpoint(A, B, Z, H) {
    const lambdaA = lambdaBudget(A);
    const { r: rA, v: vA } = shipState(A);
    const { r: rB, v: vB } = shipState(B);
    const cA = add(rA, mul(vA, H));
    const cB = add(rB, mul(vB, H));
    const rReach = Math.sqrt(lambdaA / 3) * Math.pow(H, 1.5);
    const awayB = sub(cA, cB);
    const awayZ = sub(cA, [Z.x, Z.y]);
    const dir = normalize(add(awayB, mul(awayZ, 0.35)), normalize(awayB, [1, 0]));
    const endpoint = add(cA, mul(dir, rReach));
    const gap = norm(awayB) + rReach - Math.sqrt(lambdaBudget(B) / 3) * Math.pow(H, 1.5);
    return { endpoint, gap, dir, cA, cB, rReach };
  }
  function candidateEndpoint(A, B, Z, H, theta) {
    const lambdaA = lambdaBudget(A);
    const { r: rA, v: vA } = shipState(A);
    const cA = add(rA, mul(vA, H));
    const rReach = Math.sqrt(lambdaA / 3) * Math.pow(H, 1.5);
    const endpoint = add(cA, [Math.cos(theta) * rReach, Math.sin(theta) * rReach]);
    const { r: rB, v: vB } = shipState(B);
    const cB = add(rB, mul(vB, H));
    const gap = norm(sub(endpoint, cB)) - Math.sqrt(lambdaBudget(B) / 3) * Math.pow(H, 1.5);
    const awayZ = norm(sub(endpoint, [Z.x, Z.y]));
    return { endpoint, gap, awayZ, cA, cB, rReach };
  }
  function searchEvadeOrDelayStrategy(A, B, Z, mode, TA) {
    const lambdaA = lambdaBudget(A);
    const lambdaB = lambdaBudget(B);
    const asymptoticAdvantage = lambdaA > lambdaB + 1e-8;
    const Hmin = Math.max(6, isFinite(TA) ? TA * 0.45 : 6);
    const Hmax = Math.max(180, isFinite(TA) ? TA * 7 : 220);
    const Hs = [];
    for (let i = 0; i < 28; i++) {
      const u = i / 27;
      Hs.push(Hmin * Math.pow(Hmax / Hmin, u));
    }
    const { r: rA, v: vA } = shipState(A);
    const guide = chooseEscapeEndpoint(A, B, Z, Math.max(Hmin, Math.min(Hmax, isFinite(TA) ? TA : 24)));
    let thetaSeed = Math.atan2(guide.dir[1], guide.dir[0]);
    let bestDraw = null;
    let bestDelay = null;
    for (const H of Hs) {
      const dirCount = 28;
      for (let j = 0; j < dirCount; j++) {
        const theta = thetaSeed + 2 * Math.PI * j / dirCount;
        const cand = candidateEndpoint(A, B, Z, H, theta);
        const trajA = freePositionTrajectory(rA, vA, cand.endpoint, H);
        const intercept = analyzeIntercept(B, (t) => trajA.stateAt(t), H, mode);
        const surviveScore = H + 0.25 * Math.max(0, cand.gap) + 0.015 * cand.awayZ;
        const delayScore = (intercept.earliestT ?? 0) + 0.08 * Math.max(0, intercept.minCost / Math.max(lambdaB, EPS) - 1) + 3e-3 * cand.awayZ;
        const entry = { H, theta, trajA, intercept, escape: cand, surviveScore, delayScore };
        if (!intercept.wins && asymptoticAdvantage && cand.gap > 0) {
          if (!bestDraw || surviveScore > bestDraw.surviveScore) bestDraw = entry;
        }
        if (intercept.wins) {
          if (!bestDelay || delayScore > bestDelay.delayScore) bestDelay = entry;
        } else if (!bestDelay || H > (bestDelay.intercept?.earliestT ?? -Infinity)) {
          if (!bestDelay || H > (bestDelay.intercept?.earliestT ?? -Infinity)) bestDelay = entry;
        }
      }
    }
    if (bestDraw) {
      return {
        kind: "draw",
        horizon: bestDraw.H,
        trajectoryA: bestDraw.trajA,
        trajectoryB: bestDraw.intercept.trajectory,
        intercept: bestDraw.intercept,
        note: `${mode === "boarding" ? "A keeps B outside the rendezvous set" : "A stays outside B's hit-reachable disk"} over the displayed horizon, and \u039B_A > \u039B_B gives the long-run escape edge.`
      };
    }
    if (!bestDelay) return null;
    return {
      kind: "delay",
      horizon: bestDelay.intercept.wins ? bestDelay.intercept.earliestT : bestDelay.H,
      displayHorizon: bestDelay.H,
      trajectoryA: bestDelay.trajA,
      trajectoryB: bestDelay.intercept.trajectory,
      intercept: bestDelay.intercept,
      note: bestDelay.intercept.wins ? `${mode === "boarding" ? "A maximizes the earliest feasible rendezvous time" : "A maximizes the earliest feasible hit time"} within the applet's analytic family of break-away trajectories.` : `${mode === "boarding" ? "No rendezvous is found inside the displayed horizon" : "No hit is found inside the displayed horizon"}; this trajectory is the best delay candidate found by the analytic search family.`
    };
  }
  function solveMode(A, B, Z, mode) {
    const lambdaA = lambdaBudget(A);
    const lambdaB = lambdaBudget(B);
    const { r: rA, v: vA } = shipState(A);
    const base = [Z.x, Z.y];
    const TA = minTimeToState(lambdaA, rA, vA, base, [0, 0], "fixed");
    const result = {
      mode,
      lambdaA,
      lambdaB,
      deltaA: deltaBudget(A),
      deltaB: deltaBudget(B),
      TA
    };
    if (!isFinite(TA)) {
      const fallback2 = searchEvadeOrDelayStrategy(A, B, Z, mode, Infinity);
      if (fallback2?.kind === "draw") return { ...result, outcome: "draw", strategy: "evade", ...fallback2 };
      if (fallback2?.kind === "delay") return { ...result, outcome: "b_win", strategy: "delay", ...fallback2 };
      return { ...result, outcome: "b_win", strategy: "no-base", reason: "A cannot reach base Z with zero terminal velocity." };
    }
    const trajBase = fixedTrajectory(rA, vA, base, [0, 0], TA);
    const interceptBase = analyzeIntercept(B, (t) => trajBase.stateAt(t), TA, mode);
    if (!interceptBase.wins) {
      return {
        ...result,
        outcome: "a_win",
        strategy: "base",
        horizon: TA,
        trajectoryA: trajBase,
        trajectoryB: interceptBase.trajectory,
        intercept: interceptBase,
        note: "B cannot reach A's optimal base-transfer trajectory before A reaches Z."
      };
    }
    const fallback = searchEvadeOrDelayStrategy(A, B, Z, mode, TA);
    if (fallback?.kind === "draw") {
      return { ...result, outcome: "draw", strategy: "evade", ...fallback };
    }
    if (fallback?.kind === "delay") {
      return { ...result, outcome: "b_win", strategy: "delay", ...fallback };
    }
    return {
      ...result,
      outcome: "b_win",
      strategy: "base-blocked",
      horizon: TA,
      trajectoryA: trajBase,
      trajectoryB: interceptBase.trajectory,
      intercept: interceptBase,
      note: mode === "boarding" ? "B has a feasible rendezvous with A's fastest route to base, and no better break-away delay trajectory was found by the applet's analytic search family." : "B has a feasible hit on A's fastest route to base, and no better break-away delay trajectory was found by the applet's analytic search family."
    };
  }
  function fuelSeries(solution, samples = 180) {
    if (!solution.trajectoryA) return [];
    const rows = [];
    const H = solution.horizon;
    for (let i = 0; i <= samples; i++) {
      const t = H * i / samples;
      const a = solution.trajectoryA ? solution.trajectoryA.costUntil(t) / Math.max(solution.lambdaA, EPS) : 0;
      const b = solution.trajectoryB ? solution.trajectoryB.costUntil(Math.min(t, solution.trajectoryB.T)) / Math.max(solution.lambdaB, EPS) : 0;
      rows.push({ t, a, b });
    }
    return rows;
  }
  function extentFromSolutions(solutions, A, B, Z) {
    let xs = [A.x, B.x, Z.x];
    let ys = [A.y, B.y, Z.y];
    for (const sol of solutions) {
      if (!sol.trajectoryA) continue;
      for (const p of sampleTrajectory(sol.trajectoryA, 180)) {
        xs.push(p[0]);
        ys.push(p[1]);
      }
      if (sol.trajectoryB) for (const p of sampleTrajectory(sol.trajectoryB, 160)) {
        xs.push(p[0]);
        ys.push(p[1]);
      }
    }
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const padX = Math.max(15, 0.12 * (maxX - minX + 1));
    const padY = Math.max(15, 0.12 * (maxY - minY + 1));
    return { minX: minX - padX, maxX: maxX + padX, minY: minY - padY, maxY: maxY + padY };
  }
  function Slider({ label, value, onChange, min, max, step, unit }) {
    return /* @__PURE__ */ React.createElement("div", { style: { marginBottom: 6 } }, /* @__PURE__ */ React.createElement("div", { style: { display: "flex", justifyContent: "space-between", fontSize: 11, color: "#8da3bc", marginBottom: 1 } }, /* @__PURE__ */ React.createElement("span", null, label), /* @__PURE__ */ React.createElement("span", { style: { color: "#dcecff" } }, fmt(value, step >= 1 ? 0 : 2), unit || "")), /* @__PURE__ */ React.createElement(
      "input",
      {
        type: "range",
        min,
        max,
        step,
        value,
        onChange: (e) => onChange(parseFloat(e.target.value)),
        style: { width: "100%", accentColor: "#4a9eff" }
      }
    ));
  }
  function ShipPanel({ title, color, params, onChange, withDynamics = true }) {
    const set = (k, val) => onChange({ ...params, [k]: val });
    return /* @__PURE__ */ React.createElement("div", { style: { background: "rgba(8,18,32,0.95)", border: `1px solid ${color}40`, borderLeft: `3px solid ${color}`, borderRadius: 8, padding: 10, marginBottom: 10 } }, /* @__PURE__ */ React.createElement("div", { style: { color, fontSize: 13, fontWeight: 700, letterSpacing: 1, marginBottom: 6 } }, title), /* @__PURE__ */ React.createElement("div", { style: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 } }, /* @__PURE__ */ React.createElement(Slider, { label: "x\u2080", value: params.x, onChange: (v) => set("x", v), min: -100, max: 200, step: 1 }), /* @__PURE__ */ React.createElement(Slider, { label: "y\u2080", value: params.y, onChange: (v) => set("y", v), min: -100, max: 100, step: 1 }), withDynamics && /* @__PURE__ */ React.createElement(Slider, { label: "vx\u2080", value: params.vx, onChange: (v) => set("vx", v), min: -5, max: 5, step: 0.1 }), withDynamics && /* @__PURE__ */ React.createElement(Slider, { label: "vy\u2080", value: params.vy, onChange: (v) => set("vy", v), min: -5, max: 5, step: 0.1 })), withDynamics && /* @__PURE__ */ React.createElement(React.Fragment, null, /* @__PURE__ */ React.createElement(Slider, { label: "Power P", value: params.P, onChange: (v) => set("P", v), min: 0.1, max: 5, step: 0.1, unit: " GW" }), /* @__PURE__ */ React.createElement("div", { style: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 } }, /* @__PURE__ */ React.createElement(Slider, { label: "m_wet", value: params.mWet, onChange: (v) => set("mWet", v), min: 1.1, max: 10, step: 0.1, unit: " kt" }), /* @__PURE__ */ React.createElement(Slider, { label: "m_dry", value: params.mDry, onChange: (v) => set("mDry", Math.min(v, params.mWet - 0.1)), min: 0.5, max: Math.max(0.6, params.mWet - 0.1), step: 0.1, unit: " kt" })), /* @__PURE__ */ React.createElement("div", { style: { fontSize: 10, color: "#6f859d" } }, "\u0394 = ", fmt(deltaBudget(params), 4), " \xB7 \u039B = ", fmt(lambdaBudget(params), 4))));
  }
  function OutcomeCard({ title, color, solution }) {
    const label = solution.outcome === "a_win" ? "A reaches base" : solution.outcome === "draw" ? "Forced draw / evasion" : solution.strategy === "delay" ? "B intercepts after A delays" : "B intercepts";
    return /* @__PURE__ */ React.createElement("div", { style: { background: "rgba(8,18,32,0.95)", border: `1px solid ${color}40`, borderLeft: `3px solid ${color}`, borderRadius: 8, padding: 10, marginBottom: 10 } }, /* @__PURE__ */ React.createElement("div", { style: { color, fontSize: 13, fontWeight: 700, marginBottom: 6 } }, title), /* @__PURE__ */ React.createElement("div", { style: { color, fontWeight: 700, marginBottom: 6 } }, label), /* @__PURE__ */ React.createElement("div", { style: { fontSize: 11, color: "#9eb4ca", lineHeight: 1.5 } }, "A strategy: ", solution.strategy, /* @__PURE__ */ React.createElement("br", null), isFinite(solution.TA) ? /* @__PURE__ */ React.createElement(React.Fragment, null, "A minimum base time: t = ", fmt(solution.TA, 2), /* @__PURE__ */ React.createElement("br", null)) : /* @__PURE__ */ React.createElement(React.Fragment, null, "A cannot complete the Z transfer.", /* @__PURE__ */ React.createElement("br", null)), solution.intercept && /* @__PURE__ */ React.createElement(React.Fragment, null, "B best intercept cost ratio: ", fmt(solution.intercept.minCost / Math.max(solution.lambdaB, EPS), 3), /* @__PURE__ */ React.createElement("br", null)), solution.intercept?.earliestT != null && /* @__PURE__ */ React.createElement(React.Fragment, null, "Earliest feasible intercept: t = ", fmt(solution.intercept.earliestT, 2), /* @__PURE__ */ React.createElement("br", null)), solution.strategy === "delay" && solution.displayHorizon && solution.displayHorizon > (solution.intercept?.earliestT || 0) && /* @__PURE__ */ React.createElement(React.Fragment, null, "Displayed search horizon: t = ", fmt(solution.displayHorizon, 2), /* @__PURE__ */ React.createElement("br", null)), /* @__PURE__ */ React.createElement("span", { style: { color: "#7f94ac" } }, solution.note || solution.reason)));
  }
  function MapView({ A, B, Z, solution, selectedMode }) {
    const solutions = solution ? [solution] : [];
    const extent = extentFromSolutions(solutions, A, B, Z);
    const width = 980;
    const height = 420;
    const sx = width / Math.max(extent.maxX - extent.minX, 1);
    const sy = height / Math.max(extent.maxY - extent.minY, 1);
    const toXY = (p) => [(p[0] - extent.minX) * sx, height - (p[1] - extent.minY) * sy];
    const grid = [];
    for (let x = Math.ceil(extent.minX / 10) * 10; x <= extent.maxX; x += 10) {
      const [gx] = toXY([x, 0]);
      grid.push(/* @__PURE__ */ React.createElement("line", { key: `vx${x}`, x1: gx, y1: 0, x2: gx, y2: height, stroke: "rgba(74,158,255,0.08)", strokeWidth: "1" }));
    }
    for (let y = Math.ceil(extent.minY / 10) * 10; y <= extent.maxY; y += 10) {
      const [, gy] = toXY([0, y]);
      grid.push(/* @__PURE__ */ React.createElement("line", { key: `hy${y}`, x1: 0, y1: gy, x2: width, y2: gy, stroke: "rgba(74,158,255,0.08)", strokeWidth: "1" }));
    }
    const styles = {
      boarding: { a: "#20e3a2", b: "#ffb01f", dash: "" },
      shooting: { a: "#19c2ff", b: "#ff4d7a", dash: "7 5" }
    };
    const captureMarkers = solutions.flatMap((sol) => {
      if (!sol.trajectoryA || !sol.intercept?.wins) return [];
      const tCapture = sol.intercept.earliestT ?? sol.trajectoryB?.T;
      if (!(tCapture > 0)) return [];
      const [x, y] = toXY(sol.trajectoryA.stateAt(tCapture).pos);
      return [{ mode: sol.mode, x, y }];
    });
    return /* @__PURE__ */ React.createElement("div", { style: { background: "#031224", border: "1px solid #14304d", borderRadius: 8, overflow: "hidden" } }, /* @__PURE__ */ React.createElement("svg", { viewBox: `0 0 ${width} ${height}`, style: { width: "100%", height: 420, display: "block", background: "linear-gradient(180deg,#04121f,#02101e)" } }, grid, solutions.map((sol) => {
      const st = styles[sol.mode];
      if (!sol.trajectoryA) return null;
      const pathA = sampleTrajectory(sol.trajectoryA, 240).map((p, i) => {
        const [x, y] = toXY(p);
        return `${i === 0 ? "M" : "L"}${x},${y}`;
      }).join(" ");
      const pathB = sol.trajectoryB ? sampleTrajectory(sol.trajectoryB, 200).map((p, i) => {
        const [x, y] = toXY(p);
        return `${i === 0 ? "M" : "L"}${x},${y}`;
      }).join(" ") : "";
      const opacity = sol.mode === selectedMode ? 1 : 0.6;
      return /* @__PURE__ */ React.createElement("g", { key: sol.mode, opacity }, /* @__PURE__ */ React.createElement("path", { d: pathA, fill: "none", stroke: st.a, strokeWidth: sol.mode === selectedMode ? 3 : 2.1, strokeDasharray: st.dash }), pathB && /* @__PURE__ */ React.createElement("path", { d: pathB, fill: "none", stroke: st.b, strokeWidth: sol.mode === selectedMode ? 2.6 : 2, strokeDasharray: st.dash }));
    }), (() => {
      const [x, y] = toXY([A.x, A.y]);
      return /* @__PURE__ */ React.createElement("g", null, /* @__PURE__ */ React.createElement("circle", { cx: x, cy: y, r: 7, fill: "#20e3a2" }), /* @__PURE__ */ React.createElement("text", { x: x + 10, y: y + 4, fill: "#20e3a2", fontSize: "14", fontWeight: "700" }, "A"));
    })(), (() => {
      const [x, y] = toXY([B.x, B.y]);
      return /* @__PURE__ */ React.createElement("g", null, /* @__PURE__ */ React.createElement("circle", { cx: x, cy: y, r: 7, fill: "#ff4d7a" }), /* @__PURE__ */ React.createElement("text", { x: x + 10, y: y + 4, fill: "#ff4d7a", fontSize: "14", fontWeight: "700" }, "B"));
    })(), captureMarkers.map(({ mode, x, y }) => /* @__PURE__ */ React.createElement("g", { key: `capture-${mode}`, opacity: mode === selectedMode ? 1 : 0.7 }, /* @__PURE__ */ React.createElement("circle", { cx: x, cy: y, r: 7, fill: "none", stroke: "#ff3b30", strokeWidth: "1.5" }), /* @__PURE__ */ React.createElement("circle", { cx: x, cy: y, r: 3, fill: "#ff3b30" }), /* @__PURE__ */ React.createElement("line", { x1: x - 10, y1: y, x2: x - 4, y2: y, stroke: "#ff3b30", strokeWidth: "1.5", strokeLinecap: "round" }), /* @__PURE__ */ React.createElement("line", { x1: x + 4, y1: y, x2: x + 10, y2: y, stroke: "#ff3b30", strokeWidth: "1.5", strokeLinecap: "round" }), /* @__PURE__ */ React.createElement("line", { x1: x, y1: y - 10, x2: x, y2: y - 4, stroke: "#ff3b30", strokeWidth: "1.5", strokeLinecap: "round" }), /* @__PURE__ */ React.createElement("line", { x1: x, y1: y + 4, x2: x, y2: y + 10, stroke: "#ff3b30", strokeWidth: "1.5", strokeLinecap: "round" }))), (() => {
      const [x, y] = toXY([Z.x, Z.y]);
      return /* @__PURE__ */ React.createElement("g", null, /* @__PURE__ */ React.createElement("polygon", { points: `${x},${y - 13} ${x + 11},${y - 6.5} ${x + 11},${y + 6.5} ${x},${y + 13} ${x - 11},${y + 6.5} ${x - 11},${y - 6.5}`, fill: "none", stroke: "#4a9eff", strokeWidth: "2" }), /* @__PURE__ */ React.createElement("text", { x: x + 14, y: y + 4, fill: "#4a9eff", fontSize: "14", fontWeight: "700" }, "Z"));
    })()), /* @__PURE__ */ React.createElement("div", { style: { padding: "10px 14px", borderTop: "1px solid #14304d", color: "#9ab0c7", fontSize: 12 } }, selectedMode === "boarding" ? "Showing only the boarding solution." : "Showing only the shooting solution."));
  }
  function FuelPlot({ boarding, shooting, selectedMode }) {
    const rows1 = fuelSeries(boarding, 180);
    const rows2 = fuelSeries(shooting, 180);
    const width = 980;
    const height = 300;
    const maxT = Math.max(boarding.horizon || 1, shooting.horizon || 1, 1);
    const padL = 50, padR = 12, padT = 12, padB = 28;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;
    const x = (t) => padL + plotW * t / maxT;
    const y = (f) => padT + plotH * (1 - clamp(f, 0, 1));
    const line = (rows, key) => rows.map((r, i) => `${i === 0 ? "M" : "L"}${x(r.t)},${y(r[key])}`).join(" ");
    const ys = [0, 0.25, 0.5, 0.75, 1];
    return /* @__PURE__ */ React.createElement("div", { style: { background: "#031224", border: "1px solid #14304d", borderRadius: 8, overflow: "hidden" } }, /* @__PURE__ */ React.createElement("svg", { viewBox: `0 0 ${width} ${height}`, style: { width: "100%", height: 300, display: "block" } }, ys.map((f) => /* @__PURE__ */ React.createElement("line", { key: f, x1: padL, y1: y(f), x2: width - padR, y2: y(f), stroke: "rgba(74,158,255,0.08)" })), [0, 0.2, 0.4, 0.6, 0.8, 1].map((u) => /* @__PURE__ */ React.createElement("line", { key: u, x1: x(maxT * u), y1: padT, x2: x(maxT * u), y2: height - padB, stroke: "rgba(74,158,255,0.08)" })), /* @__PURE__ */ React.createElement("path", { d: line(rows1, "a"), fill: "none", stroke: "#20e3a2", strokeWidth: selectedMode === "boarding" ? 3 : 2.2 }), /* @__PURE__ */ React.createElement("path", { d: line(rows1, "b"), fill: "none", stroke: "#ffb01f", strokeWidth: selectedMode === "boarding" ? 2.8 : 2 }), /* @__PURE__ */ React.createElement("path", { d: line(rows2, "a"), fill: "none", stroke: "#19c2ff", strokeWidth: selectedMode === "shooting" ? 3 : 2.2, strokeDasharray: "7 5" }), /* @__PURE__ */ React.createElement("path", { d: line(rows2, "b"), fill: "none", stroke: "#ff4d7a", strokeWidth: selectedMode === "shooting" ? 2.8 : 2, strokeDasharray: "7 5" }), ys.map((f) => /* @__PURE__ */ React.createElement("text", { key: `yt${f}`, x: 8, y: y(f) + 4, fill: "#6c8299", fontSize: "11" }, fmt(f, 2))), /* @__PURE__ */ React.createElement("text", { x: width / 2 - 30, y: height - 6, fill: "#7f95ab", fontSize: "12" }, "Simulation time (analytic horizon)"), /* @__PURE__ */ React.createElement("text", { transform: `translate(14 ${height / 2 + 32}) rotate(-90)`, fill: "#7f95ab", fontSize: "12" }, "Fuel fraction spent")), /* @__PURE__ */ React.createElement("div", { style: { padding: "10px 14px", borderTop: "1px solid #14304d", color: "#9ab0c7", fontSize: 12 } }, "V1 A fuel ", /* @__PURE__ */ React.createElement("span", { style: { color: "#20e3a2", fontWeight: 700 } }, "\u25A0"), " \xB7 V1 B fuel ", /* @__PURE__ */ React.createElement("span", { style: { color: "#ffb01f", fontWeight: 700 } }, "\u25A0"), " \xB7 V2 A fuel ", /* @__PURE__ */ React.createElement("span", { style: { color: "#19c2ff", fontWeight: 700 } }, "\u25A0"), " \xB7 V2 B fuel ", /* @__PURE__ */ React.createElement("span", { style: { color: "#ff4d7a", fontWeight: 700 } }, "\u25A0")));
  }
  function SpaceChaseSimulator() {
    const [presetName, setPresetName] = useState("Balanced duel");
    const [A, setA] = useState(PRESETS["Balanced duel"].A);
    const [B, setB] = useState(PRESETS["Balanced duel"].B);
    const [Z, setZ] = useState(PRESETS["Balanced duel"].Z);
    const [mode, setMode] = useState(DEFAULT_MODE);
    const [view, setView] = useState("both");
    const boarding = useMemo(() => solveMode(A, B, Z, "boarding"), [A, B, Z]);
    const shooting = useMemo(() => solveMode(A, B, Z, "shooting"), [A, B, Z]);
    const current = mode === "boarding" ? boarding : shooting;
    const applyPreset = (name) => {
      const p = PRESETS[name];
      setPresetName(name);
      setA({ ...p.A });
      setB({ ...p.B });
      setZ({ ...p.Z });
    };
    const buttonStyle = (active) => ({
      padding: "8px 12px",
      borderRadius: 6,
      border: `1px solid ${active ? "#4a9eff" : "#1b3653"}`,
      background: active ? "rgba(74,158,255,0.10)" : "transparent",
      color: active ? "#dcecff" : "#8ea4bc",
      cursor: "pointer",
      fontFamily: "inherit",
      fontSize: 12
    });
    return /* @__PURE__ */ React.createElement("div", { style: { height: "100%", minHeight: 700, background: "#020c18", color: "#c8dbee", fontFamily: "IBM Plex Mono, Fira Code, monospace", display: "grid", gridTemplateRows: "auto 1fr" } }, /* @__PURE__ */ React.createElement("div", { style: { padding: "14px 16px", borderBottom: "1px solid #17314b", display: "flex", justifyContent: "space-between", gap: 16, alignItems: "flex-start" } }, /* @__PURE__ */ React.createElement("div", null, /* @__PURE__ */ React.createElement("div", { style: { fontSize: 18, fontWeight: 800, letterSpacing: 1.5, color: "#eef7ff" } }, "PURSUIT\u2013EVASION"), /* @__PURE__ */ React.createElement("div", { style: { fontSize: 12, color: "#4a9eff", letterSpacing: 1.2 } }, "ANALYTIC FREE-SPACE GAME \xB7 VAR-Isp \xB7 CONST POWER")), /* @__PURE__ */ React.createElement("div", { style: { display: "flex", flexWrap: "wrap", gap: 8, justifyContent: "flex-end" } }, Object.keys(PRESETS).map((name) => /* @__PURE__ */ React.createElement("button", { key: name, style: buttonStyle(presetName === name), onClick: () => applyPreset(name) }, name)))), /* @__PURE__ */ React.createElement("div", { style: { display: "grid", gridTemplateColumns: "300px 1fr", minHeight: 0 } }, /* @__PURE__ */ React.createElement("div", { style: { padding: 12, borderRight: "1px solid #17314b", overflowY: "auto" } }, /* @__PURE__ */ React.createElement(ShipPanel, { title: "\u25C6 SHIP A \u2014 EVADER", color: "#20e3a2", params: A, onChange: setA }), /* @__PURE__ */ React.createElement(ShipPanel, { title: "\u25C6 SHIP B \u2014 PURSUER", color: "#ff4d7a", params: B, onChange: setB }), /* @__PURE__ */ React.createElement(ShipPanel, { title: "\u25C7 BASE Z", color: "#4a9eff", params: Z, onChange: setZ, withDynamics: false }), /* @__PURE__ */ React.createElement("div", { style: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 10 } }, /* @__PURE__ */ React.createElement("button", { style: buttonStyle(mode === "boarding"), onClick: () => setMode("boarding") }, "V1: Boarding"), /* @__PURE__ */ React.createElement("button", { style: buttonStyle(mode === "shooting"), onClick: () => setMode("shooting") }, "V2: Shooting")), /* @__PURE__ */ React.createElement("div", { style: { display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8, marginBottom: 10 } }, /* @__PURE__ */ React.createElement("button", { style: buttonStyle(view === "map"), onClick: () => setView("map") }, "Map"), /* @__PURE__ */ React.createElement("button", { style: buttonStyle(view === "fuel"), onClick: () => setView("fuel") }, "Fuel"), /* @__PURE__ */ React.createElement("button", { style: buttonStyle(view === "both"), onClick: () => setView("both") }, "Both")), /* @__PURE__ */ React.createElement(OutcomeCard, { title: "V1 \u2014 BOARDING", color: "#ffb01f", solution: boarding }), /* @__PURE__ */ React.createElement(OutcomeCard, { title: "V2 \u2014 SHOOTING", color: "#19c2ff", solution: shooting }), /* @__PURE__ */ React.createElement("div", { style: { fontSize: 11, color: "#758ca4", lineHeight: 1.6, padding: "2px 4px" } }, "The applet no longer integrates a chase step-by-step. It builds A and B trajectories from the closed-form transfer equations, then solves scalar time searches for base runs, interception feasibility, and in the losing branch a delay-maximizing search over analytic break-away trajectories.")), /* @__PURE__ */ React.createElement("div", { style: { padding: 12, overflow: "auto" } }, /* @__PURE__ */ React.createElement("div", { style: { marginBottom: 10, fontSize: 13, color: current.outcome === "a_win" ? "#20e3a2" : current.outcome === "draw" ? "#d1e6ff" : "#ffb8c5", fontWeight: 700 } }, mode === "boarding" ? "V1 \u2022 Boarding" : "V2 \u2022 Shooting", " \u2014 ", current.outcome === "a_win" ? "A commits to Z successfully" : current.outcome === "draw" ? "A breaks away and forces a draw" : current.strategy === "delay" ? "A cannot escape, but delays capture" : "B can intercept"), (view === "map" || view === "both") && /* @__PURE__ */ React.createElement(MapView, { A, B, Z, solution: current, selectedMode: mode }), (view === "fuel" || view === "both") && /* @__PURE__ */ React.createElement("div", { style: { marginTop: 12 } }, /* @__PURE__ */ React.createElement(FuelPlot, { boarding, shooting, selectedMode: mode })), /* @__PURE__ */ React.createElement("div", { style: { marginTop: 12, fontSize: 12, color: "#7f95ab", lineHeight: 1.7 } }, "\u039B_A = ", fmt(current.lambdaA, 4), " \xB7 \u039B_B = ", fmt(current.lambdaB, 4), " \xB7 A minimum base time ", isFinite(current.TA) ? `t = ${fmt(current.TA, 3)}` : "is infeasible", ". ", current.note || current.reason))));
  }
  window.SpaceChaseSimulator = SpaceChaseSimulator;
  var rootNode = document.getElementById("applet-root");
  if (rootNode) {
    const root = ReactDOM.createRoot(rootNode);
    root.render(React.createElement(SpaceChaseSimulator));
  }
})();
