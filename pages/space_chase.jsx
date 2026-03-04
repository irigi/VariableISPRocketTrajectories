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

// ─── Main solver ───
function solveChase(A, B, baseZ) {
  const DeltaA = 1 / A.mDry - 1 / A.mWet;
  const DeltaB = 1 / B.mDry - 1 / B.mWet;

  const TA = findMinTime(A.P, DeltaA, A.pos, A.vel, baseZ, [0, 0], true);
  if (!isFinite(TA)) return { TA: Infinity, feasibleA: false, DeltaA, DeltaB };

  const N = 500;
  const scanResults = [];
  let bestV1 = { tInt: null, fuelNeeded: Infinity };  // min-fuel point
  let bestV2 = { tInt: null, fuelNeeded: Infinity };
  let earliestV1 = null;  // earliest feasible interception
  let earliestV2 = null;

  for (let i = 1; i <= N; i++) {
    const tInt = (i / N) * TA;
    const stateA = getTrajectory2D(A.pos, A.vel, baseZ, [0, 0], TA, tInt);
    const fuelV1 = fuelCost2D_fixed(B.P, B.pos, B.vel, stateA.pos, stateA.vel, tInt);
    const fuelV2 = fuelCost2D_freeVf(B.P, B.pos, B.vel, stateA.pos, tInt);
    scanResults.push({ tInt, fuelV1, fuelV2 });
    if (fuelV1 < bestV1.fuelNeeded) bestV1 = { tInt, fuelNeeded: fuelV1 };
    if (fuelV2 < bestV2.fuelNeeded) bestV2 = { tInt, fuelNeeded: fuelV2 };
    // Track earliest feasible intercept (B's strategic optimum: catch A ASAP)
    if (fuelV1 <= DeltaB && earliestV1 === null) earliestV1 = { tInt, fuelNeeded: fuelV1 };
    if (fuelV2 <= DeltaB && earliestV2 === null) earliestV2 = { tInt, fuelNeeded: fuelV2 };
  }

  const v1Wins = bestV1.fuelNeeded <= DeltaB;
  const v2Wins = bestV2.fuelNeeded <= DeltaB;

  // For trajectory display: use earliest feasible if B can win, else min-fuel attempt
  const displayV1 = earliestV1 || bestV1;
  const displayV2 = earliestV2 || bestV2;

  const steps = 120;
  const trajectoryA = [];
  for (let i = 0; i <= steps; i++) {
    const t = (i / steps) * TA;
    trajectoryA.push(getTrajectory2D(A.pos, A.vel, baseZ, [0, 0], TA, t).pos);
  }

  const buildTrajB = (tInt, fixed) => {
    if (!tInt || !isFinite(tInt)) return [];
    const stateA = getTrajectory2D(A.pos, A.vel, baseZ, [0, 0], TA, tInt);
    const pts = [];
    for (let i = 0; i <= steps; i++) {
      const t = (i / steps) * tInt;
      if (fixed) {
        const s = getTrajectory2D(B.pos, B.vel, stateA.pos, stateA.vel, tInt, t);
        pts.push(s.pos);
      } else {
        const Lx = stateA.pos[0] - B.pos[0], Ly = stateA.pos[1] - B.pos[1];
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

  return {
    TA, DeltaA, DeltaB, feasibleA: true,
    v1: {
      wins: v1Wins, minFuelTInt: bestV1.tInt, fuelNeeded: bestV1.fuelNeeded, fuelRatio: bestV1.fuelNeeded / DeltaB,
      displayTInt: displayV1.tInt, displayFuel: displayV1.fuelNeeded,
      earliestTInt: earliestV1?.tInt ?? null,
    },
    v2: {
      wins: v2Wins, minFuelTInt: bestV2.tInt, fuelNeeded: bestV2.fuelNeeded, fuelRatio: bestV2.fuelNeeded / DeltaB,
      displayTInt: displayV2.tInt, displayFuel: displayV2.fuelNeeded,
      earliestTInt: earliestV2?.tInt ?? null,
    },
    trajectoryA,
    trajectoryB_v1: buildTrajB(displayV1.tInt, true),
    trajectoryB_v2: buildTrajB(displayV2.tInt, false),
    scanResults,
  };
}

// ─── Presets ───
const PRESETS = {
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

  const trajB = activeVariant === 1 ? result.trajectoryB_v1 : result.trajectoryB_v2;
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
  // B trajectory
  if (trajB?.length > 1) {
    ctx.beginPath(); ctx.strokeStyle = "#ff4466"; ctx.lineWidth = 2; ctx.setLineDash([7, 5]);
    trajB.forEach(([x, y], i) => i === 0 ? ctx.moveTo(tx(x), ty(y)) : ctx.lineTo(tx(x), ty(y)));
    ctx.stroke(); ctx.setLineDash([]);
  }

  // Interception point
  const vRes = activeVariant === 1 ? result.v1 : result.v2;
  if (vRes?.displayTInt && result.feasibleA) {
    const st = getTrajectory2D([shipA.x, shipA.y], [shipA.vx, shipA.vy], [baseZ.x, baseZ.y], [0, 0], result.TA, vRes.displayTInt);
    const ix = tx(st.pos[0]), iy = ty(st.pos[1]);
    ctx.beginPath(); ctx.arc(ix, iy, 7, 0, Math.PI * 2);
    ctx.strokeStyle = vRes.wins ? "#ff4466" : "#445566"; ctx.lineWidth = 2; ctx.stroke();
    if (vRes.wins) {
      ctx.beginPath(); ctx.arc(ix, iy, 12, 0, Math.PI * 2);
      ctx.strokeStyle = "#ff446644"; ctx.lineWidth = 1; ctx.stroke();
      ctx.fillStyle = "#ff446699"; ctx.font = "bold 10px monospace";
      ctx.fillText("intercept t=" + vRes.displayTInt.toFixed(1), ix + 16, iy + 3);
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
  const maxT = result.TA;

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

  // Best intercept dots (min-fuel)
  const drawDot = (vr, col, label) => {
    if (!vr?.minFuelTInt) return;
    if (vr.fuelNeeded <= yMax) {
      const cx = px(vr.minFuelTInt), cy = py(vr.fuelNeeded);
      ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI * 2);
      ctx.fillStyle = col; ctx.fill();
      ctx.strokeStyle = "#060d18"; ctx.lineWidth = 2; ctx.stroke();
      ctx.fillStyle = col + "cc"; ctx.font = "9px monospace"; ctx.textAlign = "left";
      ctx.fillText(`min Δ=${vr.fuelNeeded.toFixed(4)}`, cx + 10, cy + 3);
    }
  };
  drawDot(result.v1, "#ffaa22", "V1");
  drawDot(result.v2, "#ff4466", "V2");

  // Earliest feasible interception markers
  const drawEarliest = (vr, col) => {
    if (!vr?.earliestTInt) return;
    const fuel = result.scanResults.find(d => Math.abs(d.tInt - vr.earliestTInt) < maxT / 400);
    const fuelVal = col === "#ffaa22" ? fuel?.fuelV1 : fuel?.fuelV2;
    if (!fuelVal || fuelVal > yMax) return;
    const cx = px(vr.earliestTInt), cy = py(fuelVal);
    // Diamond marker
    ctx.save(); ctx.translate(cx, cy); ctx.rotate(Math.PI / 4);
    ctx.fillStyle = col; ctx.fillRect(-4, -4, 8, 8);
    ctx.strokeStyle = "#060d18"; ctx.lineWidth = 1.5; ctx.strokeRect(-4, -4, 8, 8);
    ctx.restore();
    // Vertical line to show the interception time
    ctx.beginPath(); ctx.moveTo(cx, mT); ctx.lineTo(cx, mT + pH);
    ctx.strokeStyle = col + "33"; ctx.setLineDash([3, 3]); ctx.lineWidth = 1; ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = col; ctx.font = "bold 9px monospace"; ctx.textAlign = "left";
    ctx.fillText(`earliest t=${vr.earliestTInt.toFixed(1)}`, cx + 10, cy - 8);
  };
  drawEarliest(result.v1, "#ffaa22");
  drawEarliest(result.v2, "#ff4466");

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
      return solveChase(
        { pos: [shipA.x, shipA.y], vel: [shipA.vx, shipA.vy], P: shipA.P, mWet: shipA.mWet, mDry: shipA.mDry },
        { pos: [shipB.x, shipB.y], vel: [shipB.vx, shipB.vy], P: shipB.P, mWet: shipB.mWet, mDry: shipB.mDry },
        [baseZ.x, baseZ.y]
      );
    } catch { return null; }
  }, [shipA.x, shipA.y, shipA.vx, shipA.vy, shipA.P, shipA.mWet, shipA.mDry,
      shipB.x, shipB.y, shipB.vx, shipB.vy, shipB.P, shipB.mWet, shipB.mDry,
      baseZ.x, baseZ.y]);

  useEffect(() => {
    if (viewMode !== "fuel") drawMap(mapRef.current, result, shipA, shipB, baseZ, activeVariant);
    if (viewMode !== "map") drawFuelPlot(fuelRef.current, result);
  });

  const vRes = result ? (activeVariant === 1 ? result.v1 : result.v2) : null;

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
              {!result.feasibleA ? (
                <div style={{ color: "#ff6644", fontSize: 11 }}>
                  Ship A cannot reach base Z.<br />
                  <span style={{ fontSize: 10, color: "#886644" }}>Insufficient fuel (Δ_A = {result.DeltaA.toFixed(4)})</span>
                </div>
              ) : (
                <>
                  <div style={{ fontSize: 11, marginBottom: 2 }}>
                    <span style={{ color: "#22dd88" }}>A</span> flight time: <b style={{ color: "#e0f0ff" }}>{result.TA.toFixed(2)}</b> t.u.
                  </div>
                  <div style={{ fontSize: 10, color: "#556677", marginBottom: 6 }}>
                    Δ_A = {result.DeltaA.toFixed(4)} · Δ_B = {result.DeltaB.toFixed(4)}
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
                          Min-fuel t = {r.minFuelTInt?.toFixed(2)} · Δ = {r.fuelNeeded.toFixed(4)} ({(r.fuelRatio * 100).toFixed(1)}% of Δ_B)<br />
                          {r.wins && r.earliestTInt != null
                            ? <>Earliest intercept t = {r.earliestTInt.toFixed(2)} ({(r.earliestTInt / result.TA * 100).toFixed(0)}% of A's trip)</>
                            : <>No feasible interception window</>}
                        </div>
                        <div style={{ width: "100%", height: 8, background: "#0a1525", borderRadius: 4, marginTop: 4, overflow: "hidden", position: "relative" }}>
                          <div style={{
                            width: `${Math.min(r.fuelRatio * 100, 100)}%`, height: "100%",
                            background: r.wins
                              ? `linear-gradient(90deg, ${v === 1 ? "#ffaa22" : "#ff4466"}, ${v === 1 ? "#ff8800" : "#cc2244"})`
                              : "#223344",
                            borderRadius: 4, transition: "width 0.3s",
                          }} />
                        </div>
                        <div style={{
                          fontSize: 13, fontWeight: 800, marginTop: 5,
                          color: r.wins ? "#ff4466" : "#22dd88",
                        }}>
                          {r.wins ? "⚠ B INTERCEPTS" : "✓ A ESCAPES"}
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
            A takes min-time brachistochrone to Z (v_f=0).
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
