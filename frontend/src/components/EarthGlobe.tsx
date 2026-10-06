import { useEffect, useRef, useState } from "react";
import { hexToRgb } from "../lib/color";
import { subSolarPoint, sunElevation, unitVector } from "../lib/earth";
import { estimatePlace, pinpoint } from "../lib/location";
import type { Place } from "../lib/location";
import { loadWorld } from "../lib/worldmap";
import type { Polylines, World } from "../lib/worldmap";
import { events } from "../state/events";

type Mode = "focus" | "spin";

const LEVELS = 10;
const SPIN_RATE = 0.075; // radians per second: far faster than the real Earth, so it is visible
const SPIN_TILT = 0.34; // how far the north pole leans toward you while spinning
const SUN_WHILE_SPINNING = normalise([-0.62, 0.3, 0.72]);
const FRAME_MS = 1000 / 30;

function normalise([x, y, z]: number[]): [number, number, number] {
  const length = Math.hypot(x, y, z) || 1;
  return [x / length, y / length, z / length];
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function wrapAngle(a: number): number {
  return Math.atan2(Math.sin(a), Math.cos(a));
}

function formatCoordinates(lat: number, lon: number): string {
  return `${Math.abs(lat).toFixed(2)}° ${lat >= 0 ? "N" : "S"} · ${Math.abs(lon).toFixed(2)}° ${lon >= 0 ? "E" : "W"}`;
}

/** Draw every line in a buffer that faces the viewer, as one path. */
function strokeLines(
  ctx: CanvasRenderingContext2D,
  lines: Polylines,
  rotation: { cs: number; ss: number; ct: number; st: number },
  cx: number,
  cy: number,
  r: number,
): void {
  const { xyz, starts } = lines;
  const { cs, ss, ct, st } = rotation;
  ctx.beginPath();
  for (let line = 0; line < starts.length - 1; line++) {
    let pen = false;
    for (let v = starts[line]; v < starts[line + 1]; v++) {
      const xe = xyz[v * 3];
      const ye = xyz[v * 3 + 1];
      const ze = xyz[v * 3 + 2];
      const x0 = xe * cs + ze * ss;
      const z0 = ze * cs - xe * ss;
      const y1 = ye * ct - z0 * st;
      const z1 = ye * st + z0 * ct;
      if (z1 < 0) {
        pen = false;
        continue;
      }
      if (pen) ctx.lineTo(cx + x0 * r, cy - y1 * r);
      else {
        ctx.moveTo(cx + x0 * r, cy - y1 * r);
        pen = true;
      }
    }
  }
  ctx.stroke();
}

function UtcAndSun({ place }: { place: Place }) {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);
  const elevation = sunElevation(place.lon, place.lat, now);
  return (
    <dl className="core-readout" aria-label="Your position">
      <div>
        <dt>You</dt>
        <dd>
          {place.approximate ? "~ " : ""}
          {place.label}
        </dd>
      </div>
      <div>
        <dt>Position</dt>
        <dd>{formatCoordinates(place.lat, place.lon)}</dd>
      </div>
      <div>
        <dt>Sun</dt>
        <dd>{elevation >= 0 ? `${elevation.toFixed(0)}° above · DAY` : `${Math.abs(elevation).toFixed(0)}° below · NIGHT`}</dd>
      </div>
      <div>
        <dt>UTC</dt>
        <dd>{now.toISOString().slice(11, 19)}</dd>
      </div>
    </dl>
  );
}

interface View {
  spin: number;
  tilt: number;
  /** 0: the sun is where it really is for the centred place; 1: the sun is fixed on screen. */
  sunBlend: number;
  last: number;
  started: boolean;
}

/**
 * The Earth as a hologram in the theme colour: coastlines, the borders of all 241 countries, a
 * lat/long grid and a dotted land mass, lit by the sun with a real night side. A marker shows
 * where you are: first estimated from your time zone, exactly if you press PINPOINT ME.
 * The satellites and dial are decoration; the position, borders and day/night are real.
 */
export function EarthGlobe({ animate }: { animate: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [world, setWorld] = useState<World | null>(null);
  const [place, setPlace] = useState<Place>(() => estimatePlace());
  const [mode, setMode] = useState<Mode>("focus");
  const [busy, setBusy] = useState(false);
  const view = useRef<View>({ spin: 0, tilt: SPIN_TILT, sunBlend: 0, last: 0, started: false });
  const sunNow = useRef(subSolarPoint(new Date())).current;

  useEffect(() => {
    let alive = true;
    loadWorld().then(
      (loaded) => alive && setWorld(loaded),
      () => alive && events.notify("error", "COULD NOT LOAD THE WORLD MAP", { key: "map" }),
    );
    return () => {
      alive = false;
    };
  }, []);

  const locate = async () => {
    setBusy(true);
    try {
      const found = await pinpoint();
      setPlace(found);
      setMode("focus");
      events.notify("success", "POSITION FOUND", { detail: `Accurate to about ${found.accuracyMeters} m.` });
    } catch (error) {
      events.notify("warning", "COULD NOT FIND YOUR POSITION", { detail: (error as Error).message });
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context || !world) return;
    const ctx = context;
    const state = view.current;
    const placeVector = unitVector(place.lon, place.lat);
    const sunEarth = unitVector(sunNow.lon, sunNow.lat);
    const levelBuckets = Array.from({ length: LEVELS }, () => new Float32Array(world.dots.count * 2));
    const levelCounts = new Int32Array(LEVELS);

    let width = 0;
    let height = 0;
    const resize = () => {
      const box = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      width = box.width;
      height = box.height;
      canvas.width = Math.max(1, Math.round(width * dpr));
      canvas.height = Math.max(1, Math.round(height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    const observer = new ResizeObserver(() => {
      resize();
      if (!animate) draw(0);
    });
    observer.observe(canvas);
    resize();

    function step(seconds: number, dt: number) {
      const targetSpin = -(place.lon * Math.PI) / 180 + 0.1 * Math.sin(seconds * 0.5);
      const targetTilt = Math.max(-1.1, Math.min(1.1, (place.lat * Math.PI) / 180));
      if (!state.started) {
        state.spin = -(place.lon * Math.PI) / 180;
        state.tilt = targetTilt;
        state.sunBlend = mode === "spin" ? 1 : 0;
        state.started = true;
        return;
      }
      if (mode === "focus") {
        state.spin += wrapAngle(targetSpin - state.spin) * Math.min(1, dt * 2.4);
        state.tilt += (targetTilt - state.tilt) * Math.min(1, dt * 2.4);
        state.sunBlend += (0 - state.sunBlend) * Math.min(1, dt * 2.4);
      } else {
        state.spin += SPIN_RATE * dt;
        state.tilt += (SPIN_TILT - state.tilt) * Math.min(1, dt * 2.4);
        state.sunBlend += (1 - state.sunBlend) * Math.min(1, dt * 2.4);
      }
    }

    function draw(seconds: number) {
      const css = getComputedStyle(canvas!);
      const [pr, pg, pb] = hexToRgb(css.getPropertyValue("--primary"));
      const [ar, ag, ab] = hexToRgb(css.getPropertyValue("--accent"));
      const r = Math.min(width, height) * 0.37;
      const cx = width / 2;
      const cy = height / 2;
      const cs = Math.cos(state.spin);
      const ss = Math.sin(state.spin);
      const ct = Math.cos(state.tilt);
      const st = Math.sin(state.tilt);
      const rotation = { cs, ss, ct, st };
      const primary = (alpha: number) => `rgba(${pr},${pg},${pb},${alpha})`;
      ctx.clearRect(0, 0, width, height);

      // Where the sun is on screen: the real direction for the centred place, or fixed while spinning.
      const xe = sunEarth[0] * cs + sunEarth[2] * ss;
      const ze0 = sunEarth[2] * cs - sunEarth[0] * ss;
      const real: [number, number, number] = [xe, sunEarth[1] * ct - ze0 * st, sunEarth[1] * st + ze0 * ct];
      const sun = normalise([
        real[0] + (SUN_WHILE_SPINNING[0] - real[0]) * state.sunBlend,
        real[1] + (SUN_WHILE_SPINNING[1] - real[1]) * state.sunBlend,
        real[2] + (SUN_WHILE_SPINNING[2] - real[2]) * state.sunBlend,
      ]);

      // Halo.
      const halo = ctx.createRadialGradient(cx, cy, r * 0.95, cx, cy, r * 1.16);
      halo.addColorStop(0, primary(0.34));
      halo.addColorStop(1, primary(0));
      ctx.fillStyle = halo;
      ctx.fillRect(0, 0, width, height);

      // Dial of tick marks and a scan arc.
      const dial = r * 1.2;
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(-seconds * 0.04);
      for (let k = 0; k < 90; k++) {
        const angle = (k / 90) * Math.PI * 2;
        const long = k % 5 === 0;
        ctx.beginPath();
        ctx.moveTo(Math.cos(angle) * dial, Math.sin(angle) * dial);
        ctx.lineTo(Math.cos(angle) * (dial + (long ? 10 : 5)), Math.sin(angle) * (dial + (long ? 10 : 5)));
        ctx.strokeStyle = primary(long ? 0.55 : 0.24);
        ctx.lineWidth = long ? 1.4 : 1;
        ctx.stroke();
      }
      ctx.restore();
      ctx.beginPath();
      ctx.arc(cx, cy, dial - 7, seconds * 0.8 - 1, seconds * 0.8);
      ctx.strokeStyle = primary(0.6);
      ctx.lineWidth = 2;
      ctx.stroke();

      // The planet: a dark glass ball.
      const ball = ctx.createRadialGradient(cx - r * 0.3, cy - r * 0.35, r * 0.1, cx, cy, r);
      ball.addColorStop(0, primary(0.1));
      ball.addColorStop(1, "rgba(2,6,12,0.96)");
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = ball;
      ctx.fill();

      // Lat/long grid, then the land as dots, shaded by the sun.
      ctx.lineWidth = 0.6;
      ctx.strokeStyle = primary(0.16);
      strokeLines(ctx, world!.graticule, rotation, cx, cy, r);

      levelCounts.fill(0);
      const { dots } = world!;
      for (let i = 0; i < dots.count; i++) {
        const xe2 = dots.x[i];
        const ye = dots.y[i];
        const ze = dots.z[i];
        const x0 = xe2 * cs + ze * ss;
        const z0 = ze * cs - xe2 * ss;
        const y1 = ye * ct - z0 * st;
        const z1 = ye * st + z0 * ct;
        if (z1 <= 0) continue;
        const day = smoothstep(-0.12, 0.35, x0 * sun[0] + y1 * sun[1] + z1 * sun[2]);
        const level = Math.min(LEVELS - 1, Math.round(day * (0.5 + 0.5 * Math.sqrt(z1)) * (LEVELS - 1)));
        const n = levelCounts[level]++;
        levelBuckets[level][n * 2] = cx + x0 * r;
        levelBuckets[level][n * 2 + 1] = cy - y1 * r;
      }
      const dot = r * 0.011 + 0.5;
      for (let level = 0; level < LEVELS; level++) {
        const n = levelCounts[level];
        if (!n) continue;
        const t = level / (LEVELS - 1);
        // Brighter toward white on the day side, so the lit land stands out from the night.
        ctx.fillStyle = `rgba(${Math.round(pr + (255 - pr) * t * 0.45)},${Math.round(pg + (255 - pg) * t * 0.45)},${Math.round(pb + (255 - pb) * t * 0.45)},${0.1 + t * 0.7})`;
        const data = levelBuckets[level];
        for (let k = 0; k < n; k++) ctx.fillRect(data[k * 2] - dot / 2, data[k * 2 + 1] - dot / 2, dot, dot);
      }

      // Country borders, then the coastlines on top.
      ctx.lineWidth = 0.7;
      ctx.strokeStyle = primary(0.6);
      strokeLines(ctx, world!.borders, rotation, cx, cy, r);
      ctx.lineWidth = 1.1;
      ctx.strokeStyle = `rgba(${Math.round(pr + (255 - pr) * 0.4)},${Math.round(pg + (255 - pg) * 0.4)},${Math.round(pb + (255 - pb) * 0.4)},0.95)`;
      strokeLines(ctx, world!.coast, rotation, cx, cy, r);

      // The night side: a shadow between the limb and the terminator (the day/night line).
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(Math.atan2(-sun[1], sun[0]));
      const a = r * sun[2];
      ctx.beginPath();
      const samples = 64;
      for (let k = 0; k <= samples; k++) {
        const v = -r + (2 * r * k) / samples;
        const edge = Math.sqrt(Math.max(0, 1 - (v / r) * (v / r)));
        const u = -r * edge;
        if (k === 0) ctx.moveTo(u, v);
        else ctx.lineTo(u, v);
      }
      for (let k = samples; k >= 0; k--) {
        const v = -r + (2 * r * k) / samples;
        ctx.lineTo(-a * Math.sqrt(Math.max(0, 1 - (v / r) * (v / r))), v);
      }
      ctx.closePath();
      ctx.fillStyle = "rgba(1,4,10,0.6)";
      ctx.fill();
      ctx.beginPath();
      for (let k = 0; k <= samples; k++) {
        const v = -r + (2 * r * k) / samples;
        const u = -a * Math.sqrt(Math.max(0, 1 - (v / r) * (v / r)));
        if (k === 0) ctx.moveTo(u, v);
        else ctx.lineTo(u, v);
      }
      ctx.strokeStyle = primary(0.3);
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 5]);
      ctx.stroke();
      ctx.restore();

      // Rim.
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.strokeStyle = primary(0.55);
      ctx.lineWidth = 1.4;
      ctx.stroke();

      // Two satellites on tilted orbits; the part of an orbit behind the planet is hidden.
      const orbits = [
        { radius: 1.1, inclination: 0.9, node: 0.5 + seconds * 0.01, speed: 0.55, phase: 0 },
        { radius: 1.17, inclination: 2.2, node: 2.4, speed: -0.38, phase: 2 },
      ];
      for (const orbit of orbits) {
        const cosN = Math.cos(orbit.node);
        const sinN = Math.sin(orbit.node);
        const cosI = Math.cos(orbit.inclination);
        const sinI = Math.sin(orbit.inclination);
        const at = (angle: number) => {
          const p = Math.cos(angle);
          const q = Math.sin(angle);
          const x = orbit.radius * (p * cosN - q * sinN * cosI);
          const y = orbit.radius * (p * sinN + q * cosN * cosI);
          const z = orbit.radius * q * sinI;
          return { x: cx + x * r, y: cy - y * r, hidden: z < 0 && x * x + y * y < 1 };
        };
        ctx.beginPath();
        let pen = false;
        for (let k = 0; k <= 160; k++) {
          const p = at((k / 160) * Math.PI * 2);
          if (p.hidden) pen = false;
          else if (!pen) {
            ctx.moveTo(p.x, p.y);
            pen = true;
          } else ctx.lineTo(p.x, p.y);
        }
        ctx.strokeStyle = primary(0.28);
        ctx.lineWidth = 1;
        ctx.stroke();
        const angle = orbit.phase + seconds * orbit.speed;
        for (let k = 12; k >= 0; k--) {
          const p = at(angle - k * 0.035 * Math.sign(orbit.speed));
          if (p.hidden) continue;
          ctx.beginPath();
          ctx.arc(p.x, p.y, k === 0 ? 3 : 1.6, 0, Math.PI * 2);
          ctx.fillStyle = k === 0 ? "#ffffff" : primary(0.5 - k * 0.04);
          ctx.fill();
        }
      }

      // YOU ARE HERE.
      const ex = placeVector[0] * cs + placeVector[2] * ss;
      const ez0 = placeVector[2] * cs - placeVector[0] * ss;
      const ey = placeVector[1] * ct - ez0 * st;
      const ez = placeVector[1] * st + ez0 * ct;
      if (ez > 0.02) {
        const px = cx + ex * r;
        const py = cy - ey * r;
        const accent = (alpha: number) => `rgba(${ar},${ag},${ab},${alpha})`;
        for (let k = 0; k < 2; k++) {
          const t = ((seconds * 0.7 + k * 0.5) % 1 + 1) % 1;
          ctx.beginPath();
          ctx.arc(px, py, 4 + t * 26, 0, Math.PI * 2);
          ctx.strokeStyle = accent((1 - t) * 0.85);
          ctx.lineWidth = 1.6;
          ctx.stroke();
        }
        ctx.beginPath();
        ctx.arc(px, py, 4.5, 0, Math.PI * 2);
        ctx.fillStyle = accent(1);
        ctx.shadowColor = accent(1);
        ctx.shadowBlur = 14;
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.strokeStyle = accent(0.95);
        ctx.lineWidth = 1.3;
        for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          ctx.beginPath();
          ctx.moveTo(px + dx * 9, py + dy * 9);
          ctx.lineTo(px + dx * 15, py + dy * 15);
          ctx.stroke();
        }
        // A leader line to a label, on whichever side has more room.
        const toLeft = px > cx;
        const bendX = px + (toLeft ? -34 : 34);
        const bendY = py - 30;
        ctx.beginPath();
        ctx.moveTo(px + (toLeft ? -6 : 6), py - 6);
        ctx.lineTo(bendX, bendY);
        ctx.lineTo(bendX + (toLeft ? -22 : 22), bendY);
        ctx.strokeStyle = accent(0.9);
        ctx.lineWidth = 1.2;
        ctx.stroke();
        const text =
          place.source === "gps" ? `YOU · ±${place.accuracyMeters} m` : `YOU · ${place.label} (approx.)`;
        ctx.font = '600 11px "JetBrains Mono", Consolas, monospace';
        const textWidth = ctx.measureText(text).width;
        const boxX = toLeft ? bendX - 22 - textWidth - 14 : bendX + 22;
        ctx.fillStyle = "rgba(2,6,12,0.88)";
        ctx.fillRect(boxX, bendY - 11, textWidth + 14, 22);
        ctx.strokeStyle = accent(0.8);
        ctx.strokeRect(boxX + 0.5, bendY - 10.5, textWidth + 13, 21);
        ctx.fillStyle = "#ffffff";
        ctx.textBaseline = "middle";
        ctx.fillText(text, boxX + 7, bendY + 1);
      }
    }

    if (!animate) {
      step(0, 0);
      draw(0);
      return () => observer.disconnect();
    }
    let frame = 0;
    let last = 0;
    const start = performance.now();
    const loop = (now: number) => {
      frame = requestAnimationFrame(loop);
      if (document.hidden || now - last < FRAME_MS) return;
      const dt = last ? Math.min(0.1, (now - last) / 1000) : 0;
      last = now;
      const seconds = (now - start) / 1000;
      step(seconds, dt);
      draw(seconds);
    };
    frame = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [animate, world, place, mode, sunNow]);

  return (
    <>
      <canvas ref={canvasRef} className="core-canvas" aria-label="The Earth, with your position marked" role="img" />
      {!world && <p className="core-loading">LOADING THE WORLD MAP...</p>}
      <UtcAndSun place={place} />
      <div className="core-controls">
        <div className="seg" role="group" aria-label="View">
          <button aria-pressed={mode === "focus"} className={mode === "focus" ? "seg--on" : undefined} onClick={() => setMode("focus")}>
            FOCUS ON ME
          </button>
          <button aria-pressed={mode === "spin"} className={mode === "spin" ? "seg--on" : undefined} onClick={() => setMode("spin")}>
            SPIN
          </button>
        </div>
        <button
          className="btn btn--ghost"
          onClick={() => void locate()}
          disabled={busy}
          title="Asks your browser for your position. The browser may contact its own location service; Reyleight sends it nowhere and keeps it only while this page is open."
        >
          {busy ? "LOCATING..." : place.source === "gps" ? "UPDATE POSITION" : "PINPOINT ME"}
        </button>
      </div>
      <p className="core-note">
        {place.source === "gps"
          ? `Your browser's position, accurate to about ${place.accuracyMeters} m`
          : "Position estimated from your time zone · PINPOINT ME for the exact spot"}
      </p>
    </>
  );
}
