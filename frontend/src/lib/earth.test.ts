import topology from "world-atlas/countries-50m.json";
import { beforeAll, describe, expect, it } from "vitest";
import { sunElevation, subSolarPoint } from "./earth";
import { KNOWN_ZONES, estimatePlace, placeFromTimeZone } from "./location";
import { buildWorld } from "./worldmap";
import type { Topology, World } from "./worldmap";

let world: World;
beforeAll(() => {
  world = buildWorld(topology as unknown as Topology);
});

describe("world map", () => {
  it.each([
    ["Paris", 2.3, 48.9],
    ["Cairo", 31.2, 30],
    ["the Sahara", 10, 25],
    ["Tunis", 10.18, 36.8],
    ["Moscow", 37.6, 55.7],
    ["Delhi", 77.2, 28.6],
    ["Beijing", 116.4, 39.9],
    ["Tokyo", 139.7, 35.7],
    ["Sydney", 151.2, -33.9],
    ["the Amazon", -60, -3],
    ["New York", -74, 40.7],
    ["Johannesburg", 28, -26],
    ["Greenland", -40, 72],
    ["Antarctica", 0, -80],
  ])("%s is land", (_name, lon, lat) => {
    expect(world.isLand(lon, lat)).toBe(true);
  });

  it.each([
    ["the middle of the Pacific", -150, 0],
    ["the middle of the Atlantic", -30, 30],
    ["the Indian Ocean", 80, -20],
    ["the Mediterranean", 18, 35],
    ["the Gulf of Guinea", 0, 0],
    ["the Arctic Ocean", 0, 88],
  ])("%s is water", (_name, lon, lat) => {
    expect(world.isLand(lon, lat)).toBe(false);
  });

  it("has every country and about the real share of land (29%)", () => {
    expect(world.countryCount).toBeGreaterThan(230);
    let land = 0;
    for (let i = 0; i < 20000; i++) {
      const lat = (Math.asin(1 - ((i + 0.5) / 20000) * 2) * 180) / Math.PI;
      const lon = ((i * 137.508) % 360) - 180;
      if (world.isLand(lon, lat)) land++;
    }
    expect(land / 20000).toBeGreaterThan(0.24);
    expect(land / 20000).toBeLessThan(0.36);
  });

  it("draws coastlines and the borders between countries as separate lines", () => {
    expect(world.coast.starts.length).toBeGreaterThan(100);
    expect(world.borders.starts.length).toBeGreaterThan(100);
    expect(world.borders.xyz.length).toBeGreaterThan(10000);
    // Every vertex is a point on the unit sphere.
    for (const lines of [world.coast, world.borders, world.graticule]) {
      for (let i = 0; i < lines.xyz.length; i += 3 * 97) {
        expect(Math.hypot(lines.xyz[i], lines.xyz[i + 1], lines.xyz[i + 2])).toBeCloseTo(1, 4);
      }
    }
  });

  it("has dots only on land", () => {
    expect(world.dots.count).toBeGreaterThan(8000);
    expect(world.dots.count).toBeLessThan(13000);
    for (let i = 0; i < world.dots.count; i += 101) {
      const lat = (Math.asin(world.dots.y[i]) * 180) / Math.PI;
      const lon = (Math.atan2(world.dots.x[i], world.dots.z[i]) * 180) / Math.PI;
      expect(world.isLand(lon, lat)).toBe(true);
    }
  });
});

describe("sun position", () => {
  it("is over the prime meridian at noon UTC and 90 degrees west at 18:00 UTC", () => {
    expect(subSolarPoint(new Date(Date.UTC(2026, 2, 20, 12, 0, 0))).lon).toBeCloseTo(0, 5);
    expect(subSolarPoint(new Date(Date.UTC(2026, 2, 20, 18, 0, 0))).lon).toBeCloseTo(-90, 5);
  });

  it("is near the equator at the equinox and near the tropics at the solstices", () => {
    expect(Math.abs(subSolarPoint(new Date(Date.UTC(2026, 2, 20, 12))).lat)).toBeLessThan(3);
    expect(subSolarPoint(new Date(Date.UTC(2026, 5, 21, 12))).lat).toBeGreaterThan(22);
    expect(subSolarPoint(new Date(Date.UTC(2026, 11, 21, 12))).lat).toBeLessThan(-22);
  });

  it("is high at local noon and below the horizon at local midnight", () => {
    const noonInTunis = new Date(Date.UTC(2026, 2, 20, 11, 20));
    expect(sunElevation(10.18, 36.8, noonInTunis)).toBeGreaterThan(45);
    expect(sunElevation(10.18, 36.8, new Date(Date.UTC(2026, 2, 20, 23, 20)))).toBeLessThan(-45);
  });
});

describe("estimating where you are", () => {
  it("places well-known time zones in the right city", () => {
    const tunis = placeFromTimeZone("Africa/Tunis", 60);
    expect(tunis).toMatchObject({ label: "TUNIS", source: "timezone", approximate: true });
    expect(tunis.lat).toBeCloseTo(36.81, 1);
    expect(tunis.lon).toBeCloseTo(10.18, 1);
    expect(placeFromTimeZone("America/Argentina/Buenos_Aires", -180).label).toBe("BUENOS AIRES");
  });

  it("falls back to the clock's offset when the zone is not in the table", () => {
    const place = placeFromTimeZone("Nowhere/Unknown", 120);
    expect(place.source).toBe("offset");
    expect(place.lon).toBe(30);
    expect(place.label).toBe("UTC+2");
  });

  it("keeps every table entry on the planet and on land or coast", () => {
    expect(KNOWN_ZONES.length).toBeGreaterThan(150);
    for (const zone of KNOWN_ZONES) {
      const place = placeFromTimeZone(zone, 0);
      expect(place.lat).toBeGreaterThanOrEqual(-90);
      expect(place.lat).toBeLessThanOrEqual(90);
      expect(place.lon).toBeGreaterThanOrEqual(-180);
      expect(place.lon).toBeLessThanOrEqual(180);
    }
  });

  it("puts every table city on land (a typo would drop a marker into the sea)", () => {
    // Small islands (Malta, Bermuda, Maldives...) are too small for a half-degree map.
    const islands = new Set(["Europe/Malta", "Atlantic/Bermuda", "Indian/Maldives", "Indian/Mauritius", "Pacific/Fiji", "Pacific/Guam", "Pacific/Noumea", "Pacific/Tahiti", "Atlantic/Azores", "Atlantic/Canary", "Pacific/Honolulu", "America/Puerto_Rico", "America/Jamaica", "America/Santo_Domingo", "America/Havana", "Asia/Singapore", "Asia/Hong_Kong", "Asia/Manila", "Asia/Colombo", "Pacific/Port_Moresby"]);
    const wet = KNOWN_ZONES.filter((zone) => {
      if (islands.has(zone)) return false;
      const { lat, lon } = placeFromTimeZone(zone, 0);
      // Coastal cities may fall in a neighbouring water cell: accept land within one cell.
      return ![0, 0.5, -0.5].some((dx) => [0, 0.5, -0.5].some((dy) => world.isLand(lon + dx, lat + dy)));
    });
    expect(wet).toEqual([]);
  });

  it("always gives some place for this computer", () => {
    const place = estimatePlace();
    expect(Number.isFinite(place.lat) && Number.isFinite(place.lon)).toBe(true);
  });
});
