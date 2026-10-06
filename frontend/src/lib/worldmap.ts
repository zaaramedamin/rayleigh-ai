// Turns the public-domain Natural Earth data (the `world-atlas` package, TopoJSON) into what the
// globe draws: coastlines, country borders, a lat/long grid, and a set of dots that sit on land.
// The data ships inside the app; nothing is fetched from anywhere when it runs.

type Position = [number, number];

interface PolygonGeometry {
  type: "Polygon";
  arcs: number[][];
}
interface MultiPolygonGeometry {
  type: "MultiPolygon";
  arcs: number[][][];
}
type Geometry = PolygonGeometry | MultiPolygonGeometry;

export interface Topology {
  arcs: number[][][];
  transform: { scale: [number, number]; translate: [number, number] };
  objects: {
    countries: { geometries: Geometry[] };
    land: { geometries: Geometry[] };
  };
}

/** Many separate lines in one buffer: line i is xyz[starts[i]*3 .. starts[i+1]*3]. */
export interface Polylines {
  xyz: Float32Array;
  starts: Int32Array;
}

export interface World {
  /** Land according to a half-degree map of the continents. */
  isLand(lon: number, lat: number): boolean;
  coast: Polylines;
  borders: Polylines;
  graticule: Polylines;
  /** Points on land, as unit vectors fixed to the Earth. */
  dots: { count: number; x: Float32Array; y: Float32Array; z: Float32Array };
  countryCount: number;
}

const RESOLUTION = 0.5; // degrees per cell of the land map
const COLUMNS = 360 / RESOLUTION;
const ROWS = 180 / RESOLUTION;
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

function decodeArcs(topology: Topology): Position[][] {
  const [sx, sy] = topology.transform.scale;
  const [tx, ty] = topology.transform.translate;
  return topology.arcs.map((arc) => {
    let x = 0;
    let y = 0;
    return arc.map(([dx, dy]) => {
      x += dx;
      y += dy;
      return [x * sx + tx, y * sy + ty] as Position;
    });
  });
}

/** The points of a ring given as arc indices; a negative index means the arc runs backwards. */
function ringPoints(ring: number[], arcs: Position[][]): Position[] {
  const points: Position[] = [];
  for (const index of ring) {
    const arc = index >= 0 ? arcs[index] : [...arcs[~index]].reverse();
    points.push(...(points.length ? arc.slice(1) : arc));
  }
  return points;
}

function polygonsOf(geometry: Geometry): number[][][] {
  return geometry.type === "Polygon" ? [geometry.arcs] : geometry.arcs;
}

function toVector(lon: number, lat: number): [number, number, number] {
  const lonRad = (lon * Math.PI) / 180;
  const latRad = (lat * Math.PI) / 180;
  return [Math.cos(latRad) * Math.sin(lonRad), Math.sin(latRad), Math.cos(latRad) * Math.cos(lonRad)];
}

/** Fill the land into a grid by scanlines: a cell is land when a row crosses an odd number of edges. */
function landMask(topology: Topology, arcs: Position[][]): Uint8Array {
  const crossings: number[][] = Array.from({ length: ROWS }, () => []);
  for (const geometry of topology.objects.land.geometries) {
    for (const polygon of polygonsOf(geometry)) {
      for (const ring of polygon) {
        const points = ringPoints(ring, arcs);
        for (let i = 0; i < points.length - 1; i++) {
          const x0 = (points[i][0] + 180) / RESOLUTION;
          const y0 = (90 - points[i][1]) / RESOLUTION;
          const x1 = (points[i + 1][0] + 180) / RESOLUTION;
          const y1 = (90 - points[i + 1][1]) / RESOLUTION;
          if (y0 === y1) continue;
          const first = Math.max(0, Math.ceil(Math.min(y0, y1) - 0.5));
          const last = Math.min(ROWS - 1, Math.floor(Math.max(y0, y1) - 0.5));
          for (let row = first; row <= last; row++) {
            const centre = row + 0.5;
            if (y0 <= centre !== y1 <= centre) crossings[row].push(x0 + ((centre - y0) / (y1 - y0)) * (x1 - x0));
          }
        }
      }
    }
  }
  const mask = new Uint8Array(COLUMNS * ROWS);
  crossings.forEach((xs, row) => {
    xs.sort((a, b) => a - b);
    for (let k = 0; k + 1 < xs.length; k += 2) {
      const from = Math.max(0, Math.ceil(xs[k] - 0.5));
      const to = Math.min(COLUMNS - 1, Math.floor(xs[k + 1] - 0.5));
      for (let column = from; column <= to; column++) mask[row * COLUMNS + column] = 1;
    }
  });
  return mask;
}

function pack(lines: Position[][]): Polylines {
  const starts = new Int32Array(lines.length + 1);
  let total = 0;
  lines.forEach((line, i) => {
    starts[i] = total;
    total += line.length;
  });
  starts[lines.length] = total;
  const xyz = new Float32Array(total * 3);
  let offset = 0;
  for (const line of lines) {
    for (const [lon, lat] of line) {
      const [x, y, z] = toVector(lon, lat);
      xyz[offset++] = x;
      xyz[offset++] = y;
      xyz[offset++] = z;
    }
  }
  return { xyz, starts };
}

/** Arcs that only run along the 180th meridian or the South Pole are cuts in the map, not borders. */
function isMapSeam(arc: Position[]): boolean {
  return arc.every(([lon]) => Math.abs(lon) > 179.99) || arc.every(([, lat]) => lat < -89.9);
}

export function buildWorld(topology: Topology, dotCount = 36000): World {
  const arcs = decodeArcs(topology);

  // A country edge used by one country is a coastline; used by two, it is a border between them.
  const uses = new Int32Array(arcs.length);
  for (const geometry of topology.objects.countries.geometries) {
    for (const polygon of polygonsOf(geometry)) {
      for (const ring of polygon) for (const index of ring) uses[index >= 0 ? index : ~index]++;
    }
  }
  const coast: Position[][] = [];
  const borders: Position[][] = [];
  arcs.forEach((arc, i) => {
    if (uses[i] === 0 || isMapSeam(arc)) return;
    (uses[i] === 1 ? coast : borders).push(arc);
  });

  const graticule: Position[][] = [];
  for (let lon = -180; lon < 180; lon += 30) {
    graticule.push(Array.from({ length: 35 }, (_, k) => [lon, -85 + k * 5] as Position));
  }
  for (let lat = -60; lat <= 60; lat += 30) {
    graticule.push(Array.from({ length: 73 }, (_, k) => [-180 + k * 5, lat] as Position));
  }

  const mask = landMask(topology, arcs);
  const isLand = (lon: number, lat: number): boolean => {
    const column = Math.min(COLUMNS - 1, Math.max(0, Math.floor((lon + 180) / RESOLUTION)));
    const row = Math.min(ROWS - 1, Math.max(0, Math.floor((90 - lat) / RESOLUTION)));
    return mask[row * COLUMNS + column] === 1;
  };

  // Dots spread evenly over the sphere (a Fibonacci lattice); keep those on land.
  const x: number[] = [];
  const y: number[] = [];
  const z: number[] = [];
  for (let i = 0; i < dotCount; i++) {
    const sinLat = 1 - ((i + 0.5) / dotCount) * 2;
    const ring = Math.sqrt(1 - sinLat * sinLat);
    const theta = GOLDEN_ANGLE * i;
    const lonRad = Math.atan2(Math.sin(theta), Math.cos(theta));
    const lat = (Math.asin(sinLat) * 180) / Math.PI;
    if (isLand((lonRad * 180) / Math.PI, lat)) {
      x.push(ring * Math.sin(lonRad));
      y.push(sinLat);
      z.push(ring * Math.cos(lonRad));
    }
  }

  return {
    isLand,
    coast: pack(coast),
    borders: pack(borders),
    graticule: pack(graticule),
    dots: { count: x.length, x: Float32Array.from(x), y: Float32Array.from(y), z: Float32Array.from(z) },
    countryCount: topology.objects.countries.geometries.length,
  };
}

let loading: Promise<World> | null = null;

/** Loads and decodes the map once, on first use, as a separate file so the app starts quickly. */
export function loadWorld(): Promise<World> {
  loading ??= import("world-atlas/countries-50m.json").then((module) => buildWorld(module.default as unknown as Topology));
  return loading;
}
