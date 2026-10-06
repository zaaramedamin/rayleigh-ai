// Where the sun is, for the globe's day and night. The map itself is in worldmap.ts.

/** Longitude and latitude (degrees) of the point on Earth with the sun straight overhead. */
export function subSolarPoint(date: Date): { lon: number; lat: number } {
  const dayOfYear =
    (Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()) - Date.UTC(date.getUTCFullYear(), 0, 0)) /
    86400000;
  const declination = -23.44 * Math.cos(((2 * Math.PI) / 365) * (dayOfYear + 10));
  const utcHours = date.getUTCHours() + date.getUTCMinutes() / 60 + date.getUTCSeconds() / 3600;
  let lon = -(utcHours - 12) * 15;
  if (lon > 180) lon -= 360;
  if (lon < -180) lon += 360;
  return { lon, lat: declination };
}

/** A point on the sphere as a vector (x east, y north, z toward longitude 0). */
export function unitVector(lon: number, lat: number): [number, number, number] {
  const lonRad = (lon * Math.PI) / 180;
  const latRad = (lat * Math.PI) / 180;
  return [Math.cos(latRad) * Math.sin(lonRad), Math.sin(latRad), Math.cos(latRad) * Math.cos(lonRad)];
}

/** How high the sun is above the horizon at a place, in degrees (negative: it is night). */
export function sunElevation(lon: number, lat: number, date: Date): number {
  const sun = subSolarPoint(date);
  const [px, py, pz] = unitVector(lon, lat);
  const [sx, sy, sz] = unitVector(sun.lon, sun.lat);
  const dot = Math.max(-1, Math.min(1, px * sx + py * sy + pz * sz));
  return (Math.asin(dot) * 180) / Math.PI;
}
