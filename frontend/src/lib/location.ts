// Where you are, without asking anyone. The browser knows your time zone (for example
// "Africa/Tunis"); a small built-in table turns the biggest city of each zone into coordinates.
// That is an estimate: the right country, not your street. Exact position is opt-in (pinpoint).

export interface Place {
  lat: number;
  lon: number;
  /** Short name for the marker, e.g. "TUNIS". */
  label: string;
  source: "timezone" | "offset" | "gps";
  /** True when this is a guess from the time zone rather than a measured position. */
  approximate: boolean;
  accuracyMeters?: number;
}

// zone  latitude  longitude: the main city of each time zone.
const ZONES = `
Africa/Abidjan 5.36 -4.01
Africa/Accra 5.60 -0.19
Africa/Addis_Ababa 9.03 38.74
Africa/Algiers 36.75 3.06
Africa/Bamako 12.64 -8.00
Africa/Brazzaville -4.27 15.28
Africa/Cairo 30.04 31.24
Africa/Casablanca 33.57 -7.59
Africa/Dakar 14.72 -17.47
Africa/Dar_es_Salaam -6.79 39.21
Africa/Douala 4.05 9.77
Africa/Harare -17.83 31.05
Africa/Johannesburg -26.20 28.04
Africa/Kampala 0.35 32.58
Africa/Khartoum 15.50 32.56
Africa/Kigali -1.94 30.06
Africa/Kinshasa -4.44 15.27
Africa/Lagos 6.52 3.38
Africa/Libreville 0.42 9.47
Africa/Luanda -8.84 13.23
Africa/Lusaka -15.39 28.32
Africa/Maputo -25.97 32.57
Africa/Mogadishu 2.05 45.32
Africa/Monrovia 6.30 -10.80
Africa/Nairobi -1.29 36.82
Africa/Ndjamena 12.13 15.06
Africa/Niamey 13.51 2.11
Africa/Nouakchott 18.08 -15.98
Africa/Ouagadougou 12.37 -1.52
Africa/Tripoli 32.89 13.19
Africa/Tunis 36.81 10.18
Africa/Windhoek -22.56 17.08
America/Anchorage 61.22 -149.90
America/Argentina/Buenos_Aires -34.60 -58.38
America/Asuncion -25.26 -57.58
America/Bogota 4.71 -74.07
America/Buenos_Aires -34.60 -58.38
America/Caracas 10.48 -66.90
America/Chicago 41.88 -87.63
America/Costa_Rica 9.93 -84.08
America/Denver 39.74 -104.99
America/Edmonton 53.55 -113.49
America/El_Salvador 13.69 -89.19
America/Guatemala 14.63 -90.51
America/Guayaquil -2.19 -79.89
America/Halifax 44.65 -63.57
America/Havana 23.11 -82.37
America/Jamaica 18.00 -76.80
America/La_Paz -16.50 -68.15
America/Lima -12.05 -77.04
America/Los_Angeles 34.05 -118.24
America/Manaus -3.12 -60.02
America/Mexico_City 19.43 -99.13
America/Montevideo -34.90 -56.16
America/New_York 40.71 -74.01
America/Panama 8.98 -79.52
America/Phoenix 33.45 -112.07
America/Puerto_Rico 18.47 -66.10
America/Santiago -33.45 -70.67
America/Santo_Domingo 18.49 -69.93
America/Sao_Paulo -23.55 -46.63
America/St_Johns 47.56 -52.71
America/Toronto 43.65 -79.38
America/Vancouver 49.28 -123.12
America/Winnipeg 49.90 -97.14
Asia/Almaty 43.24 76.89
Asia/Amman 31.95 35.93
Asia/Baghdad 33.31 44.36
Asia/Baku 40.41 49.87
Asia/Bangkok 13.76 100.50
Asia/Beirut 33.89 35.50
Asia/Calcutta 22.57 88.36
Asia/Colombo 6.93 79.86
Asia/Damascus 33.51 36.29
Asia/Dhaka 23.81 90.41
Asia/Dubai 25.20 55.27
Asia/Ho_Chi_Minh 10.82 106.63
Asia/Hong_Kong 22.32 114.17
Asia/Irkutsk 52.29 104.30
Asia/Jakarta -6.20 106.85
Asia/Jerusalem 31.77 35.21
Asia/Kabul 34.53 69.17
Asia/Karachi 24.86 67.01
Asia/Kathmandu 27.72 85.32
Asia/Kolkata 22.57 88.36
Asia/Krasnoyarsk 56.01 92.85
Asia/Kuala_Lumpur 3.14 101.69
Asia/Kuwait 29.38 47.99
Asia/Manila 14.60 120.98
Asia/Muscat 23.59 58.41
Asia/Novosibirsk 55.03 82.92
Asia/Phnom_Penh 11.56 104.92
Asia/Qatar 25.29 51.53
Asia/Rangoon 16.87 96.20
Asia/Riyadh 24.71 46.68
Asia/Saigon 10.82 106.63
Asia/Seoul 37.57 126.98
Asia/Shanghai 31.23 121.47
Asia/Singapore 1.35 103.82
Asia/Taipei 25.03 121.57
Asia/Tashkent 41.30 69.24
Asia/Tbilisi 41.72 44.79
Asia/Tehran 35.69 51.39
Asia/Tokyo 35.68 139.69
Asia/Ulaanbaatar 47.89 106.91
Asia/Vladivostok 43.12 131.89
Asia/Yangon 16.87 96.20
Asia/Yekaterinburg 56.84 60.60
Asia/Yerevan 40.18 44.51
Atlantic/Azores 37.74 -25.67
Atlantic/Bermuda 32.30 -64.78
Atlantic/Canary 28.10 -15.41
Atlantic/Reykjavik 64.15 -21.94
Australia/Adelaide -34.93 138.60
Australia/Brisbane -27.47 153.03
Australia/Darwin -12.46 130.84
Australia/Hobart -42.88 147.33
Australia/Melbourne -37.81 144.96
Australia/Perth -31.95 115.86
Australia/Sydney -33.87 151.21
Europe/Amsterdam 52.37 4.90
Europe/Athens 37.98 23.73
Europe/Belgrade 44.79 20.45
Europe/Berlin 52.52 13.40
Europe/Brussels 50.85 4.35
Europe/Bucharest 44.43 26.10
Europe/Budapest 47.50 19.04
Europe/Copenhagen 55.68 12.57
Europe/Dublin 53.35 -6.26
Europe/Helsinki 60.17 24.94
Europe/Istanbul 41.01 28.98
Europe/Kaliningrad 54.71 20.51
Europe/Kiev 50.45 30.52
Europe/Kyiv 50.45 30.52
Europe/Lisbon 38.72 -9.14
Europe/London 51.51 -0.13
Europe/Luxembourg 49.61 6.13
Europe/Madrid 40.42 -3.70
Europe/Malta 35.90 14.51
Europe/Minsk 53.90 27.57
Europe/Moscow 55.76 37.62
Europe/Oslo 59.91 10.75
Europe/Paris 48.86 2.35
Europe/Prague 50.08 14.44
Europe/Riga 56.95 24.11
Europe/Rome 41.90 12.50
Europe/Samara 53.20 50.15
Europe/Sofia 42.70 23.32
Europe/Stockholm 59.33 18.07
Europe/Tallinn 59.44 24.75
Europe/Vienna 48.21 16.37
Europe/Vilnius 54.69 25.28
Europe/Warsaw 52.23 21.01
Europe/Zagreb 45.81 15.98
Europe/Zurich 47.38 8.54
Indian/Maldives 4.18 73.51
Indian/Mauritius -20.16 57.50
Pacific/Auckland -36.85 174.76
Pacific/Fiji -18.14 178.44
Pacific/Guam 13.44 144.79
Pacific/Honolulu 21.31 -157.86
Pacific/Noumea -22.28 166.46
Pacific/Port_Moresby -9.44 147.18
Pacific/Tahiti -17.53 -149.57
`;

const TABLE = new Map<string, [number, number]>();
for (const line of ZONES.trim().split("\n")) {
  const [zone, lat, lon] = line.trim().split(/\s+/);
  TABLE.set(zone, [Number(lat), Number(lon)]);
}

/** How many time zones the built-in table can place (for tests). */
export const KNOWN_ZONES = [...TABLE.keys()];

function cityOf(zone: string): string {
  return (zone.split("/").pop() ?? zone).replace(/_/g, " ").toUpperCase();
}

/**
 * A place for a time zone name. If the zone is not in the table, the longitude is still known
 * from the clock's offset from UTC (15 degrees per hour); the latitude then is a guess.
 */
export function placeFromTimeZone(zone: string | undefined, offsetMinutes: number): Place {
  const known = zone ? TABLE.get(zone) : undefined;
  if (zone && known) {
    return { lat: known[0], lon: known[1], label: cityOf(zone), source: "timezone", approximate: true };
  }
  const hours = offsetMinutes / 60;
  const sign = hours >= 0 ? "+" : "-";
  return {
    lat: 25,
    lon: Math.max(-180, Math.min(180, hours * 15)),
    label: `UTC${sign}${Math.abs(hours)}`,
    source: "offset",
    approximate: true,
  };
}

/** Your estimated place, from this computer's time zone. Works offline. */
export function estimatePlace(): Place {
  let zone: string | undefined;
  try {
    zone = Intl.DateTimeFormat().resolvedOptions().timeZone;
  } catch {
    zone = undefined;
  }
  return placeFromTimeZone(zone, -new Date().getTimezoneOffset());
}

export class LocationError extends Error {}

/**
 * Ask the browser for your real position. The person is asked first; the browser may contact its
 * own location service to answer, but Reyleight sends the position nowhere and keeps it in memory.
 */
export function pinpoint(): Promise<Place> {
  return new Promise((resolve, reject) => {
    if (!("geolocation" in navigator)) {
      reject(new LocationError("This browser cannot find your position."));
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (position) =>
        resolve({
          lat: position.coords.latitude,
          lon: position.coords.longitude,
          label: "YOUR POSITION",
          source: "gps",
          approximate: false,
          accuracyMeters: Math.round(position.coords.accuracy),
        }),
      (error) =>
        reject(
          new LocationError(
            error.code === error.PERMISSION_DENIED
              ? "Location was blocked. Allow it for this page to pinpoint yourself."
              : error.code === error.TIMEOUT
                ? "Finding your position took too long."
                : "Your position is not available right now.",
          ),
        ),
      { enableHighAccuracy: false, timeout: 15000, maximumAge: 300000 },
    );
  });
}
