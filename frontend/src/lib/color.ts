/** "#3cc4f0" (or "#3cf") to [r, g, b]. Falls back to the default cyan if the value is not a hex colour. */
export function hexToRgb(hex: string): [number, number, number] {
  const value = hex.trim().replace("#", "");
  const full = value.length === 3 ? value.replace(/./g, "$&$&") : value;
  const n = parseInt(full, 16);
  return Number.isNaN(n) ? [60, 196, 240] : [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
