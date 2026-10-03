import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The browser only ever talks to its own origin; in development Vite forwards /api to the
// backend on this machine, so the backend needs no CORS rules.
export default defineConfig({
  plugins: [react()],
  // Fonts stay real files so the strict CSP (font-src self) holds.
  build: { assetsInlineLimit: 0 },
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
  test: { environment: "node" },
});
