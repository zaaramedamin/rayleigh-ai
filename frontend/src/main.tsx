import "@fontsource/inter/400.css";
import "@fontsource/inter/500.css";
import "@fontsource/inter/600.css";
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/700.css";
import "@fontsource/orbitron/500.css";
import "@fontsource/orbitron/700.css";
import "@fontsource/rajdhani/500.css";
import "@fontsource/rajdhani/600.css";
import "@fontsource/rajdhani/700.css";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { AlertOverlay } from "./components/AlertOverlay";
import { ToastHost } from "./components/ToastHost";
import { SettingsProvider } from "./state/settings";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/hud.css";
import "./styles/animations.css";
import "./styles/futuristic.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <SettingsProvider>
      <App />
      {/* Notices live above every screen, including the lock screen. */}
      <ToastHost />
      <AlertOverlay />
    </SettingsProvider>
  </StrictMode>,
);
