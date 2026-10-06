import { useEffect } from "react";

/** Tells the panel under the pointer where the pointer is, so it can glow there. */
export function usePanelGlow(): void {
  useEffect(() => {
    const onMove = (event: PointerEvent) => {
      const panel = (event.target as Element | null)?.closest?.<HTMLElement>(".hud");
      if (!panel) return;
      const box = panel.getBoundingClientRect();
      panel.style.setProperty("--mx", `${event.clientX - box.left}px`);
      panel.style.setProperty("--my", `${event.clientY - box.top}px`);
    };
    window.addEventListener("pointermove", onMove, { passive: true });
    return () => window.removeEventListener("pointermove", onMove);
  }, []);
}
