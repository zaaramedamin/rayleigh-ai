import { useEffect } from "react";

const TARGETS = "button, [role=tab], [role=option], a, .chip, .action";

/** A ring of light spreads from the point where a button is pressed. */
export function useClickPing(enabled: boolean): void {
  useEffect(() => {
    if (!enabled) return;
    const onDown = (event: PointerEvent) => {
      const target = (event.target as Element | null)?.closest?.(TARGETS);
      if (!target || (target as HTMLButtonElement).disabled) return;
      const ring = document.createElement("span");
      ring.className = "ping";
      ring.style.left = `${event.clientX}px`;
      ring.style.top = `${event.clientY}px`;
      document.body.appendChild(ring);
      setTimeout(() => ring.remove(), 800);
    };
    window.addEventListener("pointerdown", onDown, { passive: true });
    return () => window.removeEventListener("pointerdown", onDown);
  }, [enabled]);
}
