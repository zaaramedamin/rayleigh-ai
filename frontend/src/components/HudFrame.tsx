import type { ReactNode } from "react";
import { ScrambleText } from "./ScrambleText";

interface Props {
  title: string;
  tag?: string;
  className?: string;
  children: ReactNode;
}

/**
 * The building block of the HUD: an angular, cut-corner panel with a glowing edge. The frame
 * assembles when it appears, the title deciphers itself, and a soft light follows the pointer.
 */
export function HudFrame({ title, tag, className = "", children }: Props) {
  return (
    <section className={`hud ${className}`}>
      <div className="hud-in">
        <header className="hud-head">
          <h2>
            <ScrambleText text={title} />
          </h2>
          {tag && <span className="hud-tag">{tag}</span>}
        </header>
        <div className="hud-body">{children}</div>
      </div>
    </section>
  );
}
