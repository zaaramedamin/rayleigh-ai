import type { ReactNode } from "react";

interface Props {
  title: string;
  tag?: string;
  className?: string;
  children: ReactNode;
}

/** A chamfered, corner-bracketed panel: the building block of the HUD. */
export function HudFrame({ title, tag, className = "", children }: Props) {
  return (
    <section className={`hud ${className}`}>
      <header className="hud-head">
        <h2>{title}</h2>
        {tag && <span className="hud-tag">{tag}</span>}
      </header>
      <div className="hud-body">{children}</div>
    </section>
  );
}
