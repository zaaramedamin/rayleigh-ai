import { useId } from "react";

export type ReactorState = "idle" | "boot" | "thinking" | "listening" | "speaking" | "offline";

interface Props {
  state?: ReactorState;
  size?: number;
}

const SEGMENTS = 10;

/** The arc reactor. Pure SVG + CSS: its rings spin faster while the assistant is thinking. */
export function ArcReactor({ state = "idle", size = 220 }: Props) {
  const glow = useId();
  const segments = Array.from({ length: SEGMENTS }, (_, i) => (360 / SEGMENTS) * i);
  const ticks = Array.from({ length: 60 }, (_, i) => i * 6);

  return (
    <svg
      className="reactor"
      data-state={state}
      width={size}
      height={size}
      viewBox="0 0 200 200"
      role="img"
      aria-label={`Arc reactor, ${state}`}
    >
      <defs>
        <radialGradient id={glow}>
          <stop offset="0%" stopColor="var(--core)" stopOpacity="0.95" />
          <stop offset="55%" stopColor="var(--primary)" stopOpacity="0.35" />
          <stop offset="100%" stopColor="var(--primary)" stopOpacity="0" />
        </radialGradient>
      </defs>

      <circle cx="100" cy="100" r="96" fill={`url(#${glow})`} opacity="0.25" className="reactor-halo" />

      <g className="reactor-ring reactor-ring--outer">
        <circle cx="100" cy="100" r="92" className="reactor-line" strokeDasharray="2 6" />
        <circle cx="100" cy="100" r="86" className="reactor-line reactor-line--bold" strokeDasharray="90 40 20 40" />
      </g>

      <g className="reactor-ring reactor-ring--ticks">
        {ticks.map((deg) => (
          <line
            key={deg}
            x1="100"
            y1="22"
            x2="100"
            y2={deg % 30 === 0 ? 14 : 18}
            className="reactor-line"
            transform={`rotate(${deg} 100 100)`}
          />
        ))}
      </g>

      <g className="reactor-ring reactor-ring--coil">
        {segments.map((deg) => (
          <path
            key={deg}
            d="M92 40 L108 40 L112 62 L88 62 Z"
            className="reactor-coil"
            transform={`rotate(${deg} 100 100)`}
          />
        ))}
        <circle cx="100" cy="100" r="64" className="reactor-line" />
        <circle cx="100" cy="100" r="38" className="reactor-line" />
      </g>

      <g className="reactor-ring reactor-ring--core">
        <path d="M100 74 L123 114 L77 114 Z" className="reactor-tri" />
      </g>

      <circle cx="100" cy="100" r="14" fill={`url(#${glow})`} className="reactor-core" />
      <circle cx="100" cy="100" r="6" className="reactor-dot" />
    </svg>
  );
}
