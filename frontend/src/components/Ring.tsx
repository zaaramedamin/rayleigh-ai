interface Props {
  /** 0 to 1, or null when the value is not known. */
  value: number | null;
  label: string;
  size?: number;
}

/** A circular gauge. An unknown value shows a dashed empty ring rather than a made-up number. */
export function Ring({ value, label, size = 92 }: Props) {
  const radius = 38;
  const circumference = 2 * Math.PI * radius;
  const known = value !== null;
  const shown = known ? Math.max(0, Math.min(1, value)) : 0;
  return (
    <div className="ring" style={{ width: size, height: size }}>
      <svg viewBox="0 0 100 100" width={size} height={size} role="img" aria-label={`${label}: ${known ? Math.round(shown * 100) + " percent" : "unknown"}`}>
        <circle cx="50" cy="50" r={radius} className="ring-track" strokeDasharray={known ? undefined : "3 6"} />
        {known && (
          <circle
            cx="50"
            cy="50"
            r={radius}
            className="ring-value"
            strokeDasharray={circumference}
            strokeDashoffset={circumference * (1 - shown)}
            transform="rotate(-90 50 50)"
          />
        )}
      </svg>
      <span className="ring-text">{known ? `${Math.round(shown * 100)}%` : "--"}</span>
    </div>
  );
}
