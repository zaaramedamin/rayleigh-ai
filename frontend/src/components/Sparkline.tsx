interface Props {
  /** Latest last. null marks a failed check: the line breaks there and a red tick is drawn. */
  values: Array<number | null>;
  width?: number;
  height?: number;
}

/** A tiny line chart of recent measurements. An empty history draws an empty baseline. */
export function Sparkline({ values, width = 240, height = 56 }: Props) {
  const numbers = values.filter((v): v is number => v !== null);
  const max = Math.max(40, ...numbers) * 1.15;
  const step = values.length > 1 ? width / (values.length - 1) : width;
  const y = (v: number) => height - 4 - (v / max) * (height - 10);

  const segments: string[] = [];
  let open: string[] = [];
  values.forEach((value, i) => {
    if (value === null) {
      if (open.length) segments.push(open.join(" "));
      open = [];
    } else {
      open.push(`${(i * step).toFixed(1)},${y(value).toFixed(1)}`);
    }
  });
  if (open.length) segments.push(open.join(" "));

  return (
    <svg className="spark" viewBox={`0 0 ${width} ${height}`} width="100%" height={height} role="img" aria-label="Recent response times">
      <line x1="0" y1={height - 3} x2={width} y2={height - 3} className="spark-base" />
      {segments.map((points) => (
        <polyline key={points} points={points} className="spark-line" />
      ))}
      {values.map((value, i) =>
        value === null ? <line key={i} x1={i * step} y1={6} x2={i * step} y2={height - 3} className="spark-miss" /> : null,
      )}
      {numbers.length > 0 && values[values.length - 1] !== null && (
        <circle cx={(values.length - 1) * step} cy={y(values[values.length - 1] as number)} r="3" className="spark-dot" />
      )}
    </svg>
  );
}
