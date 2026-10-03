import type { SearchResult } from "../api/types";

interface Props {
  item: SearchResult;
  marker?: number;
  active?: boolean;
  onClick?: () => void;
}

/** One retrieved note with its provenance: file, heading, lines, similarity. */
export function SourceCard({ item, marker, active = false, onClick }: Props) {
  const pct = Math.round(Math.max(0, Math.min(1, item.score)) * 100);
  return (
    <article
      className={`source${active ? " source--active" : ""}${onClick ? " source--click" : ""}`}
      onClick={onClick}
    >
      <header>
        {marker !== undefined && <span className="source-marker">{marker}</span>}
        <strong className="source-file">{item.source}</strong>
        <span className="source-lines">
          L{item.start_line}-{item.end_line}
        </span>
      </header>
      {item.heading_path && <div className="source-path">{item.heading_path}</div>}
      <div className="meter" role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct} aria-label="Similarity">
        <span style={{ width: `${pct}%` }} />
        <em>{pct}%</em>
      </div>
      <p className="source-text">{item.text}</p>
    </article>
  );
}
