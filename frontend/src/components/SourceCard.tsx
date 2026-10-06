import type { SearchResult } from "../api/types";

/** "p. 12" or "pp. 12-14" for a file with pages, "L5-9" (lines) for the others. */
export function locationLabel(item: Pick<SearchResult, "start_line" | "end_line" | "start_page" | "end_page">): string {
  if (item.start_page == null) return `L${item.start_line}-${item.end_line}`;
  return item.end_page == null || item.end_page === item.start_page
    ? `p. ${item.start_page}`
    : `pp. ${item.start_page}-${item.end_page}`;
}

interface Props {
  item: SearchResult;
  marker?: number;
  active?: boolean;
  onClick?: () => void;
}

/** One retrieved note with its provenance: file, heading, page or lines, similarity. */
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
        <span className="source-lines">{locationLabel(item)}</span>
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
