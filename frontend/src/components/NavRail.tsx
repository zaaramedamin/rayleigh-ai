export type ViewId = "console" | "knowledge" | "modules" | "privacy" | "settings";

const ITEMS: Array<{ id: ViewId; label: string; glyph: string }> = [
  { id: "console", label: "Console", glyph: "◉" },
  { id: "knowledge", label: "Knowledge", glyph: "▤" },
  { id: "modules", label: "Modules", glyph: "⬡" },
  { id: "privacy", label: "Privacy", glyph: "◈" },
  { id: "settings", label: "Settings", glyph: "⚙" },
];

interface Props {
  view: ViewId;
  onChange: (view: ViewId) => void;
}

export function NavRail({ view, onChange }: Props) {
  return (
    <nav className="nav" aria-label="Main">
      {ITEMS.map((item) => (
        <button
          key={item.id}
          className={`nav-item${view === item.id ? " nav-item--on" : ""}`}
          onClick={() => onChange(item.id)}
          aria-current={view === item.id ? "page" : undefined}
        >
          <span className="nav-glyph" aria-hidden>
            {item.glyph}
          </span>
          <span className="nav-label">{item.label}</span>
        </button>
      ))}
    </nav>
  );
}
