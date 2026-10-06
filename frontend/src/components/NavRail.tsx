import { sound } from "../audio/sound";
import { Icon } from "./Icon";
import type { IconName } from "./Icon";

export type ViewId = "home" | "chat" | "knowledge" | "profile" | "modules" | "privacy" | "settings";

export const VIEW_TITLES: Record<ViewId, string> = {
  home: "Home",
  chat: "Chat",
  knowledge: "Knowledge",
  profile: "Profile",
  modules: "Modules",
  privacy: "Privacy",
  settings: "Settings",
};

interface Item {
  id: ViewId;
  icon: IconName;
}

// Grouped by what they are for, so the bar reads in sections.
const GROUPS: Array<{ title: string; items: Item[] }> = [
  {
    title: "MAIN",
    items: [
      { id: "home", icon: "home" },
      { id: "chat", icon: "chat" },
    ],
  },
  {
    title: "YOU",
    items: [
      { id: "knowledge", icon: "knowledge" },
      { id: "profile", icon: "profile" },
    ],
  },
  {
    title: "SYSTEM",
    items: [
      { id: "modules", icon: "modules" },
      { id: "privacy", icon: "privacy" },
      { id: "settings", icon: "settings" },
    ],
  },
];

interface Props {
  view: ViewId;
  onChange: (view: ViewId) => void;
}

export function NavRail({ view, onChange }: Props) {
  return (
    <nav className="nav" aria-label="Main">
      {GROUPS.map((group) => (
        <div key={group.title} className="nav-group" role="group" aria-label={group.title}>
          <span className="nav-title" aria-hidden>
            {group.title}
          </span>
          {group.items.map((item) => (
            <button
              key={item.id}
              className={`nav-item${view === item.id ? " nav-item--on" : ""}`}
              onClick={() => {
                if (view !== item.id) sound.play("click");
                onChange(item.id);
              }}
              aria-current={view === item.id ? "page" : undefined}
            >
              <Icon name={item.icon} />
              <span className="nav-label">{VIEW_TITLES[item.id]}</span>
            </button>
          ))}
        </div>
      ))}
    </nav>
  );
}
