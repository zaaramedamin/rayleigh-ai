const PATHS = {
  home: "M3 11l9-8 9 8v9a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z",
  chat: "M4 5h16v11H9l-5 4z",
  knowledge: "M5 3h9l5 5v13H5z M14 3v5h5 M8 13h8 M8 17h8",
  modules: "M12 2l8 4.5v9L12 20l-8-4.5v-9z M12 11l8-4.5 M12 11v9 M12 11L4 6.5",
  privacy: "M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z M9 12l2 2 4-4",
  settings: "M4 7h10 M18 7h2 M4 17h2 M10 17h10 M14 5v4 M6 15v4",
  profile: "M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8z M4 21c0-4 4-6 8-6s8 2 8 6",
  alert: "M12 3l10 18H2z M12 10v5 M12 18v.4",
  check: "M5 13l4 4 10-10",
  info: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z M12 11v6 M12 7.5v.4",
  lock: "M6 11h12v9H6z M8 11V8a4 4 0 0 1 8 0v3 M12 15v2",
  sound: "M4 9v6h4l5 4V5L8 9z M16 9a4 4 0 0 1 0 6 M18.5 6.5a8 8 0 0 1 0 11",
  mute: "M4 9v6h4l5 4V5L8 9z M17 9l5 6 M22 9l-5 6",
  mic: "M12 3a3 3 0 0 0-3 3v5a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3z M6 11a6 6 0 0 0 12 0 M12 17v4 M9 21h6",
  memory: "M7 4h10v16H7z M3 8h4 M3 12h4 M3 16h4 M17 8h4 M17 12h4 M17 16h4 M10 9h4v6h-4z",
} as const;

export type IconName = keyof typeof PATHS;

/** Simple line icons drawn inline, so the interface loads no icon font or image. */
export function Icon({ name, size = 22 }: { name: IconName; size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d={PATHS[name]} />
    </svg>
  );
}
