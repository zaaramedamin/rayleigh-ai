import type { ChatMode } from "../api/types";
import { sound } from "../audio/sound";
import { Icon } from "./Icon";

interface Props {
  mode: ChatMode;
  onChange: (mode: ChatMode) => void;
  /** A smaller button, for the corner chat. */
  compact?: boolean;
}

/**
 * The switch between talking to the model and asking the notes. Off: the model answers from
 * what it knows. On: answers come only from the notes, with sources.
 */
export function ModeToggle({ mode, onChange, compact = false }: Props) {
  const on = mode === "notes";
  return (
    <button
      type="button"
      className={`mode-toggle${on ? " mode-toggle--on" : ""}${compact ? " mode-toggle--compact" : ""}`}
      aria-pressed={on}
      title={
        on
          ? "On: answers come only from your notes, with sources. Click to talk to the model without your notes."
          : "Off: the model answers from its own knowledge and does not read your notes. Click to search your notes."
      }
      onClick={() => {
        sound.play("click");
        onChange(on ? "general" : "notes");
      }}
    >
      {!compact && <Icon name="knowledge" size={14} />}
      <span>MY NOTES</span>
      <b>{on ? "ON" : "OFF"}</b>
    </button>
  );
}
