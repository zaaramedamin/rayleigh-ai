import { useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import { sound } from "../audio/sound";
import { filterCommands } from "../lib/commands";
import type { Command } from "../lib/commands";

interface Props {
  open: boolean;
  commands: Command[];
  onClose: () => void;
}

/** Ctrl+K: type what you want to do or where you want to go, press Enter. */
export function CommandPalette({ open, commands, onClose }: Props) {
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const activeRef = useRef<HTMLLIElement>(null);
  const results = useMemo(() => filterCommands(commands, query), [commands, query]);

  useEffect(() => {
    if (!open) return;
    setQuery("");
    setActive(0);
    sound.play("click");
    const timer = setTimeout(() => inputRef.current?.focus(), 0);
    return () => clearTimeout(timer);
  }, [open]);

  useEffect(() => {
    setActive(0);
  }, [query]);
  useEffect(() => {
    activeRef.current?.scrollIntoView({ block: "nearest" });
  }, [active, results]);

  if (!open) return null;

  const run = (command: Command | undefined) => {
    if (!command) return;
    sound.play("blip");
    onClose();
    command.run();
  };
  const onKey = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActive((a) => (results.length ? (a + 1) % results.length : 0));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActive((a) => (results.length ? (a - 1 + results.length) % results.length : 0));
    } else if (event.key === "Enter") {
      event.preventDefault();
      run(results[active]);
    } else if (event.key === "Escape") {
      event.preventDefault();
      onClose();
    }
  };

  return (
    <div className="palette-backdrop" onMouseDown={onClose}>
      <div className="palette" role="dialog" aria-modal="true" aria-label="Command palette" onMouseDown={(e) => e.stopPropagation()}>
        <div className="palette-input">
          <span aria-hidden>&gt;_</span>
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onKey}
            placeholder="Type a command or a page..."
            aria-label="Command"
            role="combobox"
            aria-expanded="true"
            aria-controls="palette-list"
            aria-activedescendant={results[active] ? `cmd-${results[active].id}` : undefined}
            autoComplete="off"
            spellCheck={false}
          />
          <kbd>ESC</kbd>
        </div>
        <ul id="palette-list" className="palette-list" role="listbox">
          {results.length === 0 && <li className="palette-empty">Nothing matches "{query}".</li>}
          {results.map((command, index) => (
            <li
              key={command.id}
              id={`cmd-${command.id}`}
              ref={index === active ? activeRef : undefined}
              role="option"
              aria-selected={index === active}
              className={`palette-item${index === active ? " palette-item--on" : ""}`}
              onMouseEnter={() => setActive(index)}
              onClick={() => run(command)}
            >
              <span>{command.label}</span>
              <small>{command.group}</small>
            </li>
          ))}
        </ul>
        <footer className="palette-foot">
          <span>
            <kbd>↑</kbd>
            <kbd>↓</kbd> choose
          </span>
          <span>
            <kbd>ENTER</kbd> run
          </span>
          <span>
            <kbd>CTRL</kbd>
            <kbd>K</kbd> toggle
          </span>
        </footer>
      </div>
    </div>
  );
}
