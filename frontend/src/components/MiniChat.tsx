import { useEffect, useRef, useState } from "react";
import type { FormEvent, KeyboardEvent } from "react";
import { sound } from "../audio/sound";
import { MAX_CHARS } from "../state/conversation";
import { ChatBubble, SUGGESTIONS } from "./ChatBubble";
import type { Assistant } from "./ChatBubble";
import { MicButton } from "./MicButton";
import { ModeToggle } from "./ModeToggle";
import type { Voice } from "../state/useVoice";

interface Props {
  assistant: Assistant;
  voice: Voice;
  /** Called when a citation is clicked, so the page can show the source card. */
  onShowSources: () => void;
}

/** A small chat in the corner of the page. It collapses to a single button. */
export function MiniChat({ assistant, voice, onShowSources }: Props) {
  const { messages, status, mode, setMode, ask } = assistant;
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState("");
  const maxChars = MAX_CHARS[mode];
  const tooLong = draft.length > maxChars;
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (open) endRef.current?.scrollIntoView({ block: "end" });
  }, [messages, open]);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  const toggle = (value: boolean) => {
    sound.play("click");
    setOpen(value);
  };

  const send = (event?: FormEvent) => {
    event?.preventDefault();
    if (!draft.trim() || tooLong || status === "thinking") return;
    void ask(draft);
    setDraft("");
  };
  const onKey = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  };
  const latestAnswerId = [...messages].reverse().find((m) => m.kind === "answer" || m.kind === "reply")?.id;

  if (!open) {
    return (
      <button className="minichat-pill" onClick={() => toggle(true)} aria-label="Open chat">
        <span className={`minichat-dot${status === "thinking" ? " minichat-dot--busy" : ""}`} aria-hidden />
        <span>{status === "thinking" ? "THINKING..." : voice.state === "listening" ? "LISTENING..." : `ASK ${assistant.name.toUpperCase()}`}</span>
        {messages.length > 0 && <b>{messages.filter((m) => m.kind === "user").length}</b>}
      </button>
    );
  }

  return (
    <section className="minichat" aria-label="Chat">
      <header>
        <span>ASK {assistant.name.toUpperCase()}</span>
        <ModeToggle mode={mode} onChange={setMode} compact />
        <button className="link" onClick={() => toggle(false)} aria-label="Minimise chat">
          minimise
        </button>
      </header>
      <div className="minichat-messages" aria-live="polite">
        {messages.length === 0 && (
          <div className="empty">
            <p>
              {mode === "notes"
                ? "Ask about your notes. Answers cite their sources."
                : "Talk about anything. Switch MY NOTES on to ask your notes."}
            </p>
            <div className="chips">
              {SUGGESTIONS[mode].slice(0, 2).map((s) => (
                <button key={s} className="chip" onClick={() => void ask(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m) => (
          <ChatBubble
            key={m.id}
            message={m}
            assistant={assistant}
            isLatest={m.id === latestAnswerId}
            onShowSources={onShowSources}
          />
        ))}
        <div ref={endRef} />
      </div>
      <form className="minichat-input" onSubmit={send}>
        <textarea
          ref={inputRef}
          value={draft}
          onChange={(e) => setDraft(e.target.value.slice(0, Math.max(maxChars, draft.length)))}
          onKeyDown={onKey}
          rows={1}
          placeholder={mode === "notes" ? "Ask your notes..." : `Talk to ${assistant.name}...`}
          aria-label="Your message"
        />
        <MicButton voice={voice} busy={status === "thinking"} compact />
        <button
          className="btn"
          disabled={!draft.trim() || tooLong || status === "thinking"}
          title={tooLong ? `Too long for a notes question (${draft.length}/${maxChars})` : undefined}
        >
          SEND
        </button>
      </form>
    </section>
  );
}
