import { useEffect, useRef, useState } from "react";
import type { FormEvent, KeyboardEvent } from "react";
import type { Api } from "../api/types";
import { ChatBubble, SUGGESTIONS } from "../components/ChatBubble";
import type { Assistant } from "../components/ChatBubble";
import { ConversationBar } from "../components/ConversationBar";
import { HudFrame } from "../components/HudFrame";
import { MicButton } from "../components/MicButton";
import { ModeToggle } from "../components/ModeToggle";
import { SourceCard } from "../components/SourceCard";
import { MAX_CHARS } from "../state/conversation";
import type { Voice } from "../state/useVoice";

/** The full-size conversation. The home page has a small version of this in its corner. */
export function ConsoleView({ api, assistant, voice }: { api: Api; assistant: Assistant; voice: Voice }) {
  const { messages, status, mode, setMode, ask, selectedAnswer, selectedIsGeneral, selectedMarker, select, selectedId } =
    assistant;
  const [draft, setDraft] = useState("");
  const endRef = useRef<HTMLDivElement>(null);
  const notes = mode === "notes";
  const maxChars = MAX_CHARS[mode];
  // Only possible after switching to notes with a long draft: say so instead of cutting the text.
  const tooLong = draft.length > maxChars;

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

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

  return (
    <div className="console">
      <HudFrame title="CHAT" tag={notes ? "FROM YOUR NOTES" : "GENERAL"} className="console-main">
        <ConversationBar api={api} assistant={assistant} />
        <div className="messages" aria-live="polite">
          {messages.length === 0 && (
            <div className="empty">
              <p>
                {notes
                  ? "Ask anything about your notes. Every answer cites its source, or says it does not know."
                  : "Talk about anything, or give an order such as \"open the settings\" or \"remember that...\". Type it, or press VOICE and say it. For answers taken only from your notes, with sources, switch MY NOTES on."}
              </p>
              <div className="chips">
                {SUGGESTIONS[mode].map((s) => (
                  <button key={s} className="chip" onClick={() => void ask(s)}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((m) => (
            <ChatBubble key={m.id} message={m} assistant={assistant} isLatest={m.id === latestAnswerId} />
          ))}
          <div ref={endRef} />
        </div>

        <form className="command" onSubmit={send}>
          <span className="prompt" aria-hidden>
            &gt;_
          </span>
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value.slice(0, Math.max(maxChars, draft.length)))}
            onKeyDown={onKey}
            rows={1}
            placeholder={notes ? "Ask your notes..." : `Talk to ${assistant.name}, or give it an order...`}
            aria-label="Your message"
          />
          <ModeToggle mode={mode} onChange={setMode} />
          <MicButton voice={voice} busy={status === "thinking"} />
          {assistant.canStop ? (
            <button type="button" className="btn btn--danger" onClick={assistant.stop}>
              STOP
            </button>
          ) : (
            <button className="btn" disabled={!draft.trim() || tooLong || status === "thinking"}>
              SEND
            </button>
          )}
        </form>
        <div className="command-hint">
          <span>
            Enter to send · Shift+Enter for a new line · Esc stops the voice ·{" "}
            <span className={tooLong ? "bad" : undefined}>
              {draft.length}/{maxChars}
              {tooLong && " (too long for a notes question)"}
            </span>
          </span>
          {messages.length > 0 && (
            <button className="link" onClick={assistant.clear}>
              new conversation
            </button>
          )}
        </div>
      </HudFrame>

      <HudFrame
        title="SOURCES"
        tag={selectedAnswer ? `${selectedAnswer.response.sources.length} CITED` : "IDLE"}
        className="console-sources"
      >
        {selectedAnswer && selectedAnswer.response.sources.length > 0 ? (
          <div className="source-list">
            {selectedAnswer.response.sources.map((s) => (
              <SourceCard
                key={s.citation_id}
                item={s}
                marker={s.marker}
                active={selectedMarker === s.marker}
                onClick={() => selectedId !== null && select(selectedId, s.marker)}
              />
            ))}
          </div>
        ) : (
          <p className="muted">
            {selectedAnswer
              ? "This reply used no notes."
              : selectedIsGeneral
                ? "That was a general reply: your notes were not read, so there is nothing to cite. Switch MY NOTES on for answers with sources."
                : notes
                  ? "Cited notes appear here: file, heading, lines and how closely each one matched."
                  : "General chat does not read your notes. Switch MY NOTES on and the notes behind each answer appear here."}
          </p>
        )}
      </HudFrame>
    </div>
  );
}
