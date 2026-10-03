import { useEffect, useRef, useState } from "react";
import type { FormEvent, KeyboardEvent } from "react";
import { ApiError } from "../api/client";
import type { AskReason } from "../api/types";
import { AnswerText } from "../components/AnswerText";
import { HudFrame } from "../components/HudFrame";
import { SourceCard } from "../components/SourceCard";
import type { MessageState, useAssistant } from "../state/useAssistant";

type Assistant = ReturnType<typeof useAssistant>;

const MAX_CHARS = 2000;

const SUGGESTIONS = [
  "When does my flight to Lisbon leave?",
  "Why did we choose Qdrant?",
  "What is the capital of Mars?",
];

const REFUSAL_LABEL: Record<AskReason, string> = {
  answered: "GROUNDED",
  no_relevant_notes: "INSUFFICIENT DATA // no note matched well enough",
  model_declined: "INSUFFICIENT DATA // the model declined to guess",
  no_valid_citation: "UNVERIFIED // the answer had no valid citation, so it was withheld",
};

function errorText(error: Error): string {
  if (error instanceof ApiError) {
    if (error.kind === "llm_unavailable") return `LOCAL MODEL UNAVAILABLE // ${error.message}`;
    if (error.kind === "timeout") return `LOCAL MODEL TIMED OUT // ${error.message}`;
    if (error.kind === "offline") return `LINK LOST // ${error.message}`;
  }
  return `ERROR // ${error.message}`;
}

function Bubble({ message, assistant, isLatest }: { message: MessageState; assistant: Assistant; isLatest: boolean }) {
  switch (message.kind) {
    case "user":
      return (
        <div className="msg msg--user">
          <span className="msg-who">YOU</span>
          <p>{message.text}</p>
        </div>
      );
    case "pending":
      return (
        <div className="msg msg--bot">
          <span className="msg-who">REYLEIGHT</span>
          <p className="thinking">
            searching your notes<span className="dots" aria-hidden />
          </p>
        </div>
      );
    case "error":
      return (
        <div className="msg msg--bot msg--error" role="alert">
          <span className="msg-who">REYLEIGHT</span>
          <p>{errorText(message.error)}</p>
        </div>
      );
    case "answer": {
      const { response } = message;
      const active = assistant.selectedId === message.id;
      return (
        <div className={`msg msg--bot${response.grounded ? "" : " msg--refusal"}`}>
          <span className="msg-who">
            REYLEIGHT <em>{response.grounded ? "GROUNDED" : "NO ANSWER"}</em>
          </span>
          {response.grounded ? (
            <AnswerText
              text={response.answer}
              sources={response.sources}
              animate={isLatest}
              activeMarker={active ? assistant.selectedMarker : null}
              onMarker={(marker) => assistant.select(message.id, marker)}
            />
          ) : (
            <>
              <p className="refusal-label">{REFUSAL_LABEL[response.reason]}</p>
              <p>{response.answer}</p>
            </>
          )}
          <footer className="msg-meta">
            <span>{response.notes_considered} {response.notes_considered === 1 ? "note" : "notes"} considered</span>
            <span>{(message.ms / 1000).toFixed(1)} s</span>
            {response.sources.length > 0 && (
              <button className="link" onClick={() => assistant.select(message.id)}>
                {response.sources.length} {response.sources.length === 1 ? "source" : "sources"}
              </button>
            )}
          </footer>
        </div>
      );
    }
  }
}

export function ConsoleView({ assistant }: { assistant: Assistant }) {
  const { messages, status, ask, selectedAnswer, selectedMarker, select, selectedId } = assistant;
  const [draft, setDraft] = useState("");
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  const send = (event?: FormEvent) => {
    event?.preventDefault();
    if (!draft.trim() || status === "thinking") return;
    void ask(draft);
    setDraft("");
  };
  const onKey = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  };

  const latestAnswerId = [...messages].reverse().find((m) => m.kind === "answer")?.id;

  return (
    <div className="console">
      <HudFrame title="CONSOLE" tag="ASK YOUR NOTES" className="console-main">
        <div className="messages" aria-live="polite">
          {messages.length === 0 && (
            <div className="empty">
              <p>Ask anything about your notes. Every answer cites its source, or says it does not know.</p>
              <div className="chips">
                {SUGGESTIONS.map((s) => (
                  <button key={s} className="chip" onClick={() => void ask(s)}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((m) => (
            <Bubble key={m.id} message={m} assistant={assistant} isLatest={m.id === latestAnswerId} />
          ))}
          <div ref={endRef} />
        </div>

        <form className="command" onSubmit={send}>
          <span className="prompt" aria-hidden>
            &gt;_
          </span>
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value.slice(0, MAX_CHARS))}
            onKeyDown={onKey}
            rows={1}
            placeholder="Ask Reyleight..."
            aria-label="Your question"
          />
          <button type="button" className="btn btn--ghost" disabled title="Voice input is planned (roadmap phase 6)">
            ◖ VOICE
          </button>
          <button className="btn" disabled={!draft.trim() || status === "thinking"}>
            SEND
          </button>
        </form>
        <div className="command-hint">
          Enter to send · Shift+Enter for a new line · {draft.length}/{MAX_CHARS}
          {messages.length > 0 && (
            <button className="link" onClick={assistant.clear}>
              clear conversation
            </button>
          )}
        </div>
      </HudFrame>

      <HudFrame title="SOURCES" tag={selectedAnswer ? `${selectedAnswer.response.sources.length} CITED` : "IDLE"} className="console-sources">
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
              : "Cited notes appear here: file, heading, lines and how closely each one matched."}
          </p>
        )}
      </HudFrame>
    </div>
  );
}
