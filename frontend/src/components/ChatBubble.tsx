import { ApiError } from "../api/client";
import type { AskReason, ChatMode } from "../api/types";
import { actionLabel } from "../state/actions";
import type { MessageState, useAssistant } from "../state/useAssistant";
import { AnswerText } from "./AnswerText";

export type Assistant = ReturnType<typeof useAssistant>;

export const SUGGESTIONS: Record<ChatMode, string[]> = {
  general: ["Open the knowledge page", "Give me a status report", "Explain what a vector database is"],
  notes: ["When does my flight to Lisbon leave?", "Why did we choose Qdrant?", "What is the capital of Mars?"],
};

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
    if (error.kind === "not_found") {
      return "BACKEND OUT OF DATE // The running backend does not have this yet. Restart it, then try again.";
    }
  }
  return `ERROR // ${error.message}`;
}

interface Props {
  message: MessageState;
  assistant: Assistant;
  isLatest: boolean;
  /** Where the sources are shown, so the "n sources" link can say so. */
  onShowSources?: () => void;
}

/** One message in a conversation: yours, a general reply, a cited answer, a refusal or an error. */
export function ChatBubble({ message, assistant, isLatest, onShowSources }: Props) {
  const who = assistant.name.toUpperCase();
  switch (message.kind) {
    case "user":
      return (
        <div className="msg msg--user">
          <span className="msg-who">
            YOU {message.spoken && <em>SPOKEN</em>}
          </span>
          <p>{message.text}</p>
        </div>
      );
    case "note":
      // Said by the application itself (the greeting, a status report), not by the model.
      return (
        <div className="msg msg--bot msg--note">
          <span className="msg-who">{who}</span>
          <p>{message.text}</p>
        </div>
      );
    case "pending":
      // The text of a cited answer appears as it is written. Its citations are only checked when it
      // is complete, so until then it is marked as not checked and shows no source buttons.
      return (
        <div className="msg msg--bot">
          <span className="msg-who">
            {who} {message.partial && <em>NOT CHECKED YET</em>}
          </span>
          {message.searchedFor && (
            <p className="searching-for">
              searching for: <b>{message.searchedFor}</b>
            </p>
          )}
          {message.partial ? (
            <p className="streaming">
              {message.partial}
              <span className="caret" aria-hidden />
            </p>
          ) : (
            <p className="thinking">
              {message.mode === "notes" ? "searching your notes" : "thinking"}
              <span className="dots" aria-hidden />
            </p>
          )}
        </div>
      );
    case "error":
      return (
        <div className="msg msg--bot msg--error" role="alert">
          <span className="msg-who">{who}</span>
          <p>{errorText(message.error)}</p>
        </div>
      );
    case "reply": {
      // General chat: the model alone. Nothing here is checked against the notes, so no number
      // in the text is a citation and the label says where the words came from.
      const { response } = message;
      return (
        <div className="msg msg--bot msg--general">
          <span className="msg-who">
            {who} <em>GENERAL</em>
          </span>
          <AnswerText
            text={response.answer}
            sources={[]}
            animate={isLatest}
            activeMarker={null}
            onMarker={() => undefined}
          />
          {message.actions && message.actions.length > 0 && (
            // Everything the assistant did in the application for this message.
            <div className="msg-actions">
              {message.actions.map((action, index) => (
                <span key={index} className="action-tag">
                  {actionLabel(action)}
                </span>
              ))}
            </div>
          )}
          <footer className="msg-meta">
            <span>not from your notes</span>
            <span>{response.model}</span>
            {message.ms > 0 && <span>{(message.ms / 1000).toFixed(1)} s</span>}
            {response.truncated && <span className="warn">cut off at the length limit</span>}
          </footer>
        </div>
      );
    }
    case "answer": {
      const { response } = message;
      const active = assistant.selectedId === message.id;
      return (
        <div className={`msg msg--bot${response.grounded ? "" : " msg--refusal"}`}>
          <span className="msg-who">
            {who} <em>{response.grounded ? "GROUNDED" : "NO ANSWER"}</em>
          </span>
          {response.grounded ? (
            <AnswerText
              text={response.answer}
              sources={response.sources}
              animate={isLatest}
              activeMarker={active ? assistant.selectedMarker : null}
              onMarker={(marker) => {
                assistant.select(message.id, marker);
                onShowSources?.();
              }}
            />
          ) : (
            <>
              <p className="refusal-label">{REFUSAL_LABEL[response.reason]}</p>
              <p>{response.answer}</p>
            </>
          )}
          <footer className="msg-meta">
            <span>
              {response.notes_considered} {response.notes_considered === 1 ? "note" : "notes"} considered
            </span>
            {message.ms > 0 && <span>{(message.ms / 1000).toFixed(1)} s</span>}
            {response.searched_for && (
              <span title="Your follow-up was rewritten into this question before the notes were searched">
                searched for: {response.searched_for}
              </span>
            )}
            {response.sources.length > 0 && (
              <button
                className="link"
                onClick={() => {
                  assistant.select(message.id);
                  onShowSources?.();
                }}
              >
                {response.sources.length} {response.sources.length === 1 ? "source" : "sources"}
              </button>
            )}
          </footer>
        </div>
      );
    }
  }
}
