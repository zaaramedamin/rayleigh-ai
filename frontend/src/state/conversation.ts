import type {
  AskReason,
  AskResponse,
  AskSource,
  ChatAction,
  ChatMode,
  ChatResponse,
  ChatTurn,
  FeedbackKind,
  NewFeedback,
  NewStoredMessage,
  StoredMessage,
} from "../api/types";
import { parseActions } from "./actions";
import type { MessageState } from "./useAssistant";

/** Longest message in each mode. The backend enforces the same limits. */
export const MAX_CHARS: Record<ChatMode, number> = { general: 8000, notes: 2000 };

/** Earlier turns sent with a general message. The backend trims further, to fit the model. */
const MAX_HISTORY_TURNS = 20;

/**
 * What the model is told about the conversation so far: each question with the reply it got,
 * oldest first. Refusals, errors and questions still waiting are left out, so only real
 * exchanges are sent.
 */
export function historyFor(messages: MessageState[]): ChatTurn[] {
  const turns: ChatTurn[] = [];
  for (const m of messages) {
    if (m.kind === "reply" || (m.kind === "answer" && m.response.grounded)) {
      turns.push({ role: "user", content: m.question }, { role: "assistant", content: m.response.answer });
    }
  }
  return turns.slice(-MAX_HISTORY_TURNS);
}

// --- saving and reopening conversations -----------------------------------------------------------

const REASONS: readonly AskReason[] = ["answered", "no_relevant_notes", "model_declined", "no_valid_citation"];

/**
 * The turns to save for an exchange: what was asked (when there was an asker) and what came back.
 * Only real replies and answers are saved; errors, notices and a question still waiting are not.
 */
export function storedTurns(user: { text: string; spoken?: boolean } | null, reply: MessageState): NewStoredMessage[] {
  if (reply.kind !== "reply" && reply.kind !== "answer") return [];
  const turns: NewStoredMessage[] = [];
  const mode: ChatMode = reply.kind === "answer" ? "notes" : "general";
  if (user) {
    turns.push({ role: "user", mode, content: user.text, ...(user.spoken ? { payload: { spoken: true } } : {}) });
  }
  if (reply.kind === "reply") {
    const { answer, model, truncated, actions } = reply.response;
    turns.push({ role: "assistant", mode, content: answer, payload: { model, truncated, actions: actions ?? [] } });
  } else {
    const { answer, grounded, reason, notes_considered, searched_for, sources } = reply.response;
    turns.push({
      role: "assistant",
      mode,
      content: answer,
      payload: { question: reply.question, grounded, reason, notes_considered, searched_for: searched_for ?? null, sources },
    });
  }
  return turns;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

function isSource(value: unknown): value is AskSource {
  return (
    isRecord(value) &&
    typeof value.citation_id === "string" &&
    typeof value.marker === "number" &&
    typeof value.document_id === "number" &&
    typeof value.source === "string" &&
    typeof value.text === "string"
  );
}

function answerFrom(content: string, payload: Record<string, unknown>): AskResponse {
  const grounded = payload.grounded === true;
  const reason = REASONS.find((r) => r === payload.reason) ?? (grounded ? "answered" : "model_declined");
  return {
    answer: content,
    grounded,
    reason,
    sources: Array.isArray(payload.sources) ? payload.sources.filter(isSource) : [],
    notes_considered: typeof payload.notes_considered === "number" ? payload.notes_considered : 0,
    searched_for: typeof payload.searched_for === "string" ? payload.searched_for : null,
  };
}

/** The saved actions that are plain name-and-text pairs. The interface checks them again before it shows them. */
function actionsFrom(value: unknown): ChatAction[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (!isRecord(item) || typeof item.name !== "string" || !isRecord(item.args)) return [];
    const args = Object.fromEntries(Object.entries(item.args).filter(([, v]) => typeof v === "string")) as Record<string, string>;
    return [{ name: item.name, args }];
  });
}

/**
 * The messages of a stored conversation, as the chat shows them. Ids continue from `firstId`.
 * Saved data is read defensively: a message that is not what was expected still shows its text.
 */
export function restoreMessages(stored: StoredMessage[], firstId: number): MessageState[] {
  const messages: MessageState[] = [];
  let id = firstId;
  let lastQuestion = "";
  for (const turn of stored) {
    const payload = turn.payload ?? {};
    if (turn.role === "user") {
      lastQuestion = turn.content;
      messages.push({ kind: "user", id: id++, text: turn.content, spoken: payload.spoken === true });
    } else if (payload.removed === true) {
      // The note this answer came from was removed from the library, and the answer went with it.
      messages.push({ kind: "note", id: id++, text: turn.content });
    } else if (turn.mode === "notes") {
      const question = typeof payload.question === "string" ? payload.question : lastQuestion;
      messages.push({ kind: "answer", id: id++, question, response: answerFrom(turn.content, payload), ms: 0 });
    } else {
      const response: ChatResponse = {
        answer: turn.content,
        model: typeof payload.model === "string" ? payload.model : "",
        truncated: payload.truncated === true,
        actions: actionsFrom(payload.actions),
      };
      messages.push({ kind: "reply", id: id++, question: lastQuestion, response, ms: 0, actions: parseActions(response.actions) });
    }
  }
  return messages;
}

// --- marks on answers -------------------------------------------------------------------------------

/**
 * The mark to keep for an answer: the question, the answer that was given and what the interface knew
 * about it. The sources are named (file, heading, number) but their text is not copied, because the
 * point is to review where the answer went wrong, not to keep a second copy of the notes.
 * Null for anything that is not an answer or a reply.
 */
export function feedbackFor(message: MessageState, kind: FeedbackKind): NewFeedback | null {
  if (message.kind === "reply") {
    const { model, truncated } = message.response;
    return {
      kind,
      mode: "general",
      question: message.question,
      answer: message.response.answer,
      details: { model, truncated },
    };
  }
  if (message.kind !== "answer") return null;
  const { sources, reason, grounded, notes_considered, searched_for } = message.response;
  return {
    kind,
    mode: "notes",
    question: message.question,
    answer: message.response.answer,
    details: {
      sources: sources.map((s) => ({ document_id: s.document_id, source: s.source, heading_path: s.heading_path })),
      reason,
      grounded,
      notes_considered,
      searched_for: searched_for ?? null,
    },
  };
}
