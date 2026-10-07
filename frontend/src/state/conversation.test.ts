import { describe, expect, it } from "vitest";
import type { AskResponse, NewStoredMessage, StoredMessage } from "../api/types";
import { MAX_CHARS, feedbackFor, historyFor, restoreMessages, storedTurns } from "./conversation";
import type { MessageState } from "./useAssistant";

const grounded: AskResponse = {
  answer: "It leaves at 07:45 [1].",
  grounded: true,
  reason: "answered",
  sources: [],
  notes_considered: 1,
};
const refusal: AskResponse = {
  answer: "I don't have enough information in your notes to answer that.",
  grounded: false,
  reason: "no_relevant_notes",
  sources: [],
  notes_considered: 0,
};
const reply = (answer: string) => ({ answer, model: "test", truncated: false });

describe("historyFor", () => {
  it("is empty for a new conversation", () => {
    expect(historyFor([])).toEqual([]);
  });

  it("sends each question with the reply it got, oldest first", () => {
    const messages: MessageState[] = [
      { kind: "user", id: 1, text: "Hello" },
      { kind: "reply", id: 2, question: "Hello", response: reply("Hi."), ms: 900 },
      { kind: "user", id: 3, text: "When is my flight?" },
      { kind: "answer", id: 4, question: "When is my flight?", response: grounded, ms: 2100 },
    ];

    expect(historyFor(messages)).toEqual([
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi." },
      { role: "user", content: "When is my flight?" },
      { role: "assistant", content: "It leaves at 07:45 [1]." },
    ]);
  });

  it("leaves out refusals, errors and questions that are still waiting", () => {
    const messages: MessageState[] = [
      { kind: "user", id: 1, text: "What is the capital of Mars?" },
      { kind: "answer", id: 2, question: "What is the capital of Mars?", response: refusal, ms: 300 },
      { kind: "user", id: 3, text: "Hello?" },
      { kind: "error", id: 4, error: new Error("offline") },
      { kind: "user", id: 5, text: "Still there?" },
      { kind: "pending", id: 6, mode: "general" },
    ];

    expect(historyFor(messages)).toEqual([]);
  });

  it("keeps only the most recent exchanges, as whole pairs", () => {
    const messages: MessageState[] = Array.from({ length: 15 }, (_, i) => ({
      kind: "reply" as const,
      id: i,
      question: `question ${i}`,
      response: reply(`reply ${i}`),
      ms: 1,
    }));

    const history = historyFor(messages);

    expect(history).toHaveLength(20);
    expect(history[0]).toEqual({ role: "user", content: "question 5" });
    expect(history[19]).toEqual({ role: "assistant", content: "reply 14" });
  });
});

describe("MAX_CHARS", () => {
  it("allows longer messages in general chat than in a notes question", () => {
    expect(MAX_CHARS.general).toBeGreaterThan(MAX_CHARS.notes);
    expect(MAX_CHARS.notes).toBe(2000);
  });
});

describe("saving and reopening", () => {
  const source = {
    marker: 1,
    citation_id: "4:0",
    document_id: 4,
    chunk_index: 0,
    score: 0.8,
    source: "lisbon.md",
    heading_path: "Lisbon > Flights",
    start_line: 1,
    end_line: 3,
    text: "Flight TP 1357 leaves at 07:45.",
  };
  const answer: AskResponse = { ...grounded, sources: [source], searched_for: "When does my flight leave?" };
  const general = {
    answer: "Opening it.",
    model: "qwen3.5:4b",
    truncated: false,
    actions: [{ name: "open_page", args: { page: "knowledge" } }],
  };

  /** What the backend sends back for turns that were saved. */
  const asStored = (turns: NewStoredMessage[]): StoredMessage[] =>
    turns.map((turn, index) => ({
      id: index + 1,
      position: index,
      role: turn.role,
      mode: turn.mode,
      content: turn.content,
      payload: turn.payload ?? null,
      created_at: "2026-10-05T10:00:00Z",
    }));

  it("saves what was asked and the reply, with what the interface needs to show it again", () => {
    const turns = storedTurns({ text: "Open the knowledge page" }, { kind: "reply", id: 2, question: "Open the knowledge page", response: general, ms: 900 });

    expect(turns).toEqual([
      { role: "user", mode: "general", content: "Open the knowledge page" },
      {
        role: "assistant",
        mode: "general",
        content: "Opening it.",
        payload: { model: "qwen3.5:4b", truncated: false, actions: general.actions },
      },
    ]);
  });

  it("saves a cited answer with its sources, the question and what was searched for", () => {
    const turns = storedTurns({ text: "When?", spoken: true }, { kind: "answer", id: 2, question: "When?", response: answer, ms: 2100 });

    expect(turns[0]).toEqual({ role: "user", mode: "notes", content: "When?", payload: { spoken: true } });
    expect(turns[1]).toMatchObject({
      role: "assistant",
      mode: "notes",
      payload: { question: "When?", grounded: true, searched_for: "When does my flight leave?", sources: [source] },
    });
  });

  it("saves only the answer when nobody typed the question, as when the assistant searched the notes by itself", () => {
    const turns = storedTurns(null, { kind: "answer", id: 2, question: "When is the flight?", response: answer, ms: 1 });

    expect(turns.map((t) => t.role)).toEqual(["assistant"]);
    expect(turns[0].payload).toMatchObject({ question: "When is the flight?" });
  });

  it("saves nothing for an error, a notice or a question still waiting", () => {
    const user = { text: "Hello" };

    expect(storedTurns(user, { kind: "error", id: 2, error: new Error("down") })).toEqual([]);
    expect(storedTurns(user, { kind: "note", id: 2, text: "Status report" })).toEqual([]);
    expect(storedTurns(user, { kind: "pending", id: 2, mode: "general" })).toEqual([]);
  });

  it("brings a conversation back as it was, so it can be continued", () => {
    const original: MessageState[] = [
      { kind: "user", id: 1, text: "Open the knowledge page", spoken: true },
      { kind: "reply", id: 2, question: "Open the knowledge page", response: general, ms: 900, actions: [{ name: "open_page", page: "knowledge" }] },
      { kind: "user", id: 3, text: "When does my flight leave?" },
      { kind: "answer", id: 4, question: "When does my flight leave?", response: answer, ms: 2100 },
    ];
    const turns = [
      ...storedTurns({ text: "Open the knowledge page", spoken: true }, original[1]),
      ...storedTurns({ text: "When does my flight leave?" }, original[3]),
    ];

    const restored = restoreMessages(asStored(turns), 10);

    expect(restored.map((m) => m.id)).toEqual([10, 11, 12, 13]);
    expect(restored[0]).toEqual({ kind: "user", id: 10, text: "Open the knowledge page", spoken: true });
    expect(restored[1]).toMatchObject({ kind: "reply", question: "Open the knowledge page", actions: [{ name: "open_page", page: "knowledge" }] });
    expect(restored[3]).toMatchObject({ kind: "answer", question: "When does my flight leave?", response: answer, ms: 0 });
    // The restored conversation can go on: the next question carries it as history.
    expect(historyFor(restored).map((t) => t.content)).toEqual([
      "Open the knowledge page",
      "Opening it.",
      "When does my flight leave?",
      "It leaves at 07:45 [1].",
    ]);
  });

  it("shows a removed answer as a notice", () => {
    const stored: StoredMessage[] = [
      {
        id: 1,
        position: 0,
        role: "assistant",
        mode: "notes",
        content: "This answer used a note that was removed from the library, so it was removed here too.",
        payload: { removed: true },
        created_at: "2026-10-05T10:00:00Z",
      },
    ];

    expect(restoreMessages(stored, 1)).toEqual([{ kind: "note", id: 1, text: stored[0].content }]);
  });

  it("reads damaged saved data without failing", () => {
    const stored: StoredMessage[] = [
      { id: 1, position: 0, role: "user", mode: "notes", content: "q", payload: null, created_at: "" },
      {
        id: 2,
        position: 1,
        role: "assistant",
        mode: "notes",
        content: "An answer.",
        payload: { grounded: true, reason: "made-up", sources: [source, { nonsense: true }, "x", null], notes_considered: "many" },
        created_at: "",
      },
      {
        id: 3,
        position: 2,
        role: "assistant",
        mode: "general",
        content: "Hello.",
        payload: { actions: [{ name: "open_page", args: { page: "settings", extra: 3 } }, { name: 5 }, "bad"], truncated: "yes" },
        created_at: "",
      },
      { id: 4, position: 3, role: "assistant", mode: "general", content: "No details.", payload: null, created_at: "" },
    ];

    const restored = restoreMessages(stored, 1);

    expect(restored[1]).toMatchObject({
      kind: "answer",
      question: "q", // from the question that came before
      response: { grounded: true, reason: "answered", sources: [source], notes_considered: 0, searched_for: null },
    });
    expect(restored[2]).toMatchObject({
      kind: "reply",
      response: { answer: "Hello.", truncated: false, actions: [{ name: "open_page", args: { page: "settings" } }] },
      actions: [{ name: "open_page", page: "settings" }],
    });
    expect(restored[3]).toMatchObject({ kind: "reply", response: { model: "", actions: [] }, actions: [] });
  });
});

describe("feedbackFor", () => {
  const source = {
    marker: 1,
    citation_id: "4:0",
    document_id: 4,
    chunk_index: 0,
    score: 0.8,
    source: "rice.txt",
    heading_path: "Rice > Cooking",
    start_line: 1,
    end_line: 3,
    text: "Rice needs eighteen minutes: a private sentence from the notes.",
  };
  const cited: AskResponse = {
    answer: "Rice needs eighteen minutes [1].",
    grounded: true,
    reason: "answered",
    sources: [source],
    notes_considered: 3,
    searched_for: "How long does rice cook?",
  };

  it("keeps the question, the answer and where it came from, for an answer from the notes", () => {
    const message: MessageState = { kind: "answer", id: 2, question: "How long do oats simmer?", response: cited, ms: 900 };

    expect(feedbackFor(message, "wrong_source")).toEqual({
      kind: "wrong_source",
      mode: "notes",
      question: "How long do oats simmer?",
      answer: "Rice needs eighteen minutes [1].",
      details: {
        sources: [{ document_id: 4, source: "rice.txt", heading_path: "Rice > Cooking" }],
        reason: "answered",
        grounded: true,
        notes_considered: 3,
        searched_for: "How long does rice cook?",
      },
    });
  });

  it("names the sources but does not copy the text of the notes", () => {
    const message: MessageState = { kind: "answer", id: 2, question: "q", response: cited, ms: 1 };

    expect(JSON.stringify(feedbackFor(message, "not_helpful"))).not.toContain("private sentence");
  });

  it("marks a refusal with the reason it gave, and no sources", () => {
    const refusal: AskResponse = {
      answer: "I don't have enough information in your notes to answer that.",
      grounded: false,
      reason: "no_relevant_notes",
      sources: [],
      notes_considered: 0,
    };

    const entry = feedbackFor({ kind: "answer", id: 2, question: "q", response: refusal, ms: 1 }, "missing_info");

    expect(entry?.details).toMatchObject({ sources: [], reason: "no_relevant_notes", grounded: false, searched_for: null });
  });

  it("marks a general reply by the model that wrote it", () => {
    const response = { answer: "Paris.", model: "qwen3.5:4b", truncated: false };

    expect(feedbackFor({ kind: "reply", id: 2, question: "Capital of France?", response, ms: 900 }, "helpful")).toEqual({
      kind: "helpful",
      mode: "general",
      question: "Capital of France?",
      answer: "Paris.",
      details: { model: "qwen3.5:4b", truncated: false },
    });
  });

  it("has nothing to mark in a message that is not an answer", () => {
    expect(feedbackFor({ kind: "user", id: 1, text: "hi" }, "helpful")).toBeNull();
    expect(feedbackFor({ kind: "note", id: 1, text: "status" }, "helpful")).toBeNull();
    expect(feedbackFor({ kind: "pending", id: 1, mode: "notes" }, "helpful")).toBeNull();
    expect(feedbackFor({ kind: "error", id: 1, error: new Error("down") }, "helpful")).toBeNull();
  });
});
