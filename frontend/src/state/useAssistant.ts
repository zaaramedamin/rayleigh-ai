import { useCallback, useRef, useState } from "react";
import type { RefObject } from "react";
import { ApiError, isAbort } from "../api/client";
import { sound } from "../audio/sound";
import { speech } from "../audio/speech";
import { events } from "./events";
import type { Api, AskResponse, ChatMode, ChatResponse, ChatTurn, FeedbackKind } from "../api/types";
import { parseActions } from "./actions";
import type { AppAction, AppControls } from "./actions";
import { feedbackFor, historyFor, restoreMessages, storedTurns } from "./conversation";
import { addressed } from "./greeting";
import { useSettings } from "./settings";

/** The mark the owner put on an answer: its number on the server, and what it says. */
export interface Mark {
  id: number;
  kind: FeedbackKind;
}

// "answer" comes from the notes (cited, or an explicit refusal). "reply" is general chat: the
// model alone, no notes and no sources; `actions` is what it did in the application. "note" is
// something the application says itself, such as the greeting or a status report. "pending" is
// a question being answered: for the notes it shows what was searched for and the text so far.
export type MessageState =
  | { kind: "user"; id: number; text: string; spoken?: boolean }
  | {
      kind: "pending";
      id: number;
      mode: ChatMode;
      /** What a follow-up was rewritten into before the notes were searched. */
      searchedFor?: string;
      /** The text written so far. Not checked yet: citations are only verified when it is complete. */
      partial?: string;
    }
  | { kind: "answer"; id: number; question: string; response: AskResponse; ms: number; mark?: Mark }
  | {
      kind: "reply";
      id: number;
      question: string;
      response: ChatResponse;
      ms: number;
      actions?: AppAction[];
      mark?: Mark;
    }
  | { kind: "note"; id: number; text: string }
  | { kind: "error"; id: number; error: ApiError | Error };

export type AssistantStatus = "idle" | "thinking";

export interface AskOptions {
  /**
   * The message was spoken. Spoken orders always go to the assistant, which searches the notes
   * itself when asked about them; the MY NOTES switch applies to what is typed.
   */
  spoken?: boolean;
}

/** Who asked, for saving the exchange. Null when nobody typed the question (the assistant searched by itself). */
type Asker = { text: string; spoken?: boolean } | null;

/** The longest wait for the assistant to finish its sentence before the application locks. */
const LOCK_AFTER_SPEECH_MS = 6000;

function notifyFailure(err: Error): void {
  if (err instanceof ApiError && err.kind === "offline") {
    events.notify("alert", "LINK LOST", { detail: err.message, key: "link" });
  } else if (err instanceof ApiError && err.kind === "llm_unavailable") {
    events.notify("error", "LOCAL MODEL UNAVAILABLE", { detail: err.message });
  } else if (err instanceof ApiError && err.kind === "timeout") {
    events.notify("warning", "THE MODEL TOOK TOO LONG", { detail: err.message });
  } else if (err instanceof ApiError && err.kind === "not_found") {
    // The interface is newer than the running backend, which lacks this endpoint.
    events.notify("warning", "BACKEND OUT OF DATE", { detail: "Restart the backend to use this." });
  } else {
    events.notify("error", "COULD NOT ANSWER", { detail: err.message });
  }
}

/**
 * Chat state for the console. The conversation is kept on this computer (unless saving is turned
 * off) and can be reopened later. `controls` is how the assistant operates the rest of the
 * application; `identity` is the name it goes by and how it addresses you.
 */
export function useAssistant(
  api: Api,
  controls?: RefObject<AppControls | null>,
  identity: { name: string; address: string } = { name: "Reyleight", address: "" },
) {
  const [messages, setMessages] = useState<MessageState[]>([]);
  const [status, setStatus] = useState<AssistantStatus>("idle");
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [selectedMarker, setSelectedMarker] = useState<number | null>(null);
  const [conversationId, setConversationId] = useState<number | null>(null);
  // Goes up each time something is saved, so a list of conversations knows to reload.
  const [savedVersion, setSavedVersion] = useState(0);
  const nextId = useRef(1);
  // Goes up when the conversation is cleared or another one is opened, so a reply that arrives
  // afterwards is dropped.
  const epoch = useRef(0);
  // The answer being streamed, so STOP can end it.
  const stream = useRef<AbortController | null>(null);
  const conversationRef = useRef<number | null>(null);
  const messagesRef = useRef<MessageState[]>([]);
  messagesRef.current = messages;
  // Saving happens one exchange at a time, so a new conversation is only ever created once.
  const saves = useRef<Promise<void>>(Promise.resolve());
  const warnedAboutSaving = useRef(false);

  // The mode is remembered with the other settings, so the app reopens the way it was left.
  const { settings, update } = useSettings();
  const mode = settings.chatMode;
  const setMode = useCallback((next: ChatMode) => update({ chatMode: next }), [update]);
  const saveChats = settings.saveChats;
  const savingRef = useRef(saveChats);
  savingRef.current = saveChats;
  const setSaveChats = useCallback((next: boolean) => update({ saveChats: next }), [update]);

  const replace = useCallback((id: number, next: MessageState) => {
    setMessages((m) => m.map((x) => (x.id === id ? next : x)));
  }, []);

  const detach = useCallback(() => {
    conversationRef.current = null;
    setConversationId(null);
  }, []);

  const fail = useCallback(
    (id: number, error: unknown, spoken: boolean) => {
      const err = error instanceof Error ? error : new Error("Unexpected error");
      replace(id, { kind: "error", id, error: err });
      notifyFailure(err);
      if (spoken) void speech.speak(`I could not do that${addressed(identity.address)}.`);
    },
    [replace, identity.address],
  );

  /** Something the application says itself: shown in the chat and read aloud. */
  const note = useCallback((text: string) => {
    const id = nextId.current++;
    setMessages((m) => [...m, { kind: "note", id, text }]);
    void speech.say(text);
  }, []);

  /** Keep an exchange in the stored conversation, starting one if needed. Failing to save never loses the chat. */
  const persist = useCallback(
    (asker: Asker, reply: MessageState) => {
      if (!savingRef.current) return;
      const turns = storedTurns(asker, reply);
      if (turns.length === 0) return;
      const work = async () => {
        try {
          for (let attempt = 0; attempt < 2; attempt++) {
            let id = conversationRef.current;
            if (id === null) {
              id = (await api.createConversation()).id;
              conversationRef.current = id;
              setConversationId(id);
            }
            try {
              await api.addMessages(id, turns);
              break;
            } catch (error) {
              // A conversation that is full goes on in a new one.
              if (attempt === 0 && error instanceof ApiError && error.kind === "conflict") {
                detach();
                continue;
              }
              throw error;
            }
          }
          setSavedVersion((v) => v + 1);
        } catch (error) {
          if (!warnedAboutSaving.current) {
            warnedAboutSaving.current = true;
            const detail = error instanceof Error ? error.message : undefined;
            events.notify("warning", "CONVERSATION NOT SAVED", { detail });
          }
        }
      };
      saves.current = saves.current.then(work);
    },
    [api, detach],
  );

  /** A cited answer from the notes, written as it arrives and checked when it is complete. */
  const answerFromNotes = useCallback(
    async (id: number, question: string, spoken: boolean, history: ChatTurn[], asker: Asker) => {
      const started = performance.now();
      const mine = epoch.current;
      const controller = new AbortController();
      stream.current = controller;
      const show = (change: { searchedFor?: string; partial?: string }) => {
        if (mine !== epoch.current) return;
        setMessages((m) => m.map((x) => (x.id === id && x.kind === "pending" ? { ...x, ...change } : x)));
      };
      let partial = "";
      try {
        let response: AskResponse;
        try {
          response = await api.askStream(
            question,
            {
              onSearching: (searchedFor) => show({ searchedFor }),
              onToken: (text) => {
                partial += text;
                show({ partial });
              },
            },
            { history, signal: controller.signal },
          );
        } catch (error) {
          // A backend from before streaming: ask in one piece.
          if (error instanceof ApiError && error.kind === "not_found") response = await api.ask(question, {}, history);
          else throw error;
        }
        if (mine !== epoch.current) return;
        const message: MessageState = { kind: "answer", id, question, response, ms: Math.round(performance.now() - started) };
        replace(id, message);
        setSelectedId(id);
        setSelectedMarker(null);
        sound.play("receive");
        void speech.say(response.answer);
        persist(asker, message);
      } catch (error) {
        if (mine !== epoch.current) return;
        if (isAbort(error)) {
          replace(id, { kind: "note", id, text: "Stopped." });
          return;
        }
        fail(id, error, spoken);
      } finally {
        if (stream.current === controller) stream.current = null;
      }
    },
    [api, replace, fail, persist],
  );

  /** Carry out what the assistant asked for. Each action was validated twice before this. */
  const perform = useCallback(
    async (actions: AppAction[], keep: number[], spoken: boolean) => {
      const app = controls?.current;
      for (const action of actions) {
        switch (action.name) {
          case "open_page":
            app?.navigate(action.page);
            break;
          case "set_theme":
            app?.update({ theme: action.theme });
            break;
          case "set_option":
            app?.update(action.patch);
            break;
          case "ask_notes": {
            const id = nextId.current++;
            setMessages((m) => [...m, { kind: "pending", id, mode: "notes" }]);
            await answerFromNotes(id, action.question, spoken, historyFor(messagesRef.current), null);
            break;
          }
          case "update_library":
            try {
              await api.startSync();
              events.notify("info", "LIBRARY UPDATE STARTED", { detail: "Progress is on the Knowledge page." });
              app?.changed();
            } catch (error) {
              const detail = error instanceof Error ? error.message : undefined;
              events.notify("warning", "LIBRARY UPDATE NOT STARTED", { detail });
            }
            break;
          case "report_status":
            if (app) note(app.statusReport());
            break;
          case "clear_conversation":
            // Keep only this order and its confirmation, so the new conversation starts with them.
            // The conversation that was saved stays in the history.
            setMessages((m) => m.filter((x) => keep.includes(x.id)));
            setSelectedId(null);
            setSelectedMarker(null);
            detach();
            break;
          case "lock_app":
            // Let the assistant finish its sentence first.
            await Promise.race([speech.whenQuiet(), new Promise((r) => setTimeout(r, LOCK_AFTER_SPEECH_MS))]);
            app?.lock();
            return;
          case "remember":
            events.notify("success", "MEMORY SAVED", { detail: "See or delete it on the Profile page." });
            app?.changed();
            break;
        }
      }
    },
    [api, controls, answerFromNotes, note, detach],
  );

  const ask = useCallback(
    async (question: string, options: AskOptions = {}) => {
      const text = question.trim();
      if (!text || status === "thinking") return;
      const spoken = options.spoken === true;
      const useMode: ChatMode = spoken ? "general" : mode;
      const userId = nextId.current++;
      const pendingId = nextId.current++;
      const mine = epoch.current;
      // What was said so far goes with the message: a general reply answers in context, and a
      // follow-up question to the notes is rewritten into one that stands alone.
      const history = historyFor(messages);
      const asker: Asker = { text, spoken };
      setMessages((m) => [
        ...m,
        { kind: "user", id: userId, text, spoken },
        { kind: "pending", id: pendingId, mode: useMode },
      ]);
      setStatus("thinking");
      sound.play("send");
      speech.stop();
      const started = performance.now();
      try {
        if (useMode === "notes") {
          await answerFromNotes(pendingId, text, spoken, history, asker);
        } else {
          const response = await api.chat(text, history);
          if (mine !== epoch.current) return;
          const actions = parseActions(response.actions);
          const message: MessageState = {
            kind: "reply",
            id: pendingId,
            question: text,
            response,
            ms: Math.round(performance.now() - started),
            actions,
          };
          replace(pendingId, message);
          setSelectedId(pendingId);
          setSelectedMarker(null);
          sound.play("receive");
          void speech.say(response.answer);
          persist(asker, message);
          await perform(actions, [userId, pendingId], spoken);
        }
      } catch (error) {
        if (mine === epoch.current) fail(pendingId, error, spoken);
      } finally {
        setStatus("idle");
      }
    },
    [api, status, mode, messages, answerFromNotes, perform, replace, fail, persist],
  );

  /** Start a new conversation. The one that was saved stays in the history. */
  const clear = useCallback(() => {
    epoch.current++;
    stream.current?.abort();
    speech.stop();
    setMessages([]);
    setSelectedId(null);
    setSelectedMarker(null);
    detach();
  }, [detach]);

  /** Reopen a saved conversation, so it can be read and carried on. */
  const open = useCallback(
    async (id: number) => {
      const detail = await api.conversation(id);
      epoch.current++;
      stream.current?.abort();
      speech.stop();
      setMessages(restoreMessages(detail.messages, nextId.current));
      nextId.current += detail.messages.length;
      setSelectedId(null);
      setSelectedMarker(null);
      conversationRef.current = id;
      setConversationId(id);
      return detail;
    },
    [api],
  );

  /**
   * Mark an answer. The first mark is kept; another kind changes it; the same kind again takes it
   * back. A mark keeps the question and the answer on this computer, which is why it is only made
   * when pressed, whatever the SAVE switch says about conversations.
   */
  const rating = useRef(new Set<number>());
  const rate = useCallback(
    async (id: number, kind: FeedbackKind) => {
      const message = messagesRef.current.find((m) => m.id === id);
      if (!message || (message.kind !== "answer" && message.kind !== "reply")) return;
      if (rating.current.has(id)) return; // one request at a time for one answer
      rating.current.add(id);
      const show = (mark: Mark | undefined) =>
        setMessages((all) =>
          all.map((x) => (x.id === id && (x.kind === "answer" || x.kind === "reply") ? { ...x, mark } : x)),
        );
      try {
        const current = message.mark;
        if (current && current.kind === kind) {
          await api.deleteFeedback(current.id);
          show(undefined);
        } else if (current) {
          const changed = await api.changeFeedback(current.id, kind);
          show({ id: changed.id, kind: changed.kind });
        } else {
          const entry = feedbackFor(message, kind);
          if (!entry) return;
          const created = await api.addFeedback(entry);
          show({ id: created.id, kind: created.kind });
        }
      } catch (error) {
        const detail = error instanceof Error ? error.message : undefined;
        events.notify("error", "YOUR MARK WAS NOT SAVED", { detail });
      } finally {
        rating.current.delete(id);
      }
    },
    [api],
  );

  /** End the answer that is being written. What was streamed so far is dropped. */
  const stop = useCallback(() => stream.current?.abort(), []);

  const select = useCallback((id: number, marker: number | null = null) => {
    setSelectedId(id);
    setSelectedMarker(marker);
  }, []);

  const selected = messages.find((m) => m.id === selectedId);
  return {
    messages,
    status,
    mode,
    setMode,
    ask,
    note,
    clear,
    open,
    stop,
    rate,
    /** A cited answer is being written, so it can be stopped. */
    canStop: status === "thinking" && messages.some((m) => m.kind === "pending" && m.mode === "notes"),
    select,
    selectedId,
    selectedMarker,
    /** The stored conversation this chat belongs to, or null until something is saved. */
    conversationId,
    savedVersion,
    saveChats,
    setSaveChats,
    /** The name the assistant goes by, for labels. */
    name: identity.name,
    selectedAnswer: selected && selected.kind === "answer" ? selected : null,
    /** The selected message is a general reply, which has no sources to show. */
    selectedIsGeneral: selected?.kind === "reply",
  };
}
