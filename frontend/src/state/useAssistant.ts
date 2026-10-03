import { useCallback, useRef, useState } from "react";
import { ApiError } from "../api/client";
import type { Api, AskResponse } from "../api/types";

export type MessageState =
  | { kind: "user"; id: number; text: string }
  | { kind: "pending"; id: number }
  | { kind: "answer"; id: number; question: string; response: AskResponse; ms: number }
  | { kind: "error"; id: number; error: ApiError | Error };

export type AssistantStatus = "idle" | "thinking";

/** Chat state for the console. The conversation lives in memory only: closing the app clears it. */
export function useAssistant(api: Api) {
  const [messages, setMessages] = useState<MessageState[]>([]);
  const [status, setStatus] = useState<AssistantStatus>("idle");
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [selectedMarker, setSelectedMarker] = useState<number | null>(null);
  const nextId = useRef(1);

  const ask = useCallback(
    async (question: string) => {
      const text = question.trim();
      if (!text || status === "thinking") return;
      const userId = nextId.current++;
      const pendingId = nextId.current++;
      setMessages((m) => [...m, { kind: "user", id: userId, text }, { kind: "pending", id: pendingId }]);
      setStatus("thinking");
      const started = performance.now();
      try {
        const response = await api.ask(text);
        const ms = Math.round(performance.now() - started);
        setMessages((m) =>
          m.map((x) => (x.id === pendingId ? { kind: "answer", id: pendingId, question: text, response, ms } : x)),
        );
        setSelectedId(pendingId);
        setSelectedMarker(null);
      } catch (error) {
        const err = error instanceof Error ? error : new Error("Unexpected error");
        setMessages((m) => m.map((x) => (x.id === pendingId ? { kind: "error", id: pendingId, error: err } : x)));
      } finally {
        setStatus("idle");
      }
    },
    [api, status],
  );

  const clear = useCallback(() => {
    setMessages([]);
    setSelectedId(null);
    setSelectedMarker(null);
  }, []);

  const select = useCallback((id: number, marker: number | null = null) => {
    setSelectedId(id);
    setSelectedMarker(marker);
  }, []);

  const selected = messages.find((m) => m.id === selectedId && m.kind === "answer");
  return {
    messages,
    status,
    ask,
    clear,
    select,
    selectedId,
    selectedMarker,
    selectedAnswer: selected && selected.kind === "answer" ? selected : null,
  };
}
