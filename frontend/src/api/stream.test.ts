import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, createApi } from "./client";
import type { AskResponse } from "./types";

afterEach(() => vi.unstubAllGlobals());

const ANSWER: AskResponse = {
  answer: "Simmer them in milk [1].",
  grounded: true,
  reason: "answered",
  sources: [],
  notes_considered: 1,
};

const event = (name: string, data: unknown) => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;

/** A response whose body arrives in the given pieces, one per read. */
function streamed(pieces: string[], init: ResponseInit = {}): Response {
  const encoder = new TextEncoder();
  let next = 0;
  const body = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (next < pieces.length) controller.enqueue(encoder.encode(pieces[next++]));
      else controller.close();
    },
  });
  return new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" }, ...init });
}

describe("askStream", () => {
  it("passes on what is searched for and each piece of text, and returns the checked answer", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        streamed([
          event("searching", { searched_for: "How do I cook oats?" }),
          event("token", { text: "Simmer " }),
          event("token", { text: "them" }),
          event("done", ANSWER),
        ]),
      ),
    );
    const searching = vi.fn();
    const tokens: string[] = [];

    const answer = await createApi().askStream("and how long?", { onSearching: searching, onToken: (t) => tokens.push(t) });

    expect(searching).toHaveBeenCalledWith("How do I cook oats?");
    expect(tokens).toEqual(["Simmer ", "them"]);
    expect(answer).toEqual(ANSWER);
  });

  it("sends the question, the filters and the earlier turns, with the sign-in token", async () => {
    const fetchMock = vi.fn().mockResolvedValue(streamed([event("done", ANSWER)]));
    vi.stubGlobal("fetch", fetchMock);
    const history = [
      { role: "user" as const, content: "How do I cook oats?" },
      { role: "assistant" as const, content: "Simmer them [1]." },
    ];

    await createApi("/api/v1", { getToken: () => "abc", onUnauthorized: () => undefined }).askStream(
      "and how long?",
      {},
      { filters: { top_k: 3 }, history },
    );

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/v1/ask/stream");
    expect(init.method).toBe("POST");
    expect(init.headers.Authorization).toBe("Bearer abc");
    expect(JSON.parse(init.body)).toEqual({ question: "and how long?", top_k: 3, history });
  });

  it("leaves the history out when there is none", async () => {
    const fetchMock = vi.fn().mockResolvedValue(streamed([event("done", ANSWER)]));
    vi.stubGlobal("fetch", fetchMock);

    await createApi().askStream("first question", {});

    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ question: "first question" });
  });

  it("reads an event, and a multi-byte character, that arrive split between two chunks", async () => {
    const whole = event("token", { text: "café" }) + event("done", ANSWER);
    const cut = whole.indexOf("caf") + 4; // between the two bytes of "é"
    const encoder = new TextEncoder();
    const bytes = encoder.encode(whole);
    const split = encoder.encode(whole.slice(0, cut)).length - 1;
    let sent = 0;
    const body = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (sent === 0) controller.enqueue(bytes.slice(0, split));
        else if (sent === 1) controller.enqueue(bytes.slice(split));
        else controller.close();
        sent++;
      },
    });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(body)));
    const tokens: string[] = [];

    await createApi().askStream("q", { onToken: (t) => tokens.push(t) });

    expect(tokens).toEqual(["café"]);
  });

  it("turns an error event into the same kind of error a failed request gives", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        streamed([event("token", { text: "Sim" }), event("error", { status: 504, detail: "the model was too slow" })]),
      ),
    );

    const error = await createApi().askStream("q", {}).catch((e: unknown) => e);

    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ kind: "timeout", message: "the model was too slow", status: 504 });
  });

  it("reports a problem found before anything was sent as an ordinary error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "The question is empty." }), { status: 422 })),
    );

    await expect(createApi().askStream("q", {})).rejects.toMatchObject({ kind: "invalid", message: "The question is empty." });
  });

  it("reports a backend without the endpoint as not found, so the caller can ask the usual way", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "Not Found" }), { status: 404 })));

    await expect(createApi().askStream("q", {})).rejects.toMatchObject({ kind: "not_found" });
  });

  it("tells the app when the server stops accepting the sign-in", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "Sign in required." }), { status: 401 })));
    const onUnauthorized = vi.fn();

    await expect(
      createApi("/api/v1", { getToken: () => "old", onUnauthorized }).askStream("q", {}),
    ).rejects.toMatchObject({ kind: "unauthorized" });

    expect(onUnauthorized).toHaveBeenCalledOnce();
  });

  it("reports an unreachable backend as offline", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("failed")));

    await expect(createApi().askStream("q", {})).rejects.toMatchObject({ kind: "offline" });
  });

  it("reports a connection that drops in the middle as offline, not as a finished answer", async () => {
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(event("token", { text: "Sim" })));
        controller.error(new TypeError("network error"));
      },
    });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(body)));

    await expect(createApi().askStream("q", {})).rejects.toMatchObject({ kind: "offline" });
  });

  it("does not accept a stream that ends without a result", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(streamed([event("token", { text: "Sim" })])));

    await expect(createApi().askStream("q", {})).rejects.toMatchObject({ kind: "server" });
  });

  it("passes the abort signal on and lets the abort through as it is", async () => {
    const controller = new AbortController();
    const fetchMock = vi.fn().mockImplementation((_url: string, init: RequestInit) => {
      return new Promise((_resolve, reject) => {
        init.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
      });
    });
    vi.stubGlobal("fetch", fetchMock);

    const pending = createApi().askStream("q", {}, { signal: controller.signal });
    controller.abort();

    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
    expect(fetchMock.mock.calls[0][1].signal).toBe(controller.signal);
  });

  it("lets an abort in the middle of the stream through as it is", async () => {
    const controller = new AbortController();
    const body = new ReadableStream<Uint8Array>({
      start(stream) {
        stream.enqueue(new TextEncoder().encode(event("token", { text: "Sim" })));
        controller.signal.addEventListener("abort", () => stream.error(new DOMException("Aborted", "AbortError")));
      },
    });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(body)));
    const tokens: string[] = [];

    const pending = createApi().askStream(
      "q",
      {
        onToken: (t) => {
          tokens.push(t);
          controller.abort();
        },
      },
      { signal: controller.signal },
    );

    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
    expect(tokens).toEqual(["Sim"]);
  });
});

describe("conversation calls", () => {
  const ok = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });

  it("lists, creates and reads conversations from their own endpoints", async () => {
    const fetchMock = vi.fn().mockImplementation(async () => ok({ id: 3, conversations: [] }));
    vi.stubGlobal("fetch", fetchMock);
    const api = createApi();

    await api.conversations();
    await api.createConversation("Oats");
    await api.conversation(3);

    expect(fetchMock.mock.calls.map((c) => c[0])).toEqual([
      "/api/v1/conversations",
      "/api/v1/conversations",
      "/api/v1/conversations/3",
    ]);
    expect(fetchMock.mock.calls[1][1].method).toBe("POST");
    expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({ title: "Oats" });
  });

  it("renames with PUT and adds messages with POST", async () => {
    const fetchMock = vi.fn().mockImplementation(async () => ok({ id: 3 }));
    vi.stubGlobal("fetch", fetchMock);
    const api = createApi();

    await api.renameConversation(3, "New name");
    await api.addMessages(3, [{ role: "user", mode: "notes", content: "hello" }]);

    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/conversations/3");
    expect(fetchMock.mock.calls[0][1].method).toBe("PUT");
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ title: "New name" });
    expect(fetchMock.mock.calls[1][0]).toBe("/api/v1/conversations/3/messages");
    expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({
      messages: [{ role: "user", mode: "notes", content: "hello" }],
    });
  });

  it("deletes one conversation, or all and says how many", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(null, { status: 204 }))
      .mockResolvedValueOnce(ok({ deleted: 4 }));
    vi.stubGlobal("fetch", fetchMock);
    const api = createApi();

    await expect(api.deleteConversation(3)).resolves.toBeUndefined();
    await expect(api.deleteConversations()).resolves.toBe(4);

    expect(fetchMock.mock.calls[0][1].method).toBe("DELETE");
    expect(fetchMock.mock.calls[1][0]).toBe("/api/v1/conversations");
    expect(fetchMock.mock.calls[1][1].method).toBe("DELETE");
  });

  it("reports a full conversation as a conflict", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(ok({ detail: "this conversation holds 400 messages" }, 409)));

    await expect(createApi().addMessages(3, [])).rejects.toMatchObject({ kind: "conflict" });
  });
});
