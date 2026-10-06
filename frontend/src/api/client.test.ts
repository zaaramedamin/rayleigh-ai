import { afterEach, describe, expect, it, vi } from "vitest";
import { splitCitations } from "../components/AnswerText";
import { ApiError, createApi, kindForStatus } from "./client";

afterEach(() => vi.unstubAllGlobals());

describe("kindForStatus", () => {
  it("maps backend statuses to user-meaningful kinds", () => {
    expect(kindForStatus(503)).toBe("llm_unavailable");
    expect(kindForStatus(504)).toBe("timeout");
    expect(kindForStatus(422)).toBe("invalid");
    expect(kindForStatus(500)).toBe("server");
  });
});

describe("createApi", () => {
  it("reports an unreachable backend as offline", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("failed")));
    await expect(createApi().health()).rejects.toMatchObject({ kind: "offline" });
  });

  it("surfaces the backend's error detail", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "model missing" }), { status: 503 })),
    );
    const error = await createApi().ask("hi").catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ kind: "llm_unavailable", message: "model missing" });
  });

  it("reads the system status from its own endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ llm: { state: "ready" } })));
    vi.stubGlobal("fetch", fetchMock);
    await createApi().systemStatus();
    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/system/status");
  });

  it("posts the question and unwraps search results", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ results: [{ source: "a.md" }] })));
    vi.stubGlobal("fetch", fetchMock);
    const results = await createApi().search("lisbon", { top_k: 3 });
    expect(results).toEqual([{ source: "a.md" }]);
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ query: "lisbon", top_k: 3 });
  });
});

describe("general chat", () => {
  it("posts the message with the earlier turns to its own endpoint", async () => {
    const reply = { answer: "Madrid.", model: "qwen3.5:4b", truncated: false };
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(reply)));
    vi.stubGlobal("fetch", fetchMock);
    const history = [
      { role: "user" as const, content: "What is the capital of France?" },
      { role: "assistant" as const, content: "Paris." },
    ];

    expect(await createApi().chat("And of Spain?", history)).toEqual(reply);

    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/chat");
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ message: "And of Spain?", history });
  });

  it("sends an empty history for a first message", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ answer: "Hello." })));
    vi.stubGlobal("fetch", fetchMock);

    await createApi().chat("Hi");

    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ message: "Hi", history: [] });
  });

  it("reports a backend without the endpoint as not found", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "Not Found" }), { status: 404 })));

    await expect(createApi().chat("Hi")).rejects.toMatchObject({ kind: "not_found" });
  });
});

describe("splitCitations", () => {
  it("turns only verifiable markers into citations", () => {
    expect(splitCitations("Leaves at 7 [1] and gate B [9].", new Set([1]))).toEqual([
      "Leaves at 7 ",
      1,
      " and gate B [9].",
    ]);
  });
});

describe("sign-in handling", () => {
  const unauthorized = () => new Response(JSON.stringify({ detail: "Sign in required." }), { status: 401 });

  it("sends the token and tells the app when the server stops accepting it", async () => {
    const fetchMock = vi.fn().mockResolvedValue(unauthorized());
    vi.stubGlobal("fetch", fetchMock);
    const onUnauthorized = vi.fn();
    const api = createApi("/api/v1", { getToken: () => "abc", onUnauthorized });

    await expect(api.documents()).rejects.toMatchObject({ kind: "unauthorized" });

    expect(fetchMock.mock.calls[0][1].headers.Authorization).toBe("Bearer abc");
    expect(onUnauthorized).toHaveBeenCalledOnce();
  });

  it("treats a wrong password as an answer, not as a lost session", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "Wrong password." }), { status: 401 })));
    const onUnauthorized = vi.fn();
    const api = createApi("/api/v1", { getToken: () => null, onUnauthorized });

    await expect(api.login("nope")).rejects.toMatchObject({ kind: "unauthorized", message: "Wrong password." });

    expect(onUnauthorized).not.toHaveBeenCalled();
  });

  it("returns the token from a successful login", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ token: "t0k3n" }))));

    expect(await createApi().login("right")).toBe("t0k3n");
  });

  it("reports too many attempts as throttled", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "Try again in 4 seconds." }), { status: 429 })));

    await expect(createApi().login("x")).rejects.toMatchObject({ kind: "throttled" });
  });
});

describe("assistant and voice calls", () => {
  it("saves the identity with PUT and reads memories from their own page", async () => {
    const fetchMock = vi
      .fn()
      .mockImplementation(async () => new Response(JSON.stringify({ memories: [], limit: 200, name: "Jarvis" })));
    vi.stubGlobal("fetch", fetchMock);
    const api = createApi();

    await api.memories();
    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/assistant/memories");

    await api.saveAssistant({ name: "Jarvis", address: "sir", role: "", use_profile: true, use_memory: true });
    expect(fetchMock.mock.calls[1][1].method).toBe("PUT");
    expect(JSON.parse(fetchMock.mock.calls[1][1].body).name).toBe("Jarvis");
  });

  it("deletes a memory by id", async () => {
    const fetchMock = vi.fn().mockImplementation(async () => new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);

    await createApi().deleteMemory(4);

    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/assistant/memories/4");
    expect(fetchMock.mock.calls[0][1].method).toBe("DELETE");
  });

  it("sends a recording as raw audio, not JSON", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ text: "hello", seconds: 1 })));
    vi.stubGlobal("fetch", fetchMock);
    const wav = new Blob([new Uint8Array([1, 2, 3])], { type: "audio/wav" });

    expect((await createApi().transcribe(wav)).text).toBe("hello");

    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/voice/transcribe");
    expect(fetchMock.mock.calls[0][1].body).toBe(wav);
    expect(fetchMock.mock.calls[0][1].headers["Content-Type"]).toBe("audio/wav");
  });
});

describe("library calls", () => {
  it("accepts an empty 204 answer when deleting", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status: 204 })));

    await expect(createApi().deleteDocument(7)).resolves.toBeUndefined();
  });

  it("encodes the folder path and the remove-documents flag in the query", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ folders: [], documents_removed: 2 })));
    vi.stubGlobal("fetch", fetchMock);

    const result = await createApi().removeFolder("C:\My Notes\a&b", true);

    const url = new URL(fetchMock.mock.calls[0][0], "http://x");
    expect(url.pathname).toBe("/api/v1/library/folders");
    expect(url.searchParams.get("path")).toBe("C:\My Notes\a&b");
    expect(url.searchParams.get("remove_documents")).toBe("true");
    expect(fetchMock.mock.calls[0][1].method).toBe("DELETE");
    expect(result.documents_removed).toBe(2);
  });
});
