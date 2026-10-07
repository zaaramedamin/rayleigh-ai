import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, createApi } from "./client";
import { createMockApi } from "./mock";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

const ok = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });

describe("summarizeDocument", () => {
  it("asks for the summary of one document by its number", async () => {
    const summary = { text: "A trip.", document_id: 7, name: "lisbon-trip.md", parts: 1, covered_parts: 1, truncated: false };
    const fetchMock = vi.fn().mockResolvedValue(ok(summary));
    vi.stubGlobal("fetch", fetchMock);

    const result = await createApi().summarizeDocument(7);

    expect(result).toEqual(summary);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/v1/tasks/summarize");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ document_id: 7 });
  });

  it.each([
    [404, "not_found", "That document is not in the library."],
    [422, "invalid", "that document is not current: it is an older version"],
    [503, "llm_unavailable", "Ollama is not running"],
    [504, "timeout", "The local model took too long"],
  ])("reports a %i as %s with the server's sentence", async (status, kind, detail) => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(ok({ detail }, status)));

    const failed = await createApi().summarizeDocument(7).catch((error: unknown) => error);

    expect(failed).toBeInstanceOf(ApiError);
    expect((failed as ApiError).kind).toBe(kind);
    expect((failed as ApiError).message).toBe(detail);
  });

  it("is answered by the demo backend from the demo notes, and refuses an unknown document", async () => {
    vi.useFakeTimers();
    const api = createMockApi();

    const pending = api.summarizeDocument(1);
    await vi.advanceTimersByTimeAsync(1000);
    const summary = await pending;

    expect(summary.document_id).toBe(1);
    expect(summary.name).toBe("lisbon-trip.md");
    expect(summary.text).toContain("lisbon-trip.md");
    expect(summary.truncated).toBe(false);

    const unknown = api.summarizeDocument(999).catch((error: unknown) => error);
    await vi.advanceTimersByTimeAsync(1000);
    expect(await unknown).toEqual(new Error("That document is not in the library."));
  });
});
