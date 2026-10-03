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

  it("posts the question and unwraps search results", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ results: [{ source: "a.md" }] })));
    vi.stubGlobal("fetch", fetchMock);
    const results = await createApi().search("lisbon", { top_k: 3 });
    expect(results).toEqual([{ source: "a.md" }]);
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ query: "lisbon", top_k: 3 });
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
