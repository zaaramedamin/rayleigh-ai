import { afterEach, describe, expect, it, vi } from "vitest";
import { createApi } from "./client";
import { createMockApi } from "./mock";

afterEach(() => vi.unstubAllGlobals());

const ok = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });

describe("feedback calls", () => {
  it("sends a mark with the question, the answer and what the interface knew", async () => {
    const fetchMock = vi.fn().mockResolvedValue(ok({ id: 5, kind: "wrong_source" }, 201));
    vi.stubGlobal("fetch", fetchMock);

    const mark = await createApi().addFeedback({
      kind: "wrong_source",
      mode: "notes",
      question: "How long do oats simmer?",
      answer: "Rice needs eighteen minutes [1].",
      details: { sources: [{ document_id: 4, source: "rice.txt" }] },
    });

    expect(mark.id).toBe(5);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/v1/feedback");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({
      kind: "wrong_source",
      mode: "notes",
      question: "How long do oats simmer?",
      answer: "Rice needs eighteen minutes [1].",
      details: { sources: [{ document_id: 4, source: "rice.txt" }] },
    });
  });

  it("changes a mark with PUT, sending a note only when there is one", async () => {
    const fetchMock = vi.fn().mockImplementation(async () => ok({ id: 5 }));
    vi.stubGlobal("fetch", fetchMock);
    const api = createApi();

    await api.changeFeedback(5, "missing_info");
    await api.changeFeedback(5, "wrong_source", "it used the rice note");

    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/feedback/5");
    expect(fetchMock.mock.calls[0][1].method).toBe("PUT");
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ kind: "missing_info" });
    expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({ kind: "wrong_source", note: "it used the rice note" });
  });

  it("takes a mark back with DELETE and accepts the empty answer", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(createApi().deleteFeedback(5)).resolves.toBeUndefined();

    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/feedback/5");
    expect(fetchMock.mock.calls[0][1].method).toBe("DELETE");
  });

  it("reports a full store as a conflict and a missing mark as not found", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValueOnce(ok({ detail: "5000 marks are kept" }, 409)));
    await expect(
      createApi().addFeedback({ kind: "helpful", mode: "notes", question: "q", answer: "a" }),
    ).rejects.toMatchObject({ kind: "conflict" });

    vi.stubGlobal("fetch", vi.fn().mockResolvedValueOnce(ok({ detail: "That mark does not exist." }, 404)));
    await expect(createApi().deleteFeedback(9)).rejects.toMatchObject({ kind: "not_found" });
  });
});

describe("the demo backend keeps marks in memory", () => {
  it("adds, changes and takes back a mark", async () => {
    const api = createMockApi();

    const mark = await api.addFeedback({ kind: "not_helpful", mode: "notes", question: "q", answer: "a" });
    expect(mark).toMatchObject({ id: 1, kind: "not_helpful", note: null });
    const changed = await api.changeFeedback(mark.id, "wrong_source", "wrong note");
    expect(changed).toMatchObject({ kind: "wrong_source", note: "wrong note" });
    await api.deleteFeedback(mark.id);

    await expect(api.deleteFeedback(mark.id)).rejects.toMatchObject({ kind: "not_found" });
    await expect(api.changeFeedback(mark.id, "helpful")).rejects.toMatchObject({ kind: "not_found" });
  });
});
