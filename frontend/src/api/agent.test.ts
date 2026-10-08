import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, createApi } from "./client";
import { createMockApi } from "./mock";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

const ok = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });

describe("the agent calls", () => {
  it("reads and changes the switches with GET and PUT, sending only what changed", async () => {
    const fetchMock = vi.fn().mockImplementation(async () => ok({ enabled: true, web: false, tools: [], active_run: null }));
    vi.stubGlobal("fetch", fetchMock);
    const api = createApi();

    await api.agentSettings();
    await api.changeAgent({ enabled: true });
    await api.changeAgent({ tools: ["calculator", "open_path"] });

    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/agent");
    expect(fetchMock.mock.calls[1][1].method).toBe("PUT");
    expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({ enabled: true });
    expect(JSON.parse(fetchMock.mock.calls[2][1].body)).toEqual({ tools: ["calculator", "open_path"] });
  });

  it("starts a task, watches it and stops it", async () => {
    const fetchMock = vi.fn().mockImplementation(async () => ok({ run_id: "abc" }, 202));
    vi.stubGlobal("fetch", fetchMock);
    const api = createApi();

    await api.startAgentRun("open my plan");
    await api.agentRun("abc");
    await api.stopAgentRun("abc");

    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/agent/runs");
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ task: "open my plan" });
    expect(fetchMock.mock.calls[1][0]).toBe("/api/v1/agent/runs/abc");
    expect(fetchMock.mock.calls[2][0]).toBe("/api/v1/agent/runs/abc/stop");
    expect(fetchMock.mock.calls[2][1].method).toBe("POST");
  });

  it("answers a question with its number and the chosen option, and accepts the empty answer", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);

    await createApi().answerAgent("abc", 7, "allow");

    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/agent/runs/abc/answer");
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ question_id: 7, choice: "allow" });
  });

  it("does not let a task number change the address it is sent to", async () => {
    const fetchMock = vi.fn().mockResolvedValue(ok({}));
    vi.stubGlobal("fetch", fetchMock);

    await createApi().agentRun("a/../../auth/logout?x=1");

    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/agent/runs/a%2F..%2F..%2Fauth%2Flogout%3Fx%3D1");
  });

  it("reads the log with a limit, for one task or for all, and erases it", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(ok({ events: [{ id: 1 }] }))
      .mockResolvedValueOnce(ok({ events: [] }))
      .mockResolvedValueOnce(ok({ erased: 5 }));
    vi.stubGlobal("fetch", fetchMock);
    const api = createApi();

    expect(await api.agentLog()).toEqual([{ id: 1 }]);
    await api.agentLog("abc", 10);
    expect(await api.eraseAgentLog()).toBe(5);

    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/agent/log?limit=50");
    expect(fetchMock.mock.calls[1][0]).toBe("/api/v1/agent/log?limit=10&run_id=abc");
    expect(fetchMock.mock.calls[2][1].method).toBe("DELETE");
  });

  it.each([
    [409, "conflict", "The agent is switched off."],
    [422, "invalid", "No such tool: delete_all."],
    [404, "not_found", "There is no such task."],
  ])("reports a %i as %s with the server's sentence", async (status, kind, detail) => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(ok({ detail }, status)));

    const failed = await createApi().startAgentRun("x").catch((error: unknown) => error);

    expect(failed).toBeInstanceOf(ApiError);
    expect((failed as ApiError).kind).toBe(kind);
    expect((failed as ApiError).message).toBe(detail);
  });
});

describe("the demo agent", () => {
  /** Let an API call finish (each waits a fraction of a second) without letting the demo agent move on. */
  const ready = () => {
    vi.useFakeTimers();
    const api = createMockApi();
    const call = async <T>(promise: Promise<T>): Promise<T> => {
      await vi.advanceTimersByTimeAsync(200);
      return promise;
    };
    /** Let time pass for the demo agent's own steps. */
    const pass = (ms: number) => vi.advanceTimersByTimeAsync(ms);
    return { api, call, pass };
  };

  const rejected = async (promise: Promise<unknown>) => {
    const caught = promise.catch((error: unknown) => error);
    await vi.advanceTimersByTimeAsync(200);
    return caught;
  };

  it("starts switched off, with every tool listed and off", async () => {
    const { api, call } = ready();

    const settings = await call(api.agentSettings());

    expect(settings.enabled).toBe(false);
    expect(settings.web).toBe(false);
    expect(settings.active_run).toBeNull();
    expect(settings.tools.map((t) => t.name)).toEqual([
      "calculator",
      "current_time",
      "search_notes",
      "open_path",
      "open_app",
      "fetch_web_page",
    ]);
    expect(settings.tools.some((t) => t.enabled)).toBe(false);
  });

  it("refuses a task while it is off and refuses unknown tools", async () => {
    const { api, call } = ready();

    expect(((await rejected(api.startAgentRun("hello"))) as ApiError).message).toMatch(/switched off/);
    expect(((await rejected(api.changeAgent({ tools: ["delete_all"] }))) as Error).message).toBe("No such tool: delete_all.");
    expect((await call(api.agentSettings())).tools.some((t) => t.enabled)).toBe(false);
  });

  it("finishes a task that needs nothing by itself", async () => {
    const { api, call, pass } = ready();
    await call(api.changeAgent({ enabled: true, tools: ["calculator"] }));

    const started = await call(api.startAgentRun("add up my costs"));
    expect(started.status).toBe("running");
    await pass(1500);
    const done = await call(api.agentRun(started.run_id));

    expect(done.status).toBe("done");
    expect(done.answer).toContain("Nothing was changed");
    expect(done.question).toBeNull();
  });

  it("asks before opening something, and opens nothing until allowed", async () => {
    const { api, call, pass } = ready();
    await call(api.changeAgent({ enabled: true, tools: ["open_path"] }));

    const started = await call(api.startAgentRun("open my plan"));
    await pass(1000);
    const waiting = await call(api.agentRun(started.run_id));

    expect(waiting.status).toBe("waiting");
    expect(waiting.question?.kind).toBe("approve");
    expect(waiting.question?.options).toEqual(["allow", "deny", "stop"]);
    expect(waiting.question?.effect).toContain("plan.txt");
    expect(waiting.steps).toEqual([]);
    expect((await call(api.agentSettings())).active_run?.status).toBe("waiting");
  });

  it("goes on when allowed, and does not when refused or stopped", async () => {
    const { api, call, pass } = ready();
    await call(api.changeAgent({ enabled: true, tools: ["open_path"] }));
    const newRun = async () => {
      const started = await call(api.startAgentRun("open my plan"));
      await pass(1000);
      return call(api.agentRun(started.run_id));
    };
    const answer = async (id: string, question: number, choice: string) => {
      await call(api.answerAgent(id, question, choice));
      await pass(1500);
      return call(api.agentRun(id));
    };

    const first = await newRun();
    const allowed = await answer(first.run_id, first.question!.id, "allow");
    expect(allowed.status).toBe("done");
    expect(allowed.steps.map((s) => s.outcome)).toEqual(["done"]);

    const second = await newRun();
    const denied = await answer(second.run_id, second.question!.id, "deny");
    expect(denied.status).toBe("done");
    expect(denied.steps.map((s) => s.outcome)).toEqual(["not allowed by you"]);

    const third = await newRun();
    const stopped = await answer(third.run_id, third.question!.id, "stop");
    expect(stopped.status).toBe("stopped");
    expect(stopped.steps).toEqual([]);
  });

  it("refuses an answer that is not an option or is for an old question", async () => {
    const { api, call, pass } = ready();
    await call(api.changeAgent({ enabled: true, tools: ["open_path"] }));
    const started = await call(api.startAgentRun("open my plan"));
    await pass(1000);
    const waiting = await call(api.agentRun(started.run_id));

    const bad = (await rejected(api.answerAgent(started.run_id, waiting.question!.id, "yes"))) as ApiError;
    const stale = (await rejected(api.answerAgent(started.run_id, waiting.question!.id + 5, "allow"))) as ApiError;

    expect(bad.kind).toBe("invalid");
    expect(stale.kind).toBe("conflict");
    expect((await call(api.agentRun(started.run_id))).status).toBe("waiting");
  });

  it("can be stopped while waiting, but not once it has finished", async () => {
    const { api, call, pass } = ready();
    await call(api.changeAgent({ enabled: true, tools: ["open_path"] }));
    const started = await call(api.startAgentRun("open my plan"));
    await pass(1000);

    await call(api.stopAgentRun(started.run_id));

    expect((await call(api.agentRun(started.run_id))).status).toBe("stopped");
    expect(((await rejected(api.stopAgentRun(started.run_id))) as ApiError).kind).toBe("conflict");
  });

  it("allows only one task at a time", async () => {
    const { api, call } = ready();
    await call(api.changeAgent({ enabled: true }));
    await call(api.startAgentRun("first"));

    expect(((await rejected(api.startAgentRun("second"))) as ApiError).message).toMatch(/already working/);
  });

  it("keeps a log of what happened, newest first, and can erase it", async () => {
    const { api, call, pass } = ready();
    await call(api.changeAgent({ enabled: true }));
    const started = await call(api.startAgentRun("say hi"));
    await pass(1500);

    expect((await call(api.agentLog())).map((e) => e.kind)).toEqual(["run_finished", "run_started"]);
    expect((await call(api.agentLog(started.run_id))).map((e) => e.kind)).toEqual(["run_started", "run_finished"]);
    expect(await call(api.eraseAgentLog())).toBe(2);
    expect(await call(api.agentLog())).toEqual([]);
  });
});
