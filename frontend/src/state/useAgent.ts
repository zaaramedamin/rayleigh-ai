import { useCallback, useEffect, useRef, useState } from "react";
import type { AgentEvent, AgentRun, AgentSettings, AgentSettingsChange, Api } from "../api/types";
import { isActive } from "../lib/agent";
import { events } from "./events";

const message = (error: unknown, fallback: string) => (error instanceof Error ? error.message : fallback);

/**
 * The agent as the interface sees it: what it may do, the task it is working on, and the question
 * it is waiting for you to answer. While a task is going the screen asks how it is getting on, and
 * when the task stops to ask you something, a notice says so even if you are on another page.
 */
export function useAgent(api: Api, pollMs = 800) {
  const [settings, setSettings] = useState<AgentSettings | null>(null);
  const [run, setRun] = useState<AgentRun | null>(null);
  const [log, setLog] = useState<AgentEvent[] | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const announced = useRef<number | null>(null);

  const guard = useCallback(async <T,>(work: () => Promise<T>, fallback: string): Promise<T | undefined> => {
    setBusy(true);
    setError("");
    try {
      return await work();
    } catch (err) {
      setError(message(err, fallback));
      return undefined;
    } finally {
      setBusy(false);
    }
  }, []);

  const load = useCallback(async () => {
    try {
      const loaded = await api.agentSettings();
      setSettings(loaded);
      if (loaded.active_run) setRun(loaded.active_run);
    } catch (err) {
      setError(message(err, "Could not read the agent's settings."));
    }
  }, [api]);

  useEffect(() => {
    void load();
  }, [load]);

  // While a task is going, keep asking how it is getting on.
  const activeId = isActive(run) ? run?.run_id : undefined;
  useEffect(() => {
    if (!activeId) return;
    let cancelled = false;
    const timer = setInterval(async () => {
      try {
        const latest = await api.agentRun(activeId);
        if (!cancelled) setRun(latest);
      } catch (err) {
        if (!cancelled) setError(message(err, "Lost track of the task."));
      }
    }, pollMs);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [api, activeId, pollMs]);

  // A question is waiting: say so once, wherever the owner is in the interface.
  const questionId = run?.question?.id ?? null;
  useEffect(() => {
    if (questionId === null || announced.current === questionId) return;
    announced.current = questionId;
    events.notify("warning", "THE AGENT NEEDS YOU", {
      detail: run?.question?.kind === "approve" ? "It is waiting for your approval." : "Something went wrong.",
      sound: "alert",
    });
  }, [questionId, run?.question?.kind]);

  const change = useCallback(
    (patch: AgentSettingsChange) =>
      guard(async () => {
        setSettings(await api.changeAgent(patch));
      }, "Could not change the agent's settings."),
    [api, guard],
  );

  const start = useCallback(
    (task: string) =>
      guard(async () => {
        setLog(null);
        setRun(await api.startAgentRun(task));
      }, "Could not start the task."),
    [api, guard],
  );

  const answer = useCallback(
    (choice: string) => {
      const question = run?.question;
      if (!run || !question) return Promise.resolve(undefined);
      return guard(async () => {
        await api.answerAgent(run.run_id, question.id, choice);
        setRun(await api.agentRun(run.run_id));
      }, "Could not send your answer.");
    },
    [api, guard, run],
  );

  const stop = useCallback(() => {
    if (!run) return Promise.resolve(undefined);
    return guard(async () => {
      await api.stopAgentRun(run.run_id);
      setRun(await api.agentRun(run.run_id));
    }, "Could not stop the task.");
  }, [api, guard, run]);

  const showLog = useCallback(
    (runId?: string) => guard(async () => setLog(await api.agentLog(runId, 50)), "Could not read the log."),
    [api, guard],
  );

  const eraseLog = useCallback(
    () =>
      guard(async () => {
        const erased = await api.eraseAgentLog();
        setLog([]);
        events.notify("success", "AGENT LOG ERASED", { detail: `${erased} entries removed.`, sound: null });
      }, "Could not erase the log."),
    [api, guard],
  );

  return { settings, run, log, error, busy, change, start, answer, stop, showLog, eraseLog, hideLog: () => setLog(null) };
}
