import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import type { AgentQuestion, AgentRun, Api } from "../api/types";
import {
  argumentLines,
  choiceLabel,
  choiceTone,
  eventLine,
  isActive,
  levelLabel,
  readWarning,
  statusLabel,
} from "../lib/agent";
import { useAgent } from "../state/useAgent";

/** What the agent wants to do, or what went wrong, and the only answers that mean anything. */
function QuestionCard({
  question,
  busy,
  onAnswer,
}: {
  question: AgentQuestion;
  busy: boolean;
  onAnswer: (choice: string) => void;
}) {
  const warning = readWarning(question);
  const card = useRef<HTMLElement>(null);
  // A question the owner cannot see is no question: bring it into view and give it the keyboard.
  useEffect(() => {
    card.current?.scrollIntoView?.({ block: "center", behavior: "smooth" });
    card.current?.focus({ preventScroll: true });
  }, [question.id]);
  return (
    <section
      ref={card}
      tabIndex={-1}
      className={`agent-card agent-card--${question.kind}`}
      role="alertdialog"
      aria-label={question.title}
      aria-describedby="agent-card-message"
    >
      <b className="agent-card-title">{question.kind === "approve" ? "THE AGENT WANTS YOUR APPROVAL" : "SOMETHING WENT WRONG"}</b>
      <p id="agent-card-message" className="agent-card-message">
        {question.message}
      </p>
      {question.kind === "approve" && (
        <>
          {argumentLines(question.arguments).length > 0 && (
            <div className="agent-exact">
              <span className="muted">EXACTLY</span>
              {argumentLines(question.arguments).map((line) => (
                <code key={line} className="mono">
                  {line}
                </code>
              ))}
            </div>
          )}
          <p className="muted">Why you are asked: {question.reason}</p>
          {warning && <p className="warn">{warning}</p>}
        </>
      )}
      <div className="agent-choices">
        {question.options.map((choice) => (
          <button
            key={choice}
            className={`btn ${choiceTone(choice) === "go" ? "" : "btn--ghost"}`}
            disabled={busy}
            onClick={() => onAnswer(choice)}
          >
            {choiceLabel(choice)}
          </button>
        ))}
      </div>
    </section>
  );
}

function RunView({ run, busy, onAnswer, onStop }: { run: AgentRun; busy: boolean; onAnswer: (c: string) => void; onStop: () => void }) {
  const lines = run.progress.slice(-4);
  return (
    <div className="agent-run" aria-live="polite">
      <div className="agent-run-head">
        <span className={`badge agent-status agent-status--${run.status}`}>{statusLabel(run.status)}</span>
        <span className="ellipsis" title={run.task}>
          {run.task}
        </span>
        {isActive(run) && (
          <button className="btn btn--ghost" disabled={busy} onClick={onStop}>
            STOP
          </button>
        )}
      </div>
      {run.question ? (
        <QuestionCard question={run.question} busy={busy} onAnswer={onAnswer} />
      ) : (
        isActive(run) && (
          <ul className="agent-progress muted">
            {lines.map((line, index) => (
              <li key={`${index}-${line}`}>{line}</li>
            ))}
          </ul>
        )
      )}
      {run.steps.length > 0 && (
        <ul className="agent-steps">
          {run.steps.map((step, index) => (
            <li key={`${index}-${step.effect}`}>
              <span className="ellipsis" title={step.effect}>
                {step.effect}
              </span>
              <span className={step.outcome === "done" ? "ok" : "warn"}>{step.outcome.toUpperCase()}</span>
            </li>
          ))}
        </ul>
      )}
      {!isActive(run) && <p className={`agent-answer ${run.status === "done" ? "" : "warn"}`}>{run.answer}</p>}
    </div>
  );
}

/** Switch the agent on, choose what it may do, give it a task, and answer when it asks. */
export function AgentPanel({ api }: { api: Api }) {
  const agent = useAgent(api);
  const [task, setTask] = useState("");
  const [confirmErase, setConfirmErase] = useState(false);
  const { settings, run } = agent;

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!task.trim()) return;
    await agent.start(task);
    setTask("");
  };

  if (!settings) {
    return agent.error ? (
      <p className="bad" role="alert">
        {agent.error}
      </p>
    ) : (
      <p className="muted">Loading...</p>
    );
  }

  const working = isActive(run);
  const internetTool = settings.tools.some((tool) => tool.level === "external_read");
  return (
    <div className="agent">
      <p className="muted">
        The agent carries out a task by itself, using only the tools you switch on. It is off until you switch it on, and
        anything that opens a file, starts a program or reads the internet asks you first, every time, showing exactly what
        it will do. It never deletes anything, and every request is written to a log you can read.
      </p>

      <label className="toggle">
        <input
          type="checkbox"
          checked={settings.enabled}
          disabled={agent.busy}
          onChange={(e) => void agent.change({ enabled: e.target.checked })}
        />
        <span>
          <b>Agent {settings.enabled ? "ON" : "OFF"}</b> (the master switch: nothing runs while it is off)
        </span>
      </label>
      {internetTool && (
        <label className="toggle">
          <input
            type="checkbox"
            checked={settings.web}
            disabled={agent.busy}
            onChange={(e) => void agent.change({ web: e.target.checked })}
          />
          <span>Allow the internet (a page is still read only after you approve its exact address)</span>
        </label>
      )}

      <h4 className="inspector-title">TOOLS</h4>
      <ul className="agent-tools">
        {settings.tools.map((tool) => (
          <li key={tool.name}>
            <label className="toggle">
              <input
                type="checkbox"
                checked={tool.enabled}
                disabled={agent.busy || tool.level === "destructive"}
                onChange={(e) =>
                  void agent.change({
                    tools: settings.tools
                      .filter((other) => (other.name === tool.name ? e.target.checked : other.enabled))
                      .map((other) => other.name),
                  })
                }
              />
              <span>
                <b className="mono">{tool.name}</b> <span className="badge">{levelLabel(tool.level)}</span>
                <br />
                <span className="muted">{tool.description}</span>
                {tool.level === "external_read" && !settings.web && <span className="warn"> Needs the internet switch above.</span>}
              </span>
            </label>
          </li>
        ))}
      </ul>

      <h4 className="inspector-title">TASK</h4>
      <form className="agent-task" onSubmit={(e) => void submit(e)}>
        <input
          value={task}
          maxLength={2000}
          disabled={!settings.enabled || working}
          placeholder={settings.enabled ? "What should the agent do?" : "Switch the agent on first"}
          aria-label="Task for the agent"
          onChange={(e) => setTask(e.target.value)}
        />
        <button className="btn" type="submit" disabled={!settings.enabled || working || agent.busy || !task.trim()}>
          START
        </button>
      </form>

      {agent.error && (
        <p className="bad" role="alert">
          {agent.error}
        </p>
      )}
      {run && <RunView run={run} busy={agent.busy} onAnswer={(c) => void agent.answer(c)} onStop={() => void agent.stop()} />}

      <h4 className="inspector-title">LOG</h4>
      <p className="muted">Everything the agent was asked to do, including what was refused. Entries can be erased as a whole, never edited.</p>
      <div className="agent-log-actions">
        <button className="btn btn--ghost" disabled={agent.busy} onClick={() => void agent.showLog(run?.run_id)}>
          {run ? "SHOW THIS TASK" : "SHOW LOG"}
        </button>
        <button className="btn btn--ghost" disabled={agent.busy} onClick={() => void agent.showLog()}>
          SHOW LATEST
        </button>
        {agent.log && (
          <button className="link" onClick={agent.hideLog}>
            hide
          </button>
        )}
        {confirmErase ? (
          <>
            <span className="warn">Erase the whole log?</span>
            <button
              className="btn"
              disabled={agent.busy}
              onClick={() => {
                setConfirmErase(false);
                void agent.eraseLog();
              }}
            >
              YES, ERASE
            </button>
            <button className="btn btn--ghost" onClick={() => setConfirmErase(false)}>
              CANCEL
            </button>
          </>
        ) : (
          <button className="btn btn--ghost" disabled={agent.busy} onClick={() => setConfirmErase(true)}>
            ERASE LOG
          </button>
        )}
      </div>
      {agent.log &&
        (agent.log.length === 0 ? (
          <p className="muted">The log is empty.</p>
        ) : (
          <ul className="agent-log mono">
            {agent.log.map((entry) => (
              <li key={entry.id}>{eventLine(entry)}</li>
            ))}
          </ul>
        ))}
    </div>
  );
}
