import type { AgentEvent, AgentLevel, AgentQuestion, AgentRun, AgentRunStatus } from "../api/types";

/** What each answer to a question is called on its button. */
const CHOICES: Record<string, string> = {
  allow: "ALLOW ONCE",
  deny: "DO NOT ALLOW",
  stop: "STOP THE TASK",
  retry: "TRY AGAIN",
  skip: "SKIP IT",
  continue: "LET IT GO ON",
};

/** The button text for an answer. An answer this build does not know is shown as it is, never hidden. */
export function choiceLabel(choice: string): string {
  return CHOICES[choice] ?? choice.toUpperCase();
}

/** Which answers get the warning colour: refusing and stopping are always safe, so they are never red. */
export function choiceTone(choice: string): "go" | "safe" {
  return choice === "allow" || choice === "retry" || choice === "continue" ? "go" : "safe";
}

const STATUS: Record<AgentRunStatus, string> = {
  running: "WORKING",
  waiting: "WAITING FOR YOU",
  done: "DONE",
  stopped: "STOPPED",
  failed: "FAILED",
};

export function statusLabel(status: AgentRunStatus): string {
  return STATUS[status];
}

/** True while the task is still going, so a screen keeps asking how it is getting on. */
export function isActive(run: Pick<AgentRun, "status"> | null | undefined): boolean {
  return run?.status === "running" || run?.status === "waiting";
}

const LEVELS: Record<AgentLevel, string> = {
  read_local: "READS ONLY",
  open_local: "ASKS EVERY TIME",
  external_read: "INTERNET, ASKS EVERY TIME",
  write_local: "ASKS EVERY TIME",
  destructive: "NEVER ALLOWED",
};

export function levelLabel(level: AgentLevel): string {
  return LEVELS[level];
}

/** The exact arguments of an action, one `name = value` line each, for the approval card. */
export function argumentLines(args: Record<string, unknown>): string[] {
  return Object.entries(args).map(([name, value]) => `${name} = ${typeof value === "string" ? value : JSON.stringify(value)}`);
}

/** A sentence about what the model read before asking, or null when it read nothing. */
export function readWarning(question: Pick<AgentQuestion, "read_sources">): string | null {
  if (question.read_sources.length === 0) return null;
  return `The model has read the results of: ${question.read_sources.join(", ")}. Text it read could have influenced this request, so check it is what you wanted.`;
}

/** One line of the log: when, what, who decided. */
export function eventLine(event: Pick<AgentEvent, "time" | "kind" | "tool" | "decision" | "decided_by" | "detail">): string {
  const when = new Date(event.time);
  const time = Number.isNaN(when.getTime()) ? event.time : when.toLocaleTimeString();
  const who = event.decision ? ` ${event.decision}${event.decided_by ? ` by ${event.decided_by}` : ""}` : "";
  const detail = event.detail ? `  ${event.detail}` : "";
  return `${time}  ${event.kind.replace("_", " ")}${event.tool ? ` ${event.tool}` : ""}${who}${detail}`;
}
