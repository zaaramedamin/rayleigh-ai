import { describe, expect, it } from "vitest";
import { argumentLines, choiceLabel, choiceTone, eventLine, isActive, levelLabel, readWarning, statusLabel } from "./agent";

describe("choiceLabel", () => {
  it("names every answer the agent can ask for", () => {
    expect(["allow", "deny", "stop", "retry", "skip", "continue"].map(choiceLabel)).toEqual([
      "ALLOW ONCE",
      "DO NOT ALLOW",
      "STOP THE TASK",
      "TRY AGAIN",
      "SKIP IT",
      "LET IT GO ON",
    ]);
  });

  it("shows an answer it does not know as it is, never hides it", () => {
    expect(choiceLabel("escalate")).toBe("ESCALATE");
  });
});

describe("choiceTone", () => {
  it("marks only the answers that let something go ahead", () => {
    expect(["allow", "retry", "continue"].map(choiceTone)).toEqual(["go", "go", "go"]);
    expect(["deny", "stop", "skip", "anything"].map(choiceTone)).toEqual(["safe", "safe", "safe", "safe"]);
  });
});

describe("status", () => {
  it("has a plain label for every status", () => {
    expect((["running", "waiting", "done", "stopped", "failed"] as const).map(statusLabel)).toEqual([
      "WORKING",
      "WAITING FOR YOU",
      "DONE",
      "STOPPED",
      "FAILED",
    ]);
  });

  it("is active only while the task is going", () => {
    expect(isActive({ status: "running" })).toBe(true);
    expect(isActive({ status: "waiting" })).toBe(true);
    expect(isActive({ status: "done" })).toBe(false);
    expect(isActive({ status: "stopped" })).toBe(false);
    expect(isActive({ status: "failed" })).toBe(false);
    expect(isActive(null)).toBe(false);
    expect(isActive(undefined)).toBe(false);
  });
});

describe("levelLabel", () => {
  it("says what a tool will do, and that anything above reading asks every time", () => {
    expect(levelLabel("read_local")).toBe("READS ONLY");
    expect(levelLabel("open_local")).toBe("ASKS EVERY TIME");
    expect(levelLabel("external_read")).toContain("INTERNET");
    expect(levelLabel("destructive")).toBe("NEVER ALLOWED");
  });
});

describe("argumentLines", () => {
  it("shows every argument exactly, one per line", () => {
    expect(argumentLines({ url: "https://example.com/a?b=1", count: 3, deep: { a: [1] } })).toEqual([
      "url = https://example.com/a?b=1",
      "count = 3",
      'deep = {"a":[1]}',
    ]);
  });

  it("shows nothing for an action with no arguments", () => {
    expect(argumentLines({})).toEqual([]);
  });
});

describe("readWarning", () => {
  it("is silent when the model read nothing", () => {
    expect(readWarning({ read_sources: [] })).toBeNull();
  });

  it("names what it read and says to check", () => {
    const text = readWarning({ read_sources: ["search_notes", "fetch_web_page"] });

    expect(text).toContain("search_notes, fetch_web_page");
    expect(text).toContain("could have influenced this request");
  });
});

describe("eventLine", () => {
  const base = { time: "2026-10-08T12:00:00Z", kind: "decision", tool: null, decision: null, decided_by: null, detail: null };

  it("writes who decided what about which tool", () => {
    const line = eventLine({ ...base, tool: "open_path", decision: "deny", decided_by: "owner", detail: "deny" });

    expect(line).toContain("decision open_path deny by owner  deny");
  });

  it("writes a kind with underscores as words and leaves out what is missing", () => {
    expect(eventLine({ ...base, kind: "run_started" })).toMatch(/run started$/);
  });

  it("keeps a time it cannot read", () => {
    expect(eventLine({ ...base, time: "not a time" })).toMatch(/^not a time {2}decision/);
  });
});
