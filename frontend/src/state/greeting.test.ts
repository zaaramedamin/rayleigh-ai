import { describe, expect, it } from "vitest";
import type { SystemStatus } from "../api/types";
import { addressed, greetingFor, statusReport } from "./greeting";

const healthy: SystemStatus = {
  embedding: { model: "m", downloaded: true },
  library: { documents: 12, chunks: 90, searchable_documents: 12, pending_documents: 0 },
  llm: { model: "q", state: "ready", hint: null },
};
const at = (hour: number) => new Date(2026, 9, 5, hour, 0);

describe("addressed", () => {
  it("adds the form of address, or nothing", () => {
    expect(addressed(" sir ")).toBe(", sir");
    expect(addressed("")).toBe("");
  });
});

describe("greetingFor", () => {
  it("greets by the time of day, like a butler", () => {
    expect(greetingFor(at(8), "sir", healthy, true)).toMatch(/^Good morning, sir\./);
    expect(greetingFor(at(14), "sir", healthy, true)).toMatch(/^Good afternoon, sir\./);
    expect(greetingFor(at(22), "sir", healthy, true)).toMatch(/^Good evening, sir\./);
    expect(greetingFor(at(3), "sir", healthy, true)).toMatch(/^Good evening/);
  });

  it("reports the systems and asks how it can serve", () => {
    expect(greetingFor(at(9), "sir", healthy, true)).toBe(
      "Good morning, sir. All systems are online. Your library holds 12 documents, all searchable. How can I serve you?",
    );
  });

  it("works without a form of address or a known status", () => {
    expect(greetingFor(at(9), "", null, true)).toBe("Good morning. How can I serve you?");
  });
});

describe("statusReport", () => {
  it("says only what the backend reported", () => {
    expect(statusReport(null, false)).toMatch(/backend is not answering/);
    expect(statusReport(null, true)).toMatch(/still checking/);
  });

  it("explains each problem", () => {
    const stopped = { ...healthy, llm: { model: "q", state: "not_running" as const, hint: null } };
    expect(statusReport(stopped, true)).toMatch(/local model is not running/);
    const missing = { ...healthy, embedding: { model: "m", downloaded: false } };
    expect(statusReport(missing, true)).toMatch(/embedding model is missing/);
    const pending = { ...healthy, library: { ...healthy.library, pending_documents: 1 } };
    expect(statusReport(pending, true)).toMatch(/1 still needs indexing/);
    const empty = { ...healthy, library: { documents: 0, chunks: 0, searchable_documents: 0, pending_documents: 0 } };
    expect(statusReport(empty, true)).toMatch(/library is empty/);
  });
});
