import { describe, expect, it } from "vitest";
import { jobSummary, locationPath, locationState } from "./documents";

describe("locationPath", () => {
  it("joins the folder and the path with one slash", () => {
    expect(locationPath({ folder: "C:\\Users\\me\\notes\\", path: "work\\plan.md" })).toBe("C:\\Users\\me\\notes/work/plan.md");
    expect(locationPath({ folder: "/home/me/notes", path: "/plan.md" })).toBe("/home/me/notes/plan.md");
  });

  it("shows only the folder when the path is empty", () => {
    expect(locationPath({ folder: "/home/me/notes", path: "" })).toBe("/home/me/notes");
  });
});

describe("locationState", () => {
  it("says a present file is found", () => {
    expect(locationState({ status: "present", misses: 0 })).toBe("found");
  });

  it("says how many updates did not find a missing file", () => {
    expect(locationState({ status: "missing", misses: 1 })).toBe("not found in the last update");
    expect(locationState({ status: "missing", misses: 3 })).toBe("not found in the last 3 updates");
  });
});

describe("jobSummary", () => {
  const done = { state: "done", message: null, added: 0, replaced: 0, missing: 0, failed_files: 0, indexed: 0 } as const;

  it("lists only what changed", () => {
    expect(jobSummary({ ...done, added: 3, indexed: 40 })).toBe("3 added, 40 made searchable");
    expect(jobSummary({ ...done, replaced: 1, missing: 2, failed_files: 1 })).toBe(
      "1 replaced by an edit, 2 file not found, 1 could not be read",
    );
  });

  it("says so when nothing changed", () => {
    expect(jobSummary(done)).toBe("nothing changed");
  });

  it("explains a running, interrupted or failed update without counts", () => {
    expect(jobSummary({ ...done, state: "running" })).toBe("running now");
    expect(jobSummary({ ...done, state: "interrupted" })).toContain("carries on");
    expect(jobSummary({ ...done, state: "failed", message: "The folder was not found." })).toBe("The folder was not found.");
    expect(jobSummary({ ...done, state: "failed" })).toBe("stopped with an error");
  });
});
