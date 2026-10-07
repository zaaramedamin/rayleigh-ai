import { describe, expect, it } from "vitest";
import { summaryBlocked, summaryNote } from "./DocumentSummary";

describe("summaryNote", () => {
  it("says nothing about a document that was read whole", () => {
    expect(summaryNote({ truncated: false, parts: 2, covered_parts: 2 })).toBeNull();
  });

  it("says how much of a long document was read", () => {
    expect(summaryNote({ truncated: true, parts: 11, covered_parts: 8 })).toBe(
      "This document is long: only the first 8 of its 11 parts were summarized.",
    );
  });
});

describe("summaryBlocked", () => {
  it("allows a current document that has passages", () => {
    expect(summaryBlocked({ status: "active", chunks: 3 })).toBeNull();
  });

  it("explains why a missing file cannot be summarized", () => {
    expect(summaryBlocked({ status: "missing", chunks: 3 })).toMatch(/not found in the last two updates/);
  });

  it("explains that a document with no passages needs an update first", () => {
    expect(summaryBlocked({ status: "active", chunks: 0 })).toMatch(/UPDATE LIBRARY/);
  });

  it("names the missing file before the missing passages", () => {
    expect(summaryBlocked({ status: "missing", chunks: 0 })).toMatch(/not found/);
  });
});
