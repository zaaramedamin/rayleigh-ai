import { describe, expect, it } from "vitest";
import { locationLabel } from "./SourceCard";

describe("locationLabel", () => {
  it("names the lines for a file without pages", () => {
    expect(locationLabel({ start_line: 5, end_line: 9 })).toBe("L5-9");
    expect(locationLabel({ start_line: 5, end_line: 9, start_page: null, end_page: null })).toBe("L5-9");
  });

  it("names the page for a file with pages", () => {
    expect(locationLabel({ start_line: 5, end_line: 9, start_page: 12, end_page: 12 })).toBe("p. 12");
    expect(locationLabel({ start_line: 5, end_line: 9, start_page: 12 })).toBe("p. 12");
  });

  it("names the range when a passage runs over a page break", () => {
    expect(locationLabel({ start_line: 5, end_line: 9, start_page: 12, end_page: 14 })).toBe("pp. 12-14");
  });
});
