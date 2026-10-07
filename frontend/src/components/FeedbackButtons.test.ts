import { describe, expect, it } from "vitest";
import { kindsToOffer } from "./FeedbackButtons";

describe("kindsToOffer", () => {
  it("offers a thumbs up and a thumbs down before anything is marked", () => {
    expect(kindsToOffer(undefined, "notes")).toEqual(["helpful", "not_helpful"]);
    expect(kindsToOffer(undefined, "general")).toEqual(["helpful", "not_helpful"]);
  });

  it("does not ask what went wrong of an answer that helped", () => {
    expect(kindsToOffer("helpful", "notes")).toEqual(["helpful", "not_helpful"]);
  });

  it("asks what went wrong after a thumbs down on an answer from the notes", () => {
    expect(kindsToOffer("not_helpful", "notes")).toEqual(["helpful", "not_helpful", "wrong_source", "missing_info"]);
  });

  it("keeps the chosen reason visible once it is chosen", () => {
    expect(kindsToOffer("wrong_source", "notes")).toContain("wrong_source");
    expect(kindsToOffer("missing_info", "notes")).toContain("missing_info");
  });

  it("has no sources or notes to blame in a general reply, so only the thumbs", () => {
    expect(kindsToOffer("not_helpful", "general")).toEqual(["helpful", "not_helpful"]);
  });
});
