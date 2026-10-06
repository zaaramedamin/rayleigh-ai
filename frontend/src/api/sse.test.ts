import { describe, expect, it } from "vitest";
import { splitEvents } from "./sse";

describe("splitEvents", () => {
  it("reads complete events and keeps the unfinished one", () => {
    const { events, rest } = splitEvents(
      'event: token\ndata: {"text":"Sim"}\n\nevent: token\ndata: {"text":"mer"}\n\nevent: do',
    );

    expect(events).toEqual([
      { event: "token", data: '{"text":"Sim"}' },
      { event: "token", data: '{"text":"mer"}' },
    ]);
    expect(rest).toBe("event: do");
  });

  it("reads an event that arrives in two pieces", () => {
    const first = splitEvents('event: done\ndata: {"answer":');
    expect(first.events).toEqual([]);

    const second = splitEvents(first.rest + '"x"}\n\n');

    expect(second).toEqual({ events: [{ event: "done", data: '{"answer":"x"}' }], rest: "" });
  });

  it("accepts Windows line endings, even when a pair is split between pieces", () => {
    const first = splitEvents("event: a\r\ndata: 1\r");
    const second = splitEvents(first.rest + "\n\r\n");

    expect(second.events).toEqual([{ event: "a", data: "1" }]);
  });

  it("joins several data lines and ignores comments and events without data", () => {
    const { events } = splitEvents(": keep-alive\n\nevent: x\ndata: one\ndata: two\n\nevent: empty\n\n");

    expect(events).toEqual([{ event: "x", data: "one\ntwo" }]);
  });

  it("calls an event without a name a message and keeps text after the colon as it is", () => {
    const { events } = splitEvents("data:  two spaces\n\n");

    expect(events).toEqual([{ event: "message", data: " two spaces" }]);
  });

  it("returns nothing for nothing", () => {
    expect(splitEvents("")).toEqual({ events: [], rest: "" });
  });
});
