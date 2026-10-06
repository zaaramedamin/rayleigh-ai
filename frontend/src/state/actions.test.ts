import { describe, expect, it } from "vitest";
import type { ChatAction } from "../api/types";
import { actionLabel, parseAction, parseActions } from "./actions";

describe("parseAction", () => {
  it("accepts the known actions with valid values", () => {
    expect(parseAction({ name: "open_page", args: { page: "settings" } })).toEqual({ name: "open_page", page: "settings" });
    expect(parseAction({ name: "set_theme", args: { theme: "mark3" } })).toEqual({ name: "set_theme", theme: "mark3" });
    expect(parseAction({ name: "lock_app", args: {} })).toEqual({ name: "lock_app" });
    expect(parseAction({ name: "ask_notes", args: { question: " When is my flight? " } })).toEqual({
      name: "ask_notes",
      question: "When is my flight?",
    });
  });

  it("turns options into the settings they change", () => {
    expect(parseAction({ name: "set_option", args: { option: "interface_sounds", value: "off" } })).toMatchObject({
      patch: { sound: false },
    });
    expect(parseAction({ name: "set_option", args: { option: "spoken_replies", value: "on" } })).toMatchObject({
      patch: { voice: true },
    });
    // "animations on" means motion is not reduced.
    expect(parseAction({ name: "set_option", args: { option: "animations", value: "on" } })).toMatchObject({
      patch: { reducedMotion: false },
    });
    expect(parseAction({ name: "set_option", args: { option: "notes_mode", value: "on" } })).toMatchObject({
      patch: { chatMode: "notes" },
    });
  });

  it.each<ChatAction>([
    { name: "delete_document", args: { id: "1" } },
    { name: "open_page", args: { page: "C:\\Windows" } },
    { name: "open_page", args: {} },
    { name: "set_theme", args: { theme: "pink" } },
    { name: "set_option", args: { option: "demo", value: "on" } },
    { name: "set_option", args: { option: "animations", value: "maybe" } },
    { name: "ask_notes", args: { question: "  " } },
    { name: "remember", args: { fact: "" } },
  ])("refuses %j", (action) => {
    expect(parseAction(action)).toBeNull();
  });

  it("keeps only the valid actions of a reply", () => {
    expect(
      parseActions([
        { name: "open_page", args: { page: "chat" } },
        { name: "nonsense", args: {} },
      ]),
    ).toEqual([{ name: "open_page", page: "chat" }]);
    expect(parseActions(undefined)).toEqual([]);
  });

  it("describes every action", () => {
    expect(actionLabel({ name: "open_page", page: "profile" })).toBe("OPENED PROFILE");
    expect(actionLabel({ name: "remember", fact: "likes tea" })).toBe("REMEMBERED: likes tea");
    expect(actionLabel({ name: "lock_app" })).toBe("LOCKED");
  });
});
