import { describe, expect, it } from "vitest";
import { filterCommands } from "./commands";
import type { Command } from "./commands";

const make = (id: string, label: string, group: string, keywords?: string): Command => ({
  id,
  label,
  group,
  keywords,
  run: () => undefined,
});

const COMMANDS = [
  make("home", "Go to Home", "NAVIGATE"),
  make("chat", "Go to Chat", "NAVIGATE", "ask question talk"),
  make("lock", "Lock the application", "SYSTEM", "password sign out"),
  make("update", "Update the library", "LIBRARY", "read folders index"),
  make("theme", "Switch theme", "SYSTEM", "colour gold cyan"),
];

describe("filterCommands", () => {
  it("returns everything for an empty query", () => {
    expect(filterCommands(COMMANDS, "  ")).toHaveLength(COMMANDS.length);
  });

  it("matches words in the label, group or keywords", () => {
    expect(filterCommands(COMMANDS, "password").map((c) => c.id)).toEqual(["lock"]);
    expect(filterCommands(COMMANDS, "navigate").map((c) => c.id)).toEqual(["home", "chat"]);
    expect(filterCommands(COMMANDS, "folders").map((c) => c.id)).toEqual(["update"]);
  });

  it("needs every word to match", () => {
    expect(filterCommands(COMMANDS, "go chat").map((c) => c.id)).toEqual(["chat"]);
    expect(filterCommands(COMMANDS, "go password")).toEqual([]);
  });

  it("is not case sensitive", () => {
    expect(filterCommands(COMMANDS, "LOCK").map((c) => c.id)).toEqual(["lock"]);
  });

  it("puts matches at the start of the label first", () => {
    const ordered = filterCommands(COMMANDS, "the").map((c) => c.id);
    expect(ordered).toEqual(["lock", "update", "theme"]);
    const first = filterCommands([make("a", "Open settings", "X"), make("b", "Settings", "X")], "settings");
    expect(first.map((c) => c.id)).toEqual(["b", "a"]);
  });

  it("finds nothing when nothing matches", () => {
    expect(filterCommands(COMMANDS, "zzz")).toEqual([]);
  });
});
