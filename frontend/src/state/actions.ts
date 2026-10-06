import type { ChatAction } from "../api/types";
import type { ViewId } from "../components/NavRail";
import type { Settings, ThemeName } from "./settings";

// What the assistant may do in the interface. The backend only lets through requests from its
// own fixed list; this file checks each one again, so nothing outside the list can ever run
// here. Deleting things, changing the password and anything outside the application are not on
// the list at all.

const PAGES: readonly ViewId[] = ["home", "chat", "knowledge", "profile", "modules", "privacy", "settings"];
const THEMES: readonly ThemeName[] = ["reactor", "mark3"];

export type AppAction =
  | { name: "open_page"; page: ViewId }
  | { name: "set_theme"; theme: ThemeName }
  | { name: "set_option"; patch: Partial<Settings>; label: string }
  | { name: "ask_notes"; question: string }
  | { name: "update_library" }
  | { name: "report_status" }
  | { name: "clear_conversation" }
  | { name: "lock_app" }
  | { name: "remember"; fact: string };

function optionPatch(option: string, on: boolean): { patch: Partial<Settings>; label: string } | null {
  const state = on ? "ON" : "OFF";
  switch (option) {
    case "interface_sounds":
      return { patch: { sound: on }, label: `SOUNDS ${state}` };
    case "spoken_replies":
      return { patch: { voice: on }, label: `VOICE ${state}` };
    case "animations":
      return { patch: { reducedMotion: !on }, label: `ANIMATIONS ${state}` };
    case "notes_mode":
      return { patch: { chatMode: on ? "notes" : "general" }, label: `MY NOTES ${state}` };
    default:
      return null;
  }
}

/** The action a request stands for, or null if it is not exactly one the interface knows. */
export function parseAction(action: ChatAction): AppAction | null {
  const args = action.args ?? {};
  switch (action.name) {
    case "open_page": {
      const page = PAGES.find((p) => p === args.page);
      return page ? { name: "open_page", page } : null;
    }
    case "set_theme": {
      const theme = THEMES.find((t) => t === args.theme);
      return theme ? { name: "set_theme", theme } : null;
    }
    case "set_option": {
      if (args.value !== "on" && args.value !== "off") return null;
      const option = optionPatch(args.option, args.value === "on");
      return option ? { name: "set_option", ...option } : null;
    }
    case "ask_notes": {
      const question = typeof args.question === "string" ? args.question.trim() : "";
      return question ? { name: "ask_notes", question } : null;
    }
    case "remember": {
      const fact = typeof args.fact === "string" ? args.fact.trim() : "";
      return fact ? { name: "remember", fact } : null;
    }
    case "update_library":
    case "report_status":
    case "clear_conversation":
    case "lock_app":
      return { name: action.name };
    default:
      return null;
  }
}

export function parseActions(actions: ChatAction[] | undefined): AppAction[] {
  return (actions ?? []).flatMap((action) => parseAction(action) ?? []);
}

/** A few words for the chat, so every action the assistant took is visible. */
export function actionLabel(action: AppAction): string {
  switch (action.name) {
    case "open_page":
      return `OPENED ${action.page.toUpperCase()}`;
    case "set_theme":
      return `THEME ${action.theme === "mark3" ? "MARK III" : "ARC REACTOR"}`;
    case "set_option":
      return action.label;
    case "ask_notes":
      return "SEARCHED YOUR NOTES";
    case "update_library":
      return "LIBRARY UPDATE STARTED";
    case "report_status":
      return "STATUS REPORT";
    case "clear_conversation":
      return "NEW CONVERSATION";
    case "lock_app":
      return "LOCKED";
    case "remember":
      return `REMEMBERED: ${action.fact}`;
  }
}

/** What the rest of the application lets the assistant operate. Provided by App. */
export interface AppControls {
  navigate(view: ViewId): void;
  update(patch: Partial<Settings>): void;
  lock(): void;
  /** The current state of the system as a sentence, from what the backend reported. */
  statusReport(): string;
  /** The library or the memory changed: reload what depends on them. */
  changed(): void;
}
