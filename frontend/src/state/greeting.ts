import type { SystemStatus } from "../api/types";

/** "sir" becomes ", sir"; no form of address becomes nothing. */
export function addressed(address: string): string {
  const trimmed = address.trim();
  return trimmed ? `, ${trimmed}` : "";
}

function partOfDay(hour: number): string {
  if (hour >= 5 && hour < 12) return "Good morning";
  if (hour >= 12 && hour < 18) return "Good afternoon";
  return "Good evening";
}

/**
 * What the system looks like right now, in one or two spoken sentences. Built here from what
 * the backend reported, not by the model, so it is always true. Null status: nothing is known.
 */
export function statusReport(status: SystemStatus | null, online: boolean): string {
  if (!online) return "The backend is not answering, so I cannot reach your notes or the local model.";
  if (!status) return "I am still checking the systems.";
  const parts: string[] = [];
  const { llm, library, embedding } = status;
  if (llm.state === "ready" && embedding.downloaded) parts.push("All systems are online.");
  else if (llm.state !== "ready") {
    parts.push(
      llm.state === "model_missing"
        ? "The local model is not installed, so I can search your notes but not talk."
        : "The local model is not running, so I can search your notes but not talk.",
    );
  } else parts.push("The embedding model is missing, so I cannot search your notes yet.");

  if (library.documents === 0) parts.push("Your library is empty.");
  else {
    const documents = `${library.documents} ${library.documents === 1 ? "document" : "documents"}`;
    parts.push(
      library.pending_documents > 0
        ? `Your library holds ${documents}, and ${library.pending_documents} still ${library.pending_documents === 1 ? "needs" : "need"} indexing.`
        : `Your library holds ${documents}, all searchable.`,
    );
  }
  return parts.join(" ");
}

/** What the assistant says when the application opens. */
export function greetingFor(now: Date, address: string, status: SystemStatus | null, online: boolean): string {
  const hello = `${partOfDay(now.getHours())}${addressed(address)}.`;
  const systems = status && online ? ` ${statusReport(status, online)}` : "";
  return `${hello}${systems} How can I serve you?`;
}
