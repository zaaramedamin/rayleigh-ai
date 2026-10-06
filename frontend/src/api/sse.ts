/** One server-sent event: its name and the text after `data:`. */
export interface SseEvent {
  event: string;
  data: string;
}

/**
 * Cuts the text received so far into complete events and the unfinished rest.
 *
 * An event ends with a blank line. Keep the rest and put the next piece after it, so an event that
 * arrives in two chunks (or a multi-byte character split between them) is read once, whole.
 */
export function splitEvents(received: string): { events: SseEvent[]; rest: string } {
  const parts = received.replace(/\r\n/g, "\n").split("\n\n");
  const rest = parts.pop() ?? "";
  const events: SseEvent[] = [];
  for (const block of parts) {
    let event = "message";
    const data: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith(":")) continue; // a comment
      const colon = line.indexOf(":");
      const field = colon < 0 ? line : line.slice(0, colon);
      const value = colon < 0 ? "" : line.slice(colon + 1).replace(/^ /, "");
      if (field === "event") event = value;
      else if (field === "data") data.push(value);
    }
    if (data.length > 0) events.push({ event, data: data.join("\n") });
  }
  return { events, rest };
}
