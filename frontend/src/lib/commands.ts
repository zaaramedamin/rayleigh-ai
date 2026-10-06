export interface Command {
  id: string;
  label: string;
  /** Shown beside the label, and searched too. */
  group: string;
  keywords?: string;
  run: () => void;
}

/**
 * The commands that match what was typed. Every word typed must appear somewhere in the label,
 * group or keywords; matches at the start of the label come first.
 */
export function filterCommands(commands: Command[], query: string): Command[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (words.length === 0) return commands;
  const scored: Array<{ command: Command; score: number; order: number }> = [];
  commands.forEach((command, order) => {
    const label = command.label.toLowerCase();
    const haystack = `${label} ${command.group.toLowerCase()} ${(command.keywords ?? "").toLowerCase()}`;
    if (!words.every((word) => haystack.includes(word))) return;
    const score = words.reduce((sum, word) => {
      const inLabel = label.indexOf(word);
      return sum + (inLabel === 0 ? 0 : inLabel > 0 ? 1 : 3);
    }, 0);
    scored.push({ command, score, order });
  });
  return scored.sort((a, b) => a.score - b.score || a.order - b.order).map((entry) => entry.command);
}
