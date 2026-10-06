import { useEffect } from "react";
import { sound } from "../audio/sound";
import { useSettings } from "../state/settings";
import type { MessageState } from "../state/useAssistant";
import type { Link } from "../state/useHealth";
import { useReveal } from "../state/useReveal";
import type { StatusState } from "../state/useSystemStatus";
import { ArcReactor } from "./ArcReactor";
import { HudFrame } from "./HudFrame";

interface Props {
  link: Link;
  thinking: boolean;
  /** The microphone is open. */
  listening?: boolean;
  /** The assistant is talking. */
  speaking?: boolean;
  demo: boolean;
  messages: MessageState[];
  system: StatusState;
  /** True while the start-up reactor is still flying to this spot: the panel waits, all OFFLINE. */
  reactorHidden?: boolean;
}

type Tone = "ok" | "bad" | "warn" | undefined;

interface RowData {
  label: string;
  value: string;
  tone?: Tone;
  hint?: string | null;
}

const OFFLINE: Pick<RowData, "value" | "tone"> = { value: "OFFLINE", tone: "bad" };
const STEP_MS = 240;

const LLM_LABEL = {
  ready: "READY",
  model_missing: "NOT INSTALLED",
  not_running: "NOT RUNNING",
  error: "PROBLEM",
} as const;

/** What the backend reports, in order. Unknown stays unknown: nothing is invented. */
function buildRows(link: Link, system: StatusState, messages: MessageState[]): RowData[] {
  // General replies count as queries, but only answers from the notes can be grounded.
  const turns = messages.flatMap((m) => (m.kind === "answer" || m.kind === "reply" ? [m] : []));
  const answers = turns.flatMap((m) => (m.kind === "answer" ? [m] : []));
  const grounded = answers.filter((m) => m.response.grounded).length;
  const last = turns.length ? turns[turns.length - 1] : undefined;
  const online = link.state === "online";
  const known = online && system.state === "ready" ? system.status : null;
  const unknown: Pick<RowData, "value" | "tone"> = online ? { value: "UNKNOWN", tone: "warn" } : OFFLINE;

  return [
    {
      label: "BACKEND",
      value: online ? "ONLINE" : link.state === "checking" ? "SCANNING" : "OFFLINE",
      tone: online ? "ok" : link.state === "checking" ? "warn" : "bad",
    },
    { label: "VERSION", ...(online ? { value: link.health.version } : OFFLINE) },
    { label: "PING", ...(online ? { value: `${link.latencyMs} ms` } : OFFLINE) },
    known
      ? {
          label: "MODEL",
          value: LLM_LABEL[known.llm.state],
          tone: known.llm.state === "ready" ? "ok" : "bad",
          hint: known.llm.state === "ready" ? known.llm.model : `${known.llm.model}. ${known.llm.hint ?? ""}`.trim(),
        }
      : { label: "MODEL", ...unknown },
    known
      ? {
          label: "EMBEDDING",
          value: known.embedding.downloaded ? "READY" : "MISSING",
          tone: known.embedding.downloaded ? "ok" : "bad",
          hint: known.embedding.downloaded ? null : "Run python -m app download-model",
        }
      : { label: "EMBEDDING", ...unknown },
    known
      ? { label: "DOCUMENTS", value: String(known.library.documents) }
      : { label: "DOCUMENTS", ...unknown },
    known
      ? {
          label: "SEARCHABLE",
          value: `${known.library.searchable_documents}/${known.library.documents}`,
          tone: known.library.pending_documents > 0 ? "warn" : "ok",
          hint:
            known.library.pending_documents > 0
              ? `${known.library.pending_documents} waiting. Run python -m app index`
              : null,
        }
      : { label: "SEARCHABLE", ...unknown },
    { label: "QUERIES", ...(online ? { value: String(turns.length) } : OFFLINE) },
    {
      label: "GROUNDED",
      ...(online ? { value: answers.length ? `${grounded}/${answers.length}` : "-" } : OFFLINE),
    },
    {
      label: "LAST REPLY",
      ...(online ? { value: last ? `${(last.ms / 1000).toFixed(1)} s` : "-" } : OFFLINE),
    },
  ];
}

/**
 * The system panel. Every row starts as OFFLINE and switches to what the backend actually
 * reports, one row at a time, once the start-up reactor has landed. A row only ever shows data
 * the backend returned; if the backend is down, the rows stay OFFLINE.
 */
export function StatusPanel({
  link,
  thinking,
  listening = false,
  speaking = false,
  demo,
  messages,
  system,
  reactorHidden = false,
}: Props) {
  const { settings } = useSettings();
  const rows = buildRows(link, system, messages);
  const revealed = useReveal(rows.length, !reactorHidden, settings.reducedMotion ? 0 : STEP_MS);
  const reactor =
    link.state === "offline"
      ? "offline"
      : listening
        ? "listening"
        : thinking
          ? "thinking"
          : speaking
            ? "speaking"
            : "idle";
  const word =
    reactorHidden || revealed === 0
      ? "OFFLINE"
      : reactor === "listening"
        ? "LISTENING"
        : reactor === "thinking"
          ? "PROCESSING"
          : reactor === "speaking"
            ? "SPEAKING"
            : reactor === "offline"
              ? "NO LINK"
              : "STANDBY";

  useEffect(() => {
    if (revealed > 0 && !settings.reducedMotion) sound.play("blip");
  }, [revealed, settings.reducedMotion]);

  return (
    <HudFrame title="SYSTEM" tag={demo ? "DEMO" : "LIVE"} className="status">
      <div className="status-reactor">
        <div data-reactor-target className={`reactor-slot${reactorHidden ? " reactor-slot--hidden" : ""}`}>
          <ArcReactor state={reactor} size={170} />
        </div>
        <div className={`status-word${word === "OFFLINE" ? " status-word--off" : ""}`}>{word}</div>
      </div>
      {rows.map((row, index) => {
        const live = index < revealed;
        const shown: RowData = live ? row : { label: row.label, ...OFFLINE };
        return (
          <div key={row.label}>
            <div className="row">
              <span>{shown.label}</span>
              <b key={live ? "live" : "off"} className={`${shown.tone ?? ""}${live ? " flip" : ""}`}>
                {shown.value}
              </b>
            </div>
            {live && shown.hint && <p className="hint">{shown.hint}</p>}
          </div>
        );
      })}
    </HudFrame>
  );
}
