import type { Link } from "../state/useHealth";
import type { MessageState } from "../state/useAssistant";
import { ArcReactor } from "./ArcReactor";
import { HudFrame } from "./HudFrame";

interface Props {
  link: Link;
  thinking: boolean;
  demo: boolean;
  messages: MessageState[];
}

function Row({ label, value, tone }: { label: string; value: string; tone?: "ok" | "bad" | "warn" }) {
  return (
    <div className="row">
      <span>{label}</span>
      <b className={tone}>{value}</b>
    </div>
  );
}

/** Only facts the interface can actually know: link state, version, and this session's queries. */
export function StatusPanel({ link, thinking, demo, messages }: Props) {
  const answers = messages.filter((m) => m.kind === "answer");
  const grounded = answers.filter((m) => m.kind === "answer" && m.response.grounded).length;
  const last = answers.length ? answers[answers.length - 1] : undefined;
  const reactor = link.state === "offline" ? "offline" : thinking ? "thinking" : "idle";

  return (
    <HudFrame title="SYSTEM" tag={demo ? "DEMO" : "LIVE"} className="status">
      <div className="status-reactor">
        <ArcReactor state={reactor} size={170} />
        <div className="status-word">{reactor === "thinking" ? "PROCESSING" : reactor === "offline" ? "NO LINK" : "STANDBY"}</div>
      </div>
      <Row
        label="BACKEND"
        value={link.state === "online" ? "ONLINE" : link.state === "checking" ? "SCANNING" : "OFFLINE"}
        tone={link.state === "online" ? "ok" : link.state === "offline" ? "bad" : "warn"}
      />
      {link.state === "online" && (
        <>
          <Row label="VERSION" value={link.health.version} />
          <Row label="ENV" value={link.health.env.toUpperCase()} />
          <Row label="PING" value={`${link.latencyMs} ms`} />
        </>
      )}
      <Row label="QUERIES" value={String(answers.length)} />
      <Row label="GROUNDED" value={answers.length ? `${grounded}/${answers.length}` : "-"} />
      {last && last.kind === "answer" && <Row label="LAST REPLY" value={`${(last.ms / 1000).toFixed(1)} s`} />}
    </HudFrame>
  );
}
