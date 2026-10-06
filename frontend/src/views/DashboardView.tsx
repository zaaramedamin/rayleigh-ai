import { HudFrame } from "../components/HudFrame";
import type { Assistant } from "../components/ChatBubble";
import { EarthGlobe } from "../components/EarthGlobe";
import { ScrambleText } from "../components/ScrambleText";
import { Sparkline } from "../components/Sparkline";
import type { ViewId } from "../components/NavRail";
import { Ring } from "../components/Ring";
import { SourceCard } from "../components/SourceCard";
import { useEvents } from "../state/events";
import type { Link } from "../state/useHealth";
import type { StatusState } from "../state/useSystemStatus";

interface Props {
  assistant: Assistant;
  link: Link;
  history: Array<number | null>;
  system: StatusState;
  animate: boolean;
  onNavigate: (view: ViewId) => void;
  onRecheck: () => void;
}

function pct(part: number, whole: number): number | null {
  return whole > 0 ? part / whole : null;
}

/** The home page: what the knowledge base holds and how this session is going. All real values. */
export function DashboardView({ assistant, link, history, system, animate, onNavigate, onRecheck }: Props) {
  const { messages, selectedAnswer, selectedIsGeneral, selectedMarker, select, selectedId } = assistant;
  const feed = useEvents().slice(0, 8);
  const measured = history.filter((v): v is number => v !== null);
  const average = measured.length ? Math.round(measured.reduce((a, b) => a + b, 0) / measured.length) : null;
  const library = system.state === "ready" ? system.status.library : null;
  const share = library ? pct(library.searchable_documents, library.documents) : null;

  // Everything that got a reply. Only answers from the notes can be grounded; general replies
  // never read the notes, so they are counted as conversation but not as grounded or ungrounded.
  const turns = messages.flatMap((m) => (m.kind === "answer" || m.kind === "reply" ? [m] : []));
  const answers = turns.flatMap((m) => (m.kind === "answer" ? [m] : []));
  const grounded = answers.filter((m) => m.response.grounded).length;
  const averageMs = turns.length ? turns.reduce((sum, m) => sum + m.ms, 0) / turns.length : null;

  return (
    <div className="dash">
      <HudFrame title="KNOWLEDGE CORE" tag={link.state === "online" ? "LIVE" : "NO LINK"} className="dash-core">
        <EarthGlobe animate={animate} />
        <div className="core-overlay">
          <h2 className="core-title">REYLEIGHT</h2>
          <p className="core-sub">PRIVATE KNOWLEDGE ASSISTANT</p>
        </div>
      </HudFrame>

      <HudFrame title="INDEX HEALTH" tag="LIBRARY" className="dash-card">
        <div className="gauge">
          <Ring value={share} label="Searchable documents" size={76} />
          <div>
            <b>{library ? `${library.searchable_documents} of ${library.documents}` : "No data"}</b>
            <span>documents searchable</span>
          </div>
        </div>
        <dl className="stat-tiles">
          <div>
            <dt>Documents</dt>
            <dd>
              <ScrambleText text={library ? library.documents.toLocaleString() : "--"} duration={900} />
            </dd>
          </div>
          <div>
            <dt>Chunks</dt>
            <dd>
              <ScrambleText text={library ? library.chunks.toLocaleString() : "--"} duration={1100} />
            </dd>
          </div>
          <div>
            <dt>Searchable</dt>
            <dd>
              <ScrambleText text={library ? library.searchable_documents.toLocaleString() : "--"} duration={1300} />
            </dd>
          </div>
        </dl>
        {library && library.pending_documents > 0 && (
          <p className="hint">{library.pending_documents} waiting. Run python -m app index</p>
        )}
        {library && library.documents === 0 && <p className="hint">Nothing ingested yet. Run python -m app ingest</p>}
      </HudFrame>

      <HudFrame title="THIS SESSION" tag="CHAT" className="dash-card">
        <div className="gauge">
          <Ring value={pct(grounded, answers.length)} label="Grounded answers" size={76} />
          <div>
            <b>
              {turns.length === 0
                ? "No questions yet"
                : answers.length === 0
                  ? `${turns.length} general ${turns.length === 1 ? "reply" : "replies"}`
                  : `${grounded} of ${answers.length} grounded`}
            </b>
            <span>
              {averageMs === null
                ? "answers from your notes cite their sources"
                : `average reply ${(averageMs / 1000).toFixed(1)} s`}
            </span>
          </div>
        </div>
      </HudFrame>

      <HudFrame title="LINK PULSE" tag={link.state === "online" ? `${link.latencyMs} MS` : "NO LINK"} className="dash-card">
        <Sparkline values={history} />
        <p className="pulse-note">
          {history.length === 0
            ? "Measuring..."
            : average === null
              ? "No answer from the backend."
              : `Average ${average} ms over the last ${history.length} checks`}
        </p>
      </HudFrame>

      <HudFrame title="QUICK COMMANDS" tag="ACTIONS" className="dash-card">
        <div className="actions actions--stack">
          <button className="action" onClick={() => onNavigate("chat")}>
            Open full chat
          </button>
          <button className="action" onClick={() => onNavigate("knowledge")}>
            Search my notes
          </button>
          <button className="action" onClick={() => onNavigate("privacy")}>
            Privacy check
          </button>
          <button className="action" onClick={onRecheck}>
            Recheck backend
          </button>
        </div>
      </HudFrame>

      <HudFrame title="RECENT QUESTIONS" tag={`${turns.length}`} className="dash-card">
        {turns.length === 0 ? (
          <p className="muted">Nothing asked yet. Questions are kept in memory only and cleared when you close the app.</p>
        ) : (
          <ul className="recent">
            {[...turns].reverse().map((m) => (
              <li key={m.id}>
                <button
                  onClick={() => select(m.id)}
                  className={selectedId === m.id ? "recent--on" : undefined}
                  title={
                    m.kind === "reply"
                      ? "General reply, not from your notes"
                      : m.response.grounded
                        ? "Answered from your notes"
                        : "Your notes had no answer"
                  }
                >
                  {/* Grey: general. Green: grounded in the notes. Amber: the notes had no answer. */}
                  <span
                    className={`dot${m.kind === "reply" ? "" : m.response.grounded ? " dot--online" : " dot--checking"}`}
                    aria-hidden
                  />
                  <span className="recent-text">{m.question}</span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </HudFrame>

      <HudFrame
        title="CITED SOURCES"
        tag={selectedAnswer ? `${selectedAnswer.response.sources.length} CITED` : "IDLE"}
        className="dash-wide"
      >
        {selectedAnswer && selectedAnswer.response.sources.length > 0 ? (
          <div className="source-list source-list--grid">
            {selectedAnswer.response.sources.map((s) => (
              <SourceCard
                key={s.citation_id}
                item={s}
                marker={s.marker}
                active={selectedMarker === s.marker}
                onClick={() => selectedId !== null && select(selectedId, s.marker)}
              />
            ))}
          </div>
        ) : (
          <p className="muted">
            {selectedAnswer
              ? "That reply used no notes."
              : selectedIsGeneral
                ? "That was a general reply: your notes were not read, so there is nothing to cite."
                : "Ask the chat in the corner with MY NOTES on. The notes behind each answer appear here."}
          </p>
        )}
      </HudFrame>

      <HudFrame title="ACTIVITY" tag="THIS SESSION" className="dash-card">
        {feed.length === 0 ? (
          <p className="muted">Nothing has happened yet. Alerts, successes and warnings appear here as they occur.</p>
        ) : (
          <ul className="feed">
            {feed.map((event) => (
              <li key={event.id} className={`feed-item feed-item--${event.level}`}>
                <time>{new Date(event.at).toLocaleTimeString([], { hour12: false })}</time>
                <b>{event.title}</b>
                {event.detail && <span>{event.detail}</span>}
              </li>
            ))}
          </ul>
        )}
      </HudFrame>
    </div>
  );
}
