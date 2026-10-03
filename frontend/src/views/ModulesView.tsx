import { HudFrame } from "../components/HudFrame";

interface Module {
  name: string;
  phase: string;
  summary: string;
  state: "live" | "next" | "planned";
}

// Keep in step with docs/product-roadmap.md. "live" means it works in this build.
const MODULES: Module[] = [
  { name: "Cited answers", phase: "MVP", summary: "Ask your notes, get sources or an honest 'I don't know'.", state: "live" },
  { name: "Semantic search", phase: "MVP", summary: "Find passages by meaning, with file and line provenance.", state: "live" },
  { name: "Library manager", phase: "Phase 1", summary: "Add folders, see every document, re-index, remove.", state: "next" },
  { name: "Streaming + chat memory", phase: "Phase 1", summary: "Token streaming, saved conversations, follow-up questions.", state: "next" },
  { name: "Better retrieval", phase: "Phase 2", summary: "Hybrid search, reranking, query rewriting, PDF and DOCX.", state: "planned" },
  { name: "Long-term memory", phase: "Phase 3", summary: "Facts and preferences you approve, viewable and deletable.", state: "planned" },
  { name: "Voice", phase: "Phase 4", summary: "Local speech-to-text and text-to-speech, wake word optional.", state: "planned" },
  { name: "Tools (MCP)", phase: "Phase 5", summary: "Approval-gated tools; every call explained before it runs.", state: "planned" },
  { name: "Agents + schedules", phase: "Phase 6", summary: "Bounded background helpers with a visible activity log.", state: "planned" },
  { name: "Phone link", phase: "Phase 7", summary: "Your phone talks to your own computer over a private tunnel.", state: "planned" },
];

const LABEL = { live: "ONLINE", next: "NEXT", planned: "LOCKED" } as const;

export function ModulesView() {
  return (
    <HudFrame title="MODULES" tag="ROADMAP">
      <p className="muted">
        What this build does today, and what is coming. Locked modules are not built yet: nothing here pretends otherwise.
      </p>
      <div className="grid">
        {MODULES.map((m) => (
          <article key={m.name} className={`module module--${m.state}`}>
            <header>
              <h3>{m.name}</h3>
              <span className="badge">{LABEL[m.state]}</span>
            </header>
            <p>{m.summary}</p>
            <small>{m.phase}</small>
          </article>
        ))}
      </div>
    </HudFrame>
  );
}
