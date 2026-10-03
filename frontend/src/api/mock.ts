import type { Api, AskResponse, SearchResult } from "./types";

// Simulated data so the interface can be explored without the backend. It is made up, never
// comes from the owner's real notes, and the UI always labels it DEMO MODE.

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const NOTES: Omit<SearchResult, "score">[] = [
  {
    citation_id: "demo:1:0",
    document_id: 1,
    chunk_index: 0,
    source: "lisbon-trip.md",
    heading_path: "Lisbon trip > Flights",
    start_line: 4,
    end_line: 11,
    text: "Flight TP 1357 leaves Berlin on 14 March at 07:45 and lands in Lisbon at 10:05. Seat 12A. Luggage: one cabin bag.",
  },
  {
    citation_id: "demo:2:0",
    document_id: 2,
    chunk_index: 0,
    source: "project-atlas.md",
    heading_path: "Project Atlas > Decisions",
    start_line: 12,
    end_line: 22,
    text: "We chose Qdrant over pgvector because it runs locally with no server and supports payload filters. Review again after the first evaluation run.",
  },
  {
    citation_id: "demo:3:0",
    document_id: 3,
    chunk_index: 1,
    source: "oats.md",
    heading_path: "Oats > Breakfast",
    start_line: 2,
    end_line: 9,
    text: "Overnight oats: 50 g oats, 150 ml milk, 1 tbsp chia seeds. About 320 kcal. Keeps three days in the fridge.",
  },
];

const STOP_WORDS = new Set("the and for are was what when where which who why how does did has have with from that this about into our your you my me of to in on is it at a an".split(" "));

function rank(query: string): SearchResult[] {
  const words = query
    .toLowerCase()
    .split(/\W+/)
    .filter((w) => w.length > 2 && !STOP_WORDS.has(w));
  return NOTES.map((note) => {
    const haystack = `${note.source} ${note.heading_path} ${note.text}`.toLowerCase();
    const hits = words.filter((w) => haystack.includes(w)).length;
    return { ...note, score: Math.min(0.92, hits === 0 ? 0.12 : 0.45 + hits * 0.17) };
  }).sort((a, b) => b.score - a.score);
}

export function createMockApi(): Api {
  return {
    health: async () => {
      await wait(120);
      return { status: "ok", env: "demo", version: "demo" };
    },
    fileTypes: async () => [
      { name: "text", extensions: [".txt"], description: "Plain text" },
      { name: "markdown", extensions: [".md"], description: "Markdown" },
      { name: "csv", extensions: [".csv"], description: "Tables" },
      { name: "html", extensions: [".html"], description: "Web pages" },
      { name: "json", extensions: [".json"], description: "Structured data" },
    ],
    search: async (query, filters) => {
      await wait(350);
      return rank(query).slice(0, filters?.top_k ?? 5);
    },
    ask: async (question): Promise<AskResponse> => {
      await wait(1200);
      const top = rank(question).filter((r) => r.score >= 0.5);
      if (top.length === 0) {
        return {
          answer: "I don't have enough information in your notes to answer that.",
          grounded: false,
          reason: "no_relevant_notes",
          sources: [],
          notes_considered: 0,
        };
      }
      const sources = top.slice(0, 2).map((r, i) => ({ ...r, marker: i + 1 }));
      return {
        answer: sources.map((s) => `${s.text} [${s.marker}]`).join("\n\n"),
        grounded: true,
        reason: "answered",
        sources,
        notes_considered: top.length,
      };
    },
  };
}
