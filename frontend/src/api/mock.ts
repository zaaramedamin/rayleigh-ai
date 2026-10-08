import { ApiError } from "./client";
import type {
  AgentEvent,
  AgentRun,
  AgentSettings,
  AgentToolInfo,
  Api,
  AskResponse,
  AssistantIdentity,
  ChatAction,
  ChatResponse,
  ChunkPage,
  ConversationDetail,
  ConversationInfo,
  DocumentDetail,
  FeedbackMark,
  JobInfo,
  LibraryDocument,
  LibraryFolder,
  Memory,
  Profile,
  ProfileValues,
  SearchResult,
  SettingsInfo,
  Summary,
  SyncStatus,
  VoiceStatus,
} from "./types";

// Simulated data so the interface can be explored without the backend. It is made up, never
// comes from the owner's real notes, lives only in memory, and the UI always labels it DEMO MODE.

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

const DEMO_FOLDER = "C:\\Demo\\notes";

function demoDocuments(): LibraryDocument[] {
  return NOTES.map((note, index) => ({
    id: index + 1,
    name: note.source,
    size_bytes: 900 + index * 340,
    media_type: "text/markdown",
    created_at: new Date(Date.UTC(2026, 9, 1 + index)).toISOString(),
    chunks: 1 + index,
    searchable: true,
    source: `${DEMO_FOLDER}\\${note.source}`,
    is_profile: false,
    status: "active" as const,
  }));
}

const EMPTY_PROFILE: ProfileValues = {
  name: "",
  location: "",
  occupation: "",
  languages: "",
  interests: "",
  preferences: "",
  about: "",
};

const PROFILE_FIELDS: Profile["fields"] = [
  { key: "name", label: "My name", hint: "What should it call you?" },
  { key: "location", label: "Where I live", hint: "City, country, time zone" },
  { key: "occupation", label: "What I do", hint: "Work, studies, projects" },
  { key: "languages", label: "Languages I speak", hint: "And which one you want answers in" },
  { key: "interests", label: "My interests", hint: "Hobbies, topics you care about" },
  { key: "preferences", label: "How I like answers", hint: "Short or detailed, tone, format" },
  { key: "about", label: "More about me", hint: "Anything else it should know" },
];

const DEMO_ROLE = "You are my personal assistant. You help me with my notes and operate this application when I ask.";

const DEMO_VOICE: VoiceStatus = {
  model: "demo/speech",
  state: "not_downloaded",
  language: null,
  hint: "Demo mode has no speech model. Type your orders instead.",
};

const DEMO_PAGES = ["home", "chat", "knowledge", "profile", "modules", "privacy", "settings"];

/** A stand-in for the model's judgement: recognises a few plainly worded orders. */
export function demoOrder(message: string): ChatAction | null {
  const text = message.toLowerCase();
  const fact = /\bremember (?:that )?(.+)/i.exec(message)?.[1]?.trim();
  if (fact) return { name: "remember", args: { fact } };
  if (/\block\b/.test(text)) return { name: "lock_app", args: {} };
  if (/\b(status|systems?) (report|check)\b|\bstatus\b/.test(text)) return { name: "report_status", args: {} };
  if (/\b(open|show|go to)\b/.test(text)) {
    const page = DEMO_PAGES.find((p) => text.includes(p));
    if (page) return { name: "open_page", args: { page } };
  }
  return null;
}

const infoOf = (c: ConversationDetail): ConversationInfo => ({
  id: c.id,
  title: c.title,
  created_at: c.created_at,
  updated_at: c.updated_at,
  message_count: c.messages.length,
});

export function createMockApi(): Api {
  // Conversations are kept in memory only, so the history list can be tried without the backend.
  let conversations: ConversationDetail[] = [];
  let nextConversationId = 1;
  let nextMessageId = 1;
  // Marks on answers are kept in memory only, like everything else in the demo.
  let marks: FeedbackMark[] = [];
  let nextMarkId = 1;

  // The demo agent: nothing is really opened or fetched. A task that mentions "open" asks for approval
  // first, as the real one does; any other task finishes by itself.
  const DEMO_TOOLS: AgentToolInfo[] = [
    { name: "calculator", description: "Calculate an arithmetic expression exactly.", level: "read_local", what_it_does: "only reads", enabled: false },
    { name: "current_time", description: "The current date and time on this computer.", level: "read_local", what_it_does: "only reads", enabled: false },
    { name: "search_notes", description: "Search the saved notes and documents.", level: "read_local", what_it_does: "only reads", enabled: false },
    { name: "open_path", description: "Open a file or a folder with its usual program.", level: "open_local", what_it_does: "opens things, asks every time", enabled: false },
    { name: "open_app", description: "Start Notepad, Calculator, Paint or File Explorer.", level: "open_local", what_it_does: "opens things, asks every time", enabled: false },
    { name: "fetch_web_page", description: "Read the text of one public web page.", level: "external_read", what_it_does: "reads the web, asks every time", enabled: false },
  ];
  const agent = { enabled: false, web: false, tools: DEMO_TOOLS };
  const runs = new Map<string, AgentRun>();
  let agentEvents: AgentEvent[] = [];
  let nextRun = 1;
  let nextQuestion = 1;
  const record = (run: AgentRun, kind: string, fields: Partial<AgentEvent> = {}) => {
    agentEvents = [
      ...agentEvents,
      {
        id: agentEvents.length + 1,
        time: new Date().toISOString(),
        run_id: run.run_id,
        step: run.model_turns,
        kind,
        tool: null,
        level: null,
        decision: null,
        decided_by: null,
        detail: null,
        ...fields,
      },
    ];
  };
  const finishRun = (run: AgentRun, status: AgentRun["status"], answer: string) => {
    run.status = status;
    run.answer = answer;
    run.question = null;
    run.finished_at = new Date().toISOString();
    record(run, "run_finished", { detail: `${status}: ${answer}` });
  };
  const findConversation = (id: number): ConversationDetail => {
    const found = conversations.find((c) => c.id === id);
    if (!found) throw new ApiError("not_found", "That conversation does not exist.", 404);
    return found;
  };
  let identity: AssistantIdentity = {
    name: "Reyleight",
    address: "sir",
    role: DEMO_ROLE,
    use_profile: true,
    use_memory: true,
  };
  let memories: Memory[] = [];
  let nextMemoryId = 1;
  const newMemory = (text: string, origin: Memory["origin"]): Memory => ({
    id: nextMemoryId++,
    text,
    origin,
    created_at: new Date().toISOString(),
  });
  let documents = demoDocuments();
  let folders: LibraryFolder[] = [
    { path: DEMO_FOLDER, origin: "env", exists: true, removable: false, documents: documents.length },
  ];
  let removed = 0;
  let profile: ProfileValues = { ...EMPTY_PROFILE };
  let profileSavedAt: string | null = null;
  let sync: SyncStatus = {
    state: "idle",
    phase: "",
    message: null,
    started_at: null,
    finished_at: null,
    folders: 1,
    added: 0,
    unchanged: 0,
    skipped_excluded: 0,
    failed_files: 0,
    replaced: 0,
    missing: 0,
    indexing_total: 0,
    indexed: 0,
    chunks_total: 0,
    chunks_done: 0,
    job_id: null,
  };

  const profileOut = (): Profile => ({
    values: profile,
    fields: PROFILE_FIELDS,
    saved_as: "My profile.md",
    updated_at: profileSavedAt,
    searchable: profileSavedAt !== null,
  });
  const withoutProfile = () => documents.filter((d) => !d.is_profile);

  const api: Api = {
    health: async () => {
      await wait(120);
      return { status: "ok", env: "demo", version: "demo" };
    },
    systemStatus: async () => ({
      embedding: { model: "demo/embedding", downloaded: true },
      library: {
        documents: documents.length,
        chunks: documents.reduce((n, d) => n + d.chunks, 0),
        searchable_documents: documents.filter((d) => d.searchable).length,
        pending_documents: documents.filter((d) => !d.searchable).length,
      },
      llm: { model: "demo/llm", state: "ready" as const, hint: null },
    }),
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
    askStream: async (question, handlers, options = {}): Promise<AskResponse> => {
      // The demo has no model: the answer is the same as `ask`, handed out word by word.
      const answer = await api.ask(question, options.filters, options.history);
      for (const word of answer.answer.split(/(?<=\s)/)) {
        if (options.signal?.aborted) throw new DOMException("Aborted", "AbortError");
        handlers.onToken?.(word);
        await wait(25);
      }
      return answer;
    },
    chat: async (message): Promise<ChatResponse> => {
      await wait(900);
      const sir = identity.address ? `, ${identity.address}` : "";
      const order = demoOrder(message);
      if (order?.name === "remember") {
        if (!memories.some((m) => m.text === order.args.fact)) {
          memories = [...memories, newMemory(order.args.fact, "assistant")];
        }
        return { answer: `I will remember that${sir}.`, model: "demo/llm", truncated: false, actions: [order] };
      }
      if (order) {
        return { answer: `Right away${sir}.`, model: "demo/llm", truncated: false, actions: [order] };
      }
      return {
        answer:
          "Demo mode has no model, so this reply is simulated. With the backend running, the local model " +
          "answers here from its own knowledge, without reading your notes. Try an order such as " +
          '"open the settings" or "remember that I like tea", or switch MY NOTES on to see a cited answer.',
        model: "demo/llm",
        truncated: false,
        actions: [],
      };
    },

    authStatus: async () => ({ required: false, configured: true, authenticated: true }),
    setupPassword: async () => "demo",
    login: async () => "demo",
    logout: async () => undefined,
    changePassword: async () => "demo",

    documents: async () => {
      await wait(150);
      return { documents: [...documents], removed_count: removed, total: documents.length };
    },
    documentDetail: async (id): Promise<DocumentDetail> => {
      await wait(120);
      const doc = documents.find((d) => d.id === id);
      if (!doc) throw new Error("That document is not in the library.");
      const [folder, ...rest] = (doc.source ?? "").split("\\");
      return {
        ...doc,
        supersedes_id: null,
        locations: doc.source
          ? [{ folder: DEMO_FOLDER, path: rest.join("/") || folder, status: "present", last_seen_at: doc.created_at, misses: 0 }]
          : [],
        older_versions: [],
        indexed_chunks: doc.searchable ? doc.chunks : 0,
      };
    },
    documentChunks: async (id): Promise<ChunkPage> => {
      await wait(120);
      const note = NOTES.find((n) => n.document_id === id);
      const chunks = note
        ? [
            {
              index: note.chunk_index,
              heading_path: note.heading_path,
              start_line: note.start_line,
              end_line: note.end_line,
              start_page: null,
              end_page: null,
              char_count: note.text.length,
              text: note.text,
              searchable: true,
            },
          ]
        : [];
      return { total: chunks.length, chunks };
    },
    summarizeDocument: async (id): Promise<Summary> => {
      await wait(900);
      const doc = documents.find((d) => d.id === id);
      if (!doc) throw new Error("That document is not in the library.");
      const note = NOTES.find((n) => n.document_id === id);
      return {
        text: note ? `${doc.name}: ${note.text}` : `${doc.name} has no passages to summarize yet.`,
        document_id: doc.id,
        name: doc.name,
        parts: 1,
        covered_parts: 1,
        truncated: false,
      };
    },
    deleteDocument: async (id) => {
      await wait(200);
      documents = documents.filter((d) => d.id !== id);
      removed += 1;
    },
    restoreRemoved: async () => {
      const restored = removed;
      removed = 0;
      return restored;
    },
    folders: async () =>
      folders.map((f) => ({ ...f, documents: documents.filter((d) => d.source?.startsWith(f.path)).length })),
    addFolder: async (path) => {
      await wait(200);
      if (!/^[A-Za-z]:\\|^\//.test(path)) throw new Error("Use the full path, for example C:\\Users\\you\\Documents\\notes.");
      if (folders.some((f) => f.path.toLowerCase() === path.toLowerCase())) throw new Error("That folder is already on the list.");
      folders = [...folders, { path, origin: "ui", exists: true, removable: true, documents: 0 }];
      return folders;
    },
    removeFolder: async (path, removeDocuments) => {
      await wait(200);
      const before = documents.length;
      if (removeDocuments) documents = documents.filter((d) => !d.source?.startsWith(path));
      folders = folders.filter((f) => f.path !== path);
      return { folders, documents_removed: before - documents.length };
    },
    startSync: async () => {
      sync = { ...sync, state: "running", phase: "indexing", started_at: new Date().toISOString(), indexing_total: 2, indexed: 0 };
      setTimeout(() => (sync = { ...sync, indexed: 1 }), 700);
      setTimeout(
        () =>
          (sync = {
            ...sync,
            state: "done",
            phase: "",
            indexed: 2,
            unchanged: withoutProfile().length,
            finished_at: new Date().toISOString(),
          }),
        1500,
      );
      return sync;
    },
    syncStatus: async () => sync,
    jobs: async (): Promise<JobInfo[]> => [],
    settings: async (): Promise<SettingsInfo> => ({
      version: "demo",
      embedding_model: "demo/embedding",
      llm_model: "demo/llm",
      ollama_url: "http://127.0.0.1:11434",
      speech_model: "demo/speech",
      search_mode: "hybrid",
      retrieval_top_k: 5,
      answer_min_score: 0.3,
      chunk_size_chars: 1000,
      chunk_overlap_chars: 150,
      max_file_size_mb: 5,
      parser_timeout_seconds: 120,
      pdf_max_pages: 2000,
      pdf_keep_headers_footers: false,
      max_request_mb: 8,
      rate_limit_per_minute: 1200,
      access_required: false,
      library_encrypted: false,
    }),

    conversations: async () => ({
      conversations: [...conversations].sort((a, b) => b.updated_at.localeCompare(a.updated_at)).map(infoOf),
      total: conversations.length,
      limit: 500,
      retention_days: 0,
    }),
    createConversation: async (title = "") => {
      const now = new Date().toISOString();
      const created: ConversationDetail = {
        id: nextConversationId++,
        title,
        created_at: now,
        updated_at: now,
        message_count: 0,
        messages: [],
        message_limit: 400,
      };
      conversations = [...conversations, created];
      return infoOf(created);
    },
    conversation: async (id) => ({ ...findConversation(id), message_count: findConversation(id).messages.length }),
    renameConversation: async (id, title) => {
      const found = findConversation(id);
      found.title = title.trim();
      return infoOf(found);
    },
    addMessages: async (id, messages) => {
      const found = findConversation(id);
      const now = new Date().toISOString();
      for (const message of messages) {
        found.messages.push({
          id: nextMessageId++,
          position: found.messages.length,
          role: message.role,
          mode: message.mode,
          content: message.content,
          payload: message.payload ?? null,
          created_at: now,
        });
      }
      if (!found.title) found.title = messages.find((m) => m.role === "user")?.content.slice(0, 60) ?? "";
      found.updated_at = now;
      return infoOf(found);
    },
    deleteConversation: async (id) => {
      findConversation(id);
      conversations = conversations.filter((c) => c.id !== id);
    },
    deleteConversations: async () => {
      const count = conversations.length;
      conversations = [];
      return count;
    },

    agentSettings: async (): Promise<AgentSettings> => {
      await wait(120);
      const active = [...runs.values()].find((r) => r.status === "running" || r.status === "waiting") ?? null;
      return { enabled: agent.enabled, web: agent.web, tools: agent.tools.map((t) => ({ ...t })), active_run: active && structuredClone(active) };
    },
    changeAgent: async (change): Promise<AgentSettings> => {
      await wait(120);
      if (change.enabled !== undefined) agent.enabled = change.enabled;
      if (change.web !== undefined) agent.web = change.web;
      if (change.tools) {
        const unknown = change.tools.filter((n) => !agent.tools.some((t) => t.name === n));
        if (unknown.length) throw new Error(`No such tool: ${unknown.join(", ")}.`);
        agent.tools = agent.tools.map((t) => ({ ...t, enabled: change.tools!.includes(t.name) }));
      }
      const active = [...runs.values()].find((r) => r.status === "running" || r.status === "waiting") ?? null;
      return { enabled: agent.enabled, web: agent.web, tools: agent.tools.map((t) => ({ ...t })), active_run: active && structuredClone(active) };
    },
    startAgentRun: async (task): Promise<AgentRun> => {
      await wait(150);
      if (!agent.enabled) throw new ApiError("conflict", "The agent is switched off. Switch it on in the settings first.", 409);
      if ([...runs.values()].some((r) => r.status === "running" || r.status === "waiting")) {
        throw new ApiError("conflict", "The agent is already working on a task. Wait for it or stop it.", 409);
      }
      const text = task.split(/\s+/).filter(Boolean).join(" ");
      if (!text) throw new ApiError("invalid", "Say what the agent should do.", 422);
      const run: AgentRun = {
        run_id: `demo${String(nextRun++).padStart(4, "0")}`,
        task: text,
        status: "running",
        answer: "",
        started_at: new Date().toISOString(),
        finished_at: null,
        progress: ["thinking ..."],
        question: null,
        steps: [],
        model_turns: 1,
      };
      runs.set(run.run_id, run);
      record(run, "run_started", { detail: text });
      if (/\bopen\b/i.test(text) && agent.tools.some((t) => t.name === "open_path" && t.enabled)) {
        setTimeout(() => {
          if (run.status !== "running") return;
          const path = "C:\\Demo\\notes\\plan.txt";
          record(run, "request", { tool: "open_path", detail: JSON.stringify({ path }) });
          record(run, "decision", { tool: "open_path", level: "open_local", decision: "ask", decided_by: "policy" });
          run.status = "waiting";
          run.question = {
            id: nextQuestion++,
            kind: "approve",
            title: "Allow open_path?",
            message: `Open ${path} with the program Windows uses for it`,
            options: ["allow", "deny", "stop"],
            tool: "open_path",
            arguments: { path },
            effect: `Open ${path} with the program Windows uses for it`,
            reason: "This opens something on your computer, so I need your approval each time.",
            read_sources: [],
          };
        }, 700);
      } else {
        setTimeout(() => {
          if (run.status !== "running") return;
          run.progress.push("done");
          finishRun(run, "done", `Demo: I would carry out "${text}" here. Nothing was changed.`);
        }, 1200);
      }
      return structuredClone(run);
    },
    agentRun: async (runId): Promise<AgentRun> => {
      await wait(80);
      const run = runs.get(runId);
      if (!run) throw new ApiError("not_found", "There is no such task.", 404);
      return structuredClone(run);
    },
    answerAgent: async (runId, questionId, choice): Promise<void> => {
      await wait(100);
      const run = runs.get(runId);
      if (!run) throw new ApiError("not_found", "There is no such task.", 404);
      const question = run.question;
      if (!question || question.id !== questionId) throw new ApiError("conflict", "That question is no longer waiting for an answer.", 409);
      if (!question.options.includes(choice)) throw new ApiError("invalid", `The answer must be one of: ${question.options.join(", ")}`, 422);
      record(run, "approval", { tool: question.tool, decision: choice === "allow" ? "allow" : "deny", decided_by: "owner", detail: choice });
      run.question = null;
      if (choice === "stop") return finishRun(run, "stopped", "You stopped the task.");
      run.status = "running";
      if (choice === "deny") {
        run.steps.push({ tool: question.tool ?? "", effect: question.effect, outcome: "not allowed by you" });
        setTimeout(() => finishRun(run, "done", "You did not allow that, so I did not open it."), 500);
        return;
      }
      run.progress.push(`running: ${question.effect}`);
      setTimeout(() => {
        run.steps.push({ tool: question.tool ?? "", effect: question.effect, outcome: "done" });
        record(run, "result", { tool: question.tool, detail: "Demo: nothing was really opened." });
        finishRun(run, "done", "Demo: I would have opened it now. Nothing was really opened.");
      }, 700);
    },
    stopAgentRun: async (runId): Promise<void> => {
      await wait(80);
      const run = runs.get(runId);
      if (!run) throw new ApiError("not_found", "There is no such task.", 404);
      if (run.status !== "running" && run.status !== "waiting") throw new ApiError("conflict", "That task has already finished.", 409);
      finishRun(run, "stopped", "You stopped the task.");
    },
    agentLog: async (runId, limit = 50): Promise<AgentEvent[]> => {
      await wait(100);
      return runId ? agentEvents.filter((e) => e.run_id === runId).slice(0, limit) : [...agentEvents].reverse().slice(0, limit);
    },
    eraseAgentLog: async (): Promise<number> => {
      await wait(100);
      const erased = agentEvents.length;
      agentEvents = [];
      return erased;
    },

    addFeedback: async (mark) => {
      const now = new Date().toISOString();
      const created: FeedbackMark = {
        id: nextMarkId++,
        kind: mark.kind,
        mode: mark.mode,
        question: mark.question,
        answer: mark.answer,
        note: mark.note ?? null,
        details: mark.details ?? null,
        created_at: now,
        updated_at: now,
      };
      marks = [...marks, created];
      return created;
    },
    changeFeedback: async (id, kind, note) => {
      const found = marks.find((m) => m.id === id);
      if (!found) throw new ApiError("not_found", "That mark does not exist.", 404);
      found.kind = kind;
      if (note) found.note = note;
      found.updated_at = new Date().toISOString();
      return { ...found };
    },
    deleteFeedback: async (id) => {
      if (!marks.some((m) => m.id === id)) throw new ApiError("not_found", "That mark does not exist.", 404);
      marks = marks.filter((m) => m.id !== id);
    },

    profile: async () => profileOut(),
    saveProfile: async (values) => {
      await wait(250);
      profile = { ...values };
      const hasAny = Object.values(values).some((v) => v.trim());
      profileSavedAt = hasAny ? new Date().toISOString() : null;
      documents = documents.filter((d) => !d.is_profile);
      if (hasAny) {
        documents = [
          ...documents,
          {
            id: 99,
            name: "My profile.md",
            size_bytes: 400,
            media_type: "text/markdown",
            created_at: profileSavedAt ?? new Date().toISOString(),
            chunks: 3,
            searchable: true,
            source: null,
            is_profile: true,
            status: "active",
          },
        ];
      }
      return profileOut();
    },
    clearProfile: async () => {
      profile = { ...EMPTY_PROFILE };
      profileSavedAt = null;
      documents = documents.filter((d) => !d.is_profile);
    },

    assistant: async () => ({ ...identity, default_role: DEMO_ROLE }),
    saveAssistant: async (values) => {
      await wait(200);
      identity = {
        ...values,
        name: values.name.trim() || "Reyleight",
        address: values.address.trim(),
        role: values.role.trim() || DEMO_ROLE,
      };
      return { ...identity, default_role: DEMO_ROLE };
    },
    memories: async () => ({ memories: [...memories], limit: 200 }),
    addMemory: async (text) => {
      const cleaned = text.trim();
      if (!cleaned) throw new Error("There is nothing to remember.");
      const existing = memories.find((m) => m.text.toLowerCase() === cleaned.toLowerCase());
      if (existing) return existing;
      const memory = newMemory(cleaned, "owner");
      memories = [...memories, memory];
      return memory;
    },
    deleteMemory: async (id) => {
      memories = memories.filter((m) => m.id !== id);
    },
    clearMemories: async () => {
      const count = memories.length;
      memories = [];
      return count;
    },

    voiceStatus: async () => DEMO_VOICE,
    prepareVoice: async () => DEMO_VOICE,
    transcribe: async () => {
      throw new Error("Demo mode has no speech model. Type your orders instead.");
    },
  };
  return api;
}
