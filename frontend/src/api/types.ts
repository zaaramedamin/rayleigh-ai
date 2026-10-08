// Mirrors the backend's Pydantic models (backend/app/api/v1).

export interface Health {
  status: string;
  env: string;
  version: string;
}

export interface FileTypeInfo {
  name: string;
  extensions: string[];
  description: string;
}

/** How notes are found: by meaning, by the words in the query, or both merged. */
export type SearchMode = "vector" | "keyword" | "hybrid";

export interface SearchFilters {
  top_k?: number;
  mode?: SearchMode;
  document_ids?: number[];
  file_types?: string[];
}

/** A passage of a document with where it comes from. */
export interface SourcePassage {
  citation_id: string;
  document_id: number;
  chunk_index: number;
  score: number;
  source: string;
  heading_path: string;
  start_line: number;
  end_line: number;
  text: string;
  /** For files with pages (PDF): the first and last page the text comes from. */
  start_page?: number | null;
  end_page?: number | null;
}

export interface SearchResult extends SourcePassage {
  /** How well the words of the query match, when the passage was found by its words. */
  keyword_score?: number | null;
}

export interface AskSource extends SourcePassage {
  marker: number;
}

export type AskReason = "answered" | "no_relevant_notes" | "model_declined" | "no_valid_citation";

export interface AskResponse {
  answer: string;
  grounded: boolean;
  reason: AskReason;
  sources: AskSource[];
  notes_considered: number;
  /** The standalone question the notes were searched for, when a follow-up was rewritten. */
  searched_for?: string | null;
}

/** How a message is answered: by the model alone, or from the notes with sources. */
export type ChatMode = "general" | "notes";

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

/** Something the assistant asks the interface to do. The backend has validated it already. */
export interface ChatAction {
  name: string;
  args: Record<string, string>;
}

/** A general reply: written by the local model from what it knows. No notes, so no sources. */
export interface ChatResponse {
  answer: string;
  model: string;
  truncated: boolean;
  actions?: ChatAction[];
}

/** Who the assistant is. Written by you on the Profile page. */
export interface AssistantIdentity {
  name: string;
  /** How it addresses you ("sir", your name, ...). May be empty. */
  address: string;
  role: string;
  /** Tell the model what the Profile page says about you. */
  use_profile: boolean;
  /** Tell the model its memories, and let it save new ones. */
  use_memory: boolean;
}

export interface AssistantInfo extends AssistantIdentity {
  default_role: string;
}

export interface Memory {
  id: number;
  text: string;
  /** Who saved it: you on the Profile page, or the assistant during a conversation. */
  origin: "owner" | "assistant";
  created_at: string;
}

export interface MemoryList {
  memories: Memory[];
  limit: number;
}

export interface VoiceStatus {
  model: string;
  state: "ready" | "loading" | "not_loaded" | "not_downloaded";
  language: string | null;
  hint: string | null;
}

export interface Transcription {
  text: string;
  seconds: number;
}

export interface EmbeddingStatus {
  model: string;
  downloaded: boolean;
}

export interface LibraryStatus {
  documents: number;
  chunks: number;
  searchable_documents: number;
  pending_documents: number;
  /** Documents whose file was not found in the last two updates. */
  missing_documents?: number;
}

export interface LLMStatus {
  model: string;
  state: "ready" | "model_missing" | "not_running" | "error";
  hint: string | null;
}

export interface SystemStatus {
  embedding: EmbeddingStatus;
  library: LibraryStatus;
  llm: LLMStatus;
}

export interface AuthStatus {
  required: boolean;
  configured: boolean;
  authenticated: boolean;
}

export interface LibraryDocument {
  id: number;
  name: string;
  size_bytes: number;
  media_type: string;
  created_at: string;
  chunks: number;
  searchable: boolean;
  source: string | null;
  is_profile: boolean;
  /** "missing": the file was not found in the last two updates, so it is no longer searched. */
  status: "active" | "missing";
}

export interface DocumentList {
  documents: LibraryDocument[];
  removed_count: number;
  /** Documents that match the filter, before paging. */
  total: number;
}

/** A place a file was found: a folder and the path inside it. */
export interface LocationInfo {
  folder: string;
  path: string;
  status: "present" | "missing";
  last_seen_at: string;
  /** Updates in a row that did not find the file. */
  misses: number;
}

export interface VersionInfo {
  id: number;
  created_at: string;
  size_bytes: number;
  chunks: number;
}

export interface DocumentDetail extends LibraryDocument {
  supersedes_id: number | null;
  locations: LocationInfo[];
  /** Earlier versions of this file, newest first. Kept as history, never searched. */
  older_versions: VersionInfo[];
  indexed_chunks: number;
}

/** A short summary of one document, written by the local model from the document's own text. */
export interface Summary {
  text: string;
  document_id: number;
  name: string;
  /** How many parts the document was cut into. */
  parts: number;
  /** How many of them were summarized. */
  covered_parts: number;
  /** True when the document was longer than one summary reads, so only its start was summarized. */
  truncated: boolean;
}

/** How much a tool can do. Anything above "read_local" asks you every time. */
export type AgentLevel = "read_local" | "open_local" | "external_read" | "write_local" | "destructive";

export interface AgentToolInfo {
  name: string;
  description: string;
  level: AgentLevel;
  /** In plain words: "only reads", "opens things, asks every time". */
  what_it_does: string;
  enabled: boolean;
}

/** What a task asks you when it cannot go on without you: approve an action, or something went wrong. */
export interface AgentQuestion {
  /** Send it back with the answer, so an answer to an old question is refused. */
  id: number;
  kind: "approve" | "problem";
  title: string;
  message: string;
  /** The only answers that mean anything: allow, deny, stop, retry, skip or continue. */
  options: string[];
  tool: string | null;
  /** The exact arguments, as checked. */
  arguments: Record<string, unknown>;
  effect: string;
  reason: string;
  /** Tools whose results the model has read so far in this task. */
  read_sources: string[];
}

export interface AgentStep {
  tool: string;
  effect: string;
  outcome: string;
}

export type AgentRunStatus = "running" | "waiting" | "done" | "stopped" | "failed";

export interface AgentRun {
  run_id: string;
  task: string;
  status: AgentRunStatus;
  answer: string;
  started_at: string;
  finished_at: string | null;
  progress: string[];
  question: AgentQuestion | null;
  steps: AgentStep[];
  model_turns: number;
}

export interface AgentSettings {
  enabled: boolean;
  web: boolean;
  tools: AgentToolInfo[];
  /** The task that is running now, if any. */
  active_run: AgentRun | null;
}

/** Only the fields sent are changed. */
export interface AgentSettingsChange {
  enabled?: boolean;
  web?: boolean;
  /** The tools to have switched on. */
  tools?: string[];
}

/** One entry of the record of everything the agent was asked to do. */
export interface AgentEvent {
  id: number;
  time: string;
  run_id: string;
  step: number;
  kind: string;
  tool: string | null;
  level: string | null;
  decision: string | null;
  decided_by: string | null;
  detail: string | null;
}

/** One passage a document was cut into. */
export interface ChunkInfo {
  index: number;
  heading_path: string;
  start_line: number;
  end_line: number;
  start_page: number | null;
  end_page: number | null;
  char_count: number;
  text: string;
  searchable: boolean;
}

export interface ChunkPage {
  total: number;
  chunks: ChunkInfo[];
}

/** One run of a library update, kept so a restart still shows what happened. */
export interface JobInfo {
  id: number;
  kind: string;
  state: "running" | "done" | "failed" | "interrupted";
  message: string | null;
  started_at: string;
  finished_at: string | null;
  added: number;
  unchanged: number;
  replaced: number;
  missing: number;
  skipped_excluded: number;
  failed_files: number;
  indexed: number;
  chunks_done: number;
}

export interface LibraryFolder {
  path: string;
  origin: "env" | "ui";
  exists: boolean;
  removable: boolean;
  documents: number;
}

export interface FolderRemoval {
  folders: LibraryFolder[];
  documents_removed: number;
}

export interface SyncStatus {
  state: "idle" | "running" | "done" | "failed";
  phase: string;
  message: string | null;
  started_at: string | null;
  finished_at: string | null;
  folders: number;
  added: number;
  unchanged: number;
  skipped_excluded: number;
  failed_files: number;
  /** Documents replaced by a newer version of their file. */
  replaced: number;
  /** Documents whose file was not found again. */
  missing: number;
  indexing_total: number;
  indexed: number;
  /** The same progress in chunks, which keeps moving while one large document is indexed. */
  chunks_total: number;
  chunks_done: number;
  /** This run's record in the job history. */
  job_id: number | null;
}

/** A stored conversation, without its messages. */
export interface ConversationInfo {
  id: number;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface ConversationList {
  conversations: ConversationInfo[];
  /** All stored conversations, before paging. */
  total: number;
  /** The most conversations that can be kept. */
  limit: number;
  /** Conversations untouched for this many days are deleted. 0: kept until deleted. */
  retention_days: number;
}

/** One stored turn. `payload` is what the interface showed beside it (sources and so on). */
export interface StoredMessage {
  id: number;
  position: number;
  role: "user" | "assistant";
  mode: ChatMode;
  content: string;
  payload: Record<string, unknown> | null;
  created_at: string;
}

export interface ConversationDetail extends ConversationInfo {
  messages: StoredMessage[];
  /** The most messages one conversation can hold. */
  message_limit: number;
}

export interface NewStoredMessage {
  role: "user" | "assistant";
  mode: ChatMode;
  content: string;
  payload?: Record<string, unknown> | null;
}

/** What a streamed answer reports while it is written. */
export interface AskStreamHandlers {
  /** A follow-up was rewritten: this is what the notes are searched for. */
  onSearching?: (searchedFor: string) => void;
  /** Text so far. Unverified: citations are only checked when the answer is complete. */
  onToken?: (text: string) => void;
}

/** What a mark on an answer says. The last three say it went wrong. */
export type FeedbackKind = "helpful" | "not_helpful" | "wrong_source" | "missing_info";

/** A mark you put on an answer, kept on this computer so a failure can become an evaluation question. */
export interface FeedbackMark {
  id: number;
  kind: FeedbackKind;
  mode: ChatMode;
  question: string;
  answer: string;
  note: string | null;
  details: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
}

export interface NewFeedback {
  kind: FeedbackKind;
  mode: ChatMode;
  question: string;
  answer: string;
  note?: string | null;
  /** What the interface knew about the answer: its sources, why it was refused. */
  details?: Record<string, unknown> | null;
}

/** What the program is set to do (read-only). */
export interface SettingsInfo {
  version: string;
  embedding_model: string;
  llm_model: string;
  ollama_url: string;
  speech_model: string;
  search_mode: SearchMode;
  retrieval_top_k: number;
  answer_min_score: number;
  chunk_size_chars: number;
  chunk_overlap_chars: number;
  max_file_size_mb: number;
  parser_timeout_seconds: number;
  pdf_max_pages: number;
  pdf_keep_headers_footers: boolean;
  max_request_mb: number;
  rate_limit_per_minute: number;
  access_required: boolean;
  library_encrypted: boolean;
}

export type ProfileKey =
  | "name"
  | "location"
  | "occupation"
  | "languages"
  | "interests"
  | "preferences"
  | "about";

export type ProfileValues = Record<ProfileKey, string>;

export interface Profile {
  values: ProfileValues;
  fields: Array<{ key: ProfileKey; label: string; hint: string }>;
  saved_as: string;
  updated_at: string | null;
  searchable: boolean;
}

export interface Api {
  health(): Promise<Health>;
  systemStatus(): Promise<SystemStatus>;
  fileTypes(): Promise<FileTypeInfo[]>;
  search(query: string, filters?: SearchFilters): Promise<SearchResult[]>;
  ask(question: string, filters?: SearchFilters, history?: ChatTurn[]): Promise<AskResponse>;
  chat(message: string, history?: ChatTurn[]): Promise<ChatResponse>;
  /**
   * `ask`, delivered while the model writes it. Resolves with the checked answer, which replaces what
   * was streamed. Rejects with an AbortError when `signal` is aborted.
   */
  askStream(
    question: string,
    handlers: AskStreamHandlers,
    options?: { filters?: SearchFilters; history?: ChatTurn[]; signal?: AbortSignal },
  ): Promise<AskResponse>;

  authStatus(): Promise<AuthStatus>;
  setupPassword(password: string): Promise<string>;
  login(password: string): Promise<string>;
  logout(): Promise<void>;
  changePassword(current: string, next: string): Promise<string>;

  documents(): Promise<DocumentList>;
  documentDetail(id: number): Promise<DocumentDetail>;
  documentChunks(id: number, offset?: number, limit?: number): Promise<ChunkPage>;
  /** Ask the local model for a short summary of one document. Can take a minute for a long one. */
  summarizeDocument(id: number): Promise<Summary>;
  deleteDocument(id: number): Promise<void>;
  restoreRemoved(): Promise<number>;
  folders(): Promise<LibraryFolder[]>;
  addFolder(path: string): Promise<LibraryFolder[]>;
  removeFolder(path: string, removeDocuments: boolean): Promise<FolderRemoval>;
  startSync(): Promise<SyncStatus>;
  syncStatus(): Promise<SyncStatus>;
  jobs(): Promise<JobInfo[]>;
  settings(): Promise<SettingsInfo>;

  conversations(): Promise<ConversationList>;
  createConversation(title?: string): Promise<ConversationInfo>;
  conversation(id: number): Promise<ConversationDetail>;
  renameConversation(id: number, title: string): Promise<ConversationInfo>;
  addMessages(id: number, messages: NewStoredMessage[]): Promise<ConversationInfo>;
  deleteConversation(id: number): Promise<void>;
  deleteConversations(): Promise<number>;

  agentSettings(): Promise<AgentSettings>;
  changeAgent(change: AgentSettingsChange): Promise<AgentSettings>;
  /** Start a task. It runs in the background: watch it with agentRun. */
  startAgentRun(task: string): Promise<AgentRun>;
  agentRun(runId: string): Promise<AgentRun>;
  /** Answer the question a task is waiting on, with one of its options. */
  answerAgent(runId: string, questionId: number, choice: string): Promise<void>;
  stopAgentRun(runId: string): Promise<void>;
  agentLog(runId?: string, limit?: number): Promise<AgentEvent[]>;
  /** Erase the whole log; says how many entries went. */
  eraseAgentLog(): Promise<number>;

  addFeedback(mark: NewFeedback): Promise<FeedbackMark>;
  changeFeedback(id: number, kind: FeedbackKind, note?: string | null): Promise<FeedbackMark>;
  deleteFeedback(id: number): Promise<void>;

  profile(): Promise<Profile>;
  saveProfile(values: ProfileValues): Promise<Profile>;
  clearProfile(): Promise<void>;

  assistant(): Promise<AssistantInfo>;
  saveAssistant(values: AssistantIdentity): Promise<AssistantInfo>;
  memories(): Promise<MemoryList>;
  addMemory(text: string): Promise<Memory>;
  deleteMemory(id: number): Promise<void>;
  clearMemories(): Promise<number>;

  voiceStatus(): Promise<VoiceStatus>;
  /** Start loading the speech model, so the first spoken order is not slow. */
  prepareVoice(): Promise<VoiceStatus>;
  /** Turn a recording (16-bit PCM WAV, 16 kHz) into text, on this machine. */
  transcribe(wav: Blob): Promise<Transcription>;
}
