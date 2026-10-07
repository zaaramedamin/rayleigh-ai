import type {
  Api,
  AskResponse,
  AssistantIdentity,
  AssistantInfo,
  AuthStatus,
  ChatResponse,
  ChatTurn,
  ChunkPage,
  ConversationDetail,
  ConversationInfo,
  ConversationList,
  DocumentDetail,
  DocumentList,
  FeedbackKind,
  FeedbackMark,
  FileTypeInfo,
  FolderRemoval,
  Health,
  JobInfo,
  LibraryFolder,
  Memory,
  MemoryList,
  NewFeedback,
  NewStoredMessage,
  Profile,
  ProfileValues,
  SearchFilters,
  SearchResult,
  SettingsInfo,
  SyncStatus,
  SystemStatus,
  Transcription,
  VoiceStatus,
} from "./types";

import { splitEvents } from "./sse";

export type ApiErrorKind =
  | "offline"
  | "unauthorized"
  | "throttled"
  | "llm_unavailable"
  | "timeout"
  | "invalid"
  | "conflict"
  | "not_found"
  | "server";

export class ApiError extends Error {
  constructor(
    readonly kind: ApiErrorKind,
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export function kindForStatus(status: number): ApiErrorKind {
  if (status === 401) return "unauthorized";
  if (status === 404) return "not_found";
  if (status === 409) return "conflict";
  if (status === 429) return "throttled";
  if (status === 503) return "llm_unavailable";
  if (status === 504) return "timeout";
  if (status === 422) return "invalid";
  return "server";
}

/** How the client learns the sign-in token, and what to do when the server stops accepting it. */
export interface AuthHooks {
  getToken(): string | null;
  onUnauthorized(): void;
}

interface RequestOptions extends RequestInit {
  /** A 401 here is an answer (wrong password), not a sign that the session ended. */
  expectUnauthorized?: boolean;
}

/** The error for an HTTP answer that was not a success. */
async function failure(response: Response, hooks: AuthHooks | undefined, expectUnauthorized = false): Promise<ApiError> {
  let detail = `Request failed (${response.status}).`;
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") detail = body.detail;
  } catch {
    // keep the generic message
  }
  if (response.status === 401 && !expectUnauthorized) hooks?.onUnauthorized();
  return new ApiError(kindForStatus(response.status), detail, response.status);
}

/** True for the error a fetch or a stream raises when its AbortSignal was triggered. */
export function isAbort(error: unknown): boolean {
  return typeof error === "object" && error !== null && (error as { name?: unknown }).name === "AbortError";
}

async function request<T>(
  base: string,
  path: string,
  hooks: AuthHooks | undefined,
  options: RequestOptions = {},
): Promise<T> {
  const { expectUnauthorized, ...init } = options;
  const token = hooks?.getToken();
  let response: Response;
  try {
    response = await fetch(`${base}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...init.headers,
      },
    });
  } catch {
    throw new ApiError("offline", "The Reyleight backend is not reachable on this machine.");
  }
  if (!response.ok) throw await failure(response, hooks, expectUnauthorized);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

const json = (value: unknown): RequestOptions => ({ method: "POST", body: JSON.stringify(value) });

/** Talks to the real backend. The default base is same-origin, proxied to 127.0.0.1 in dev. */
export function createApi(base = "/api/v1", hooks?: AuthHooks): Api {
  const call = <T>(path: string, options?: RequestOptions) => request<T>(base, path, hooks, options);
  return {
    health: () => call<Health>("/health"),
    systemStatus: () => call<SystemStatus>("/system/status"),
    fileTypes: () => call<FileTypeInfo[]>("/ingestion/file-types"),
    search: async (query: string, filters: SearchFilters = {}) => {
      const body = await call<{ results: SearchResult[] }>("/search", json({ query, ...filters }));
      return body.results;
    },
    ask: (question: string, filters: SearchFilters = {}, history: ChatTurn[] = []) =>
      call<AskResponse>("/ask", json({ question, ...filters, ...(history.length ? { history } : {}) })),
    chat: (message: string, history: ChatTurn[] = []) => call<ChatResponse>("/chat", json({ message, history })),
    askStream: async (question, handlers, options = {}) => {
      const { filters = {}, history = [], signal } = options;
      const token = hooks?.getToken();
      let response: Response;
      try {
        response = await fetch(`${base}/ask/stream`, {
          method: "POST",
          signal,
          headers: { "Content-Type": "application/json", ...(token ? { Authorization: `Bearer ${token}` } : {}) },
          body: JSON.stringify({ question, ...filters, ...(history.length ? { history } : {}) }),
        });
      } catch (error) {
        if (isAbort(error)) throw error;
        throw new ApiError("offline", "The Reyleight backend is not reachable on this machine.");
      }
      if (!response.ok) throw await failure(response, hooks);
      if (!response.body) throw new ApiError("server", "The answer could not be read.");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffered = "";
      let answer: AskResponse | null = null;
      try {
        for (;;) {
          const { done, value } = await reader.read();
          buffered += value ? decoder.decode(value, { stream: !done }) : done ? decoder.decode() : "";
          const { events, rest } = splitEvents(buffered);
          buffered = rest;
          for (const event of events) {
            const data = JSON.parse(event.data) as Record<string, unknown>;
            if (event.event === "searching") handlers.onSearching?.(String(data.searched_for ?? ""));
            else if (event.event === "token") handlers.onToken?.(String(data.text ?? ""));
            else if (event.event === "done") answer = data as unknown as AskResponse;
            else if (event.event === "error") {
              const status = typeof data.status === "number" ? data.status : 500;
              throw new ApiError(kindForStatus(status), String(data.detail ?? "The answer failed."), status);
            }
          }
          if (done) break;
        }
      } catch (error) {
        if (isAbort(error) || error instanceof ApiError) throw error;
        throw new ApiError("offline", "The connection to the backend was lost while the answer was being written.");
      } finally {
        await reader.cancel().catch(() => undefined);
      }
      if (!answer) throw new ApiError("server", "The answer ended without a result.");
      return answer;
    },

    authStatus: () => call<AuthStatus>("/auth/status", { expectUnauthorized: true }),
    setupPassword: async (password) =>
      (await call<{ token: string }>("/auth/setup", { ...json({ password }), expectUnauthorized: true })).token,
    login: async (password) =>
      (await call<{ token: string }>("/auth/login", { ...json({ password }), expectUnauthorized: true })).token,
    logout: () => call<void>("/auth/logout", { method: "POST", expectUnauthorized: true }),
    changePassword: async (current, next) =>
      (
        await call<{ token: string }>(
          "/auth/change-password",
          { ...json({ current_password: current, new_password: next }), expectUnauthorized: true },
        )
      ).token,

    documents: () => call<DocumentList>("/library/documents"),
    documentDetail: (id) => call<DocumentDetail>(`/library/documents/${id}`),
    documentChunks: (id, offset = 0, limit = 20) =>
      call<ChunkPage>(`/library/documents/${id}/chunks?${new URLSearchParams({ offset: String(offset), limit: String(limit) })}`),
    deleteDocument: (id) => call<void>(`/library/documents/${id}`, { method: "DELETE" }),
    restoreRemoved: async () => (await call<{ restored: number }>("/library/removed/restore", { method: "POST" })).restored,
    folders: async () => (await call<{ folders: LibraryFolder[] }>("/library/folders")).folders,
    addFolder: async (path) => (await call<{ folders: LibraryFolder[] }>("/library/folders", json({ path }))).folders,
    removeFolder: (path, removeDocuments) =>
      call<FolderRemoval>(
        `/library/folders?${new URLSearchParams({ path, remove_documents: String(removeDocuments) })}`,
        { method: "DELETE" },
      ),
    startSync: () => call<SyncStatus>("/library/sync", { method: "POST" }),
    syncStatus: () => call<SyncStatus>("/library/sync"),
    jobs: () => call<JobInfo[]>("/library/jobs"),
    settings: () => call<SettingsInfo>("/system/settings"),

    conversations: () => call<ConversationList>("/conversations"),
    createConversation: (title = "") => call<ConversationInfo>("/conversations", json({ title })),
    conversation: (id) => call<ConversationDetail>(`/conversations/${id}`),
    renameConversation: (id, title) =>
      call<ConversationInfo>(`/conversations/${id}`, { method: "PUT", body: JSON.stringify({ title }) }),
    addMessages: (id, messages: NewStoredMessage[]) => call<ConversationInfo>(`/conversations/${id}/messages`, json({ messages })),
    deleteConversation: (id) => call<void>(`/conversations/${id}`, { method: "DELETE" }),
    deleteConversations: async () => (await call<{ deleted: number }>("/conversations", { method: "DELETE" })).deleted,

    addFeedback: (mark: NewFeedback) => call<FeedbackMark>("/feedback", json(mark)),
    changeFeedback: (id: number, kind: FeedbackKind, note?: string | null) =>
      call<FeedbackMark>(`/feedback/${id}`, { method: "PUT", body: JSON.stringify({ kind, ...(note ? { note } : {}) }) }),
    deleteFeedback: (id: number) => call<void>(`/feedback/${id}`, { method: "DELETE" }),

    profile: () => call<Profile>("/profile"),
    saveProfile: (values: ProfileValues) => call<Profile>("/profile", { method: "PUT", body: JSON.stringify(values) }),
    clearProfile: () => call<void>("/profile", { method: "DELETE" }),

    assistant: () => call<AssistantInfo>("/assistant"),
    saveAssistant: (values: AssistantIdentity) =>
      call<AssistantInfo>("/assistant", { method: "PUT", body: JSON.stringify(values) }),
    memories: () => call<MemoryList>("/assistant/memories"),
    addMemory: (text) => call<Memory>("/assistant/memories", json({ text })),
    deleteMemory: (id) => call<void>(`/assistant/memories/${id}`, { method: "DELETE" }),
    clearMemories: async () => (await call<{ deleted: number }>("/assistant/memories", { method: "DELETE" })).deleted,

    voiceStatus: () => call<VoiceStatus>("/voice/status"),
    prepareVoice: () => call<VoiceStatus>("/voice/prepare", { method: "POST" }),
    // The recording is the request body itself: raw audio, not JSON.
    transcribe: (wav) =>
      call<Transcription>("/voice/transcribe", { method: "POST", body: wav, headers: { "Content-Type": "audio/wav" } }),
  };
}
