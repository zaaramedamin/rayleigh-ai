import type { Api, AskResponse, FileTypeInfo, Health, SearchFilters, SearchResult } from "./types";

export type ApiErrorKind = "offline" | "llm_unavailable" | "timeout" | "invalid" | "server";

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
  if (status === 503) return "llm_unavailable";
  if (status === 504) return "timeout";
  if (status === 422) return "invalid";
  return "server";
}

async function request<T>(base: string, path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${base}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new ApiError("offline", "The Reyleight backend is not reachable on this machine.");
  }
  if (!response.ok) {
    let detail = `Request failed (${response.status}).`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // keep the generic message
    }
    throw new ApiError(kindForStatus(response.status), detail, response.status);
  }
  return (await response.json()) as T;
}

/** Talks to the real backend. The default base is same-origin, proxied to 127.0.0.1 in dev. */
export function createApi(base = "/api/v1"): Api {
  return {
    health: () => request<Health>(base, "/health"),
    fileTypes: () => request<FileTypeInfo[]>(base, "/ingestion/file-types"),
    search: async (query: string, filters: SearchFilters = {}) => {
      const body = await request<{ results: SearchResult[] }>(base, "/search", {
        method: "POST",
        body: JSON.stringify({ query, ...filters }),
      });
      return body.results;
    },
    ask: (question: string, filters: SearchFilters = {}) =>
      request<AskResponse>(base, "/ask", {
        method: "POST",
        body: JSON.stringify({ question, ...filters }),
      }),
  };
}
