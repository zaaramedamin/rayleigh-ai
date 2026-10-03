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

export interface SearchFilters {
  top_k?: number;
  document_ids?: number[];
  file_types?: string[];
}

export interface SearchResult {
  citation_id: string;
  document_id: number;
  chunk_index: number;
  score: number;
  source: string;
  heading_path: string;
  start_line: number;
  end_line: number;
  text: string;
}

export interface AskSource extends SearchResult {
  marker: number;
}

export type AskReason = "answered" | "no_relevant_notes" | "model_declined" | "no_valid_citation";

export interface AskResponse {
  answer: string;
  grounded: boolean;
  reason: AskReason;
  sources: AskSource[];
  notes_considered: number;
}

export interface Api {
  health(): Promise<Health>;
  fileTypes(): Promise<FileTypeInfo[]>;
  search(query: string, filters?: SearchFilters): Promise<SearchResult[]>;
  ask(question: string, filters?: SearchFilters): Promise<AskResponse>;
}
