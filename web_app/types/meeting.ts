export interface AskRequest {
  message: string;
}

export interface AskCitation {
  document_id?: string;
  content_snippet?: string;
  page_number?: number;
  source_system?: string;
}

export interface AskResponse {
  answer: string;
  source: 'rag' | 'openai';
  citations?: AskCitation[];
}

export interface DocumentLookupRequest {
  meeting_transcript: string;
  topic_keywords?: string[];
  context_window?: number;
  top_k?: number;
  use_pageindex?: boolean;
}

export interface DocumentLookupResult {
  document_id: string;
  content: string;
  page_number?: number;
  section_title?: string;
  confidence_score: number;
  source_file?: string;
  source_system: string;
  retrieval_method: string;
}

export interface UserDatasetInfo {
  user_id: string;
  user_email: string;
  dataset_id: string;
  pageindex: {
    available: boolean;
    enabled?: boolean;
    stats?: unknown;
    error?: string;
    message?: string;
  } | null;
}

export interface IndexDocumentResponse {
  success: boolean;
  message: string;
  document_id: string;
  user_id: string;
  node_count?: number;
  indexing_time?: number;
}
