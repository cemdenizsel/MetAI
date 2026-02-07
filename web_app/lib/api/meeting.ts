import { apiClient, ApiError } from './client';
import type {
  AskResponse,
  DocumentLookupRequest,
  UserDatasetInfo,
  IndexDocumentResponse,
} from '@/types/meeting';

const MEETING_API = '/api/meeting';

export async function ask(message: string): Promise<AskResponse> {
  return apiClient<AskResponse>(`${MEETING_API}/ask`, {
    method: 'POST',
    body: { message },
  });
}

/**
 * Start document lookup (SSE). Returns the Response so the caller can read
 * response.body and parse Server-Sent Events (data: {...}\n\n).
 */
export async function documentLookup(
  body: DocumentLookupRequest
): Promise<Response> {
  const token =
    typeof window !== 'undefined' ? localStorage.getItem('auth_token') : null;
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  const response = await fetch(`${MEETING_API}/document-lookup`, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    let detail = 'Document lookup failed';
    try {
      const data = await response.json();
      detail = (data as { detail?: string }).detail || detail;
    } catch {
      // ignore
    }
    throw new ApiError(response.status, detail);
  }
  return response;
}

export async function indexDocument(
  file: File,
  documentId?: string
): Promise<IndexDocumentResponse> {
  const formData = new FormData();
  formData.append('document', file);
  if (documentId) {
    formData.append('document_id', documentId);
  }
  return apiClient<IndexDocumentResponse>(`${MEETING_API}/index-document`, {
    method: 'POST',
    body: formData,
    isFormData: true,
  });
}

export async function deleteIndexedDocument(
  documentId: string
): Promise<{ success: boolean; message: string }> {
  return apiClient(`${MEETING_API}/index-document/${encodeURIComponent(documentId)}`, {
    method: 'DELETE',
  });
}

export async function getUserDatasetInfo(): Promise<UserDatasetInfo> {
  return apiClient<UserDatasetInfo>(`${MEETING_API}/user-dataset-info`, {
    method: 'GET',
  });
}

export async function getRetrievalStatus(): Promise<{
  retrieval: { enabled: boolean; merge_strategy: string };
  pageindex: { enabled: boolean; available: boolean };
}> {
  return apiClient(`${MEETING_API}/retrieval-status`, { method: 'GET' });
}
