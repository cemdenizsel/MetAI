import { apiClient } from './client';
import type {
  MultiModelResponse,
  AnalysisJob,
  JobSubmitResponse,
  ModelInfo,
  AnalysisOptions,
} from '@/types/emotion';

export async function analyzeVideo(
  file: File,
  options: AnalysisOptions = {}
): Promise<MultiModelResponse> {
  const formData = new FormData();
  formData.append('video', file);

  // Map model to fusion_strategy if provided
  if (options.model) {
    formData.append('fusion_strategy', options.model);
  }
  if (options.include_ai_analysis !== undefined) {
    formData.append('run_ai_analysis', String(options.include_ai_analysis));
  }
  if (options.llm_provider) {
    formData.append('llm_provider', options.llm_provider);
  }

  return apiClient<MultiModelResponse>('/api/emotion/analyze', {
    method: 'POST',
    body: formData,
    isFormData: true,
  });
}

export async function submitJob(
  file: File,
  options: AnalysisOptions = {}
): Promise<JobSubmitResponse> {
  const formData = new FormData();
  formData.append('video', file);

  // Map model to fusion_strategy if provided
  if (options.model) {
    formData.append('fusion_strategy', options.model);
  }
  if (options.include_ai_analysis !== undefined) {
    formData.append('run_ai_analysis', String(options.include_ai_analysis));
  }
  if (options.llm_provider) {
    formData.append('llm_provider', options.llm_provider);
  }

  return apiClient<JobSubmitResponse>('/api/jobs/submit', {
    method: 'POST',
    body: formData,
    isFormData: true,
  });
}

export async function getJobStatus(
  jobId: string
): Promise<{ status: string; progress?: number }> {
  return apiClient<{ status: string; progress?: number }>(
    `/api/jobs/${jobId}/status`
  );
}

export async function getJobResult(jobId: string): Promise<AnalysisJob> {
  return apiClient<AnalysisJob>(`/api/jobs/${jobId}/result`);
}

/** Backend response shape for list jobs */
interface MyJobsResponse {
  success: boolean;
  count: number;
  limit: number;
  skip: number;
  jobs: Array<{
    job_id: string;
    filename?: string;
    status: string;
    created_at: string | null;
    completed_at?: string;
    processing_time?: number;
    result?: unknown;
    error?: string;
  }>;
}

export async function getMyJobs(
  limit: number = 10,
  skip: number = 0
): Promise<AnalysisJob[]> {
  const res = await apiClient<MyJobsResponse>(
    `/api/jobs/my-jobs?limit=${limit}&skip=${skip}`
  );
  if (!res?.jobs || !Array.isArray(res.jobs)) return [];
  const statusMap: Record<string, AnalysisJob['status']> = {
    PENDING: 'pending',
    STARTED: 'processing',
    PROGRESS: 'processing',
    SUCCESS: 'success',
    FAILURE: 'failed',
    REVOKED: 'failed',
    RETRY: 'processing',
  };
  return res.jobs.map((job) => ({
    id: job.job_id,
    user_id: '',
    filename: job.filename ?? '',
    status: statusMap[job.status] ?? (job.status.toLowerCase() as AnalysisJob['status']) ?? 'pending',
    created_at: job.created_at ?? new Date().toISOString(),
    completed_at: job.completed_at,
    result: job.result as AnalysisJob['result'],
    error: job.error,
  }));
}

export async function cancelJob(jobId: string): Promise<void> {
  return apiClient<void>(`/api/jobs/${jobId}/cancel`, { method: 'POST' });
}

export async function deleteJob(jobId: string): Promise<void> {
  return apiClient<void>(`/api/jobs/${jobId}`, { method: 'DELETE' });
}

export async function getModels(): Promise<ModelInfo[]> {
  return apiClient<ModelInfo[]>('/api/emotion/models');
}

export async function checkHealth(): Promise<{ status: string }> {
  return apiClient<{ status: string }>('/api/emotion/health');
}
