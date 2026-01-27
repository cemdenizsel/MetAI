'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import * as emotionApi from '@/lib/api/emotion';
import type { AnalysisJob } from '@/types/emotion';

interface JobTrackingState {
  job: AnalysisJob | null;
  isPolling: boolean;
  error: string | null;
}

export function useJobTracking(jobId: string | null) {
  const [state, setState] = useState<JobTrackingState>({
    job: null,
    isPolling: false,
    error: null,
  });
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const stopPolling = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setState((prev) => ({ ...prev, isPolling: false }));
  }, []);

  const fetchJobStatus = useCallback(async (id: string) => {
    try {
      const status = await emotionApi.getJobStatus(id);

      if (status.status === 'success' || status.status === 'failed') {
        const result = await emotionApi.getJobResult(id);
        setState({
          job: result,
          isPolling: false,
          error: status.status === 'failed' ? result.error || 'Job failed' : null,
        });
        stopPolling();
      } else {
        setState((prev) => ({
          ...prev,
          job: prev.job
            ? { ...prev.job, status: status.status as AnalysisJob['status'] }
            : null,
        }));
      }
    } catch (err) {
      setState((prev) => ({
        ...prev,
        error: err instanceof Error ? err.message : 'Failed to fetch job status',
      }));
      stopPolling();
    }
  }, [stopPolling]);

  const startPolling = useCallback(
    (id: string) => {
      setState({
        job: {
          id,
          user_id: '',
          filename: '',
          status: 'pending',
          created_at: new Date().toISOString(),
        },
        isPolling: true,
        error: null,
      });

      fetchJobStatus(id);

      intervalRef.current = setInterval(() => {
        fetchJobStatus(id);
      }, 2000);
    },
    [fetchJobStatus]
  );

  useEffect(() => {
    if (jobId) {
      startPolling(jobId);
    }

    return () => {
      stopPolling();
    };
  }, [jobId, startPolling, stopPolling]);

  return {
    job: state.job,
    isPolling: state.isPolling,
    error: state.error,
    stopPolling,
    retry: jobId ? () => startPolling(jobId) : undefined,
  };
}
