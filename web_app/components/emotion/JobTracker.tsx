'use client';

import { Loader2, CheckCircle, AlertCircle, Clock } from 'lucide-react';
import type { AnalysisJob } from '@/types/emotion';

interface JobTrackerProps {
  job: AnalysisJob | null;
  isPolling: boolean;
  error: string | null;
}

export function JobTracker({ job, isPolling, error }: JobTrackerProps) {
  if (!job) return null;

  const getStatusDisplay = () => {
    switch (job.status) {
      case 'pending':
        return {
          icon: Clock,
          text: 'Waiting in queue...',
          color: 'text-gray-500',
          bgColor: 'bg-gray-100 dark:bg-gray-700',
        };
      case 'processing':
        return {
          icon: Loader2,
          text: 'Processing video...',
          color: 'text-blue-500',
          bgColor: 'bg-blue-100 dark:bg-blue-900/30',
          animate: true,
        };
      case 'success':
        return {
          icon: CheckCircle,
          text: 'Analysis complete!',
          color: 'text-green-500',
          bgColor: 'bg-green-100 dark:bg-green-900/30',
        };
      case 'failed':
        return {
          icon: AlertCircle,
          text: 'Analysis failed',
          color: 'text-red-500',
          bgColor: 'bg-red-100 dark:bg-red-900/30',
        };
      default:
        return {
          icon: Clock,
          text: 'Unknown status',
          color: 'text-gray-500',
          bgColor: 'bg-gray-100 dark:bg-gray-700',
        };
    }
  };

  const status = getStatusDisplay();
  const StatusIcon = status.icon;

  return (
    <div className={`p-6 rounded-xl ${status.bgColor}`}>
      <div className="flex items-center gap-4">
        <div
          className={`w-12 h-12 rounded-full flex items-center justify-center ${status.bgColor}`}
        >
          <StatusIcon
            className={`w-6 h-6 ${status.color} ${
              status.animate ? 'animate-spin' : ''
            }`}
          />
        </div>
        <div className="flex-1">
          <h3 className={`text-lg font-semibold ${status.color}`}>
            {status.text}
          </h3>
          {job.filename && (
            <p className="text-sm text-gray-600 dark:text-gray-400">
              {job.filename}
            </p>
          )}
          {error && (
            <p className="text-sm text-red-500 mt-1">{error}</p>
          )}
        </div>
        {isPolling && job.status !== 'success' && job.status !== 'failed' && (
          <div className="text-sm text-gray-500 dark:text-gray-400">
            Checking status...
          </div>
        )}
      </div>

      {/* Progress Steps */}
      <div className="mt-6 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div
            className={`w-8 h-8 rounded-full flex items-center justify-center ${
              job.status === 'pending' ||
              job.status === 'processing' ||
              job.status === 'success'
                ? 'bg-teal-500 text-white'
                : 'bg-gray-300 dark:bg-gray-600 text-gray-500'
            }`}
          >
            1
          </div>
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Queued
          </span>
        </div>
        <div className="flex-1 h-1 mx-4 bg-gray-200 dark:bg-gray-600 rounded">
          <div
            className={`h-full rounded transition-all duration-500 ${
              job.status === 'processing' || job.status === 'success'
                ? 'bg-teal-500 w-full'
                : 'bg-gray-300 w-0'
            }`}
          />
        </div>
        <div className="flex items-center gap-2">
          <div
            className={`w-8 h-8 rounded-full flex items-center justify-center ${
              job.status === 'processing' || job.status === 'success'
                ? 'bg-teal-500 text-white'
                : 'bg-gray-300 dark:bg-gray-600 text-gray-500'
            }`}
          >
            2
          </div>
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Processing
          </span>
        </div>
        <div className="flex-1 h-1 mx-4 bg-gray-200 dark:bg-gray-600 rounded">
          <div
            className={`h-full rounded transition-all duration-500 ${
              job.status === 'success'
                ? 'bg-teal-500 w-full'
                : 'bg-gray-300 w-0'
            }`}
          />
        </div>
        <div className="flex items-center gap-2">
          <div
            className={`w-8 h-8 rounded-full flex items-center justify-center ${
              job.status === 'success'
                ? 'bg-teal-500 text-white'
                : 'bg-gray-300 dark:bg-gray-600 text-gray-500'
            }`}
          >
            3
          </div>
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Complete
          </span>
        </div>
      </div>
    </div>
  );
}
