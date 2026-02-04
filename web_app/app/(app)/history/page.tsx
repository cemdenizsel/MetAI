'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import {
  History,
  Search,
  Filter,
  CheckCircle,
  AlertCircle,
  Clock,
  Loader2,
  Trash2,
  Eye,
  Upload,
  Calendar,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import * as emotionApi from '@/lib/api/emotion';
import type { AnalysisJob } from '@/types/emotion';
import { AnalysisResults } from '@/components/emotion/AnalysisResults';

const ITEMS_PER_PAGE = 10;

export default function HistoryPage() {
  const [jobs, setJobs] = useState<AnalysisJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedJob, setSelectedJob] = useState<AnalysisJob | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [page, setPage] = useState(0);
  const [hasMore, setHasMore] = useState(true);

  useEffect(() => {
    fetchJobs();
  }, [page]);

  const fetchJobs = async () => {
    setLoading(true);
    try {
      const data = await emotionApi.getMyJobs(ITEMS_PER_PAGE, page * ITEMS_PER_PAGE);
      // Ensure data is always an array
      const jobsArray = Array.isArray(data) ? data : [];
      setJobs(jobsArray);
      setHasMore(jobsArray.length === ITEMS_PER_PAGE);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load history');
      setJobs([]); // Reset to empty array on error
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (jobId: string) => {
    if (!confirm('Are you sure you want to delete this analysis?')) return;

    try {
      await emotionApi.deleteJob(jobId);
      setJobs((prevJobs) => (Array.isArray(prevJobs) ? prevJobs : []).filter((j) => j.id !== jobId));
      if (selectedJob?.id === jobId) {
        setSelectedJob(null);
      }
    } catch (err) {
      alert('Failed to delete job');
    }
  };

  const getStatusIcon = (status: AnalysisJob['status']) => {
    switch (status) {
      case 'success':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'failed':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      case 'processing':
        return <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />;
      default:
        return <Clock className="w-4 h-4 text-gray-400" />;
    }
  };

  const getStatusBadge = (status: AnalysisJob['status']) => {
    const colors = {
      success: 'bg-green-100 dark:bg-green-900/20 text-green-700 dark:text-green-400',
      failed: 'bg-red-100 dark:bg-red-900/20 text-red-700 dark:text-red-400',
      processing: 'bg-blue-100 dark:bg-blue-900/20 text-blue-700 dark:text-blue-400',
      pending: 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-400',
    };
    return colors[status] || colors.pending;
  };

  const filteredJobs = (Array.isArray(jobs) ? jobs : []).filter((job) => {
    const matchesStatus = statusFilter === 'all' || job.status === statusFilter;
    const matchesSearch =
      !searchQuery ||
      job.filename?.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesStatus && matchesSearch;
  });

  if (selectedJob) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-4">
          <button
            onClick={() => setSelectedJob(null)}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
          >
            <ChevronLeft className="w-5 h-5 text-gray-600 dark:text-gray-400" />
          </button>
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              {selectedJob.filename}
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              {new Date(selectedJob.created_at).toLocaleString()}
            </p>
          </div>
        </div>

        {selectedJob.result ? (
          <AnalysisResults result={selectedJob.result} />
        ) : (
          <div className="p-8 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 text-center">
            <AlertCircle className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-600 dark:text-gray-400">
              {selectedJob.status === 'failed'
                ? selectedJob.error || 'Analysis failed'
                : 'No results available'}
            </p>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            Analysis History
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            View and manage your past emotion analyses
          </p>
        </div>
        <Link
          href="/upload"
          className="inline-flex items-center justify-center gap-2 px-4 py-2 bg-teal-600 hover:bg-teal-700 text-white rounded-lg font-medium transition-colors"
        >
          <Upload className="w-4 h-4" />
          New Analysis
        </Link>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="relative flex-1">
          <Search className="w-5 h-5 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search by filename..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white placeholder-gray-500 focus:ring-2 focus:ring-teal-500"
          />
        </div>
        <div className="flex items-center gap-2">
          <Filter className="w-5 h-5 text-gray-400" />
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="px-4 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white focus:ring-2 focus:ring-teal-500"
          >
            <option value="all">All Status</option>
            <option value="success">Completed</option>
            <option value="processing">Processing</option>
            <option value="pending">Pending</option>
            <option value="failed">Failed</option>
          </select>
        </div>
      </div>

      {/* Jobs List */}
      <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
        {loading ? (
          <div className="p-8 text-center">
            <Loader2 className="w-8 h-8 text-teal-600 animate-spin mx-auto mb-4" />
            <p className="text-gray-600 dark:text-gray-400">Loading history...</p>
          </div>
        ) : error ? (
          <div className="p-8 text-center">
            <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
            <p className="text-red-600 dark:text-red-400">{error}</p>
            <button
              onClick={fetchJobs}
              className="mt-4 text-teal-600 dark:text-teal-400 hover:underline"
            >
              Try again
            </button>
          </div>
        ) : filteredJobs.length === 0 ? (
          <div className="p-8 text-center">
            <History className="w-12 h-12 text-gray-300 dark:text-gray-600 mx-auto mb-4" />
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              {jobs.length === 0
                ? 'No analyses yet'
                : 'No analyses match your filters'}
            </p>
            {jobs.length === 0 && (
              <Link
                href="/upload"
                className="inline-flex items-center gap-2 px-4 py-2 bg-teal-600 hover:bg-teal-700 text-white rounded-lg font-medium transition-colors"
              >
                <Upload className="w-4 h-4" />
                Start Your First Analysis
              </Link>
            )}
          </div>
        ) : (
          <>
            <div className="divide-y divide-gray-200 dark:divide-gray-700">
              {filteredJobs.map((job) => (
                <div
                  key={job.id}
                  className="flex items-center gap-4 px-6 py-4 hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
                >
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-gray-900 dark:text-white truncate">
                      {job.filename}
                    </p>
                    <div className="flex items-center gap-3 text-sm text-gray-500 dark:text-gray-400">
                      <span className="flex items-center gap-1">
                        <Calendar className="w-3 h-3" />
                        {new Date(job.created_at).toLocaleDateString()}
                      </span>
                      <span>
                        {new Date(job.created_at).toLocaleTimeString()}
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    {getStatusIcon(job.status)}
                    <span
                      className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusBadge(
                        job.status
                      )}`}
                    >
                      {job.status}
                    </span>
                  </div>

                  {job.status === 'success' && job.result && (
                    <div className="hidden md:block text-right min-w-[100px]">
                      <p className="text-sm font-medium text-gray-900 dark:text-white capitalize">
                        {job.result.predicted_emotion}
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        {Math.round(job.result.confidence * 100)}%
                      </p>
                    </div>
                  )}

                  <div className="flex items-center gap-1">
                    {job.status === 'success' && job.result && (
                      <button
                        onClick={() => setSelectedJob(job)}
                        className="p-2 text-gray-400 hover:text-teal-600 hover:bg-teal-50 dark:hover:bg-teal-900/20 rounded-lg transition-colors"
                        title="View details"
                      >
                        <Eye className="w-5 h-5" />
                      </button>
                    )}
                    <button
                      onClick={() => handleDelete(job.id)}
                      className="p-2 text-gray-400 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors"
                      title="Delete"
                    >
                      <Trash2 className="w-5 h-5" />
                    </button>
                  </div>
                </div>
              ))}
            </div>

            {/* Pagination */}
            <div className="flex items-center justify-between px-6 py-4 border-t border-gray-200 dark:border-gray-700">
              <button
                onClick={() => setPage(Math.max(0, page - 1))}
                disabled={page === 0}
                className="flex items-center gap-1 px-3 py-1 text-gray-600 dark:text-gray-400 hover:text-teal-600 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronLeft className="w-4 h-4" />
                Previous
              </button>
              <span className="text-sm text-gray-600 dark:text-gray-400">
                Page {page + 1}
              </span>
              <button
                onClick={() => setPage(page + 1)}
                disabled={!hasMore}
                className="flex items-center gap-1 px-3 py-1 text-gray-600 dark:text-gray-400 hover:text-teal-600 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Next
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
