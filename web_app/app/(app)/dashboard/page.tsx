'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import {
  Brain,
  Upload,
  History,
  TrendingUp,
  Clock,
  CheckCircle,
  AlertCircle,
  ArrowRight,
  BarChart3,
} from 'lucide-react';
import { useAuthStore } from '@/lib/stores/authStore';
import * as emotionApi from '@/lib/api/emotion';
import type { AnalysisJob } from '@/types/emotion';

export default function DashboardPage() {
  const { user } = useAuthStore();
  const [recentJobs, setRecentJobs] = useState<AnalysisJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    totalAnalyses: 0,
    completed: 0,
    pending: 0,
    failed: 0,
  });

  useEffect(() => {
    async function fetchData() {
      try {
        const jobs = await emotionApi.getMyJobs(5, 0);
        setRecentJobs(jobs);

        const total = jobs.length;
        const completed = jobs.filter((j) => j.status === 'success').length;
        const pending = jobs.filter(
          (j) => j.status === 'pending' || j.status === 'processing'
        ).length;
        const failed = jobs.filter((j) => j.status === 'failed').length;

        setStats({ totalAnalyses: total, completed, pending, failed });
      } catch (error) {
        console.error('Failed to fetch jobs:', error);
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, []);

  const getStatusIcon = (status: AnalysisJob['status']) => {
    switch (status) {
      case 'success':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'failed':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      case 'processing':
        return <Clock className="w-4 h-4 text-blue-500 animate-pulse" />;
      default:
        return <Clock className="w-4 h-4 text-gray-400" />;
    }
  };

  const getStatusColor = (status: AnalysisJob['status']) => {
    switch (status) {
      case 'success':
        return 'bg-green-100 dark:bg-green-900/20 text-green-700 dark:text-green-400';
      case 'failed':
        return 'bg-red-100 dark:bg-red-900/20 text-red-700 dark:text-red-400';
      case 'processing':
        return 'bg-blue-100 dark:bg-blue-900/20 text-blue-700 dark:text-blue-400';
      default:
        return 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-400';
    }
  };

  return (
    <div className="space-y-8">
      {/* Welcome Section */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          Welcome back{user?.name ? `, ${user.name}` : ''}!
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Analyze emotions from video content with AI-powered insights
        </p>
      </div>

      {/* Quick Actions */}
      <div className="grid md:grid-cols-2 gap-4">
        <Link
          href="/upload"
          className="group flex items-center gap-4 p-6 bg-gradient-to-br from-teal-500 to-teal-600 rounded-xl text-white hover:from-teal-600 hover:to-teal-700 transition-all shadow-lg"
        >
          <div className="w-12 h-12 bg-white/20 rounded-lg flex items-center justify-center">
            <Upload className="w-6 h-6" />
          </div>
          <div className="flex-1">
            <h3 className="text-lg font-semibold mb-1">New Analysis</h3>
            <p className="text-teal-100 text-sm">
              Upload a video to analyze emotions
            </p>
          </div>
          <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
        </Link>

        <Link
          href="/history"
          className="group flex items-center gap-4 p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 hover:border-teal-500 dark:hover:border-teal-500 transition-all"
        >
          <div className="w-12 h-12 bg-teal-100 dark:bg-teal-900/30 rounded-lg flex items-center justify-center">
            <History className="w-6 h-6 text-teal-600 dark:text-teal-400" />
          </div>
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-1">
              View History
            </h3>
            <p className="text-gray-600 dark:text-gray-400 text-sm">
              Browse your past analyses
            </p>
          </div>
          <ArrowRight className="w-5 h-5 text-gray-400 group-hover:text-teal-500 group-hover:translate-x-1 transition-all" />
        </Link>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 bg-teal-100 dark:bg-teal-900/30 rounded-lg flex items-center justify-center">
              <Brain className="w-5 h-5 text-teal-600 dark:text-teal-400" />
            </div>
          </div>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">
            {loading ? '-' : stats.totalAnalyses}
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Total Analyses
          </p>
        </div>

        <div className="p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 bg-green-100 dark:bg-green-900/30 rounded-lg flex items-center justify-center">
              <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" />
            </div>
          </div>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">
            {loading ? '-' : stats.completed}
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400">Completed</p>
        </div>

        <div className="p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center">
              <Clock className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            </div>
          </div>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">
            {loading ? '-' : stats.pending}
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400">In Progress</p>
        </div>

        <div className="p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 bg-amber-100 dark:bg-amber-900/30 rounded-lg flex items-center justify-center">
              <TrendingUp className="w-5 h-5 text-amber-600 dark:text-amber-400" />
            </div>
          </div>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">
            {loading
              ? '-'
              : stats.totalAnalyses > 0
              ? Math.round((stats.completed / stats.totalAnalyses) * 100)
              : 0}
            %
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Success Rate
          </p>
        </div>
      </div>

      {/* Recent Analyses */}
      <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            Recent Analyses
          </h2>
          <Link
            href="/history"
            className="text-sm text-teal-600 dark:text-teal-400 hover:underline"
          >
            View all
          </Link>
        </div>

        {loading ? (
          <div className="p-8 text-center text-gray-500 dark:text-gray-400">
            Loading...
          </div>
        ) : recentJobs.length === 0 ? (
          <div className="p-8 text-center">
            <BarChart3 className="w-12 h-12 text-gray-300 dark:text-gray-600 mx-auto mb-4" />
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              No analyses yet
            </p>
            <Link
              href="/upload"
              className="inline-flex items-center gap-2 px-4 py-2 bg-teal-600 hover:bg-teal-700 text-white rounded-lg font-medium transition-colors"
            >
              <Upload className="w-4 h-4" />
              Start Your First Analysis
            </Link>
          </div>
        ) : (
          <div className="divide-y divide-gray-200 dark:divide-gray-700">
            {recentJobs.length > 0 && recentJobs.map((job) => (
              <div
                key={job.id}
                className="flex items-center gap-4 px-6 py-4 hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
              >
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-gray-900 dark:text-white truncate">
                    {job.filename}
                  </p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    {new Date(job.created_at).toLocaleDateString()} at{' '}
                    {new Date(job.created_at).toLocaleTimeString()}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  {getStatusIcon(job.status)}
                  <span
                    className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(
                      job.status
                    )}`}
                  >
                    {job.status}
                  </span>
                </div>
                {job.status === 'success' && job.result && (
                  <div className="hidden sm:block text-right">
                    <p className="text-sm font-medium text-gray-900 dark:text-white capitalize">
                      {job.result.predicted_emotion}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      {Math.round(job.result.confidence * 100)}% confidence
                    </p>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
