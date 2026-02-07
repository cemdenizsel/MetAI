'use client';

import { useState, useEffect } from 'react';
import {
  FileText,
  Upload,
  Search,
  Loader2,
  CheckCircle,
  AlertCircle,
  Info,
} from 'lucide-react';
import * as meetingApi from '@/lib/api/meeting';
import type { UserDatasetInfo, AskCitation } from '@/types/meeting';

export default function DocumentsPage() {
  const [datasetInfo, setDatasetInfo] = useState<UserDatasetInfo | null>(null);
  const [datasetLoading, setDatasetLoading] = useState(true);
  const [datasetError, setDatasetError] = useState<string | null>(null);

  const [indexFile, setIndexFile] = useState<File | null>(null);
  const [indexDocId, setIndexDocId] = useState('');
  const [indexing, setIndexing] = useState(false);
  const [indexSuccess, setIndexSuccess] = useState<string | null>(null);
  const [indexError, setIndexError] = useState<string | null>(null);

  const [searchQuery, setSearchQuery] = useState('');
  const [searching, setSearching] = useState(false);
  const [searchStatus, setSearchStatus] = useState<string | null>(null);
  const [searchAnswer, setSearchAnswer] = useState<string | null>(null);
  const [searchCitations, setSearchCitations] = useState<AskCitation[]>([]);
  const [searchError, setSearchError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      setDatasetLoading(true);
      setDatasetError(null);
      try {
        const info = await meetingApi.getUserDatasetInfo();
        setDatasetInfo(info);
      } catch (err) {
        setDatasetError(err instanceof Error ? err.message : 'Failed to load dataset info');
      } finally {
        setDatasetLoading(false);
      }
    }
    load();
  }, []);

  const handleIndexSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!indexFile) return;
    setIndexing(true);
    setIndexSuccess(null);
    setIndexError(null);
    try {
      const res = await meetingApi.indexDocument(
        indexFile,
        indexDocId.trim() || undefined
      );
      setIndexSuccess(res.message);
      setIndexFile(null);
      setIndexDocId('');
      const info = await meetingApi.getUserDatasetInfo();
      setDatasetInfo(info);
    } catch (err) {
      setIndexError(err instanceof Error ? err.message : 'Indexing failed');
    } finally {
      setIndexing(false);
    }
  };

  const handleSearchSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;
    setSearching(true);
    setSearchStatus('Asking your documents...');
    setSearchAnswer(null);
    setSearchCitations([]);
    setSearchError(null);
    try {
      const res = await meetingApi.ask(searchQuery.trim());
      setSearchAnswer(res.answer);
      setSearchCitations(res.citations ?? []);
      setSearchStatus(null);
    } catch (err) {
      setSearchError(err instanceof Error ? err.message : 'Ask failed');
      setSearchStatus(null);
    } finally {
      setSearching(false);
    }
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2 flex items-center gap-3">
          <FileText className="w-8 h-8 text-teal-500" />
          Documents
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Upload PDFs to your knowledge base and search them during meetings. Used by the Real-Time chat (RAG-first answers).
        </p>
      </div>

      {/* Dataset info */}
      <div className="p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
          <Info className="w-5 h-5 text-teal-500" />
          Dataset status
        </h2>
        {datasetLoading && (
          <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400">
            <Loader2 className="w-4 h-4 animate-spin" />
            Loading...
          </div>
        )}
        {datasetError && (
          <p className="text-red-600 dark:text-red-400">{datasetError}</p>
        )}
        {!datasetLoading && datasetInfo && (
          <div className="p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg max-w-md">
            <p className="font-medium text-gray-900 dark:text-white mb-1">PageIndex</p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              {datasetInfo.pageindex?.available
                ? 'Available'
                : datasetInfo.pageindex?.message ?? datasetInfo.pageindex?.error ?? 'Not available'}
            </p>
          </div>
        )}
      </div>

      {/* Index document */}
      <div className="p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
          <Upload className="w-5 h-5 text-teal-500" />
          Index document (PDF)
        </h2>
        <form onSubmit={handleIndexSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              PDF file
            </label>
            <input
              type="file"
              accept=".pdf"
              onChange={(e) => setIndexFile(e.target.files?.[0] ?? null)}
              className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-teal-50 file:text-teal-700 dark:file:bg-teal-900/20 dark:file:text-teal-400"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Document ID (optional)
            </label>
            <input
              type="text"
              value={indexDocId}
              onChange={(e) => setIndexDocId(e.target.value)}
              placeholder="e.g. handbook-2024"
              className="w-full px-4 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white focus:ring-2 focus:ring-teal-500"
            />
          </div>
          <button
            type="submit"
            disabled={!indexFile || indexing}
            className="flex items-center gap-2 px-4 py-2 bg-teal-600 hover:bg-teal-700 disabled:opacity-50 text-white rounded-lg font-medium transition-colors"
          >
            {indexing ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Upload className="w-4 h-4" />
            )}
            {indexing ? 'Indexing...' : 'Index document'}
          </button>
        </form>
        {indexSuccess && (
          <p className="mt-4 flex items-center gap-2 text-green-600 dark:text-green-400">
            <CheckCircle className="w-4 h-4" />
            {indexSuccess}
          </p>
        )}
        {indexError && (
          <p className="mt-4 flex items-center gap-2 text-red-600 dark:text-red-400">
            <AlertCircle className="w-4 h-4" />
            {indexError}
          </p>
        )}
      </div>

      {/* Ask your documents (Q&A) */}
      <div className="p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
          <Search className="w-5 h-5 text-teal-500" />
          Ask your documents
        </h2>
        <form onSubmit={handleSearchSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Question
            </label>
            <textarea
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              rows={4}
              placeholder="Type a question about your uploaded documents..."
              className="w-full px-4 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white focus:ring-2 focus:ring-teal-500 resize-y"
            />
          </div>
          <button
            type="submit"
            disabled={!searchQuery.trim() || searching}
            className="flex items-center gap-2 px-4 py-2 bg-teal-600 hover:bg-teal-700 disabled:opacity-50 text-white rounded-lg font-medium transition-colors"
          >
            {searching ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Search className="w-4 h-4" />
            )}
            {searching ? 'Asking...' : 'Ask'}
          </button>
        </form>
        {searching && searchStatus && (
          <p className="mt-4 text-sm text-gray-600 dark:text-gray-400">{searchStatus}</p>
        )}
        {searchError && (
          <p className="mt-4 flex items-center gap-2 text-red-600 dark:text-red-400">
            <AlertCircle className="w-4 h-4" />
            {searchError}
          </p>
        )}
        {searchAnswer != null && (
          <div className="mt-6 space-y-4">
            <h3 className="font-medium text-gray-900 dark:text-white">Answer</h3>
            <div className="p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg border border-gray-200 dark:border-gray-600">
              <p className="text-gray-700 dark:text-gray-300 whitespace-pre-wrap">{searchAnswer}</p>
            </div>
            {searchCitations.length > 0 && (
              <>
                <h3 className="font-medium text-gray-900 dark:text-white">References</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Sources used for this answer so you can verify or read more.
                </p>
                <div className="space-y-2">
                  {searchCitations.map((c, i) => (
                    <div
                      key={i}
                      className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg border border-gray-200 dark:border-gray-600 text-sm"
                    >
                      <span className="font-medium text-gray-700 dark:text-gray-300">
                        Document: {c.document_id || 'Unknown'}
                        {c.page_number != null && ` · Page ${c.page_number}`}
                      </span>
                      {c.content_snippet && (
                        <p className="mt-1 text-gray-600 dark:text-gray-400 line-clamp-2">
                          {c.content_snippet}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
