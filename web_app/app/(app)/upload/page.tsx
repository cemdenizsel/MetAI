'use client';

import { useState, useEffect } from 'react';
import {
  Brain,
  Cloud,
  Server,
  Loader2,
  Settings2,
  Sparkles,
} from 'lucide-react';
import { VideoUploader } from '@/components/emotion/VideoUploader';
import { AnalysisResults } from '@/components/emotion/AnalysisResults';
import { JobTracker } from '@/components/emotion/JobTracker';
import { useJobTracking } from '@/lib/hooks/useJobTracking';
import * as emotionApi from '@/lib/api/emotion';
import type { MultiModelResponse, ModelInfo, AnalysisOptions } from '@/types/emotion';

export default function UploadPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<MultiModelResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [showOptions, setShowOptions] = useState(false);
  const [options, setOptions] = useState<AnalysisOptions>({
    include_ai_analysis: true,
    llm_provider: 'cloud',
  });

  const { job, isPolling, error: jobError } = useJobTracking(jobId);

  useEffect(() => {
    async function fetchModels() {
      try {
        const modelList = await emotionApi.getModels();
        setModels(modelList);
        if (modelList.length > 0) {
          setOptions((prev) => ({ ...prev, model: modelList[0].name }));
        }
      } catch (err) {
        console.error('Failed to fetch models:', err);
      }
    }
    fetchModels();
  }, []);

  useEffect(() => {
    if (job?.status === 'success' && job.result) {
      // Job result might be in legacy format, convert if needed
      if ('results' in job.result) {
        setResult(job.result);
      } else {
        // Convert legacy format to new format if needed
        setResult(job.result as any);
      }
      setIsAnalyzing(false);
    }
    if (job?.status === 'failed') {
      setError(job.error || 'Analysis failed');
      setIsAnalyzing(false);
    }
  }, [job]);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setResult(null);
    setError(null);
    setJobId(null);
  };

  const handleClear = () => {
    setSelectedFile(null);
    setResult(null);
    setError(null);
    setJobId(null);
    setIsAnalyzing(false);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    try {
      const response = await emotionApi.submitJob(selectedFile, options);
      setJobId(response.job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
      setIsAnalyzing(false);
    }
  };

  const handleDirectAnalyze = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    try {
      const response = await emotionApi.analyzeVideo(selectedFile, options);
      // Response is already in the correct format from the API
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          New Analysis
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Upload a video to analyze emotional content
        </p>
      </div>

      {/* Upload Section */}
      {!result && (
        <div className="space-y-6">
          <VideoUploader
            onFileSelect={handleFileSelect}
            selectedFile={selectedFile}
            onClear={handleClear}
            disabled={isAnalyzing}
          />

          {/* Options */}
          {selectedFile && !isAnalyzing && (
            <div className="p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
              <button
                onClick={() => setShowOptions(!showOptions)}
                className="flex items-center gap-2 text-gray-700 dark:text-gray-300 hover:text-teal-600 dark:hover:text-teal-400 transition-colors"
              >
                <Settings2 className="w-5 h-5" />
                <span className="font-medium">Analysis Options</span>
                <span className="text-xs text-gray-500 dark:text-gray-400">
                  ({showOptions ? 'Hide' : 'Show'})
                </span>
              </button>

              {showOptions && (
                <div className="mt-4 space-y-4">
                  {/* Model Selection */}
                  {models.length > 0 && (
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Analysis Model
                      </label>
                      <select
                        value={options.model || ''}
                        onChange={(e) =>
                          setOptions((prev) => ({
                            ...prev,
                            model: e.target.value,
                          }))
                        }
                        className="w-full px-4 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white focus:ring-2 focus:ring-teal-500"
                      >
                        {models.map((model) => (
                          <option key={model.name} value={model.name}>
                            {model.name} - {model.description}
                          </option>
                        ))}
                      </select>
                    </div>
                  )}

                  {/* AI Analysis Toggle */}
                  <div className="flex items-center justify-between">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                        Include AI Analysis
                      </label>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        Get detailed insights and recommendations
                      </p>
                    </div>
                    <button
                      onClick={() =>
                        setOptions((prev) => ({
                          ...prev,
                          include_ai_analysis: !prev.include_ai_analysis,
                        }))
                      }
                      className={`relative w-12 h-6 rounded-full transition-colors ${
                        options.include_ai_analysis
                          ? 'bg-teal-500'
                          : 'bg-gray-300 dark:bg-gray-600'
                      }`}
                    >
                      <span
                        className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-transform ${
                          options.include_ai_analysis
                            ? 'translate-x-7'
                            : 'translate-x-1'
                        }`}
                      />
                    </button>
                  </div>

                  {/* LLM Provider */}
                  {options.include_ai_analysis && (
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        AI Provider
                      </label>
                      <div className="flex gap-3">
                        <button
                          onClick={() =>
                            setOptions((prev) => ({
                              ...prev,
                              llm_provider: 'cloud',
                            }))
                          }
                          className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-lg border transition-colors ${
                            options.llm_provider === 'cloud'
                              ? 'border-teal-500 bg-teal-50 dark:bg-teal-900/20 text-teal-700 dark:text-teal-400'
                              : 'border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:border-teal-400'
                          }`}
                        >
                          <Cloud className="w-5 h-5" />
                          <span className="font-medium">Cloud AI</span>
                        </button>
                        <button
                          onClick={() =>
                            setOptions((prev) => ({
                              ...prev,
                              llm_provider: 'local',
                            }))
                          }
                          className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-lg border transition-colors ${
                            options.llm_provider === 'local'
                              ? 'border-teal-500 bg-teal-50 dark:bg-teal-900/20 text-teal-700 dark:text-teal-400'
                              : 'border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:border-teal-400'
                          }`}
                        >
                          <Server className="w-5 h-5" />
                          <span className="font-medium">Local AI</span>
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Analyze Buttons */}
          {selectedFile && !isAnalyzing && (
            <div className="space-y-4">
              {/* Direct Analysis Button */}
              <button
                onClick={handleDirectAnalyze}
                className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-teal-600 hover:bg-teal-700 text-white rounded-xl font-semibold shadow-lg transition-colors"
              >
                <Sparkles className="w-5 h-5" />
                <span>Analyze Video</span>
              </button>

              {/* Async Analysis - Coming Soon */}
              <div className="relative">
                <button
                  disabled
                  className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-gray-100 dark:bg-gray-700 text-gray-400 dark:text-gray-500 rounded-xl font-semibold cursor-not-allowed opacity-60"
                >
                  <Brain className="w-5 h-5" />
                  <span>Analyze (Background Job)</span>
                </button>
                <div className="absolute -top-2 -right-2 bg-amber-500 text-white text-xs font-semibold px-2 py-1 rounded-full">
                  Coming Soon
                </div>
              </div>
              <p className="text-xs text-center text-gray-500 dark:text-gray-400">
                Background job analysis will be available in a future update
              </p>
            </div>
          )}

          {/* Job Tracker */}
          {isAnalyzing && jobId && (
            <JobTracker job={job} isPolling={isPolling} error={jobError} />
          )}

          {/* Loading State for Direct Analysis */}
          {isAnalyzing && !jobId && (
            <div className="p-8 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 text-center">
              <Loader2 className="w-12 h-12 text-teal-600 animate-spin mx-auto mb-4" />
              <p className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                Analyzing Video...
              </p>
              <p className="text-gray-600 dark:text-gray-400">
                This may take a moment depending on video length
              </p>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl">
              <p className="text-red-600 dark:text-red-400">{error}</p>
              <button
                onClick={handleClear}
                className="mt-2 text-sm text-red-700 dark:text-red-300 hover:underline"
              >
                Try again
              </button>
            </div>
          )}
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
              Analysis Results
            </h2>
            <button
              onClick={handleClear}
              className="px-4 py-2 text-teal-600 dark:text-teal-400 hover:bg-teal-50 dark:hover:bg-teal-900/20 rounded-lg font-medium transition-colors"
            >
              New Analysis
            </button>
          </div>
          <AnalysisResults result={result} />
        </div>
      )}
    </div>
  );
}
