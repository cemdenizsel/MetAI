'use client';

import { useCallback, useEffect, useState } from 'react';
import { Radio, AlertCircle, Info, Wifi, WifiOff, Send, MessageSquare, FileText } from 'lucide-react';
import { WebcamCapture } from '@/components/emotion/WebcamCapture';
import { RealtimeEmotionDisplay } from '@/components/emotion/RealtimeEmotionDisplay';
import { useRealtimeAnalysis } from '@/lib/hooks/useRealtimeAnalysis';
import { useAuthStore } from '@/lib/stores/authStore';
import * as meetingApi from '@/lib/api/meeting';
import type { AskCitation } from '@/types/meeting';

type ChatMessage = {
  role: 'user' | 'assistant';
  content: string;
  source?: 'rag' | 'openai';
  citations?: AskCitation[];
};

export default function RealtimePage() {
  const { token } = useAuthStore();
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [pendingStart, setPendingStart] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);

  const {
    isConnected,
    isConnecting,
    sessionId,
    currentEmotion,
    predictions,
    summary,
    error,
    chunksSent,
    connect,
    disconnect,
    sendChunk,
    complete,
    clearError,
    reset,
  } = useRealtimeAnalysis();

  // Handle chunk ready from webcam
  const handleChunkReady = useCallback(
    (chunk: Blob, timestamp: number, index: number) => {
      if (isConnected) {
        sendChunk(chunk, timestamp, index);
      }
    },
    [isConnected, sendChunk]
  );

  // Start analysis: connect first; recording starts when isConnected (see effect below)
  const handleStartAnalysis = useCallback(() => {
    const authToken = token || (typeof window !== 'undefined' ? localStorage.getItem('auth_token') : null);
    if (!authToken) {
      console.error('No auth token available');
      alert('Not authenticated. Please log out and log in again.');
      return;
    }
    setPendingStart(true);
    connect(authToken);
  }, [token, connect]);

  // Stop analysis
  const handleStopAnalysis = useCallback(() => {
    setPendingStart(false);
    setIsAnalyzing(false);
    complete();
    setTimeout(() => {
      disconnect();
    }, 1000);
  }, [complete, disconnect]);

  // Start recording only after WebSocket is connected
  useEffect(() => {
    if (isConnected && pendingStart) {
      setIsAnalyzing(true);
      setPendingStart(false);
    }
  }, [isConnected, pendingStart]);

  // Clear pending if connection fails or closes before we started recording
  useEffect(() => {
    if (error || (!isConnecting && !isConnected && pendingStart)) {
      setPendingStart(false);
    }
  }, [error, isConnecting, isConnected, pendingStart]);

  // Reset on unmount
  useEffect(() => {
    return () => {
      if (isConnected) {
        disconnect();
      }
      reset();
    };
  }, []);

  const handleChatSend = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      const text = chatInput.trim();
      if (!text || chatLoading) return;
      setChatInput('');
      setChatMessages((prev) => [...prev, { role: 'user', content: text }]);
      setChatLoading(true);
      setChatError(null);
      try {
        const res = await meetingApi.ask(text);
        setChatMessages((prev) => [
          ...prev,
          { role: 'assistant', content: res.answer, source: res.source, citations: res.citations },
        ]);
      } catch (err) {
        setChatError(err instanceof Error ? err.message : 'Failed to get answer');
      } finally {
        setChatLoading(false);
      }
    },
    [chatInput, chatLoading]
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2 flex items-center gap-3">
            <Radio className="w-8 h-8 text-teal-500" />
            Real-Time Analysis
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Analyze emotions from your webcam in real-time using AI
          </p>
        </div>

        {/* Connection Status */}
        <div className="flex items-center gap-2">
          {isConnected ? (
            <span className="flex items-center gap-2 px-3 py-1.5 bg-green-100 dark:bg-green-900/20 text-green-700 dark:text-green-400 text-sm font-medium rounded-full">
              <Wifi className="w-4 h-4" />
              Connected
            </span>
          ) : isConnecting ? (
            <span className="flex items-center gap-2 px-3 py-1.5 bg-amber-100 dark:bg-amber-900/20 text-amber-700 dark:text-amber-400 text-sm font-medium rounded-full">
              <Wifi className="w-4 h-4 animate-pulse" />
              Connecting...
            </span>
          ) : (
            <span className="flex items-center gap-2 px-3 py-1.5 bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 text-sm font-medium rounded-full">
              <WifiOff className="w-4 h-4" />
              Disconnected
            </span>
          )}
        </div>
      </div>

      {/* Info Banner */}
      <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-xl">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-blue-500 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-blue-700 dark:text-blue-300">
            <p className="font-medium mb-1">How it works:</p>
            <ul className="list-disc list-inside space-y-1 text-blue-600 dark:text-blue-400">
              <li>Enable your camera and click "Start Analysis"</li>
              <li>Video is processed in 4-second chunks locally</li>
              <li>Each chunk is analyzed by our AI for emotion detection</li>
              <li>Results appear in real-time on the right panel</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="text-red-700 dark:text-red-300 font-medium">Error</p>
              <p className="text-sm text-red-600 dark:text-red-400 mt-1">{error}</p>
            </div>
            <button
              onClick={clearError}
              className="text-red-500 hover:text-red-700 text-sm font-medium"
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Left Column - Webcam */}
        <div className="space-y-4">
          <div className="p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Camera Feed
            </h2>
            <WebcamCapture
              onChunkReady={handleChunkReady}
              isAnalyzing={isAnalyzing && isConnected}
              onStartAnalysis={handleStartAnalysis}
              onStopAnalysis={handleStopAnalysis}
              disabled={isConnecting}
            />
          </div>

          {/* Session Info / Connecting */}
          {pendingStart && isConnecting && (
            <div className="p-4 bg-gray-50 dark:bg-gray-800/50 rounded-xl border border-gray-200 dark:border-gray-700">
              <p className="text-sm text-gray-500 dark:text-gray-400">Connecting…</p>
            </div>
          )}
          {sessionId && (
            <div className="p-4 bg-gray-50 dark:bg-gray-800/50 rounded-xl border border-gray-200 dark:border-gray-700">
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Session ID: <code className="text-xs bg-gray-200 dark:bg-gray-700 px-2 py-0.5 rounded">{sessionId}</code>
              </p>
            </div>
          )}
        </div>

        {/* Right Column - Results */}
        <div>
          <RealtimeEmotionDisplay
            currentEmotion={currentEmotion}
            predictions={predictions}
            summary={summary}
            isAnalyzing={isAnalyzing && isConnected}
            chunksSent={chunksSent}
          />
        </div>
      </div>

      {/* Chat panel: ask questions (RAG-first, then OpenAI) */}
      <div className="p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
          <MessageSquare className="w-5 h-5 text-teal-500" />
          Meeting chat
        </h2>
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
          Ask questions during your meeting. Answers come from your Documents first; if not found, we use general AI.
        </p>
        <div className="flex flex-col gap-4">
          <div className="min-h-[200px] max-h-[320px] overflow-y-auto space-y-3 p-3 bg-gray-50 dark:bg-gray-700/30 rounded-lg border border-gray-200 dark:border-gray-600">
            {chatMessages.length === 0 && (
              <p className="text-sm text-gray-500 dark:text-gray-400 text-center py-4">
                Type a question and press Send to get answers from your documents or AI.
              </p>
            )}
            {chatMessages.map((msg, i) => (
              <div
                key={i}
                className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}
              >
                <div
                  className={`max-w-[85%] rounded-lg px-4 py-2 ${
                    msg.role === 'user'
                      ? 'bg-teal-600 text-white'
                      : 'bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 text-gray-900 dark:text-white'
                  }`}
                >
                  <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                  {msg.role === 'assistant' && msg.source && (
                    <>
                      <span
                        className={`inline-flex items-center gap-1 mt-2 text-xs ${
                          msg.source === 'rag'
                            ? 'text-teal-600 dark:text-teal-400'
                            : 'text-gray-500 dark:text-gray-400'
                        }`}
                      >
                        {msg.source === 'rag' ? (
                          <>
                            <FileText className="w-3 h-3" />
                            From your documents
                          </>
                        ) : (
                          'General answer'
                        )}
                      </span>
                      {msg.source === 'rag' && msg.citations && msg.citations.length > 0 && (
                        <details className="mt-2 text-xs">
                          <summary className="cursor-pointer text-teal-600 dark:text-teal-400 hover:underline">
                            Sources ({msg.citations.length})
                          </summary>
                          <ul className="mt-1 space-y-1 pl-2 border-l-2 border-teal-200 dark:border-teal-700">
                            {msg.citations.map((c, j) => (
                              <li key={j} className="text-gray-600 dark:text-gray-300">
                                {c.document_id && <span className="font-medium">{c.document_id}</span>}
                                {c.page_number != null && ` · p.${c.page_number}`}
                                {c.content_snippet && (
                                  <p className="mt-0.5 text-gray-500 dark:text-gray-400 line-clamp-2">
                                    {c.content_snippet}
                                  </p>
                                )}
                              </li>
                            ))}
                          </ul>
                        </details>
                      )}
                    </>
                  )}
                </div>
              </div>
            ))}
            {chatLoading && (
              <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
                <span className="inline-block w-2 h-2 rounded-full bg-teal-500 animate-pulse" />
                Getting answer...
              </div>
            )}
          </div>
          {chatError && (
            <p className="text-sm text-red-600 dark:text-red-400">{chatError}</p>
          )}
          <form onSubmit={handleChatSend} className="flex gap-2">
            <input
              type="text"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              placeholder="Ask a question..."
              disabled={chatLoading}
              className="flex-1 px-4 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white placeholder-gray-500 focus:ring-2 focus:ring-teal-500 disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={!chatInput.trim() || chatLoading}
              className="flex items-center gap-2 px-4 py-2 bg-teal-600 hover:bg-teal-700 disabled:opacity-50 text-white rounded-lg font-medium transition-colors"
            >
              <Send className="w-4 h-4" />
              Send
            </button>
          </form>
        </div>
      </div>

      {/* Tips Section */}
      <div className="p-6 bg-gray-50 dark:bg-gray-800/50 rounded-xl border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
          Tips for Best Results
        </h3>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
          <div className="p-3 bg-white dark:bg-gray-800 rounded-lg">
            <p className="font-medium text-gray-900 dark:text-white mb-1">Good Lighting</p>
            <p className="text-gray-500 dark:text-gray-400">
              Ensure your face is well-lit from the front
            </p>
          </div>
          <div className="p-3 bg-white dark:bg-gray-800 rounded-lg">
            <p className="font-medium text-gray-900 dark:text-white mb-1">Face the Camera</p>
            <p className="text-gray-500 dark:text-gray-400">
              Keep your face centered and visible
            </p>
          </div>
          <div className="p-3 bg-white dark:bg-gray-800 rounded-lg">
            <p className="font-medium text-gray-900 dark:text-white mb-1">Stable Position</p>
            <p className="text-gray-500 dark:text-gray-400">
              Minimize movement for better detection
            </p>
          </div>
          <div className="p-3 bg-white dark:bg-gray-800 rounded-lg">
            <p className="font-medium text-gray-900 dark:text-white mb-1">Clear Background</p>
            <p className="text-gray-500 dark:text-gray-400">
              A simple background helps focus on your face
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
