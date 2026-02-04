'use client';

import { useCallback, useEffect, useState } from 'react';
import { Radio, AlertCircle, Info, Wifi, WifiOff } from 'lucide-react';
import { WebcamCapture } from '@/components/emotion/WebcamCapture';
import { RealtimeEmotionDisplay } from '@/components/emotion/RealtimeEmotionDisplay';
import { useRealtimeAnalysis } from '@/lib/hooks/useRealtimeAnalysis';
import { useAuthStore } from '@/lib/stores/authStore';

export default function RealtimePage() {
  const { token } = useAuthStore();
  const [isAnalyzing, setIsAnalyzing] = useState(false);

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

  // Start analysis
  const handleStartAnalysis = useCallback(() => {
    // Try to get token from store first, then fallback to localStorage
    const authToken = token || (typeof window !== 'undefined' ? localStorage.getItem('auth_token') : null);
    
    if (!authToken) {
      console.error('No auth token available');
      alert('Not authenticated. Please log out and log in again.');
      return;
    }

    console.log('Starting analysis with token:', authToken.substring(0, 20) + '...');
    setIsAnalyzing(true);
    connect(authToken);
  }, [token, connect]);

  // Stop analysis
  const handleStopAnalysis = useCallback(() => {
    setIsAnalyzing(false);
    complete();
    
    // Give time for the complete message to process, then disconnect
    setTimeout(() => {
      disconnect();
    }, 1000);
  }, [complete, disconnect]);

  // Reset on unmount
  useEffect(() => {
    return () => {
      if (isConnected) {
        disconnect();
      }
      reset();
    };
  }, []);

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

          {/* Session Info */}
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
