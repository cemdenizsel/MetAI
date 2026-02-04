'use client';

import { useState, useCallback, useRef, useEffect } from 'react';

// Types for real-time analysis
export interface EmotionPrediction {
  chunk_index: number;
  timestamp: number;
  emotion: string;
  confidence: number;
  confidences: Record<string, number>;
  processing_time?: number;
  created_at?: string;
}

export interface SessionSummary {
  session_id: string;
  user_id: string;
  total_chunks: number;
  duration: number;
  dominant_emotion: string;
  average_confidence: number;
  emotion_distribution: Record<string, number>;
  predictions: EmotionPrediction[];
}

export interface RealtimeAnalysisState {
  isConnected: boolean;
  isConnecting: boolean;
  sessionId: string | null;
  currentEmotion: EmotionPrediction | null;
  predictions: EmotionPrediction[];
  summary: SessionSummary | null;
  error: string | null;
  chunksSent: number;
}

type MessageHandler = (data: unknown) => void;

const getWebSocketUrl = (token: string): string => {
  // Use environment variable or default to localhost
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8084';
  const wsProtocol = apiUrl.startsWith('https') ? 'wss' : 'ws';
  const wsHost = apiUrl.replace(/^https?:\/\//, '');
  return `${wsProtocol}://${wsHost}/ws/analyze?token=${encodeURIComponent(token)}`;
};

export function useRealtimeAnalysis() {
  const [state, setState] = useState<RealtimeAnalysisState>({
    isConnected: false,
    isConnecting: false,
    sessionId: null,
    currentEmotion: null,
    predictions: [],
    summary: null,
    error: null,
    chunksSent: 0,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const handleMessage: MessageHandler = useCallback((data: unknown) => {
    const message = data as Record<string, unknown>;
    const messageType = message.type as string;

    switch (messageType) {
      case 'connected':
        setState((prev) => ({
          ...prev,
          isConnected: true,
          isConnecting: false,
          sessionId: message.session_id as string,
          error: null,
        }));
        break;

      case 'result':
        const resultData = message.data_model as EmotionPrediction;
        setState((prev) => ({
          ...prev,
          currentEmotion: resultData,
          predictions: [...prev.predictions, resultData].slice(-60), // Keep last 60
        }));
        break;

      case 'progress':
        // Handle progress updates if needed
        break;

      case 'complete':
        const summaryData = message.summary as SessionSummary;
        setState((prev) => ({
          ...prev,
          summary: summaryData,
        }));
        break;

      case 'error':
        setState((prev) => ({
          ...prev,
          error: message.error as string,
        }));
        break;

      case 'heartbeat':
      case 'pong':
        // Heartbeat received, connection is alive
        break;

      case 'disconnected':
        setState((prev) => ({
          ...prev,
          isConnected: false,
          sessionId: null,
        }));
        break;

      default:
        console.log('Unknown message type:', messageType);
    }
  }, []);

  const connect = useCallback((token: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.log('WebSocket already connected');
      return;
    }

    setState((prev) => ({
      ...prev,
      isConnecting: true,
      error: null,
      predictions: [],
      summary: null,
      currentEmotion: null,
      chunksSent: 0,
    }));

    try {
      const wsUrl = getWebSocketUrl(token);
      console.log('Connecting to WebSocket:', wsUrl.replace(/token=.*/, 'token=***'));
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log('WebSocket connected successfully');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('WebSocket message received:', data.type);
          handleMessage(data);
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      ws.onerror = (event) => {
        console.error('WebSocket error event:', event);
        setState((prev) => ({
          ...prev,
          error: 'WebSocket connection error. Make sure the backend server is running.',
          isConnecting: false,
        }));
      };

      ws.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        setState((prev) => ({
          ...prev,
          isConnected: false,
          isConnecting: false,
          sessionId: null,
        }));
        wsRef.current = null;
      };

      wsRef.current = ws;
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      setState((prev) => ({
        ...prev,
        error: 'Failed to connect to analysis server',
        isConnecting: false,
      }));
    }
  }, [handleMessage]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      // Send complete message before closing
      if (wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'complete' }));
      }
      wsRef.current.close();
      wsRef.current = null;
    }

    setState((prev) => ({
      ...prev,
      isConnected: false,
      isConnecting: false,
      sessionId: null,
    }));
  }, []);

  const sendChunk = useCallback((
    chunkData: Blob | ArrayBuffer,
    timestamp: number,
    chunkIndex: number
  ) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.error('WebSocket not connected');
      return false;
    }

    // Convert to base64 if Blob
    if (chunkData instanceof Blob) {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = (reader.result as string).split(',')[1];
        const message = {
          type: 'chunk',
          data_model: base64,
          timestamp,
          chunk_index: chunkIndex,
        };
        wsRef.current?.send(JSON.stringify(message));
        setState((prev) => ({
          ...prev,
          chunksSent: prev.chunksSent + 1,
        }));
      };
      reader.readAsDataURL(chunkData);
    } else {
      // ArrayBuffer - convert to base64
      const bytes = new Uint8Array(chunkData);
      let binary = '';
      bytes.forEach((b) => (binary += String.fromCharCode(b)));
      const base64 = btoa(binary);
      
      const message = {
        type: 'chunk',
        data_model: base64,
        timestamp,
        chunk_index: chunkIndex,
      };
      wsRef.current.send(JSON.stringify(message));
      setState((prev) => ({
        ...prev,
        chunksSent: prev.chunksSent + 1,
      }));
    }

    return true;
  }, []);

  const complete = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'complete' }));
    }
  }, []);

  const sendPing = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'ping' }));
    }
  }, []);

  const clearError = useCallback(() => {
    setState((prev) => ({ ...prev, error: null }));
  }, []);

  const reset = useCallback(() => {
    setState({
      isConnected: false,
      isConnecting: false,
      sessionId: null,
      currentEmotion: null,
      predictions: [],
      summary: null,
      error: null,
      chunksSent: 0,
    });
  }, []);

  return {
    ...state,
    connect,
    disconnect,
    sendChunk,
    complete,
    sendPing,
    clearError,
    reset,
  };
}
