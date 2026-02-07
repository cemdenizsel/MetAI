'use client';

import { useState, useRef, useCallback, useEffect } from 'react';
import {
  Camera,
  CameraOff,
  Play,
  Square,
  AlertCircle,
  Loader2,
  RefreshCw,
} from 'lucide-react';

interface WebcamCaptureProps {
  onChunkReady: (chunk: Blob, timestamp: number, index: number) => void;
  isAnalyzing: boolean;
  onStartAnalysis: () => void;
  onStopAnalysis: () => void;
  disabled?: boolean;
}

const CHUNK_DURATION_MS = 4000; // 4 seconds per chunk

export function WebcamCapture({
  onChunkReady,
  isAnalyzing,
  onStartAnalysis,
  onStopAnalysis,
  disabled = false,
}: WebcamCaptureProps) {
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [hasAudio, setHasAudio] = useState<boolean | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isCameraOn, setIsCameraOn] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunkIndexRef = useRef(0);
  const startTimeRef = useRef(0);
  const recordingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  const startCamera = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      // Request both video and audio so real-time analysis can use your voice
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: 'user',
          },
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
          },
        });
      } catch (audioErr) {
        // If microphone is denied, fall back to video-only (face analysis only)
        if (audioErr instanceof DOMException && (audioErr.name === 'NotAllowedError' || audioErr.name === 'NotFoundError')) {
          stream = await navigator.mediaDevices.getUserMedia({
            video: {
              width: { ideal: 640 },
              height: { ideal: 480 },
              facingMode: 'user',
            },
            audio: false,
          });
        } else {
          throw audioErr;
        }
      }

      streamRef.current = stream;
      setHasAudio(stream.getAudioTracks().length > 0);

      // Attach stream to video element immediately since it's always rendered
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setHasPermission(true);
      setIsCameraOn(true);
    } catch (err) {
      console.error('Camera access error:', err);
      setHasPermission(false);
      
      if (err instanceof DOMException) {
        if (err.name === 'NotAllowedError') {
          setError('Camera access denied. Please allow camera access in your browser settings.');
        } else if (err.name === 'NotFoundError') {
          setError('No camera found. Please connect a camera and try again.');
        } else if (err.name === 'NotReadableError') {
          setError('Camera is in use by another application.');
        } else {
          setError(`Camera error: ${err.message}`);
        }
      } else {
        setError('Failed to access camera. Please try again.');
      }
    } finally {
      setIsLoading(false);
    }
  }, []);

  const stopCamera = useCallback(() => {
    // Stop recording if active
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    mediaRecorderRef.current = null;

    // Clear recording interval
    if (recordingIntervalRef.current) {
      clearInterval(recordingIntervalRef.current);
      recordingIntervalRef.current = null;
    }

    // Stop all tracks
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    // Clear video
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setIsCameraOn(false);
    setHasAudio(null);
  }, []);

  const startRecording = useCallback(() => {
    if (!streamRef.current) {
      setError('Camera not available');
      return;
    }

    chunkIndexRef.current = 0;
    startTimeRef.current = Date.now();

    // Get supported MIME type
    const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
      ? 'video/webm;codecs=vp9'
      : MediaRecorder.isTypeSupported('video/webm;codecs=vp8')
      ? 'video/webm;codecs=vp8'
      : MediaRecorder.isTypeSupported('video/webm')
      ? 'video/webm'
      : 'video/mp4';

    const recordChunk = () => {
      if (!streamRef.current || !isCameraOn) return;

      try {
        const mediaRecorder = new MediaRecorder(streamRef.current, {
          mimeType,
          videoBitsPerSecond: 1000000, // 1 Mbps
        });

        const chunks: Blob[] = [];

        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            chunks.push(event.data);
          }
        };

        mediaRecorder.onstop = () => {
          if (chunks.length > 0) {
            const blob = new Blob(chunks, { type: mimeType });
            const timestamp = (Date.now() - startTimeRef.current) / 1000;
            onChunkReady(blob, timestamp, chunkIndexRef.current);
            chunkIndexRef.current++;
          }
        };

        mediaRecorder.start();
        mediaRecorderRef.current = mediaRecorder;

        // Stop recording after chunk duration
        setTimeout(() => {
          if (mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
          }
        }, CHUNK_DURATION_MS);
      } catch (err) {
        console.error('Recording error:', err);
        setError('Failed to start recording');
      }
    };

    // Start first chunk immediately
    recordChunk();

    // Continue recording chunks
    recordingIntervalRef.current = setInterval(recordChunk, CHUNK_DURATION_MS);
  }, [isCameraOn, onChunkReady]);

  const stopRecording = useCallback(() => {
    // Stop current recording
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }

    // Clear interval
    if (recordingIntervalRef.current) {
      clearInterval(recordingIntervalRef.current);
      recordingIntervalRef.current = null;
    }

    onStopAnalysis();
  }, [onStopAnalysis]);

  // Start recording when parent sets isAnalyzing (after WebSocket is connected)
  useEffect(() => {
    if (!isAnalyzing || !isCameraOn || !streamRef.current) return;
    startRecording();
    return () => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
      mediaRecorderRef.current = null;
      if (recordingIntervalRef.current) {
        clearInterval(recordingIntervalRef.current);
        recordingIntervalRef.current = null;
      }
    };
  }, [isAnalyzing, isCameraOn, startRecording]);

  const handleStartStop = useCallback(() => {
    if (isAnalyzing) {
      stopRecording();
    } else {
      onStartAnalysis();
    }
  }, [isAnalyzing, onStartAnalysis, stopRecording]);

  // Initial permission check
  useEffect(() => {
    if (typeof navigator !== 'undefined' && navigator.permissions) {
      navigator.permissions
        .query({ name: 'camera' as PermissionName })
        .then((result) => {
          if (result.state === 'granted') {
            setHasPermission(true);
          } else if (result.state === 'denied') {
            setHasPermission(false);
          }
        })
        .catch(() => {
          // Permissions API not supported
        });
    }
  }, []);

  return (
    <div className="space-y-4">
      {/* Video Preview */}
      <div className="relative aspect-video bg-gray-900 rounded-xl overflow-hidden">
        {/* Always render video element, just hide it when camera is off */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`w-full h-full object-cover ${isCameraOn ? 'block' : 'hidden'}`}
        />
        
        {/* Overlay when camera is off */}
        {!isCameraOn && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-400">
            {isLoading ? (
              <>
                <Loader2 className="w-12 h-12 animate-spin mb-4" />
                <p>Starting camera...</p>
              </>
            ) : error ? (
              <>
                <AlertCircle className="w-12 h-12 text-red-500 mb-4" />
                <p className="text-red-400 text-center px-4">{error}</p>
                <button
                  onClick={() => {
                    setError(null);
                    startCamera();
                  }}
                  className="mt-4 flex items-center gap-2 px-4 py-2 bg-teal-600 hover:bg-teal-700 text-white rounded-lg transition-colors"
                >
                  <RefreshCw className="w-4 h-4" />
                  Try Again
                </button>
              </>
            ) : (
              <>
                <CameraOff className="w-12 h-12 mb-4" />
                <p>Camera is off</p>
                <button
                  onClick={startCamera}
                  className="mt-4 flex items-center gap-2 px-4 py-2 bg-teal-600 hover:bg-teal-700 text-white rounded-lg transition-colors"
                >
                  <Camera className="w-4 h-4" />
                  Start Camera
                </button>
              </>
            )}
          </div>
        )}

        {/* Recording indicator */}
        {isAnalyzing && (
          <div className="absolute top-4 left-4 flex items-center gap-2 px-3 py-1.5 bg-red-600 text-white text-sm font-medium rounded-full">
            <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
            Recording
          </div>
        )}

        {/* Camera toggle when on */}
        {isCameraOn && !isAnalyzing && (
          <button
            onClick={stopCamera}
            className="absolute top-4 right-4 p-2 bg-gray-800/80 hover:bg-gray-700/80 text-white rounded-lg transition-colors"
            title="Turn off camera"
          >
            <CameraOff className="w-5 h-5" />
          </button>
        )}
      </div>

      {/* Controls */}
      <div className="flex items-center justify-center gap-4">
        {!isCameraOn ? (
          <button
            onClick={startCamera}
            disabled={isLoading || disabled}
            className="flex items-center gap-2 px-6 py-3 bg-teal-600 hover:bg-teal-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors"
          >
            {isLoading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Camera className="w-5 h-5" />
            )}
            {isLoading ? 'Starting...' : 'Enable Camera'}
          </button>
        ) : (
          <>
            <button
              onClick={handleStartStop}
              disabled={disabled}
              className={`flex items-center gap-2 px-6 py-3 font-medium rounded-lg transition-colors ${
                isAnalyzing
                  ? 'bg-red-600 hover:bg-red-700 text-white'
                  : 'bg-teal-600 hover:bg-teal-700 text-white'
              } disabled:bg-gray-600 disabled:cursor-not-allowed`}
            >
              {isAnalyzing ? (
                <>
                  <Square className="w-5 h-5" />
                  Stop Analysis
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  Start Analysis
                </>
              )}
            </button>

            {!isAnalyzing && (
              <button
                onClick={stopCamera}
                className="flex items-center gap-2 px-4 py-3 bg-gray-600 hover:bg-gray-700 text-white font-medium rounded-lg transition-colors"
              >
                <CameraOff className="w-5 h-5" />
                Turn Off
              </button>
            )}
          </>
        )}
      </div>

      {/* Permission hint */}
      {hasPermission === false && !error && (
        <p className="text-center text-sm text-amber-500">
          Camera access was denied. Please enable it in your browser settings.
        </p>
      )}

      {/* Mic hint when camera is on but mic was blocked */}
      {hasPermission === true && hasAudio === false && isCameraOn && (
        <p className="text-center text-sm text-amber-500">
          Microphone not in use (blocked or denied). To analyze your voice, allow the microphone for this site in your browser settings, then turn off the camera and enable it again.
        </p>
      )}
    </div>
  );
}
