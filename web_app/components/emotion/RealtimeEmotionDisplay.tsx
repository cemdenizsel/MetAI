'use client';

import { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts';
import {
  Smile,
  Frown,
  Meh,
  Angry,
  AlertTriangle,
  Zap,
  HelpCircle,
  Clock,
  Activity,
  BarChart3,
} from 'lucide-react';
import type { EmotionPrediction, SessionSummary } from '@/lib/hooks/useRealtimeAnalysis';

interface RealtimeEmotionDisplayProps {
  currentEmotion: EmotionPrediction | null;
  predictions: EmotionPrediction[];
  summary: SessionSummary | null;
  isAnalyzing: boolean;
  chunksSent: number;
}

const EMOTION_COLORS: Record<string, string> = {
  angry: '#EF4444',
  disgust: '#22C55E',
  fear: '#A855F7',
  happy: '#EAB308',
  sad: '#3B82F6',
  surprise: '#F97316',
  neutral: '#6B7280',
};

const EMOTION_LABELS: Record<string, string> = {
  angry: 'Angry',
  disgust: 'Disgust',
  fear: 'Fear',
  happy: 'Happy',
  sad: 'Sad',
  surprise: 'Surprise',
  neutral: 'Neutral',
};

const EMOTION_ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  angry: Angry,
  disgust: Frown,
  fear: AlertTriangle,
  happy: Smile,
  sad: Frown,
  surprise: Zap,
  neutral: Meh,
};

export function RealtimeEmotionDisplay({
  currentEmotion,
  predictions,
  summary,
  isAnalyzing,
  chunksSent,
}: RealtimeEmotionDisplayProps) {
  // Calculate session duration
  const sessionDuration = useMemo(() => {
    if (predictions.length === 0) return 0;
    const lastPrediction = predictions[predictions.length - 1];
    return Math.round(lastPrediction.timestamp);
  }, [predictions]);

  // Format duration as MM:SS
  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Prepare chart data
  const chartData = useMemo(() => {
    return predictions.slice(-30).map((pred, index) => ({
      index,
      timestamp: Math.round(pred.timestamp),
      confidence: Math.round(pred.confidence * 100),
      emotion: pred.emotion,
      color: EMOTION_COLORS[pred.emotion] || EMOTION_COLORS.neutral,
    }));
  }, [predictions]);

  // Calculate emotion distribution
  const emotionDistribution = useMemo(() => {
    if (predictions.length === 0) return {};
    
    const counts: Record<string, number> = {};
    predictions.forEach((pred) => {
      counts[pred.emotion] = (counts[pred.emotion] || 0) + 1;
    });
    
    const total = predictions.length;
    const distribution: Record<string, number> = {};
    Object.entries(counts).forEach(([emotion, count]) => {
      distribution[emotion] = Math.round((count / total) * 100);
    });
    
    return distribution;
  }, [predictions]);

  // Get dominant emotion
  const dominantEmotion = useMemo(() => {
    if (predictions.length === 0) return null;
    
    const counts: Record<string, number> = {};
    predictions.forEach((pred) => {
      counts[pred.emotion] = (counts[pred.emotion] || 0) + 1;
    });
    
    let maxCount = 0;
    let dominant = 'neutral';
    Object.entries(counts).forEach(([emotion, count]) => {
      if (count > maxCount) {
        maxCount = count;
        dominant = emotion;
      }
    });
    
    return dominant;
  }, [predictions]);

  const EmotionIcon = currentEmotion
    ? EMOTION_ICONS[currentEmotion.emotion] || HelpCircle
    : HelpCircle;

  const emotionColor = currentEmotion
    ? EMOTION_COLORS[currentEmotion.emotion] || EMOTION_COLORS.neutral
    : EMOTION_COLORS.neutral;

  return (
    <div className="space-y-6">
      {/* Current Emotion Display */}
      <div className="p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5 text-teal-500" />
          Current Emotion
        </h3>

        {currentEmotion ? (
          <div className="flex items-center gap-6">
            {/* Emotion Icon */}
            <div
              className="w-20 h-20 rounded-full flex items-center justify-center"
              style={{ backgroundColor: `${emotionColor}20` }}
            >
              <EmotionIcon
                className="w-10 h-10"
                style={{ color: emotionColor }}
              />
            </div>

            {/* Emotion Details */}
            <div className="flex-1">
              <p
                className="text-3xl font-bold capitalize"
                style={{ color: emotionColor }}
              >
                {EMOTION_LABELS[currentEmotion.emotion] || currentEmotion.emotion}
              </p>
              
              {/* Confidence Bar */}
              <div className="mt-2">
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-500 dark:text-gray-400">Confidence</span>
                  <span className="font-medium text-gray-900 dark:text-white">
                    {Math.round(currentEmotion.confidence * 100)}%
                  </span>
                </div>
                <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-300"
                    style={{
                      width: `${currentEmotion.confidence * 100}%`,
                      backgroundColor: emotionColor,
                    }}
                  />
                </div>
              </div>

              {/* All Confidences */}
              {currentEmotion.confidences && (
                <div className="mt-3 flex flex-wrap gap-2">
                  {Object.entries(currentEmotion.confidences)
                    .sort(([, a], [, b]) => b - a)
                    .slice(0, 4)
                    .map(([emotion, conf]) => (
                      <span
                        key={emotion}
                        className="px-2 py-1 text-xs rounded-full"
                        style={{
                          backgroundColor: `${EMOTION_COLORS[emotion] || '#6B7280'}20`,
                          color: EMOTION_COLORS[emotion] || '#6B7280',
                        }}
                      >
                        {EMOTION_LABELS[emotion] || emotion}: {Math.round(conf * 100)}%
                      </span>
                    ))}
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="text-center py-8 text-gray-400">
            {isAnalyzing ? (
              <p>Waiting for first analysis result...</p>
            ) : (
              <p>Start analysis to see real-time emotions</p>
            )}
          </div>
        )}
      </div>

      {/* Emotion History Chart */}
      {predictions.length > 0 && (
        <div className="p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-teal-500" />
            Confidence Over Time
          </h3>

          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="confidenceGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#14B8A6" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#14B8A6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis
                  dataKey="timestamp"
                  tick={{ fill: '#6B7280', fontSize: 11 }}
                  tickFormatter={(val) => `${val}s`}
                />
                <YAxis
                  domain={[0, 100]}
                  tick={{ fill: '#6B7280', fontSize: 11 }}
                  tickFormatter={(val) => `${val}%`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    border: 'none',
                    borderRadius: '8px',
                    color: '#fff',
                  }}
                  formatter={(value: number, name: string, props: { payload: { emotion: string } }) => [
                    `${value}%`,
                    `${EMOTION_LABELS[props.payload.emotion] || props.payload.emotion}`,
                  ]}
                  labelFormatter={(label) => `Time: ${label}s`}
                />
                <Area
                  type="monotone"
                  dataKey="confidence"
                  stroke="#14B8A6"
                  strokeWidth={2}
                  fill="url(#confidenceGradient)"
                  dot={(props) => {
                    const { cx, cy, payload } = props;
                    return (
                      <circle
                        key={`dot-${payload.index}`}
                        cx={cx}
                        cy={cy}
                        r={4}
                        fill={payload.color}
                        stroke="#fff"
                        strokeWidth={1}
                      />
                    );
                  }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Session Statistics */}
      <div className="p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
          <Clock className="w-5 h-5 text-teal-500" />
          Session Statistics
        </h3>

        <div className="grid grid-cols-2 gap-4">
          <div className="p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
            <p className="text-sm text-gray-500 dark:text-gray-400">Duration</p>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              {formatDuration(sessionDuration)}
            </p>
          </div>

          <div className="p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
            <p className="text-sm text-gray-500 dark:text-gray-400">Chunks Analyzed</p>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              {chunksSent}
            </p>
          </div>

          {dominantEmotion && (
            <div className="p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg col-span-2">
              <p className="text-sm text-gray-500 dark:text-gray-400">Dominant Emotion</p>
              <p
                className="text-2xl font-bold capitalize"
                style={{ color: EMOTION_COLORS[dominantEmotion] }}
              >
                {EMOTION_LABELS[dominantEmotion] || dominantEmotion}
              </p>
            </div>
          )}
        </div>

        {/* Emotion Distribution */}
        {Object.keys(emotionDistribution).length > 0 && (
          <div className="mt-4">
            <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">Distribution</p>
            <div className="flex flex-wrap gap-2">
              {Object.entries(emotionDistribution)
                .sort(([, a], [, b]) => b - a)
                .map(([emotion, percentage]) => (
                  <span
                    key={emotion}
                    className="px-3 py-1 text-sm rounded-full font-medium"
                    style={{
                      backgroundColor: `${EMOTION_COLORS[emotion] || '#6B7280'}20`,
                      color: EMOTION_COLORS[emotion] || '#6B7280',
                    }}
                  >
                    {EMOTION_LABELS[emotion] || emotion}: {percentage}%
                  </span>
                ))}
            </div>
          </div>
        )}
      </div>

      {/* Session Summary (when complete) */}
      {summary && (
        <div className="p-6 bg-teal-50 dark:bg-teal-900/20 rounded-xl border border-teal-200 dark:border-teal-800">
          <h3 className="text-lg font-semibold text-teal-900 dark:text-teal-100 mb-4">
            Session Complete
          </h3>
          
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <p className="text-teal-600 dark:text-teal-400">Total Duration</p>
              <p className="font-semibold text-teal-900 dark:text-teal-100">
                {formatDuration(summary.duration)}
              </p>
            </div>
            <div>
              <p className="text-teal-600 dark:text-teal-400">Chunks Processed</p>
              <p className="font-semibold text-teal-900 dark:text-teal-100">
                {summary.total_chunks}
              </p>
            </div>
            <div>
              <p className="text-teal-600 dark:text-teal-400">Dominant Emotion</p>
              <p className="font-semibold capitalize text-teal-900 dark:text-teal-100">
                {EMOTION_LABELS[summary.dominant_emotion] || summary.dominant_emotion}
              </p>
            </div>
            <div>
              <p className="text-teal-600 dark:text-teal-400">Avg. Confidence</p>
              <p className="font-semibold text-teal-900 dark:text-teal-100">
                {Math.round(summary.average_confidence * 100)}%
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
