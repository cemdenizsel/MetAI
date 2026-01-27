'use client';

import {
  Brain,
  Heart,
  MessageSquare,
  Lightbulb,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  Clock,
  Video,
  BarChart3,
  PieChart,
  Activity,
  FileText,
  Layers,
  Users,
} from 'lucide-react';
import type { MultiModelResponse, ModelResults } from '@/types/emotion';
import { EmotionChart } from './EmotionChart';
import { MentalHealthCard } from './MentalHealthCard';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
  PieChart as RechartsPieChart,
  Pie,
} from 'recharts';

interface AnalysisResultsProps {
  result: MultiModelResponse;
}

const emotionEmojis: Record<string, string> = {
  angry: '😠',
  disgust: '🤢',
  fear: '😨',
  happy: '😊',
  sad: '😢',
  surprise: '😲',
  neutral: '😐',
};

const emotionColors: Record<string, string> = {
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

function ModelResultCard({ modelResult }: { modelResult: ModelResults }) {
  const confidencePercent = Math.round(modelResult.overall_prediction.confidence * 100);
  const emotion = modelResult.overall_prediction.predicted_emotion;

  return (
    <div className="space-y-6">
      {/* Primary Result */}
      <div className="p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <div className="text-5xl">{emotionEmojis[emotion] || '😐'}</div>
          <div>
            <h3 className="text-2xl font-bold text-gray-900 dark:text-white capitalize">
                {emotion}
            </h3>
            <p className="text-gray-600 dark:text-gray-400">
              Detected with {confidencePercent}% confidence
            </p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-sm text-gray-500 dark:text-gray-400">Model</div>
            <div className="text-lg font-semibold text-gray-900 dark:text-white">
              {modelResult.model_name}
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              {modelResult.fusion_strategy}
            </div>
          </div>
        </div>

        {/* Confidence Bar */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Confidence
            </span>
            <span className="text-sm font-bold text-gray-900 dark:text-white">
              {confidencePercent}%
            </span>
          </div>
          <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full transition-all duration-500"
              style={{
                width: `${confidencePercent}%`,
                backgroundColor: emotionColors[emotion] || '#6B7280',
              }}
            />
          </div>
        </div>

        {/* Processing Time */}
          <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          <Clock className="w-4 h-4" />
          <span>Processed in {modelResult.processing_time.toFixed(2)}s</span>
          </div>
      </div>

      {/* Emotion Distribution Chart */}
      {modelResult.overall_prediction.all_confidences && (
        <EmotionChart distribution={modelResult.overall_prediction.all_confidences} />
      )}

      {/* Temporal Predictions */}
      {modelResult.temporal_predictions && modelResult.temporal_predictions.length > 0 && (
        <div className="p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5 text-purple-500" />
            Emotion Over Time
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={modelResult.temporal_predictions}>
                <XAxis
                  dataKey="timestamp"
                  label={{ value: 'Time (s)', position: 'insideBottom', offset: -5 }}
                  tick={{ fill: '#6B7280', fontSize: 12 }}
                />
                <YAxis
                  tick={{ fill: '#6B7280', fontSize: 12 }}
                  label={{ value: 'Confidence', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    border: 'none',
                    borderRadius: '8px',
                    color: '#fff',
                  }}
                  formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, 'Confidence']}
                />
                <Legend />
                {Object.keys(emotionColors).map((emotion) => (
                  <Line
                    key={emotion}
                    type="monotone"
                    dataKey={`all_confidences.${emotion}`}
                    stroke={emotionColors[emotion]}
                    strokeWidth={2}
                    dot={false}
                    name={EMOTION_LABELS[emotion]}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Mental Health Analysis */}
      {modelResult.mental_health_analysis && (
        <div className="p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <Heart className="w-5 h-5 text-red-500" />
            Mental Health Analysis
          </h3>
          <div className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {modelResult.mental_health_analysis.mental_health_score.toFixed(0)}
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">Health Score</div>
              </div>
              <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <div className="text-2xl font-bold text-gray-900 dark:text-white capitalize">
                  {modelResult.mental_health_analysis.dominant_emotion}
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">Dominant</div>
              </div>
              <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                  {modelResult.mental_health_analysis.positive_percentage.toFixed(1)}%
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">Positive</div>
              </div>
              <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                  {modelResult.mental_health_analysis.negative_percentage.toFixed(1)}%
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">Negative</div>
              </div>
            </div>
            <div className="p-4 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800">
              <div className="flex items-start gap-2">
                <AlertCircle className="w-5 h-5 text-amber-600 dark:text-amber-400 mt-0.5 flex-shrink-0" />
                <div>
                  <div className="font-medium text-amber-900 dark:text-amber-200 mb-1">
                    Status: {modelResult.mental_health_analysis.status}
                  </div>
                  {modelResult.mental_health_analysis.recommendation && (
                    <div className="text-sm text-amber-800 dark:text-amber-300">
                      {modelResult.mental_health_analysis.recommendation}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Video Metadata */}
      <div className="p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
          <Video className="w-5 h-5 text-blue-500" />
          Video Information
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <div>
            <div className="text-sm text-gray-500 dark:text-gray-400">Filename</div>
            <div className="text-sm font-medium text-gray-900 dark:text-white truncate">
              {modelResult.video_metadata.filename}
            </div>
          </div>
          <div>
            <div className="text-sm text-gray-500 dark:text-gray-400">Duration</div>
            <div className="text-sm font-medium text-gray-900 dark:text-white">
              {modelResult.video_metadata.duration.toFixed(2)}s
            </div>
          </div>
          <div>
            <div className="text-sm text-gray-500 dark:text-gray-400">Resolution</div>
            <div className="text-sm font-medium text-gray-900 dark:text-white">
              {modelResult.video_metadata.width} × {modelResult.video_metadata.height}
            </div>
          </div>
          <div>
            <div className="text-sm text-gray-500 dark:text-gray-400">FPS</div>
            <div className="text-sm font-medium text-gray-900 dark:text-white">
              {modelResult.video_metadata.fps}
            </div>
          </div>
          <div>
            <div className="text-sm text-gray-500 dark:text-gray-400">Frames</div>
            <div className="text-sm font-medium text-gray-900 dark:text-white">
              {modelResult.video_metadata.frame_count}
            </div>
          </div>
        </div>
      </div>

      {/* Features */}
      <div className="p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
          <Layers className="w-5 h-5 text-teal-500" />
          Extracted Features
        </h3>
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {modelResult.features.audio_features}
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-400">Audio Features</div>
          </div>
          <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {modelResult.features.visual_features}
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-400">Visual Features</div>
          </div>
          <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {modelResult.features.text_features}
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-400">Text Features</div>
          </div>
        </div>
      </div>

      {/* Modality Weights */}
      {modelResult.modality_weights && (
        <div className="p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <PieChart className="w-5 h-5 text-purple-500" />
            Modality Weights
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <RechartsPieChart>
                <Pie
                  data={[
                    {
                      name: 'Audio',
                      value: Math.round(modelResult.modality_weights.audio * 100),
                      color: '#3B82F6',
                    },
                    {
                      name: 'Visual',
                      value: Math.round(modelResult.modality_weights.visual * 100),
                      color: '#EF4444',
                    },
                    {
                      name: 'Text',
                      value: Math.round(modelResult.modality_weights.text * 100),
                      color: '#22C55E',
                    },
                  ]}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, value }) => `${name}: ${value}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {[
                    { color: '#3B82F6' },
                    { color: '#EF4444' },
                    { color: '#22C55E' },
                  ].map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(value: number) => [`${value}%`, 'Weight']}
                  contentStyle={{
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    border: 'none',
                    borderRadius: '8px',
                    color: '#fff',
                  }}
                />
              </RechartsPieChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Model Agreement */}
      {modelResult.model_agreement && (
        <div className="p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <Users className="w-5 h-5 text-indigo-500" />
            Model Agreement
          </h3>
          <div className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">RFRBoost</div>
                <div className="text-sm font-semibold text-gray-900 dark:text-white capitalize">
                  {modelResult.model_agreement.rfrboost}
                </div>
              </div>
              <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Attention+Deep</div>
                <div className="text-sm font-semibold text-gray-900 dark:text-white capitalize">
                  {modelResult.model_agreement.attention_deep}
                </div>
              </div>
              <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">MLP Baseline</div>
                <div className="text-sm font-semibold text-gray-900 dark:text-white capitalize">
                  {modelResult.model_agreement.mlp_baseline}
                </div>
              </div>
              <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Status</div>
                <div
                  className={`text-sm font-semibold capitalize ${
                    modelResult.model_agreement.agreement_status === 'all_agree'
                      ? 'text-green-600 dark:text-green-400'
                      : modelResult.model_agreement.agreement_status === 'majority_agree'
                        ? 'text-amber-600 dark:text-amber-400'
                        : 'text-red-600 dark:text-red-400'
                  }`}
                >
                  {modelResult.model_agreement.agreement_status.replace('_', ' ')}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Transcription */}
      {modelResult.transcription && (
        <div className="p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <MessageSquare className="w-5 h-5 text-blue-500" />
            Transcription
          </h3>
          <div className="space-y-2">
            <p className="text-gray-600 dark:text-gray-400 whitespace-pre-wrap">
              {modelResult.transcription.text}
            </p>
            <div className="flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
              <span>{modelResult.transcription.word_count} words</span>
              {modelResult.transcription.language && (
                <span>Language: {modelResult.transcription.language}</span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* AI Analysis */}
      {modelResult.ai_analysis && (
        <div className="p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <Lightbulb className="w-5 h-5 text-amber-500" />
            AI Insights
          </h3>

          {modelResult.ai_analysis.summary && (
          <div className="mb-6">
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Summary
            </h4>
            <p className="text-gray-600 dark:text-gray-400">
                {modelResult.ai_analysis.summary}
            </p>
          </div>
          )}

          {modelResult.ai_analysis.key_insights &&
            modelResult.ai_analysis.key_insights.length > 0 && (
            <div className="mb-6">
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Key Insights
              </h4>
              <ul className="space-y-2">
                  {modelResult.ai_analysis.key_insights.map((insight, index) => (
                  <li
                    key={index}
                    className="flex items-start gap-2 text-gray-600 dark:text-gray-400"
                  >
                    <CheckCircle className="w-4 h-4 text-teal-500 mt-0.5 flex-shrink-0" />
                    <span>{insight}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {modelResult.ai_analysis.recommendations &&
            modelResult.ai_analysis.recommendations.length > 0 && (
            <div className="mb-6">
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Recommendations
              </h4>
              <ul className="space-y-2">
                  {modelResult.ai_analysis.recommendations.map((rec, index) => (
                  <li
                    key={index}
                    className="flex items-start gap-2 text-gray-600 dark:text-gray-400"
                  >
                    <TrendingUp className="w-4 h-4 text-blue-500 mt-0.5 flex-shrink-0" />
                    <span>{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function AnalysisResults({ result }: AnalysisResultsProps) {
  if (!result || !result.results || result.results.length === 0) {
    return (
      <div className="p-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 text-center">
        <p className="text-gray-600 dark:text-gray-400">No results available</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Summary Header */}
      <div className="p-6 bg-gradient-to-r from-teal-50 to-blue-50 dark:from-teal-900/20 dark:to-blue-900/20 rounded-xl border border-teal-200 dark:border-teal-800">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
              Analysis Complete
            </h2>
            <p className="text-gray-600 dark:text-gray-400">{result.message}</p>
          </div>
          <div className="text-right">
            <div className="text-sm text-gray-500 dark:text-gray-400">Total Processing Time</div>
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {result.total_processing_time.toFixed(2)}s
            </div>
          </div>
        </div>
      </div>

      {/* Model Results */}
      {result.results.map((modelResult, index) => (
        <div key={index} className="space-y-6">
          {index > 0 && (
            <div className="border-t border-gray-200 dark:border-gray-700 pt-6">
              <div className="text-center text-sm text-gray-500 dark:text-gray-400 mb-4">
                Additional Model Result
              </div>
        </div>
      )}
          <ModelResultCard modelResult={modelResult} />
        </div>
      ))}
    </div>
  );
}
