export type EmotionType =
  | 'angry'
  | 'disgust'
  | 'fear'
  | 'happy'
  | 'sad'
  | 'surprise'
  | 'neutral';

export interface EmotionPrediction {
  emotion: EmotionType;
  confidence: number;
  timestamp?: number;
}

export interface EmotionDistribution {
  angry: number;
  disgust: number;
  fear: number;
  happy: number;
  sad: number;
  surprise: number;
  neutral: number;
}

export interface MentalHealthAssessment {
  score: number;
  status: 'excellent' | 'good' | 'moderate' | 'concerning' | 'critical';
  recommendation: string;
  factors: string[];
}

export interface AIAnalysis {
  summary: string;
  key_insights: string[];
  recommendations: string[];
  emotional_trajectory: string;
}

export interface FrameAnalysis {
  timestamp: number;
  emotion: EmotionType;
  confidence: number;
  face_detected: boolean;
}

// API Response Types
export interface OverallPrediction {
  predicted_emotion: EmotionType;
  confidence: number;
  all_confidences: EmotionDistribution;
}

export interface TemporalPrediction {
  timestamp: number;
  emotion: EmotionType;
  confidence: number;
  all_confidences: EmotionDistribution;
}

export interface VideoMetadata {
  filename: string;
  duration: number;
  fps: number;
  width: number;
  height: number;
  frame_count: number;
}

export interface Transcription {
  text: string;
  word_count: number;
  language?: string;
}

export interface ModalityFeatures {
  audio_features: number;
  visual_features: number;
  text_features: number;
}

export interface ModalityWeights {
  audio: number;
  visual: number;
  text: number;
}

export interface ModelAgreement {
  rfrboost: EmotionType;
  attention_deep: EmotionType;
  mlp_baseline: EmotionType;
  agreement_status: 'all_agree' | 'majority_agree' | 'disagree';
}

export interface MaelfabienPredictions {
  text_cnn_lstm: EmotionType;
  audio_time_cnn: EmotionType;
  video_xception: EmotionType;
}

export interface EmotionLLaMaDetails {
  intensity: number;
  reasoning: string;
}

export interface MentalHealthAnalysis {
  mental_health_score: number;
  avg_confidence: number;
  num_frames: number;
  dominant_emotion: EmotionType;
  positive_percentage: number;
  negative_percentage: number;
  emotion_distribution: Record<EmotionType, number>;
  status: string;
  recommendation?: string;
}

export interface ModelResults {
  model_name: string;
  fusion_strategy: string;
  processing_time: number;
  overall_prediction: OverallPrediction;
  temporal_predictions: TemporalPrediction[];
  mental_health_analysis: MentalHealthAnalysis | null;
  video_metadata: VideoMetadata;
  transcription: Transcription | null;
  features: ModalityFeatures;
  modality_weights: ModalityWeights | null;
  model_agreement: ModelAgreement | null;
  maelfabien_predictions: MaelfabienPredictions | null;
  emotion_llama_details: EmotionLLaMaDetails | null;
  ai_analysis: AIAnalysis | null;
  frame_paths: string[] | null;
}

export interface MultiModelResponse {
  success: boolean;
  message: string;
  results: ModelResults[];
  total_processing_time: number;
}

// Legacy type for backward compatibility
export interface MultiModelResponseLegacy {
  job_id?: string;
  predicted_emotion: EmotionType;
  confidence: number;
  emotion_distribution: EmotionDistribution;
  frame_analyses?: FrameAnalysis[];
  mental_health_assessment?: MentalHealthAssessment;
  transcription?: string;
  ai_analysis?: AIAnalysis;
  processing_time_ms: number;
  models_used: string[];
}

export interface AnalysisJob {
  id: string;
  user_id: string;
  filename: string;
  status: 'pending' | 'processing' | 'success' | 'failed';
  created_at: string;
  completed_at?: string;
  result?: MultiModelResponse;
  error?: string;
}

export interface JobSubmitResponse {
  job_id: string;
  message: string;
  status: 'pending';
}

export interface ModelInfo {
  name: string;
  description: string;
  supported_formats: string[];
}

export interface AnalysisOptions {
  model?: string;
  include_ai_analysis?: boolean;
  llm_provider?: 'cloud' | 'local';
  extract_audio?: boolean;
}
