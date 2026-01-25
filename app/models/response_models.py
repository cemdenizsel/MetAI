"""
Response Models for Emotion Recognition API

Defines the standardized response format for all ai_models.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class EmotionConfidence(BaseModel):
    """Confidence scores for each emotion."""
    angry: float = Field(..., ge=0, le=1, description="Confidence for angry emotion")
    disgust: float = Field(..., ge=0, le=1, description="Confidence for disgust emotion")
    fear: float = Field(..., ge=0, le=1, description="Confidence for fear emotion")
    happy: float = Field(..., ge=0, le=1, description="Confidence for happy emotion")
    sad: float = Field(..., ge=0, le=1, description="Confidence for sad emotion")
    surprise: float = Field(..., ge=0, le=1, description="Confidence for surprise emotion")
    neutral: float = Field(..., ge=0, le=1, description="Confidence for neutral emotion")


class TemporalPrediction(BaseModel):
    """Emotion prediction at a specific timestamp."""
    timestamp: float = Field(..., description="Time in seconds")
    emotion: str = Field(..., description="Dominant emotion at this timestamp")
    confidence: float = Field(..., ge=0, le=1, description="Confidence of dominant emotion")
    all_confidences: EmotionConfidence = Field(..., description="Confidence scores for all emotions")


class MentalHealthAnalysis(BaseModel):
    """Mental health analysis based on emotion distribution."""
    mental_health_score: float = Field(..., ge=0, le=100, description="Mental health score (0-100)")
    avg_confidence: float = Field(..., ge=0, le=1, description="Average prediction confidence")
    num_frames: int = Field(..., description="Number of frames analyzed")
    dominant_emotion: str = Field(..., description="Most frequent emotion")
    positive_percentage: float = Field(..., ge=0, le=100, description="Percentage of positive emotions")
    negative_percentage: float = Field(..., ge=0, le=100, description="Percentage of negative emotions")
    emotion_distribution: Dict[str, float] = Field(..., description="Distribution of emotions as percentages")
    status: str = Field(..., description="Mental health status interpretation")
    recommendation: Optional[str] = Field(None, description="Recommendation based on status")


class VideoMetadata(BaseModel):
    """Video file metadata."""
    filename: str = Field(..., description="Original filename")
    duration: float = Field(..., description="Duration in seconds")
    fps: float = Field(..., description="Frames per second")
    width: int = Field(..., description="Video width in pixels")
    height: int = Field(..., description="Video height in pixels")
    frame_count: int = Field(..., description="Total number of frames")


class Transcription(BaseModel):
    """Video transcription data_model."""
    text: str = Field(..., description="Full transcription text")
    word_count: int = Field(..., description="Number of words in transcription")
    language: Optional[str] = Field("en", description="Detected language")


class ModalityFeatures(BaseModel):
    """Feature counts per modality."""
    audio_features: int = Field(..., description="Number of audio features extracted")
    visual_features: int = Field(..., description="Number of visual features extracted")
    text_features: int = Field(..., description="Number of text features extracted")


class ModalityWeights(BaseModel):
    """Modality importance weights (for Hybrid model)."""
    audio: float = Field(..., ge=0, le=1, description="Audio modality weight")
    visual: float = Field(..., ge=0, le=1, description="Visual modality weight")
    text: float = Field(..., ge=0, le=1, description="Text modality weight")


class ModelAgreement(BaseModel):
    """Individual model predictions (for Hybrid model)."""
    rfrboost: str = Field(..., description="RFRBoost prediction")
    attention_deep: str = Field(..., description="Attention+Deep prediction")
    mlp_baseline: str = Field(..., description="MLP Baseline prediction")
    agreement_status: str = Field(..., description="Agreement level (all_agree, majority_agree, disagree)")


class MaelfabienPredictions(BaseModel):
    """Individual model predictions (for Maelfabien model)."""
    text_cnn_lstm: str = Field(..., description="Text CNN-LSTM prediction")
    audio_time_cnn: str = Field(..., description="Audio Time-CNN prediction")
    video_xception: str = Field(..., description="Video XCeption prediction")


class EmotionLLaMaDetails(BaseModel):
    """Emotion-LLaMA specific details."""
    intensity: float = Field(..., ge=0, le=1, description="Emotion intensity score")
    reasoning: str = Field(..., description="Natural language reasoning for prediction")


class OverallPrediction(BaseModel):
    """Overall emotion prediction for the video."""
    predicted_emotion: str = Field(..., description="Overall predicted emotion")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    all_confidences: EmotionConfidence = Field(..., description="Confidence scores for all emotions")


class AIAnalysis(BaseModel):
    """AI agent meeting analysis."""
    summary: Optional[str] = Field(None, description="Executive summary of the meeting")
    key_insights: Optional[List[str]] = Field(None, description="Key insights from the analysis")
    emotional_dynamics: Optional[Dict[str, Any]] = Field(None, description="Analysis of emotional dynamics")
    recommendations: Optional[List[str]] = Field(None, description="Actionable recommendations")
    knowledge_base_context: Optional[List[Dict[str, Any]]] = Field(None, description="Retrieved knowledge base context")
    detailed_analysis: Optional[str] = Field(None, description="Detailed analysis text")
    raw_llm_response: Optional[str] = Field(None, description="Raw LLM response for transparency")
    llm_model: Optional[str] = Field(None, description="LLM model used")
    agent_available: bool = Field(False, description="Whether AI agent was available")


class ModelResults(BaseModel):
    """Standardized results for a single model."""
    model_name: str = Field(..., description="Name of the model")
    fusion_strategy: str = Field(..., description="Fusion strategy used")
    processing_time: float = Field(..., description="Processing time in seconds")
    
    # Core results
    overall_prediction: OverallPrediction = Field(..., description="Overall emotion prediction")
    temporal_predictions: List[TemporalPrediction] = Field(..., description="Emotion predictions over time")
    mental_health_analysis: Optional[MentalHealthAnalysis] = Field(None, description="Mental health analysis")
    
    # Video data_model
    video_metadata: VideoMetadata = Field(..., description="Video metadata")
    transcription: Optional[Transcription] = Field(None, description="Video transcription")
    
    # Features
    features: ModalityFeatures = Field(..., description="Extracted feature counts")
    
    # Model-specific data_model
    modality_weights: Optional[ModalityWeights] = Field(None, description="Modality weights (Hybrid only)")
    model_agreement: Optional[ModelAgreement] = Field(None, description="Model agreement (Hybrid only)")
    maelfabien_predictions: Optional[MaelfabienPredictions] = Field(None, description="Maelfabien predictions")
    emotion_llama_details: Optional[EmotionLLaMaDetails] = Field(None, description="Emotion-LLaMA details")
    
    # AI Analysis and Additional Data
    ai_analysis: Optional[AIAnalysis] = Field(None, description="AI agent meeting analysis")
    frame_paths: Optional[List[str]] = Field(None, description="Paths to extracted frames")


class MultiModelResponse(BaseModel):
    """Complete response with results from all ai_models."""
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="Status message")
    results: List[ModelResults] = Field(..., description="Results from all ai_models")
    total_processing_time: float = Field(..., description="Total processing time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Successfully analyzed video with all ai_models",
                "results": [
                    {
                        "model_name": "Hybrid (Best)",
                        "fusion_strategy": "RFRBoost + Deep Learning + Attention",
                        "processing_time": 15.3,
                        "overall_prediction": {
                            "predicted_emotion": "happy",
                            "confidence": 0.87,
                            "all_confidences": {
                                "angry": 0.02,
                                "disgust": 0.01,
                                "fear": 0.03,
                                "happy": 0.87,
                                "sad": 0.02,
                                "surprise": 0.03,
                                "neutral": 0.02
                            }
                        }
                    }
                ],
                "total_processing_time": 45.6
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = Field(False, description="Always False for errors")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: float = Field(..., description="Unix timestamp when message was sent")


class ChatSession(BaseModel):
    """Chat session model."""
    chat_id: str = Field(..., description="Unique chat session ID")
    context: str = Field(..., description="Initial context for the chat")
    messages: List[ChatMessage] = Field(default_factory=list, description="Chat message history")
    created_at: float = Field(..., description="Unix timestamp when chat was created")
    updated_at: float = Field(..., description="Unix timestamp when chat was last updated")


class StartChatRequest(BaseModel):
    """Request model for starting a new chat."""
    context: str = Field(..., description="Initial context for the chat session")


class StartChatResponse(BaseModel):
    """Response model for starting a new chat."""
    success: bool = Field(..., description="Whether the chat was successfully created")
    chat_id: str = Field(..., description="Unique chat session ID")
    message: str = Field(..., description="Status message")


class SendMessageRequest(BaseModel):
    """Request model for sending a message."""
    chat_id: str = Field(..., description="Chat session ID")
    message: str = Field(..., description="User message content")


class SendMessageResponse(BaseModel):
    """Response model for sending a message."""
    success: bool = Field(..., description="Whether the message was processed successfully")
    chat_id: str = Field(..., description="Chat session ID")
    user_message: ChatMessage = Field(..., description="The user's message")
    assistant_message: ChatMessage = Field(..., description="The assistant's response")
    message: str = Field(..., description="Status message")


# Real-time Analysis Models

class RealtimeChunkResult(BaseModel):
    """Result for a single video chunk in real-time analysis."""
    chunk_index: int = Field(..., description="Chunk sequence number")
    timestamp: float = Field(..., description="Timestamp in video (seconds)")
    emotion: str = Field(..., description="Detected emotion")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    confidences: Dict[str, float] = Field(..., description="Confidence scores for all emotions")
    processing_time: float = Field(..., description="Processing time for this chunk (seconds)")
    created_at: str = Field(..., description="ISO timestamp of result")


class RealtimeSessionStatus(BaseModel):
    """Current status of a real-time analysis session."""
    session_id: str = Field(..., description="Unique session identifier")
    status: str = Field(..., description="Session status (active, completed, error)")
    chunk_count: int = Field(..., description="Number of chunks processed")
    prediction_count: int = Field(..., description="Number of predictions made")
    created_at: str = Field(..., description="Session creation time")
    last_activity: str = Field(..., description="Last activity time")


class RealtimeFinalSummary(BaseModel):
    """Final summary for completed real-time analysis session."""
    session_id: str = Field(..., description="Session identifier")
    user_id: str = Field(..., description="User identifier")
    total_chunks: int = Field(..., description="Total chunks processed")
    duration: float = Field(..., description="Total video duration analyzed (seconds)")
    dominant_emotion: str = Field(..., description="Most frequent emotion")
    average_confidence: float = Field(..., ge=0, le=1, description="Average confidence score")
    emotion_distribution: Dict[str, float] = Field(..., description="Emotion distribution percentages")
    created_at: str = Field(..., description="Session start time")
    last_activity: str = Field(..., description="Session end time")
    predictions: List[Dict[str, Any]] = Field(..., description="All predictions from session")
