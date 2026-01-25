"""
Data ai_models for emotion analysis results.

These ai_models represent the output of the emotion analysis pipeline
and can be easily serialized to JSON for API responses.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class VideoMetadata:
    """Video file metadata."""
    duration: float
    fps: float
    width: int
    height: int
    frame_count: int
    filename: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class EmotionPrediction:
    """Overall emotion prediction for the video."""
    predicted_emotion: str
    predicted_label: int
    confidence: float
    all_confidences: Dict[str, float]
    fusion_method: str
    modality_weights: Optional[Dict[str, float]] = None
    individual_models: Optional[Dict[str, str]] = None
    intensity: Optional[float] = None
    reasoning: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TemporalPrediction:
    """Emotion prediction at a specific timestamp."""
    timestamp: float
    emotion: str
    confidences: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class MentalHealthAnalysis:
    """Mental health analysis based on facial expression recognition."""
    mental_health_score: float
    avg_confidence: float
    num_frames: int
    dominant_emotion: str
    positive_percentage: float
    negative_percentage: float
    emotion_distribution: Dict[str, float]
    status: str
    recommendation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TranscriptionResult:
    """Video transcription data_model."""
    text: str
    word_count: int
    language: str = "en"
    segments: Optional[List[Dict]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        if self.segments is None:
            data['segments'] = []
        return data


@dataclass
class FeatureInfo:
    """Information about extracted features."""
    audio: Optional[np.ndarray] = None
    visual: Optional[np.ndarray] = None
    text: Optional[np.ndarray] = None
    
    def get_counts(self) -> Dict[str, int]:
        """Get feature counts per modality."""
        counts = {}
        if self.audio is not None:
            counts['audio'] = len(self.audio) if hasattr(self.audio, '__len__') else 0
        if self.visual is not None:
            counts['visual'] = len(self.visual) if hasattr(self.visual, '__len__') else 0
        if self.text is not None:
            counts['text'] = len(self.text) if hasattr(self.text, '__len__') else 0
        return counts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without numpy arrays)."""
        return self.get_counts()


@dataclass
class AIAnalysisResult:
    """AI agent meeting analysis results."""
    summary: Optional[str] = None
    key_insights: Optional[List[str]] = None
    emotional_dynamics: Optional[Any] = None
    recommendations: Optional[List[str]] = None
    knowledge_base_context: Optional[List[Dict]] = None
    detailed_analysis: Optional[str] = None
    raw_llm_response: Optional[str] = None
    llm_model: Optional[str] = None
    llm_prompt: Optional[str] = None
    agent_available: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class EmotionAnalysisResult:
    """
    Complete emotion analysis result.
    
    This is the main result object returned by the EmotionAnalysisPipeline.
    Contains all information about the analyzed video.
    """
    # Core results
    metadata: VideoMetadata
    prediction: Optional[EmotionPrediction] = None
    temporal_predictions: List[TemporalPrediction] = field(default_factory=list)
    
    # Analysis results
    mental_health_analysis: Optional[MentalHealthAnalysis] = None
    transcription: Optional[TranscriptionResult] = None
    ai_analysis: Optional[AIAnalysisResult] = None
    
    # Feature information
    features: FeatureInfo = field(default_factory=FeatureInfo)
    
    # File paths
    frame_paths: List[str] = field(default_factory=list)
    frames_folder: Optional[str] = None
    audio_path: Optional[str] = None
    
    # Processing info
    processing_time: float = 0.0
    
    # Raw data_model (for compatibility with Streamlit)
    audio: Optional[np.ndarray] = None
    sample_rate: Optional[int] = None
    frames: Optional[List[np.ndarray]] = None
    timestamps: Optional[List[float]] = None
    combined_features: Optional[np.ndarray] = None
    
    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        Args:
            include_arrays: If True, include numpy arrays (not JSON serializable)
        
        Returns:
            Dictionary representation
        """
        result = {
            'metadata': self.metadata.to_dict(),
            'prediction': self.prediction.to_dict() if self.prediction else None,
            'temporal_predictions': [p.to_dict() for p in self.temporal_predictions],
            'mental_health_analysis': self.mental_health_analysis.to_dict() if self.mental_health_analysis else None,
            'transcription': self.transcription.to_dict() if self.transcription else None,
            'ai_analysis': self.ai_analysis.to_dict() if self.ai_analysis else None,
            'features': self.features.to_dict(),
            'frame_paths': self.frame_paths,
            'frames_folder': self.frames_folder,
            'audio_path': self.audio_path,
            'processing_time': self.processing_time,
        }
        
        if include_arrays:
            result['audio'] = self.audio
            result['sample_rate'] = self.sample_rate
            result['frames'] = self.frames
            result['timestamps'] = self.timestamps
            result['combined_features'] = self.combined_features
        
        return result
    
    def to_streamlit_format(self) -> Dict[str, Any]:
        """
        Convert to format expected by Streamlit UI.
        Includes numpy arrays and maintains backward compatibility.
        """
        return {
            'metadata': self.metadata.to_dict(),
            'prediction': self.prediction.to_dict() if self.prediction else None,
            'temporal_predictions': [p.to_dict() for p in self.temporal_predictions],
            'mental_health_analysis': self.mental_health_analysis.to_dict() if self.mental_health_analysis else None,
            'transcription': self.transcription.to_dict() if self.transcription else None,
            'ai_analysis': self.ai_analysis.to_dict() if self.ai_analysis else None,
            'features': {
                'audio': self.features.audio,
                'visual': self.features.visual,
                'text': self.features.text,
            },
            'frame_paths': self.frame_paths,
            'frames_folder': self.frames_folder,
            'audio_path': self.audio_path,
            'processing_time': self.processing_time,
            'audio': self.audio,
            'sample_rate': self.sample_rate,
            'frames': self.frames,
            'timestamps': self.timestamps,
            'combined_features': self.combined_features,
        }

