"""
Emotion Analysis Framework

A reusable framework for multimodal emotion recognition from video files.
This framework can be used by both API services and UI applications.
"""

__version__ = "1.0.1"
__author__ = "Emotion Analysis Team"

from .core.pipeline import EmotionAnalysisPipeline
from .core.realtime_pipeline import RealtimeEmotionAnalyzer, RealtimeSession
from .models.result_models import EmotionAnalysisResult
from .processors.realtime_processor import RealtimeVideoProcessor, RealtimeFeatureExtractor, IncrementalFusionEngine

__all__ = [
    "EmotionAnalysisPipeline",
    "EmotionAnalysisResult",
    "RealtimeEmotionAnalyzer",
    "RealtimeSession",
    "RealtimeVideoProcessor",
    "RealtimeFeatureExtractor",
    "IncrementalFusionEngine",
]

