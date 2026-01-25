"""Core framework components."""

from .pipeline import EmotionAnalysisPipeline
from .config_loader import load_framework_config
from .realtime_pipeline import RealtimeEmotionAnalyzer, RealtimeSession

__all__ = [
    "EmotionAnalysisPipeline",
    "load_framework_config",
    "RealtimeEmotionAnalyzer",
    "RealtimeSession",
]

