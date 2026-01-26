"""
Celery Tasks Package

Contains all async tasks for emotion analysis.
"""

from .emotion_tasks import (
    analyze_video_async,
    batch_analyze_videos,
    cleanup_temp_files,
)

__all__ = [
    "analyze_video_async",
    "batch_analyze_videos",
    "cleanup_temp_files",
]

