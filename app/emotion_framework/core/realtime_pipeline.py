"""
Real-time emotion analysis pipeline.

This module provides real-time video emotion analysis using sliding windows.
Designed for processing video streams in 4-second chunks with 1-second stride.
"""

import os
import sys
import time
import uuid
import logging
import tempfile
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from emotion_framework.models.result_models import (
    TemporalPrediction,
    EmotionPrediction,
)

logger = logging.getLogger(__name__)


class RealtimeSession:
    """Represents a real-time analysis session."""
    
    def __init__(self, session_id: str, user_id: str, max_predictions: int = 60):
        """
        Initialize a real-time session.
        
        Args:
            session_id: Unique session identifier
            user_id: User ID for this session
            max_predictions: Maximum number of predictions to keep in buffer
        """
        self.session_id = session_id
        self.user_id = user_id
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.chunk_count = 0
        self.predictions = deque(maxlen=max_predictions)  # Rolling buffer
        self.metadata = {}
        
    def add_prediction(self, prediction: Dict[str, Any]):
        """Add a prediction to the session buffer."""
        self.predictions.append(prediction)
        self.chunk_count += 1
        self.last_activity = datetime.now()
        
    def get_recent_predictions(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get recent predictions from buffer.
        
        Args:
            last_n: Number of recent predictions to return (None = all)
            
        Returns:
            List of prediction dictionaries
        """
        if last_n is None:
            return list(self.predictions)
        return list(self.predictions)[-last_n:]
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete session state."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "chunk_count": self.chunk_count,
            "prediction_count": len(self.predictions),
            "metadata": self.metadata,
        }
    
    def is_stale(self, max_age_hours: int = 24) -> bool:
        """Check if session is stale based on last activity."""
        age = datetime.now() - self.last_activity
        return age > timedelta(hours=max_age_hours)


class RealtimeEmotionAnalyzer:
    """
    Real-time emotion analyzer using sliding window approach.
    
    Processes video chunks (4-second windows) with 1-second stride for
    continuous emotion analysis. Maintains session state for each analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize real-time analyzer.
        
        Args:
            config: Configuration dictionary for analysis
        """
        from emotion_framework.core.config_loader import load_framework_config
        
        if config is None:
            config = load_framework_config()
        
        self.config = config
        self.emotion_labels = config.get('emotions', {}).get('labels', 
                                                              ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise'])
        
        # Session storage
        self.sessions: Dict[str, RealtimeSession] = {}
        self.max_sessions = int(os.getenv('REALTIME_MAX_SESSIONS', '100'))
        
        # Lazy load processor
        self._processor = None
        
        logger.info("RealtimeEmotionAnalyzer initialized")
    
    def create_session(self, user_id: str, session_id: Optional[str] = None) -> str:
        """
        Create a new analysis session.
        
        Args:
            user_id: User ID for the session
            session_id: Optional session ID (generated if not provided)
            
        Returns:
            Session ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        if len(self.sessions) >= self.max_sessions:
            self.cleanup_old_sessions(max_age_hours=1)
            
            if len(self.sessions) >= self.max_sessions:
                oldest_id = min(self.sessions.keys(), 
                              key=lambda k: self.sessions[k].last_activity)
                del self.sessions[oldest_id]
                logger.warning(f"Removed oldest session {oldest_id} due to limit")
        
        session = RealtimeSession(session_id, user_id)
        self.sessions[session_id] = session
        
        logger.info(f"Created session {session_id} for user {user_id}")
        return session_id
    
    def process_chunk(
        self,
        chunk_data: bytes,
        session_id: str,
        timestamp: float = 0.0,
        chunk_index: int = 0
    ) -> Dict[str, Any]:
        """
        Process a single video chunk.
        
        Args:
            chunk_data: Video chunk data_model (bytes)
            session_id: Session ID
            timestamp: Timestamp of chunk in original video
            chunk_index: Index of this chunk
            
        Returns:
            Prediction dictionary with emotion results
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        start_time = time.time()
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(chunk_data)
                chunk_path = tmp_file.name
            
            result = self._process_chunk_lightweight(chunk_path, timestamp)
            
            prediction = {
                "chunk_index": chunk_index,
                "timestamp": timestamp,
                "emotion": result.get('emotion', 'neutral'),
                "confidence": result.get('confidence', 0.0),
                "confidences": result.get('confidences', {}),
                "processing_time": time.time() - start_time,
                "created_at": datetime.now().isoformat(),
            }
            
            session.add_prediction(prediction)
            
            try:
                os.unlink(chunk_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")
            
            logger.info(f"Processed chunk {chunk_index} for session {session_id}: {result.get('emotion')}")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error processing chunk: {e}", exc_info=True)
            raise
    
    def _process_chunk_lightweight(self, chunk_path: str, timestamp: float) -> Dict[str, Any]:
        """
        Process chunk with lightweight/optimized pipeline.
        
        Args:
            chunk_path: Path to chunk file
            timestamp: Chunk timestamp
            
        Returns:
            Emotion prediction dictionary
        """
        from emotion_framework.processors.realtime_processor import RealtimeVideoProcessor
        
        if self._processor is None:
            self._processor = RealtimeVideoProcessor(self.config)
        
        result = self._processor.process_chunk(chunk_path, timestamp)
        
        return result
    
    def get_predictions(self, session_id: str, last_n: Optional[int] = 10) -> List[Dict[str, Any]]:
        """
        Get recent predictions for a session.
        
        Args:
            session_id: Session ID
            last_n: Number of recent predictions (None = all)
            
        Returns:
            List of prediction dictionaries
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        return self.sessions[session_id].get_recent_predictions(last_n)
    
    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """
        Get complete session state.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session state dictionary
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        state = self.sessions[session_id].get_state()
        state['predictions'] = self.get_predictions(session_id, last_n=None)
        
        return state
    
    def reset_session(self, session_id: str):
        """
        Reset/clear session data_model.
        
        Args:
            session_id: Session ID to reset
        """
        if session_id in self.sessions:
            user_id = self.sessions[session_id].user_id
            del self.sessions[session_id]
            self.create_session(user_id, session_id)
            logger.info(f"Reset session {session_id}")
        else:
            logger.warning(f"Session {session_id} not found for reset")
    
    def delete_session(self, session_id: str):
        """
        Delete a session.
        
        Args:
            session_id: Session ID to delete
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session {session_id}")
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """
        Remove stale sessions based on last activity.
        
        Args:
            max_age_hours: Maximum age in hours before session is considered stale
        """
        stale_sessions = [
            sid for sid, session in self.sessions.items()
            if session.is_stale(max_age_hours)
        ]
        
        for sid in stale_sessions:
            del self.sessions[sid]
        
        if stale_sessions:
            logger.info(f"Cleaned up {len(stale_sessions)} stale sessions")
    
    def get_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self.sessions)
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get state of all active sessions."""
        return [session.get_state() for session in self.sessions.values()]
