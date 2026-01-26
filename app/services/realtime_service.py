"""
Real-time Analysis Service

Service layer for real-time emotion analysis via WebSocket.
"""

import os
import sys
import logging
import base64
from typing import Dict, Any, Optional

# Add parent directory to path

from emotion_framework import RealtimeEmotionAnalyzer
from services.subs_service import get_user_subscription
from utils.general import is_localhost

logger = logging.getLogger(__name__)


class RealtimeAnalysisService:
    """
    Service for real-time video emotion analysis.
    
    Handles chunk validation, session management, and real-time processing.
    """
    
    def __init__(self):
        """Initialize real-time analysis service."""
        self.analyzer = RealtimeEmotionAnalyzer()
        self.max_chunk_size = 50 * 1024 * 1024
        self.max_chunk_duration = 5.0
        
        logger.info("RealtimeAnalysisService initialized")
    
    async def create_session(self, user_id: str, request=None) -> str:
        """
        Create a new real-time analysis session.
        
        Args:
            user_id: User ID
            request: Optional request object for localhost check
            
        Returns:
            Session ID
            
        Raises:
            Exception: If subscription validation fails
        """
        await self._validate_subscription(user_id, request)
        
        session_id = self.analyzer.create_session(user_id)
        
        logger.info(f"Created real-time session {session_id} for user {user_id}")
        
        return session_id
    
    async def process_chunk(
        self,
        session_id: str,
        chunk_data: bytes,
        timestamp: float = 0.0,
        chunk_index: int = 0,
        encoding: str = "raw"
    ) -> Dict[str, Any]:
        """
        Process a single video chunk.
        
        Args:
            session_id: Session ID
            chunk_data: Video chunk data_model
            timestamp: Timestamp in video
            chunk_index: Chunk index
            encoding: Data encoding ("raw" or "base64")
            
        Returns:
            Processing result dictionary
            
        Raises:
            ValueError: If validation fails
        """
        if encoding == "base64":
            try:
                chunk_data = base64.b64decode(chunk_data)
            except Exception as e:
                raise ValueError(f"Invalid base64 encoding: {e}")
        
        self._validate_chunk(chunk_data)
        
        try:
            result = self.analyzer.process_chunk(
                chunk_data=chunk_data,
                session_id=session_id,
                timestamp=timestamp,
                chunk_index=chunk_index
            )
            
            formatted_result = self._format_chunk_result(result)
            
            return formatted_result
            
        except Exception as e:
            logger.error(f"Error processing chunk: {e}", exc_info=True)
            raise
    
    async def finalize_session(self, session_id: str) -> Dict[str, Any]:
        """
        Finalize session and generate summary.
        
        Args:
            session_id: Session ID
            
        Returns:
            Summary dictionary
        """
        session_state = self.analyzer.get_session_state(session_id)
        summary = self._generate_session_summary(session_state)
        
        logger.info(f"Finalized session {session_id}")
        
        return summary
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get current session status.
        
        Args:
            session_id: Session ID
            
        Returns:
            Status dictionary
        """
        try:
            state = self.analyzer.get_session_state(session_id)
            
            return {
                "session_id": session_id,
                "status": "active",
                "chunk_count": state.get("chunk_count", 0),
                "prediction_count": state.get("prediction_count", 0),
                "created_at": state.get("created_at"),
                "last_activity": state.get("last_activity"),
            }
        except ValueError:
            return {
                "session_id": session_id,
                "status": "not_found",
                "error": "Session not found"
            }
    
    async def delete_session(self, session_id: str):
        """
        Delete a session.
        
        Args:
            session_id: Session ID
        """
        self.analyzer.delete_session(session_id)
        logger.info(f"Deleted session {session_id}")
    
    def _validate_chunk(self, chunk_data: bytes):
        """
        Validate chunk data_model.
        
        Args:
            chunk_data: Chunk bytes
            
        Raises:
            ValueError: If validation fails
        """
        if len(chunk_data) > self.max_chunk_size:
            raise ValueError(
                f"Chunk size {len(chunk_data)} exceeds maximum {self.max_chunk_size}"
            )
        
        if len(chunk_data) == 0:
            raise ValueError("Empty chunk data_model")
    
    def _format_chunk_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format chunk processing result for API response.
        
        Args:
            result: Raw result from analyzer
            
        Returns:
            Formatted result dictionary
        """
        return {
            "chunk_index": result.get("chunk_index", 0),
            "timestamp": result.get("timestamp", 0.0),
            "emotion": result.get("emotion", "neutral"),
            "confidence": result.get("confidence", 0.0),
            "confidences": result.get("confidences", {}),
            "processing_time": result.get("processing_time", 0.0),
            "created_at": result.get("created_at"),
        }
    
    def _generate_session_summary(self, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate session summary from state.
        
        Args:
            session_state: Session state dictionary
            
        Returns:
            Summary dictionary
        """
        predictions = session_state.get("predictions", [])
        total_chunks = len(predictions)
        
        if total_chunks > 0:
            emotion_counts = {}
            total_confidence = 0.0
            
            for pred in predictions:
                emotion = pred.get("emotion", "neutral")
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                total_confidence += pred.get("confidence", 0.0)
            
            dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "neutral"
            avg_confidence = total_confidence / total_chunks if total_chunks > 0 else 0.0
            
            emotion_distribution = {
                emotion: count / total_chunks
                for emotion, count in emotion_counts.items()
            }
        else:
            dominant_emotion = "neutral"
            avg_confidence = 0.0
            emotion_distribution = {}
        
        summary = {
            "session_id": session_state.get("session_id"),
            "user_id": session_state.get("user_id"),
            "total_chunks": total_chunks,
            "duration": session_state.get("chunk_count", 0) * 4.0,  # Approximate duration
            "dominant_emotion": dominant_emotion,
            "average_confidence": avg_confidence,
            "emotion_distribution": emotion_distribution,
            "created_at": session_state.get("created_at"),
            "last_activity": session_state.get("last_activity"),
            "predictions": predictions,
        }
        
        return summary
    
    async def _validate_subscription(self, user_id: str, request=None):
        """
        Validate user subscription for real-time analysis.
        
        Args:
            user_id: User ID
            request: Optional request object
            
        Raises:
            Exception: If subscription validation fails
        """
        try:
            if self._is_dev_or_local(request):
                logger.info(f"Skipping subscription check for local/dev: user {user_id}")
                return
            
            subscription = await get_user_subscription(user_id)
            
            if not subscription:
                raise Exception("No active subscription found")
            
            status = subscription.get('status', '')
            if status not in ['active', 'trialing']:
                raise Exception(f"Subscription status is '{status}'. Active subscription required.")
            
            logger.info(f"Subscription validated for user {user_id}")
            
        except Exception as e:
            logger.error(f"Subscription validation failed: {e}")
            raise
    
    def _is_dev_or_local(self, request=None) -> bool:
        """Check if running in dev/local environment."""
        app_env = os.getenv("APP_ENV", "").lower()
        if app_env in {"dev", "development", "local", "test", "testing"}:
            return True
        
        if request is None:
            return False
        
        try:
            return bool(is_localhost(request))
        except:
            return False
    
    def get_active_session_count(self) -> int:
        """Get number of active sessions."""
        return self.analyzer.get_session_count()
    
    def get_session_limit(self) -> int:
        """Get maximum session limit."""
        return self.analyzer.max_sessions

