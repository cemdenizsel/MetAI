"""
WebSocket Connection Manager

Manages WebSocket connections for real-time emotion analysis streaming.
"""

import logging
import asyncio
from typing import Dict, Optional, Any
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections for real-time analysis.
    
    Handles connection pooling, session mapping, and message broadcasting.
    """
    
    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_users: Dict[str, str] = {}
        self._heartbeat_task = None
        self._heartbeat_interval = 30
        
        logger.info("ConnectionManager initialized")
    
    async def connect(self, websocket: WebSocket, session_id: str, user_id: str):
        """
        Register a new WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            session_id: Unique session ID
            user_id: User ID for this connection
        """
        await websocket.accept()
        
        self.active_connections[session_id] = websocket
        self.session_users[session_id] = user_id
        
        logger.info(f"WebSocket connected: session={session_id}, user={user_id}")
        
        await self.send_message(session_id, {
            "type": "connected",
            "session_id": session_id,
            "message": "WebSocket connection established",
            "timestamp": datetime.now().isoformat()
        })
        
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
    
    async def disconnect(self, session_id: str):
        """
        Remove a WebSocket connection.
        
        Args:
            session_id: Session ID to disconnect
        """
        if session_id in self.active_connections:
            try:
                await self.send_message(session_id, {
                    "type": "disconnected",
                    "session_id": session_id,
                    "message": "WebSocket connection closed",
                    "timestamp": datetime.now().isoformat()
                })
            except Exception:
                pass
            
            websocket = self.active_connections.pop(session_id, None)
            user_id = self.session_users.pop(session_id, None)
            
            if websocket:
                try:
                    await websocket.close()
                except Exception as e:
                    logger.warning(f"Error closing WebSocket: {e}")
            
            logger.info(f"WebSocket disconnected: session={session_id}, user={user_id}")
    
    async def send_message(self, session_id: str, message: Dict[str, Any]):
        """
        Send a message to a specific client.
        
        Args:
            session_id: Session ID
            message: Message dictionary to send
        """
        if session_id not in self.active_connections:
            logger.warning(f"Session {session_id} not found for message send")
            return
        
        websocket = self.active_connections[session_id]
        
        try:
            if "timestamp" not in message:
                message["timestamp"] = datetime.now().isoformat()
            
            await websocket.send_json(message)
            
        except WebSocketDisconnect:
            logger.warning(f"WebSocket disconnected while sending: session={session_id}")
            await self.disconnect(session_id)
        except Exception as e:
            logger.error(f"Error sending message to {session_id}: {e}")
            await self.disconnect(session_id)
    
    async def send_text(self, session_id: str, text: str):
        """
        Send a text message to a specific client.
        
        Args:
            session_id: Session ID
            text: Text message to send
        """
        if session_id not in self.active_connections:
            logger.warning(f"Session {session_id} not found for text send")
            return
        
        websocket = self.active_connections[session_id]
        
        try:
            await websocket.send_text(text)
        except WebSocketDisconnect:
            logger.warning(f"WebSocket disconnected while sending text: session={session_id}")
            await self.disconnect(session_id)
        except Exception as e:
            logger.error(f"Error sending text to {session_id}: {e}")
            await self.disconnect(session_id)
    
    async def broadcast(self, message: Dict[str, Any], exclude: Optional[str] = None):
        """
        Broadcast a message to all connected clients.
        
        Args:
            message: Message dictionary to broadcast
            exclude: Optional session_id to exclude from broadcast
        """
        if "timestamp" not in message:
            message["timestamp"] = datetime.now().isoformat()
        
        disconnected = []
        
        for session_id, websocket in self.active_connections.items():
            if exclude and session_id == exclude:
                continue
            
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {session_id}: {e}")
                disconnected.append(session_id)
        
        for session_id in disconnected:
            await self.disconnect(session_id)
    
    async def send_progress(self, session_id: str, progress_data: Dict[str, Any]):
        """
        Send progress update to a client.
        
        Args:
            session_id: Session ID
            progress_data: Progress information
        """
        message = {
            "type": "progress",
            "session_id": session_id,
            "data_model": progress_data,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.send_message(session_id, message)
    
    async def send_result(self, session_id: str, result_data: Dict[str, Any]):
        """
        Send analysis result to a client.
        
        Args:
            session_id: Session ID
            result_data: Analysis result
        """
        message = {
            "type": "result",
            "session_id": session_id,
            "data_model": result_data,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.send_message(session_id, message)
    
    async def send_error(self, session_id: str, error_message: str, error_code: Optional[str] = None):
        """
        Send error message to a client.
        
        Args:
            session_id: Session ID
            error_message: Error message
            error_code: Optional error code
        """
        message = {
            "type": "error",
            "session_id": session_id,
            "error": error_message,
            "error_code": error_code,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.send_message(session_id, message)
    
    async def send_complete(self, session_id: str, summary: Optional[Dict[str, Any]] = None):
        """
        Send completion message to a client.
        
        Args:
            session_id: Session ID
            summary: Optional summary data_model
        """
        message = {
            "type": "complete",
            "session_id": session_id,
            "message": "Analysis complete",
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.send_message(session_id, message)
    
    def get_connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)
    
    def is_connected(self, session_id: str) -> bool:
        """Check if a session is connected."""
        return session_id in self.active_connections
    
    def get_user_sessions(self, user_id: str) -> list:
        """Get all session IDs for a user."""
        return [
            session_id for session_id, uid in self.session_users.items()
            if uid == user_id
        ]
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat to all connections."""
        while True:
            try:
                await asyncio.sleep(self._heartbeat_interval)
                
                if not self.active_connections:
                    continue
                
                heartbeat_msg = {
                    "type": "heartbeat",
                    "message": "ping",
                    "timestamp": datetime.now().isoformat()
                }
                
                disconnected = []
                
                for session_id in list(self.active_connections.keys()):
                    try:
                        await self.send_message(session_id, heartbeat_msg)
                    except Exception as e:
                        logger.warning(f"Heartbeat failed for {session_id}: {e}")
                        disconnected.append(session_id)
                
                for session_id in disconnected:
                    await self.disconnect(session_id)
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
    
    async def cleanup_all(self):
        """Disconnect all connections and cleanup."""
        logger.info("Cleaning up all WebSocket connections")
        
        session_ids = list(self.active_connections.keys())
        
        for session_id in session_ids:
            await self.disconnect(session_id)
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None


# Global connection manager instance
connection_manager = ConnectionManager()


async def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager instance."""
    return connection_manager

