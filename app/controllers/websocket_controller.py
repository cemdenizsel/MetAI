"""
WebSocket Controller

WebSocket endpoints for real-time emotion analysis.
"""

import logging
import json
import os
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends, status
from fastapi.responses import JSONResponse
from jose import JWTError, jwt

from services.websocket_manager import connection_manager
from services.realtime_service import RealtimeAnalysisService
from services.user_service import get_user_service, UserService
from utils.auth import SECRET_KEY, ALGORITHM

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["WebSocket"])


async def verify_websocket_token(token: str, user_service: UserService) -> Optional[dict]:
    """
    Verify JWT token for WebSocket authentication.
    
    Uses the same JWT verification method as auth_controller.
    
    Args:
        token: JWT token string
        user_service: User service instance
        
    Returns:
        User document if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        
        if not email:
            logger.warning("Token missing 'sub' claim")
            return None
        
        user_doc = await user_service.get_user_by_email(email)
        return user_doc
        
    except JWTError as e:
        logger.error(f"JWT verification failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        return None


@router.websocket("/analyze")
async def websocket_analyze(
    websocket: WebSocket,
    token: str = Query(..., description="JWT authentication token"),
    user_service: UserService = Depends(get_user_service)
):
    """
    WebSocket endpoint for real-time emotion analysis.
    
    **Authentication**: Requires JWT token in query parameter
    
    **Protocol**:
    1. Client connects with JWT token
    2. Server validates and creates session
    3. Server sends: {"type": "connected", "session_id": "..."}
    4. Client sends chunks: {"type": "chunk", "data_model": "base64...", "timestamp": 0.0, "chunk_index": 0}
    5. Server processes and sends: {"type": "result", "data_model": {...}}
    6. Client sends: {"type": "complete"}
    7. Server sends: {"type": "complete", "summary": {...}}
    8. Connection closes
    
    **Message Types from Client**:
    - `chunk`: Video chunk data_model (base64 or raw binary)
    - `complete`: Finalize analysis
    - `status`: Request session status
    - `ping`: Heartbeat ping
    
    **Message Types from Server**:
    - `connected`: Connection established
    - `result`: Chunk processing result
    - `progress`: Processing progress
    - `complete`: Analysis complete with summary
    - `error`: Error occurred
    - `heartbeat`: Keepalive ping
    - `pong`: Response to ping
    
    Args:
        websocket: WebSocket connection
        token: JWT authentication token
        user_service: User service dependency
    """
    session_id = None
    
    try:
        user_doc = await verify_websocket_token(token, user_service)
        
        if not user_doc:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token")
            return
        
        user_id = str(user_doc['_id'])
        user_email = user_doc.get('email', 'unknown')
        
        logger.info(f"WebSocket connection attempt from user: {user_email}")
        
        realtime_service = RealtimeAnalysisService()
        
        try:
            session_id = await realtime_service.create_session(user_id)
        except Exception as e:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=str(e))
            logger.error(f"Session creation failed: {e}")
            return
        
        await connection_manager.connect(websocket, session_id, user_id)
        
        logger.info(f"WebSocket connected: user={user_email}, session={session_id}")
        
        while True:
            try:
                message = await websocket.receive()
                
                if "text" in message:
                    data = json.loads(message["text"])
                    await handle_text_message(data, session_id, realtime_service)
                    
                elif "bytes" in message:
                    chunk_data = message["bytes"]
                    await handle_binary_chunk(
                        chunk_data, session_id, realtime_service
                    )
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: session={session_id}")
                break
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
                await connection_manager.send_error(
                    session_id,
                    "Invalid JSON format",
                    "INVALID_JSON"
                )
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                await connection_manager.send_error(
                    session_id,
                    str(e),
                    "PROCESSING_ERROR"
                )
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    
    finally:
        if session_id:
            await connection_manager.disconnect(session_id)
            try:
                await realtime_service.delete_session(session_id)
            except:
                pass
        
        logger.info(f"WebSocket session ended: {session_id}")


async def handle_text_message(
    data: dict,
    session_id: str,
    realtime_service: RealtimeAnalysisService
):
    """
    Handle text/JSON message from client.
    
    Args:
        data: Message data_model dictionary
        session_id: Session ID
        realtime_service: Real-time service instance
    """
    message_type = data.get("type", "unknown")
    
    if message_type == "chunk":
        chunk_b64 = data.get("data_model")
        timestamp = data.get("timestamp", 0.0)
        chunk_index = data.get("chunk_index", 0)
        
        if not chunk_b64:
            await connection_manager.send_error(
                session_id,
                "Missing chunk data_model",
                "MISSING_DATA"
            )
            return
        
        try:
            result = await realtime_service.process_chunk(
                session_id=session_id,
                chunk_data=chunk_b64.encode() if isinstance(chunk_b64, str) else chunk_b64,
                timestamp=timestamp,
                chunk_index=chunk_index,
                encoding="base64"
            )
            
            await connection_manager.send_result(session_id, result)
            
        except Exception as e:
            logger.error(f"Chunk processing error: {e}")
            await connection_manager.send_error(
                session_id,
                f"Chunk processing failed: {str(e)}",
                "CHUNK_ERROR"
            )
    
    elif message_type == "complete":
        try:
            summary = await realtime_service.finalize_session(session_id)
            await connection_manager.send_complete(session_id, summary)
        except Exception as e:
            logger.error(f"Finalization error: {e}")
            await connection_manager.send_error(
                session_id,
                f"Finalization failed: {str(e)}",
                "FINALIZE_ERROR"
            )
    
    elif message_type == "status":
        try:
            status = await realtime_service.get_session_status(session_id)
            await connection_manager.send_message(session_id, {
                "type": "status",
                "data_model": status
            })
        except Exception as e:
            logger.error(f"Status error: {e}")
            await connection_manager.send_error(
                session_id,
                f"Status check failed: {str(e)}",
                "STATUS_ERROR"
            )
    
    elif message_type == "ping":
        await connection_manager.send_message(session_id, {
            "type": "pong"
        })
    
    else:
        logger.warning(f"Unknown message type: {message_type}")
        await connection_manager.send_error(
            session_id,
            f"Unknown message type: {message_type}",
            "UNKNOWN_TYPE"
        )


async def handle_binary_chunk(
    chunk_data: bytes,
    session_id: str,
    realtime_service: RealtimeAnalysisService,
    timestamp: float = 0.0,
    chunk_index: int = 0
):
    """
    Handle binary chunk data_model from client.
    
    Args:
        chunk_data: Binary chunk data_model
        session_id: Session ID
        realtime_service: Real-time service instance
        timestamp: Timestamp
        chunk_index: Chunk index
    """
    try:
        result = await realtime_service.process_chunk(
            session_id=session_id,
            chunk_data=chunk_data,
            timestamp=timestamp,
            encoding="raw"
        )
            
        await connection_manager.send_result(session_id, result)
        
    except Exception as e:
        logger.error(f"Binary chunk error: {e}")
        await connection_manager.send_error(
            session_id,
            f"Chunk processing failed: {str(e)}",
            "CHUNK_ERROR"
        )


@router.get(
    "/health",
    summary="WebSocket Health Check",
    description="Check WebSocket service health",
    response_model=dict
)
async def websocket_health():
    """
    Health check for WebSocket service.
    
    Returns:
        Health status information
    """
    return {
        "status": "healthy",
        "service": "WebSocket Real-time Analysis",
        "version": "1.0.0",
        "active_connections": connection_manager.get_connection_count()
    }


@router.get(
    "/sessions/active",
    summary="Get Active Sessions",
    description="Get information about active WebSocket sessions",
    response_model=dict
)
async def get_active_sessions():
    """
    Get active WebSocket sessions.
    
    Returns:
        Active session information
    """
    realtime_service = RealtimeAnalysisService()
    
    return {
        "active_sessions": realtime_service.get_active_session_count(),
        "max_sessions": realtime_service.get_session_limit(),
        "connections": connection_manager.get_connection_count()
    }

