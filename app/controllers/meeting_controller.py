"""
Meeting Document Lookup Controller

Handles document lookup during meetings using RAGFlow integration.
Provides authenticated endpoints with Server-Sent Events for real-time updates.
"""

import asyncio
import json
import logging
from typing import AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from utils.auth import get_current_user_email
from services.ragflow_service import RAGFlowService, RAGFlowLookupRequest

router = APIRouter(prefix="/meeting", tags=["Meeting Document Lookup"])

logger = logging.getLogger(__name__)


class MeetingDocumentLookupRequest(BaseModel):
    """
    Request model for meeting document lookup.
    """
    meeting_transcript: str
    topic_keywords: list[str] = []
    context_window: int = 500
    top_k: int = 5


@router.post(
    "/document-lookup",
    summary="Lookup relevant documents during meeting",
    description="""
    Performs real-time document lookup based on meeting transcript.
    Returns results via Server-Sent Events for progressive loading.
    """,
    response_class=StreamingResponse
)
async def meeting_document_lookup(
    request: MeetingDocumentLookupRequest,
    current_user_email: str = Depends(get_current_user_email)
):
    """
    Lookup relevant documents during a meeting based on transcript and keywords.
    
    Args:
        request: Meeting document lookup request with transcript and keywords
        current_user_email: Authenticated user's email from JWT token
    
    Returns:
        StreamingResponse: Server-Sent Events stream with lookup progress and results
    """
    # Initialize RAGFlow service
    ragflow_service = RAGFlowService()
    
    # Create dataset ID based on user email (extract user ID from email)
    user_id = current_user_email.replace('@', '_').replace('.', '_')  # Simple transformation
    dataset_id = f"kb_user_{user_id}"
    
    async def event_generator():
        """
        Generator that yields Server-Sent Events with lookup progress and results.
        """
        try:
            # Send initial event
            yield f"data: {json.dumps({'status': 'started', 'message': 'Starting document lookup...', 'timestamp': asyncio.get_event_loop().time()})}\n\n"
            
            # Prepare the lookup request
            lookup_request = RAGFlowLookupRequest(
                query=request.meeting_transcript,
                dataset_ids=[dataset_id],
                topic_keywords=request.topic_keywords,
                context_window=request.context_window,
                top_k=request.top_k
            )
            
            # Send search initiation event
            yield f"data: {json.dumps({'status': 'searching', 'message': 'Searching relevant documents...', 'timestamp': asyncio.get_event_loop().time()})}\n\n"
            
            # Perform the lookup with progress updates
            async for result in ragflow_service.lookup_with_progress(lookup_request, current_user_email):
                yield f"data: {json.dumps(result)}\n\n"
                
        except Exception as e:
            logger.error(f"Error in meeting document lookup: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'status': 'error', 'message': f'Lookup failed: {str(e)}', 'timestamp': asyncio.get_event_loop().time()})}\n\n"
        finally:
            # Send completion event
            yield f"data: {json.dumps({'status': 'completed', 'message': 'Document lookup completed', 'timestamp': asyncio.get_event_loop().time()})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )


@router.get(
    "/user-dataset-info",
    summary="Get user's dataset information",
    description="Returns information about the user's document dataset in RAGFlow"
)
async def get_user_dataset_info(
    current_user_email: str = Depends(get_current_user_email)
):
    """
    Get information about the user's document dataset in RAGFlow.
    
    Args:
        current_user_email: Authenticated user's email from JWT token
    
    Returns:
        Dictionary with dataset information
    """
    user_id = current_user_email.replace('@', '_').replace('.', '_')
    dataset_id = f"kb_user_{user_id}"
    
    ragflow_service = RAGFlowService()
    
    try:
        dataset_info = await ragflow_service.get_dataset_info(dataset_id)
        return {
            "dataset_id": dataset_id,
            "user_email": current_user_email,
            "info": dataset_info
        }
    except Exception as e:
        logger.error(f"Error getting dataset info: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get dataset info: {str(e)}")