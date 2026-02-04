"""
Meeting Document Lookup Controller

Handles document lookup during meetings using hybrid RAGFlow + PageIndex integration.
Provides authenticated endpoints with Server-Sent Events for real-time updates.
"""

import asyncio
import json
import logging
import tempfile
import os
from pathlib import Path
from typing import AsyncGenerator, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from utils.auth import get_current_user_email
from services.ragflow_service import RAGFlowService, RAGFlowLookupRequest
from services.hybrid_retrieval_service import HybridRetrievalService, HybridLookupRequest
from services.pageindex_service import PageIndexService, is_pageindex_available
from config.pageindex_config import get_hybrid_config

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
    use_hybrid: bool = True  # Use hybrid retrieval (RAGFlow + PageIndex)
    use_ragflow: bool = True  # Use RAGFlow vector search
    use_pageindex: bool = True  # Use PageIndex reasoning search


class DocumentIndexRequest(BaseModel):
    """
    Request model for document indexing.
    """
    document_id: Optional[str] = None  # Optional custom document ID


@router.post(
    "/document-lookup",
    summary="Lookup relevant documents during meeting",
    description="""
    Performs real-time document lookup based on meeting transcript.
    Uses hybrid retrieval combining RAGFlow (vector-based) and PageIndex (reasoning-based).
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
    
    Uses hybrid retrieval that:
    1. Queries RAGFlow (vector-based) and PageIndex (reasoning-based) in parallel
    2. Merges results using LLM-based ranking
    3. Returns unified, ranked results
    
    Args:
        request: Meeting document lookup request with transcript and keywords
        current_user_email: Authenticated user's email from JWT token
    
    Returns:
        StreamingResponse: Server-Sent Events stream with lookup progress and results
    """
    # Get configuration
    hybrid_config = get_hybrid_config()
    
    # Create user ID from email
    user_id = current_user_email.replace('@', '_').replace('.', '_')
    dataset_id = f"kb_user_{user_id}"
    
    # Determine which retrieval mode to use
    use_hybrid = request.use_hybrid and hybrid_config.enabled
    use_ragflow = request.use_ragflow and hybrid_config.use_ragflow
    use_pageindex = request.use_pageindex and hybrid_config.use_pageindex and is_pageindex_available()
    
    # If hybrid mode with both systems available
    if use_hybrid and (use_ragflow or use_pageindex):
        return await _hybrid_document_lookup(
            request=request,
            user_id=user_id,
            dataset_id=dataset_id,
            current_user_email=current_user_email,
            use_ragflow=use_ragflow,
            use_pageindex=use_pageindex
        )
    else:
        # Fallback to RAGFlow only
        return await _ragflow_only_lookup(
            request=request,
            dataset_id=dataset_id,
            current_user_email=current_user_email
        )


async def _hybrid_document_lookup(
    request: MeetingDocumentLookupRequest,
    user_id: str,
    dataset_id: str,
    current_user_email: str,
    use_ragflow: bool,
    use_pageindex: bool
) -> StreamingResponse:
    """Perform hybrid document lookup using both RAGFlow and PageIndex."""
    
    hybrid_service = HybridRetrievalService()
    
    async def event_generator():
        try:
            # Send initial event
            yield f"data: {json.dumps({'status': 'started', 'message': 'Starting hybrid document lookup...', 'mode': 'hybrid', 'systems': {'ragflow': use_ragflow, 'pageindex': use_pageindex}, 'timestamp': asyncio.get_event_loop().time()})}\n\n"
            
            # Prepare hybrid lookup request
            lookup_request = HybridLookupRequest(
                query=request.meeting_transcript,
                user_id=user_id,
                dataset_id=dataset_id,
                topic_keywords=request.topic_keywords,
                context_window=request.context_window,
                top_k=request.top_k,
                use_ragflow=use_ragflow,
                use_pageindex=use_pageindex
            )
            
            # Perform hybrid lookup with progress updates
            async for result in hybrid_service.lookup_with_progress(lookup_request, current_user_email):
                yield f"data: {json.dumps(result)}\n\n"
                
        except Exception as e:
            logger.error(f"Error in hybrid document lookup: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'status': 'error', 'message': f'Lookup failed: {str(e)}', 'timestamp': asyncio.get_event_loop().time()})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )


async def _ragflow_only_lookup(
    request: MeetingDocumentLookupRequest,
    dataset_id: str,
    current_user_email: str
) -> StreamingResponse:
    """Fallback to RAGFlow-only document lookup."""
    
    ragflow_service = RAGFlowService()
    
    async def event_generator():
        try:
            yield f"data: {json.dumps({'status': 'started', 'message': 'Starting document lookup (RAGFlow only)...', 'mode': 'ragflow', 'timestamp': asyncio.get_event_loop().time()})}\n\n"
            
            lookup_request = RAGFlowLookupRequest(
                query=request.meeting_transcript,
                dataset_ids=[dataset_id],
                topic_keywords=request.topic_keywords,
                context_window=request.context_window,
                top_k=request.top_k
            )
            
            yield f"data: {json.dumps({'status': 'searching', 'message': 'Searching relevant documents...', 'timestamp': asyncio.get_event_loop().time()})}\n\n"
            
            async for result in ragflow_service.lookup_with_progress(lookup_request, current_user_email):
                yield f"data: {json.dumps(result)}\n\n"
                
        except Exception as e:
            logger.error(f"Error in RAGFlow document lookup: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'status': 'error', 'message': f'Lookup failed: {str(e)}', 'timestamp': asyncio.get_event_loop().time()})}\n\n"
        finally:
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


@router.post(
    "/index-document",
    summary="Index document for PageIndex reasoning-based search",
    description="""
    Builds a PageIndex tree structure for a document, enabling reasoning-based search.
    This runs in addition to RAGFlow indexing for hybrid retrieval.
    """
)
async def index_document_for_pageindex(
    document: UploadFile = File(..., description="PDF document to index"),
    document_id: Optional[str] = Form(None, description="Optional custom document ID"),
    current_user_email: str = Depends(get_current_user_email)
):
    """
    Index a document for PageIndex reasoning-based retrieval.
    
    This creates a hierarchical tree structure from the document that enables
    LLM reasoning-based search through the document structure.
    
    Args:
        document: PDF document to index
        document_id: Optional custom document ID (defaults to filename)
        current_user_email: Authenticated user's email from JWT token
    
    Returns:
        Dictionary with indexing results
    """
    if not is_pageindex_available():
        raise HTTPException(
            status_code=503,
            detail="PageIndex is not available. Please ensure it's installed and enabled."
        )
    
    # Validate file type
    if not document.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF documents are supported for PageIndex indexing"
        )
    
    user_id = current_user_email.replace('@', '_').replace('.', '_')
    pageindex_service = PageIndexService()
    
    # Save uploaded file temporarily
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await document.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Use filename as document_id if not provided
        doc_id = document_id or Path(document.filename).stem
        
        logger.info(f"Indexing document '{doc_id}' for user '{user_id}'")
        
        # Build PageIndex tree
        result = await pageindex_service.index_document(
            document_path=temp_path,
            user_id=user_id,
            document_id=doc_id
        )
        
        return {
            "success": True,
            "message": f"Document '{doc_id}' indexed successfully for PageIndex",
            "document_id": result["document_id"],
            "user_id": user_id,
            "node_count": result.get("node_count", 0),
            "indexing_time": result.get("indexing_time", 0)
        }
        
    except Exception as e:
        logger.error(f"Failed to index document: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to index document: {str(e)}")
    
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@router.delete(
    "/index-document/{document_id}",
    summary="Delete PageIndex tree for a document",
    description="Removes the PageIndex tree structure for a specific document"
)
async def delete_document_pageindex(
    document_id: str,
    current_user_email: str = Depends(get_current_user_email)
):
    """
    Delete a document's PageIndex tree.
    
    Args:
        document_id: Document ID to delete
        current_user_email: Authenticated user's email from JWT token
    
    Returns:
        Dictionary with deletion result
    """
    user_id = current_user_email.replace('@', '_').replace('.', '_')
    pageindex_service = PageIndexService()
    
    deleted = await pageindex_service.delete_document_tree(user_id, document_id)
    
    if deleted:
        return {
            "success": True,
            "message": f"PageIndex tree for '{document_id}' deleted successfully"
        }
    else:
        raise HTTPException(
            status_code=404,
            detail=f"No PageIndex tree found for document '{document_id}'"
        )


@router.get(
    "/user-dataset-info",
    summary="Get user's dataset information",
    description="Returns information about the user's document datasets in both RAGFlow and PageIndex"
)
async def get_user_dataset_info(
    current_user_email: str = Depends(get_current_user_email)
):
    """
    Get information about the user's document datasets.
    
    Args:
        current_user_email: Authenticated user's email from JWT token
    
    Returns:
        Dictionary with dataset information from both RAGFlow and PageIndex
    """
    user_id = current_user_email.replace('@', '_').replace('.', '_')
    dataset_id = f"kb_user_{user_id}"
    
    result = {
        "user_id": user_id,
        "user_email": current_user_email,
        "dataset_id": dataset_id,
        "ragflow": None,
        "pageindex": None
    }
    
    # Get RAGFlow info
    try:
        ragflow_service = RAGFlowService()
        ragflow_info = await ragflow_service.get_dataset_info(dataset_id)
        result["ragflow"] = {
            "available": True,
            "info": ragflow_info
        }
    except Exception as e:
        logger.error(f"Error getting RAGFlow dataset info: {str(e)}")
        result["ragflow"] = {
            "available": False,
            "error": str(e)
        }
    
    # Get PageIndex info
    if is_pageindex_available():
        try:
            pageindex_service = PageIndexService()
            pageindex_stats = await pageindex_service.get_user_stats(user_id)
            result["pageindex"] = {
                "available": True,
                "enabled": True,
                "stats": pageindex_stats
            }
        except Exception as e:
            logger.error(f"Error getting PageIndex stats: {str(e)}")
            result["pageindex"] = {
                "available": True,
                "enabled": True,
                "error": str(e)
            }
    else:
        result["pageindex"] = {
            "available": False,
            "enabled": False,
            "message": "PageIndex is not installed or disabled"
        }
    
    return result


@router.get(
    "/retrieval-status",
    summary="Get retrieval system status",
    description="Returns the status and availability of all retrieval systems"
)
async def get_retrieval_status():
    """
    Get status of all retrieval systems.
    
    Returns:
        Dictionary with system statuses
    """
    hybrid_config = get_hybrid_config()
    
    return {
        "hybrid_retrieval": {
            "enabled": hybrid_config.enabled,
            "merge_strategy": hybrid_config.merge_strategy
        },
        "ragflow": {
            "enabled": hybrid_config.use_ragflow,
            "available": True  # RAGFlow is always available if configured
        },
        "pageindex": {
            "enabled": hybrid_config.use_pageindex,
            "available": is_pageindex_available()
        }
    }
