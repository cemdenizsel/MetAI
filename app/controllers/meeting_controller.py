"""
Meeting Document Lookup Controller

Handles document lookup during meetings using PageIndex.
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
from services.hybrid_retrieval_service import HybridRetrievalService, HybridLookupRequest
from services.pageindex_service import PageIndexService, is_pageindex_available
from config.pageindex_config import get_hybrid_config

router = APIRouter(prefix="/meeting", tags=["Meeting Document Lookup"])

logger = logging.getLogger(__name__)


class MeetingDocumentLookupRequest(BaseModel):
    """
    Request model for meeting document lookup (PageIndex).
    """
    meeting_transcript: str
    topic_keywords: list[str] = []
    context_window: int = 500
    top_k: int = 5
    use_pageindex: bool = True


class DocumentIndexRequest(BaseModel):
    """
    Request model for document indexing.
    """
    document_id: Optional[str] = None  # Optional custom document ID


class AskRequest(BaseModel):
    """
    Request model for RAG-first chat (ask a question; answer from docs or OpenAI).
    """
    message: str


@router.post(
    "/ask",
    summary="Ask a question (RAG-first, then OpenAI fallback)",
    description="""
    Sends the user's question to the backend. The backend searches the user's
    indexed documents (PageIndex). If relevant chunks are found, the answer
    is generated from them (RAG). Otherwise the answer is from OpenAI only.
    Returns answer, source ('rag' or 'openai'), and optional citations.
    """
)
async def meeting_ask(
    body: AskRequest,
    current_user_email: str = Depends(get_current_user_email)
):
    """
    Answer a user question: PageIndex Chat API when PageIndex is available; else retrieval + OpenAI or OpenAI only.
    """
    user_id = current_user_email.replace('@', '_').replace('.', '_')
    dataset_id = f"kb_user_{user_id}"
    hybrid_config = get_hybrid_config()
    use_pageindex = hybrid_config.use_pageindex and is_pageindex_available()
    top_k = 5

    # PageIndex: use Chat API over all user docs (one call, no retrieval + OpenAI)
    if use_pageindex:
        pageindex_service = PageIndexService()
        chat_result = await pageindex_service.ask_chat(user_id, body.message)
        if chat_result and chat_result.get("answer"):
            # Resolve citation doc ref (filename/api_doc_id) to user-facing document_id
            user_docs = (await pageindex_service.get_user_stats(user_id)).get("documents", [])
            doc_by_filename = {d.get("filename"): d.get("document_id") for d in user_docs if d.get("filename")}
            doc_by_api_id = {d.get("api_doc_id"): d.get("document_id") for d in user_docs if d.get("api_doc_id")}

            def _resolve_document_id(c):
                ref = c.get("document_id") or c.get("doc_id") or ""
                display_id = doc_by_filename.get(ref) or doc_by_api_id.get(ref) or ref
                return {
                    "document_id": display_id,
                    "content_snippet": (c.get("content_snippet") or c.get("content") or "")[:200],
                    "page_number": c.get("page_number"),
                    "source_system": "pageindex",
                }

            citations = [_resolve_document_id(c) for c in chat_result.get("citations", [])]
            return {"answer": chat_result["answer"], "source": "rag", "citations": citations}

    # Fallback: retrieval + OpenAI when PageIndex is on
    chunks = []
    if use_pageindex:
        try:
            hybrid_service = HybridRetrievalService()
            lookup_request = HybridLookupRequest(
                query=body.message,
                user_id=user_id,
                dataset_id=dataset_id,
                top_k=top_k,
                use_pageindex=use_pageindex,
            )
            results = await hybrid_service.lookup(lookup_request)
            if results:
                chunks = [r.model_dump() if hasattr(r, 'model_dump') else r for r in results]
        except Exception as e:
            logger.warning(f"Retrieval failed for ask: {e}")

    # 3. Build prompt and call OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except ImportError:
        raise HTTPException(status_code=503, detail="OpenAI library not available")

    if chunks:
        # RAG: answer from context
        context_parts = []
        for i, c in enumerate(chunks[:5], 1):
            content = c.get("content", "")[:800]
            doc_id = c.get("document_id", "")
            page = c.get("page_number")
            context_parts.append(f"[{i}] (doc: {doc_id}" + (f", p.{page}" if page is not None else "") + f")\n{content}")
        context_text = "\n\n".join(context_parts)
        system = (
            "You are a helpful assistant. Answer the user's question using ONLY the following excerpts from their documents. "
            "If the answer is not in the excerpts, say so clearly. Keep answers concise. Cite which excerpt (number) when relevant."
        )
        user_content = f"Context from the user's documents:\n\n{context_text}\n\nUser question: {body.message}"
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            max_tokens=1000,
        )
        answer = response.choices[0].message.content if response.choices else ""
        citations = [
            {
                "document_id": c.get("document_id"),
                "content_snippet": (c.get("content") or "")[:200],
                "page_number": c.get("page_number"),
                "source_system": c.get("source_system"),
            }
            for c in chunks[:5]
        ]
        return {"answer": answer, "source": "rag", "citations": citations}
    else:
        # Fallback: answer from OpenAI only
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": body.message}],
            max_tokens=1000,
        )
        answer = response.choices[0].message.content if response.choices else ""
        return {"answer": answer, "source": "openai"}


@router.post(
    "/document-lookup",
    summary="Lookup relevant documents during meeting",
    description="""
    Performs document lookup based on meeting transcript using PageIndex.
    Returns results via Server-Sent Events for progressive loading.
    """,
    response_class=StreamingResponse
)
async def meeting_document_lookup(
    request: MeetingDocumentLookupRequest,
    current_user_email: str = Depends(get_current_user_email)
):
    """
    Lookup relevant documents during a meeting (PageIndex).
    """
    hybrid_config = get_hybrid_config()
    user_id = current_user_email.replace('@', '_').replace('.', '_')
    dataset_id = f"kb_user_{user_id}"
    use_pageindex = request.use_pageindex and hybrid_config.use_pageindex and is_pageindex_available()

    if not use_pageindex:
        async def no_system_stream():
            yield f"data: {json.dumps({'status': 'error', 'message': 'PageIndex is not enabled. Set PAGEINDEX_API_KEY.', 'timestamp': asyncio.get_event_loop().time()})}\n\n"
        return StreamingResponse(
            no_system_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "Access-Control-Allow-Origin": "*"},
        )

    hybrid_service = HybridRetrievalService()
    lookup_request = HybridLookupRequest(
        query=request.meeting_transcript,
        user_id=user_id,
        dataset_id=dataset_id,
        topic_keywords=request.topic_keywords,
        context_window=request.context_window,
        top_k=request.top_k,
        use_pageindex=True,
    )

    async def event_generator():
        try:
            async for result in hybrid_service.lookup_with_progress(lookup_request, current_user_email):
                yield f"data: {json.dumps(result)}\n\n"
        except Exception as e:
            logger.error(f"Error in document lookup: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'status': 'error', 'message': f'Lookup failed: {str(e)}', 'timestamp': asyncio.get_event_loop().time()})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "Access-Control-Allow-Origin": "*"},
    )


@router.post(
    "/index-document",
    summary="Index document for PageIndex reasoning-based search",
    description="Builds a PageIndex tree structure for a document, enabling reasoning-based search."
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
    description="Returns information about the user's document datasets (PageIndex)."
)
async def get_user_dataset_info(
    current_user_email: str = Depends(get_current_user_email)
):
    """
    Get information about the user's document datasets (PageIndex).
    """
    user_id = current_user_email.replace('@', '_').replace('.', '_')
    dataset_id = f"kb_user_{user_id}"
    result = {
        "user_id": user_id,
        "user_email": current_user_email,
        "dataset_id": dataset_id,
        "pageindex": None
    }

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
    description="Returns the status of PageIndex retrieval."
)
async def get_retrieval_status():
    """Get status of PageIndex retrieval."""
    hybrid_config = get_hybrid_config()
    return {
        "retrieval": {
            "enabled": hybrid_config.enabled,
            "merge_strategy": hybrid_config.merge_strategy
        },
        "pageindex": {
            "enabled": hybrid_config.use_pageindex,
            "available": is_pageindex_available()
        }
    }
