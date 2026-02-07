"""
Document Retrieval Service (PageIndex)

Orchestrates PageIndex lookup and optional result merging/ranking.
"""

import asyncio
import logging
import time
from typing import AsyncGenerator, Dict, List, Optional, Any
from pydantic import BaseModel

from services.pageindex_service import PageIndexService, PageIndexLookupRequest, is_pageindex_available

logger = logging.getLogger(__name__)


class HybridLookupRequest(BaseModel):
    """
    Request model for document lookup (PageIndex).
    """
    query: str
    user_id: str
    dataset_id: str
    topic_keywords: Optional[List[str]] = []
    context_window: Optional[int] = 500
    top_k: Optional[int] = 5
    use_pageindex: bool = True


class HybridDocumentResult(BaseModel):
    """
    Model for unified document results from retrieval.
    """
    document_id: str
    content: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    confidence_score: float
    source_file: Optional[str] = None
    source_system: str  # "pageindex"
    retrieval_method: str  # "reasoning"
    reasoning_path: Optional[List[str]] = None
    bbox_coordinates: Optional[List[float]] = None
    metadata: Optional[Dict] = {}


class HybridRetrievalService:
    """
    Service for PageIndex document retrieval with optional merging/ranking.
    """

    def __init__(
        self,
        pageindex_service: PageIndexService = None,
        result_merger=None
    ):
        self.pageindex_service = pageindex_service or PageIndexService()
        self._result_merger = result_merger
        logger.info("HybridRetrievalService initialized (PageIndex only)")

    @property
    def result_merger(self):
        if self._result_merger is None:
            from services.result_merger_service import ResultMergerService
            self._result_merger = ResultMergerService()
        return self._result_merger

    async def lookup(
        self,
        request: HybridLookupRequest
    ) -> List[HybridDocumentResult]:
        """
        Perform document lookup via PageIndex.
        """
        start_time = time.time()
        pageindex_results = []

        if request.use_pageindex and is_pageindex_available():
            pageindex_request = PageIndexLookupRequest(
                query=request.query,
                user_id=request.user_id,
                topic_keywords=request.topic_keywords,
                top_k=request.top_k
            )
            pageindex_results = await self._collect_pageindex_results(pageindex_request)

        if not pageindex_results:
            logger.warning("No retrieval results")
            return []

        merged_results = await self.result_merger.merge(
            query=request.query,
            ragflow_results=[],
            pageindex_results=pageindex_results,
            top_k=request.top_k
        )

        elapsed = time.time() - start_time
        logger.info(f"Lookup completed in {elapsed:.2f}s: pageindex={len(pageindex_results)}, merged={len(merged_results)}")
        return merged_results

    async def _collect_pageindex_results(
        self,
        request: PageIndexLookupRequest
    ) -> List[Dict]:
        """Collect all results from PageIndex lookup."""
        results = []
        async for update in self.pageindex_service.lookup_with_progress(request):
            if update.get("status") == "completed" and "results" in update:
                results = update["results"]
                break
            elif update.get("status") == "error":
                raise Exception(update.get("message", "PageIndex lookup failed"))
        return results

    async def lookup_with_progress(
        self,
        request: HybridLookupRequest,
        user_email: str = None
    ) -> AsyncGenerator[Dict, None]:
        """
        Perform PageIndex lookup with progress updates via async generator.
        """
        start_time = time.time()
        use_pageindex = request.use_pageindex and is_pageindex_available()

        yield {
            "status": "started",
            "message": "Starting document lookup...",
            "mode": "pageindex",
            "systems": {"pageindex": use_pageindex},
            "timestamp": time.time()
        }

        if not use_pageindex:
            yield {
                "status": "error",
                "message": "PageIndex is not available. Set PAGEINDEX_API_KEY.",
                "timestamp": time.time()
            }
            return

        yield {
            "status": "searching",
            "message": "Querying PageIndex...",
            "systems": {
                "pageindex": {"enabled": True, "status": "pending"}
            },
            "progress": 10,
            "timestamp": time.time()
        }

        pageindex_results = []
        pageindex_error = None
        try:
            pageindex_request = PageIndexLookupRequest(
                query=request.query,
                user_id=request.user_id,
                topic_keywords=request.topic_keywords,
                top_k=request.top_k
            )
            pageindex_results = await self._collect_pageindex_results(pageindex_request)
        except Exception as e:
            pageindex_error = str(e)

        yield {
            "status": "retrieving",
            "message": "Retrieval complete, merging results...",
            "systems": {
                "pageindex": {
                    "enabled": True,
                    "status": "error" if pageindex_error else "completed",
                    "result_count": len(pageindex_results),
                    "error": pageindex_error
                }
            },
            "progress": 60,
            "timestamp": time.time()
        }

        if not pageindex_results:
            message = (
                pageindex_error
                or "No results found. If you just uploaded documents, PageIndex may still be processing them. Try again shortly."
            )
            yield {"status": "error", "message": message, "timestamp": time.time()}
            return

        try:
            merged_results = await self.result_merger.merge(
                query=request.query,
                ragflow_results=[],
                pageindex_results=pageindex_results,
                top_k=request.top_k
            )
            result_dicts = []
            for r in merged_results:
                if isinstance(r, HybridDocumentResult):
                    result_dicts.append(r.model_dump())
                elif isinstance(r, dict):
                    result_dicts.append(r)

            elapsed = time.time() - start_time
            yield {
                "status": "completed",
                "message": f"Found {len(result_dicts)} results",
                "progress": 100,
                "results": result_dicts,
                "query_duration_ms": int(elapsed * 1000),
                "retrieval_stats": {
                    "pageindex_results": len(pageindex_results),
                    "merged_results": len(result_dicts),
                    "pageindex_error": pageindex_error
                },
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Result merging failed: {e}", exc_info=True)
            for r in pageindex_results:
                r.setdefault("source_system", "pageindex")
                r.setdefault("retrieval_method", "reasoning")
            fallback = pageindex_results[: request.top_k]
            elapsed = time.time() - start_time
            yield {
                "status": "completed",
                "message": f"Found {len(fallback)} results (fallback)",
                "progress": 100,
                "results": fallback,
                "query_duration_ms": int(elapsed * 1000),
                "merge_error": str(e),
                "timestamp": time.time()
            }
