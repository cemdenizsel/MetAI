"""
Hybrid Retrieval Service

Orchestrates parallel queries to both RAGFlow (vector-based) and PageIndex (reasoning-based)
retrieval systems, then merges results using LLM-based ranking.
"""

import asyncio
import json
import logging
import time
from typing import AsyncGenerator, Dict, List, Optional, Any
from pydantic import BaseModel

from services.ragflow_service import RAGFlowService, RAGFlowLookupRequest
from services.pageindex_service import PageIndexService, PageIndexLookupRequest, is_pageindex_available

logger = logging.getLogger(__name__)


class HybridLookupRequest(BaseModel):
    """
    Request model for hybrid lookup operations.
    """
    query: str
    user_id: str
    dataset_id: str
    topic_keywords: Optional[List[str]] = []
    context_window: Optional[int] = 500
    top_k: Optional[int] = 5
    use_ragflow: bool = True
    use_pageindex: bool = True


class HybridDocumentResult(BaseModel):
    """
    Model for unified document results from hybrid retrieval.
    """
    document_id: str
    content: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    confidence_score: float
    source_file: Optional[str] = None
    source_system: str  # "ragflow", "pageindex", or "both"
    retrieval_method: str  # "vector", "reasoning", or "hybrid"
    reasoning_path: Optional[List[str]] = None
    bbox_coordinates: Optional[List[float]] = None
    metadata: Optional[Dict] = {}


class HybridRetrievalService:
    """
    Service that combines RAGFlow and PageIndex for optimal document retrieval.
    
    Features:
    - Parallel querying of both systems
    - Intelligent result merging via LLM
    - Fallback handling when one system fails
    - Progress streaming via SSE
    """

    def __init__(
        self,
        ragflow_service: RAGFlowService = None,
        pageindex_service: PageIndexService = None,
        result_merger = None  # ResultMergerService, imported later to avoid circular imports
    ):
        self.ragflow_service = ragflow_service or RAGFlowService()
        self.pageindex_service = pageindex_service or PageIndexService()
        self._result_merger = result_merger
        
        logger.info("HybridRetrievalService initialized")

    @property
    def result_merger(self):
        """Lazy load result merger to avoid circular imports."""
        if self._result_merger is None:
            from services.result_merger_service import ResultMergerService
            self._result_merger = ResultMergerService()
        return self._result_merger

    async def lookup(
        self,
        request: HybridLookupRequest
    ) -> List[HybridDocumentResult]:
        """
        Perform hybrid lookup by querying both systems in parallel.
        
        Args:
            request: Hybrid lookup request
            
        Returns:
            List of merged and ranked results
        """
        start_time = time.time()
        
        ragflow_results = []
        pageindex_results = []
        errors = []
        
        # Prepare tasks for parallel execution
        tasks = []
        task_names = []
        
        if request.use_ragflow:
            ragflow_request = RAGFlowLookupRequest(
                query=request.query,
                dataset_ids=[request.dataset_id],
                topic_keywords=request.topic_keywords,
                context_window=request.context_window,
                top_k=request.top_k
            )
            tasks.append(self._collect_ragflow_results(ragflow_request, request.user_id))
            task_names.append("ragflow")
        
        if request.use_pageindex and is_pageindex_available():
            pageindex_request = PageIndexLookupRequest(
                query=request.query,
                user_id=request.user_id,
                topic_keywords=request.topic_keywords,
                top_k=request.top_k
            )
            tasks.append(self._collect_pageindex_results(pageindex_request))
            task_names.append("pageindex")
        
        if not tasks:
            logger.warning("No retrieval systems available")
            return []
        
        # Execute in parallel
        logger.info(f"Executing parallel retrieval: {task_names}")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            task_name = task_names[i]
            
            if isinstance(result, Exception):
                logger.error(f"{task_name} failed: {result}")
                errors.append({"system": task_name, "error": str(result)})
            elif task_name == "ragflow":
                ragflow_results = result
            elif task_name == "pageindex":
                pageindex_results = result
        
        # Merge results using LLM
        merged_results = await self.result_merger.merge(
            query=request.query,
            ragflow_results=ragflow_results,
            pageindex_results=pageindex_results,
            top_k=request.top_k
        )
        
        elapsed = time.time() - start_time
        logger.info(
            f"Hybrid lookup completed in {elapsed:.2f}s: "
            f"ragflow={len(ragflow_results)}, pageindex={len(pageindex_results)}, "
            f"merged={len(merged_results)}"
        )
        
        return merged_results

    async def _collect_ragflow_results(
        self,
        request: RAGFlowLookupRequest,
        user_email: str
    ) -> List[Dict]:
        """Collect all results from RAGFlow lookup."""
        results = []
        
        async for update in self.ragflow_service.lookup_with_progress(request, user_email):
            if update.get("status") == "completed" and "results" in update:
                results = update["results"]
                break
            elif update.get("status") == "error":
                raise Exception(update.get("message", "RAGFlow lookup failed"))
        
        return results

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
        Perform hybrid lookup with progress updates via async generator.
        
        Args:
            request: Hybrid lookup request
            user_email: Optional user email for logging
            
        Yields:
            Dictionary with progress updates and final results
        """
        start_time = time.time()
        
        # Initial status
        yield {
            "status": "started",
            "message": "Starting hybrid document lookup...",
            "timestamp": time.time()
        }
        
        ragflow_results = []
        pageindex_results = []
        ragflow_done = False
        pageindex_done = False
        ragflow_error = None
        pageindex_error = None
        
        # Check which systems are available
        use_ragflow = request.use_ragflow
        use_pageindex = request.use_pageindex and is_pageindex_available()
        
        if not use_ragflow and not use_pageindex:
            yield {
                "status": "error",
                "message": "No retrieval systems available",
                "timestamp": time.time()
            }
            return
        
        # Create progress updates for parallel execution
        yield {
            "status": "searching",
            "message": f"Querying retrieval systems in parallel...",
            "systems": {
                "ragflow": {"enabled": use_ragflow, "status": "pending"},
                "pageindex": {"enabled": use_pageindex, "status": "pending"}
            },
            "progress": 10,
            "timestamp": time.time()
        }
        
        # Prepare and run tasks in parallel
        async def run_ragflow():
            nonlocal ragflow_results, ragflow_done, ragflow_error
            try:
                ragflow_request = RAGFlowLookupRequest(
                    query=request.query,
                    dataset_ids=[request.dataset_id],
                    topic_keywords=request.topic_keywords,
                    context_window=request.context_window,
                    top_k=request.top_k
                )
                ragflow_results = await self._collect_ragflow_results(ragflow_request, user_email)
                ragflow_done = True
            except Exception as e:
                ragflow_error = str(e)
                ragflow_done = True
        
        async def run_pageindex():
            nonlocal pageindex_results, pageindex_done, pageindex_error
            try:
                pageindex_request = PageIndexLookupRequest(
                    query=request.query,
                    user_id=request.user_id,
                    topic_keywords=request.topic_keywords,
                    top_k=request.top_k
                )
                pageindex_results = await self._collect_pageindex_results(pageindex_request)
                pageindex_done = True
            except Exception as e:
                pageindex_error = str(e)
                pageindex_done = True
        
        # Start parallel tasks
        tasks = []
        if use_ragflow:
            tasks.append(asyncio.create_task(run_ragflow()))
        else:
            ragflow_done = True
            
        if use_pageindex:
            tasks.append(asyncio.create_task(run_pageindex()))
        else:
            pageindex_done = True
        
        # Wait for all tasks
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Report individual system results
        yield {
            "status": "retrieving",
            "message": "Retrieval complete, merging results...",
            "systems": {
                "ragflow": {
                    "enabled": use_ragflow,
                    "status": "error" if ragflow_error else "completed",
                    "result_count": len(ragflow_results),
                    "error": ragflow_error
                },
                "pageindex": {
                    "enabled": use_pageindex,
                    "status": "error" if pageindex_error else "completed",
                    "result_count": len(pageindex_results),
                    "error": pageindex_error
                }
            },
            "progress": 60,
            "timestamp": time.time()
        }
        
        # Handle complete failure
        if not ragflow_results and not pageindex_results:
            error_msg = []
            if ragflow_error:
                error_msg.append(f"RAGFlow: {ragflow_error}")
            if pageindex_error:
                error_msg.append(f"PageIndex: {pageindex_error}")
            
            yield {
                "status": "error",
                "message": f"All retrieval systems failed: {'; '.join(error_msg) if error_msg else 'No results found'}",
                "timestamp": time.time()
            }
            return
        
        # Merge results
        try:
            merged_results = await self.result_merger.merge(
                query=request.query,
                ragflow_results=ragflow_results,
                pageindex_results=pageindex_results,
                top_k=request.top_k
            )
            
            # Convert to serializable format
            result_dicts = []
            for r in merged_results:
                if isinstance(r, HybridDocumentResult):
                    result_dicts.append(r.model_dump())
                elif isinstance(r, dict):
                    result_dicts.append(r)
            
            elapsed = time.time() - start_time
            
            yield {
                "status": "completed",
                "message": f"Found {len(result_dicts)} results via hybrid search",
                "progress": 100,
                "results": result_dicts,
                "query_duration_ms": int(elapsed * 1000),
                "retrieval_stats": {
                    "ragflow_results": len(ragflow_results),
                    "pageindex_results": len(pageindex_results),
                    "merged_results": len(result_dicts),
                    "ragflow_error": ragflow_error,
                    "pageindex_error": pageindex_error
                },
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Result merging failed: {e}", exc_info=True)
            
            # Fallback: return concatenated results without LLM ranking
            fallback_results = self._fallback_merge(ragflow_results, pageindex_results, request.top_k)
            
            elapsed = time.time() - start_time
            
            yield {
                "status": "completed",
                "message": f"Found {len(fallback_results)} results (fallback merge, LLM ranking failed)",
                "progress": 100,
                "results": fallback_results,
                "query_duration_ms": int(elapsed * 1000),
                "merge_error": str(e),
                "timestamp": time.time()
            }

    def _fallback_merge(
        self,
        ragflow_results: List[Dict],
        pageindex_results: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """
        Simple fallback merge when LLM ranking fails.
        Interleaves results from both systems.
        """
        merged = []
        
        # Add source attribution
        for r in ragflow_results:
            r['source_system'] = 'ragflow'
            r['retrieval_method'] = 'vector'
        
        for r in pageindex_results:
            r['source_system'] = 'pageindex'
            r['retrieval_method'] = 'reasoning'
        
        # Interleave results, preferring higher confidence
        all_results = ragflow_results + pageindex_results
        all_results.sort(key=lambda x: x.get('confidence_score', 0), reverse=True)
        
        return all_results[:top_k]
