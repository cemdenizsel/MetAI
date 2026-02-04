"""
PageIndex Service

Handles document indexing and reasoning-based retrieval using PageIndex.
Provides vectorless, tree-based document search with LLM reasoning.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Any
from pydantic import BaseModel
import aiofiles

logger = logging.getLogger(__name__)


# Check if PageIndex is available
PAGEINDEX_AVAILABLE = False
try:
    from pageindex import PageIndex
    PAGEINDEX_AVAILABLE = True
    logger.info("PageIndex library loaded successfully")
except ImportError:
    logger.warning("PageIndex not available. Install with: pip install pageindex")


class PageIndexLookupRequest(BaseModel):
    """
    Request model for PageIndex lookup operations.
    """
    query: str
    user_id: str
    topic_keywords: Optional[List[str]] = []
    top_k: Optional[int] = 5


class PageIndexDocumentResult(BaseModel):
    """
    Model for individual document results from PageIndex.
    """
    document_id: str
    content: str
    page_number: Optional[int] = None
    page_range: Optional[List[int]] = None  # [start_page, end_page]
    section_title: Optional[str] = None
    section_path: Optional[List[str]] = None  # Reasoning path through tree
    confidence_score: float
    source_file: Optional[str] = None
    retrieval_method: str = "reasoning"
    metadata: Optional[Dict] = {}


class PageIndexConfig:
    """
    Configuration class for PageIndex service settings.
    """
    
    def __init__(self):
        self.storage_path: str = os.getenv('PAGEINDEX_STORAGE', './pageindex_trees')
        self.model: str = os.getenv('PAGEINDEX_MODEL', 'gpt-4o')
        self.enabled: bool = os.getenv('PAGEINDEX_ENABLED', 'true').lower() == 'true'
        self.max_pages_per_node: int = int(os.getenv('PAGEINDEX_MAX_PAGES_PER_NODE', '10'))
        self.max_tokens_per_node: int = int(os.getenv('PAGEINDEX_MAX_TOKENS_PER_NODE', '20000'))
        self.toc_check_pages: int = int(os.getenv('PAGEINDEX_TOC_CHECK_PAGES', '20'))
        
        # OpenAI API key for PageIndex
        self.api_key: str = os.getenv('OPENAI_API_KEY', '')
        
        # Ensure storage path exists
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)


class PageIndexService:
    """
    Service class for PageIndex-based document retrieval.
    
    Provides:
    - Document indexing (building tree structures)
    - Reasoning-based search through tree traversal
    - User-specific document isolation
    """

    def __init__(self, config: PageIndexConfig = None):
        self.config = config or PageIndexConfig()
        self.storage_path = Path(self.config.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Set OpenAI API key for PageIndex
        if self.config.api_key:
            os.environ['CHATGPT_API_KEY'] = self.config.api_key
        
        # Cache for loaded document trees
        self._tree_cache: Dict[str, Dict] = {}
        
        logger.info(f"PageIndexService initialized. Storage: {self.storage_path}, Model: {self.config.model}")

    def _get_user_storage_path(self, user_id: str) -> Path:
        """Get the storage path for a specific user."""
        user_path = self.storage_path / f"kb_user_{user_id}"
        user_path.mkdir(parents=True, exist_ok=True)
        return user_path

    def _get_tree_path(self, user_id: str, document_id: str) -> Path:
        """Get the path to a document's tree index."""
        return self._get_user_storage_path(user_id) / f"{document_id}_tree.json"

    async def index_document(
        self,
        document_path: str,
        user_id: str,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build PageIndex tree structure for a document.
        
        Args:
            document_path: Path to the PDF document
            user_id: User ID for storage isolation
            document_id: Optional custom document ID (defaults to filename)
            
        Returns:
            Dictionary with indexing results including tree structure metadata
        """
        if not PAGEINDEX_AVAILABLE:
            raise RuntimeError("PageIndex library not available. Install with: pip install pageindex")
        
        if not self.config.enabled:
            raise RuntimeError("PageIndex is disabled in configuration")
        
        start_time = time.time()
        doc_path = Path(document_path)
        
        if not doc_path.exists():
            raise FileNotFoundError(f"Document not found: {document_path}")
        
        # Generate document ID if not provided
        if not document_id:
            document_id = doc_path.stem
        
        logger.info(f"Indexing document '{document_id}' for user '{user_id}'")
        
        try:
            # Run PageIndex tree building in executor to avoid blocking
            loop = asyncio.get_event_loop()
            tree = await loop.run_in_executor(
                None,
                self._build_tree_sync,
                str(doc_path)
            )
            
            # Save tree to user's storage
            tree_path = self._get_tree_path(user_id, document_id)
            
            tree_data = {
                "document_id": document_id,
                "source_file": str(doc_path.name),
                "indexed_at": time.time(),
                "model": self.config.model,
                "tree": tree
            }
            
            async with aiofiles.open(tree_path, 'w') as f:
                await f.write(json.dumps(tree_data, indent=2))
            
            # Update cache
            cache_key = f"{user_id}:{document_id}"
            self._tree_cache[cache_key] = tree_data
            
            elapsed = time.time() - start_time
            logger.info(f"Successfully indexed '{document_id}' in {elapsed:.2f}s")
            
            return {
                "document_id": document_id,
                "user_id": user_id,
                "tree_path": str(tree_path),
                "indexing_time": elapsed,
                "node_count": self._count_nodes(tree),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Failed to index document '{document_id}': {e}", exc_info=True)
            raise

    def _build_tree_sync(self, pdf_path: str) -> Dict:
        """Synchronous tree building (runs in executor)."""
        from pageindex import PageIndex
        
        # Build tree using PageIndex
        tree = PageIndex.build_tree(
            pdf_path=pdf_path,
            model=self.config.model,
            toc_check_pages=self.config.toc_check_pages,
            max_pages_per_node=self.config.max_pages_per_node,
            max_tokens_per_node=self.config.max_tokens_per_node,
            if_add_node_id=True,
            if_add_node_summary=True,
            if_add_doc_description=True
        )
        
        return tree

    def _count_nodes(self, tree: Dict) -> int:
        """Count total nodes in tree structure."""
        count = 1
        if 'nodes' in tree and tree['nodes']:
            for child in tree['nodes']:
                count += self._count_nodes(child)
        return count

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None
    ) -> List[PageIndexDocumentResult]:
        """
        Perform reasoning-based search through document trees.
        
        Args:
            query: Search query
            user_id: User ID for document isolation
            top_k: Maximum number of results to return
            document_ids: Optional list of specific documents to search
            
        Returns:
            List of PageIndexDocumentResult objects
        """
        if not PAGEINDEX_AVAILABLE:
            logger.warning("PageIndex not available, returning empty results")
            return []
        
        if not self.config.enabled:
            logger.warning("PageIndex disabled, returning empty results")
            return []
        
        start_time = time.time()
        results = []
        
        # Get user's indexed documents
        user_trees = await self.get_user_trees(user_id)
        
        if not user_trees:
            logger.info(f"No indexed documents for user '{user_id}'")
            return []
        
        # Filter by specific document IDs if provided
        if document_ids:
            user_trees = {k: v for k, v in user_trees.items() if k in document_ids}
        
        logger.info(f"Searching {len(user_trees)} documents for user '{user_id}'")
        
        # Search each document tree
        for doc_id, tree_data in user_trees.items():
            try:
                # Run tree search in executor
                loop = asyncio.get_event_loop()
                search_result = await loop.run_in_executor(
                    None,
                    self._tree_search_sync,
                    query,
                    tree_data['tree']
                )
                
                if search_result:
                    result = PageIndexDocumentResult(
                        document_id=doc_id,
                        content=search_result.get('content', ''),
                        page_number=search_result.get('pages', [None])[0] if search_result.get('pages') else None,
                        page_range=search_result.get('pages', []),
                        section_title=search_result.get('section_title', ''),
                        section_path=search_result.get('reasoning_path', []),
                        confidence_score=search_result.get('confidence', 0.0),
                        source_file=tree_data.get('source_file', ''),
                        retrieval_method="reasoning",
                        metadata={
                            "node_id": search_result.get('node_id'),
                            "summary": search_result.get('summary'),
                            "search_time": search_result.get('search_time', 0)
                        }
                    )
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Error searching document '{doc_id}': {e}", exc_info=True)
                continue
        
        # Sort by confidence and limit to top_k
        results.sort(key=lambda x: x.confidence_score, reverse=True)
        results = results[:top_k]
        
        elapsed = time.time() - start_time
        logger.info(f"PageIndex search completed in {elapsed:.2f}s, found {len(results)} results")
        
        return results

    def _tree_search_sync(self, query: str, tree: Dict) -> Optional[Dict]:
        """Synchronous tree search (runs in executor)."""
        try:
            from pageindex import PageIndex
            
            start_time = time.time()
            
            # Perform tree search with LLM reasoning
            result = PageIndex.tree_search(
                tree=tree,
                query=query,
                model=self.config.model
            )
            
            if result:
                result['search_time'] = time.time() - start_time
                
                # Extract reasoning path from the search
                if 'path' in result:
                    result['reasoning_path'] = [node.get('title', '') for node in result['path']]
                
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Tree search error: {e}", exc_info=True)
            return None

    async def get_user_trees(self, user_id: str) -> Dict[str, Dict]:
        """
        Get all indexed document trees for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary mapping document_id to tree data
        """
        user_path = self._get_user_storage_path(user_id)
        trees = {}
        
        # Check for cached trees first
        cache_prefix = f"{user_id}:"
        for key, tree_data in self._tree_cache.items():
            if key.startswith(cache_prefix):
                doc_id = key[len(cache_prefix):]
                trees[doc_id] = tree_data
        
        # Load any trees not in cache
        if user_path.exists():
            for tree_file in user_path.glob("*_tree.json"):
                doc_id = tree_file.stem.replace('_tree', '')
                cache_key = f"{user_id}:{doc_id}"
                
                if cache_key not in self._tree_cache:
                    try:
                        async with aiofiles.open(tree_file, 'r') as f:
                            content = await f.read()
                            tree_data = json.loads(content)
                            self._tree_cache[cache_key] = tree_data
                            trees[doc_id] = tree_data
                    except Exception as e:
                        logger.error(f"Error loading tree '{tree_file}': {e}")
                elif doc_id not in trees:
                    trees[doc_id] = self._tree_cache[cache_key]
        
        return trees

    async def delete_document_tree(self, user_id: str, document_id: str) -> bool:
        """
        Delete a document's tree index.
        
        Args:
            user_id: User ID
            document_id: Document ID to delete
            
        Returns:
            True if deleted successfully
        """
        tree_path = self._get_tree_path(user_id, document_id)
        cache_key = f"{user_id}:{document_id}"
        
        # Remove from cache
        if cache_key in self._tree_cache:
            del self._tree_cache[cache_key]
        
        # Remove file
        if tree_path.exists():
            tree_path.unlink()
            logger.info(f"Deleted tree index for '{document_id}' (user: {user_id})")
            return True
        
        return False

    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get statistics about a user's indexed documents.
        
        Args:
            user_id: User ID
            
        Returns:
            Statistics dictionary
        """
        trees = await self.get_user_trees(user_id)
        
        total_nodes = 0
        for tree_data in trees.values():
            if 'tree' in tree_data:
                total_nodes += self._count_nodes(tree_data['tree'])
        
        return {
            "user_id": user_id,
            "document_count": len(trees),
            "total_nodes": total_nodes,
            "documents": list(trees.keys())
        }

    async def lookup_with_progress(
        self,
        request: PageIndexLookupRequest,
        user_email: str = None
    ) -> AsyncGenerator[Dict, None]:
        """
        Perform document lookup with progress updates via async generator.
        Compatible with the HybridRetrievalService interface.
        
        Args:
            request: PageIndex lookup request
            user_email: Optional user email for logging
            
        Yields:
            Dictionary with progress updates and final results
        """
        start_time = time.time()
        
        # Send initial progress
        yield {
            "status": "processing",
            "source": "pageindex",
            "message": "Starting reasoning-based search...",
            "progress": 10,
            "timestamp": time.time()
        }
        
        try:
            # Get user's documents
            user_trees = await self.get_user_trees(request.user_id)
            
            if not user_trees:
                yield {
                    "status": "completed",
                    "source": "pageindex",
                    "message": "No indexed documents found",
                    "progress": 100,
                    "results": [],
                    "timestamp": time.time()
                }
                return
            
            yield {
                "status": "searching",
                "source": "pageindex",
                "message": f"Searching {len(user_trees)} documents with LLM reasoning...",
                "progress": 30,
                "timestamp": time.time()
            }
            
            # Perform search
            results = await self.search(
                query=request.query,
                user_id=request.user_id,
                top_k=request.top_k
            )
            
            # Convert results to dictionaries
            result_dicts = [
                {
                    "document_id": r.document_id,
                    "content": r.content,
                    "page_number": r.page_number,
                    "page_range": r.page_range,
                    "section_title": r.section_title,
                    "section_path": r.section_path,
                    "confidence_score": r.confidence_score,
                    "source_file": r.source_file,
                    "retrieval_method": r.retrieval_method,
                    "metadata": r.metadata
                }
                for r in results
            ]
            
            elapsed = time.time() - start_time
            
            yield {
                "status": "completed",
                "source": "pageindex",
                "message": f"Found {len(results)} results via reasoning",
                "progress": 100,
                "results": result_dicts,
                "query_duration_ms": int(elapsed * 1000),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"PageIndex lookup error: {e}", exc_info=True)
            yield {
                "status": "error",
                "source": "pageindex",
                "message": f"Search failed: {str(e)}",
                "error_details": str(e),
                "timestamp": time.time()
            }


def is_pageindex_available() -> bool:
    """Check if PageIndex is available and enabled."""
    config = PageIndexConfig()
    return PAGEINDEX_AVAILABLE and config.enabled
