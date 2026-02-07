"""
PageIndex Service (API-based)

Uses PageIndex cloud API (PageIndexClient) for document indexing and retrieval.
Documents are submitted to https://api.pageindex.ai; tree generation and OCR run on their side.
User-to-document mapping is stored locally (JSON per user).
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Any
from pydantic import BaseModel

from config.pageindex_config import get_pageindex_config

logger = logging.getLogger(__name__)


def _clean_chat_answer(text: str) -> str:
    """
    Remove raw metadata and citation tags from PageIndex Chat API answer
    so the user sees clean text; citations are shown separately in References.
    """
    if not text or not isinstance(text, str):
        return text
    # Remove JSON-like blobs e.g. {"doc_name": "file.pdf", "pages": "1"}
    text = re.sub(r'\s*\{[^{}]*"doc_name"[^{}]*\}\s*', ' ', text)
    # Remove citation tags e.g. <doc=file.pdf;page=1> or <doc=file.pdf;page=1;block=...>
    text = re.sub(r'\s*<doc=[^>]+>\s*', ' ', text)
    # Remove leading meta phrase e.g. "I'll extract the email information from the document for you."
    text = re.sub(
        r"^I'll (retrieve|extract|find|get) (?:the )?[^.]+ (?:from|in) [^.]+\.[\s\n]*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Collapse multiple spaces/newlines and strip
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _parse_inline_citations(raw_content: str) -> List[Dict[str, Any]]:
    """
    Extract <doc=filename;page=N> or <doc=filename;page=N;block=...> from raw answer
    so we can show them in References even if the API didn't return a citations array.
    """
    if not raw_content or not isinstance(raw_content, str):
        return []
    # Match <doc=...;page=...> or <doc=...;page=...;block=...>
    pattern = re.compile(r"<doc=([^;>]+);page=(\d+)")
    seen = set()
    out = []
    for m in pattern.finditer(raw_content):
        doc_name, page_str = m.group(1), m.group(2)
        key = (doc_name, page_str)
        if key in seen:
            continue
        seen.add(key)
        try:
            page_num = int(page_str)
        except ValueError:
            page_num = None
        out.append({
            "document_id": doc_name,
            "page_number": page_num,
            "content_snippet": "",
            "source_system": "pageindex",
        })
    return out


def _normalize_chat_citation(c: Any) -> Dict[str, Any]:
    """
    Normalize a PageIndex Chat API citation to a consistent shape for the UI:
    document_id (filename or doc id), page_number, content_snippet.
    """
    if not isinstance(c, dict):
        return {"document_id": "", "page_number": None, "content_snippet": ""}
    doc_id = (
        c.get("document_id")
        or c.get("doc_id")
        or c.get("doc_name")
        or ""
    )
    page = c.get("page_number")
    if page is None:
        p = c.get("page") or c.get("pages")
        if p is not None:
            try:
                page = int(p) if isinstance(p, (int, float)) else int(str(p).strip())
            except (TypeError, ValueError):
                pass
    snippet = (
        c.get("content_snippet")
        or c.get("content")
        or c.get("text")
        or c.get("relevant_content")
        or ""
    )
    if isinstance(snippet, str) and len(snippet) > 300:
        snippet = snippet[:300].rstrip() + "…"
    return {
        "document_id": doc_id,
        "page_number": page,
        "content_snippet": snippet,
        "source_system": "pageindex",
    }


# Check if PageIndex API client is available
PAGEINDEX_AVAILABLE = False
PageIndexClient = None
PageIndexAPIError = None
try:
    from pageindex import PageIndexClient as _PageIndexClient
    from pageindex import PageIndexAPIError as _PageIndexAPIError
    PageIndexClient = _PageIndexClient
    PageIndexAPIError = _PageIndexAPIError
    PAGEINDEX_AVAILABLE = True
    logger.info("PageIndex API client loaded successfully")
except Exception as e:
    logger.warning("PageIndex not available. pip install pageindex and set PAGEINDEX_API_KEY. Error: %s", e)


class PageIndexLookupRequest(BaseModel):
    """Request model for PageIndex lookup operations."""
    query: str
    user_id: str
    topic_keywords: Optional[List[str]] = []
    top_k: Optional[int] = 5


class PageIndexDocumentResult(BaseModel):
    """Model for a document result from PageIndex retrieval."""
    document_id: str
    content: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    confidence_score: float
    source_file: Optional[str] = None
    source_system: str = "pageindex"
    retrieval_method: str = "reasoning"
    reasoning_path: Optional[List[str]] = None
    section_path: Optional[List[str]] = None  # alias used by merger and tests
    metadata: Optional[Dict] = {}


def _get_mapping_path(user_id: str, config) -> Path:
    """Path to the user's document mapping JSON file."""
    base = Path(config.api_docs_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base / f"kb_user_{user_id}.json"


def _load_user_mapping(user_id: str, config) -> List[Dict[str, Any]]:
    """Load user's document list: [{ document_id, api_doc_id, filename? }]."""
    path = _get_mapping_path(user_id, config)
    if not path.exists():
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data.get("documents", [])
    except Exception as e:
        logger.warning("Failed to load user mapping for %s: %s", user_id, e)
        return []


def _save_user_mapping(user_id: str, documents: List[Dict[str, Any]], config) -> None:
    """Save user's document list."""
    path = _get_mapping_path(user_id, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"documents": documents}, f, indent=2)


def _map_retrieved_nodes_to_results(
    api_doc_id: str,
    logical_doc_id: str,
    retrieved_nodes: List[Dict],
    top_k: int,
) -> List[Dict]:
    """
    Map PageIndex API retrieved_nodes to the format expected by ResultMergerService.
    API (legacy): retrieved_nodes[] with title, node_id, relevant_contents[] (page_index, relevant_content).
    Newer API: node keys may be id (not node_id); relevant_contents may be list of strings or mixed shapes.
    page_index is 1-based. Defensive: only process dict nodes; accept string or dict items in relevant_contents.
    """
    results = []
    if not isinstance(retrieved_nodes, list):
        return results
    for node in retrieved_nodes:
        if not isinstance(node, dict):
            continue
        title = node.get("title") or ""
        node_id = node.get("node_id") or node.get("id")
        reasoning_path = [title] if title else ([node_id] if node_id else None)
        relevant_contents = node.get("relevant_contents", [])
        if relevant_contents:
            for rc in relevant_contents:
                content = ""
                page_index = None
                if isinstance(rc, dict):
                    content = rc.get("relevant_content") or rc.get("content") or rc.get("text") or ""
                    page_index = rc.get("page_index")
                elif isinstance(rc, str):
                    content = rc
                    page_index = node.get("page_index")
                elif isinstance(rc, (list, tuple)) and len(rc) >= 2:
                    page_index = rc[0] if isinstance(rc[0], (int, float)) else None
                    content = str(rc[1]) if rc[1] else ""
                # If rc is another type (e.g. number), skip
                if content:
                    results.append({
                        "document_id": logical_doc_id,
                        "content": content,
                        "page_number": page_index,
                        "section_title": title or None,
                        "confidence_score": 1.0,
                        "source_file": None,
                        "source_system": "pageindex",
                        "retrieval_method": "reasoning",
                        "reasoning_path": reasoning_path,
                        "metadata": {},
                    })
        else:
            # Node-level content (e.g. API returns text/page_index on the node itself)
            content = node.get("relevant_content") or node.get("content") or node.get("text") or ""
            page_index = node.get("page_index")
            if content:
                results.append({
                    "document_id": logical_doc_id,
                    "content": content,
                    "page_number": page_index,
                    "section_title": title or None,
                    "confidence_score": 1.0,
                    "source_file": None,
                    "source_system": "pageindex",
                    "retrieval_method": "reasoning",
                    "reasoning_path": reasoning_path,
                    "metadata": {},
                })
    return results[:top_k]


class PageIndexService:
    """
    PageIndex integration via cloud API (PageIndexClient).
    - index_document: submit PDF to API, store (user_id, doc_id) -> api_doc_id in mapping.
    - lookup_with_progress: for each user doc, check retrieval_ready, submit_query, poll get_retrieval, merge results.
    - delete_document_tree: call API delete_document, remove from mapping.
    - get_user_stats: return count and list from mapping.
    """

    def __init__(self, config=None):
        self.config = config or get_pageindex_config()
        self._client = None
        if self.config.api_key and PageIndexClient:
            self._client = PageIndexClient(api_key=self.config.api_key)
        Path(self.config.api_docs_dir).mkdir(parents=True, exist_ok=True)

    @property
    def storage_path(self) -> Path:
        """Directory for user–document mapping (API mode). Backward compat for tests."""
        return Path(self.config.api_docs_dir)

    def _get_user_storage_path(self, user_id: str) -> Path:
        """Path to the user's mapping file. Backward compat for tests."""
        return _get_mapping_path(user_id, self.config)

    @property
    def client(self):
        if self._client is None and self.config.api_key and PageIndexClient:
            self._client = PageIndexClient(api_key=self.config.api_key)
        return self._client

    async def index_document(
        self,
        document_path: str,
        user_id: str,
        document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit PDF to PageIndex API; store api_doc_id in user mapping.
        Returns dict with document_id (logical), node_count/indexing_time omitted or 0.
        """
        if not self.client:
            raise RuntimeError("PageIndex API client not available. Set PAGEINDEX_API_KEY.")
        logical_id = document_id or Path(document_path).stem
        filename = Path(document_path).name

        def _submit():
            return self.client.submit_document(document_path)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _submit)
        api_doc_id = result.get("doc_id")
        if not api_doc_id:
            raise RuntimeError("PageIndex API did not return doc_id")

        documents = _load_user_mapping(user_id, self.config)
        documents.append({
            "document_id": logical_id,
            "api_doc_id": api_doc_id,
            "filename": filename,
        })
        _save_user_mapping(user_id, documents, self.config)
        logger.info("Indexed document %s for user %s -> api_doc_id %s", logical_id, user_id, api_doc_id)

        return {
            "document_id": logical_id,
            "node_count": 0,
            "indexing_time": 0,
        }

    async def lookup_with_progress(
        self,
        request: PageIndexLookupRequest,
    ) -> AsyncGenerator[Dict, None]:
        """
        For each document in user mapping: check retrieval_ready, submit_query, poll get_retrieval.
        Map retrieved_nodes to merger format; merge and cap at top_k; yield progress then completed results.
        """
        user_id = request.user_id
        query = request.query
        top_k = request.top_k or 5

        yield {"status": "started", "message": "Starting PageIndex lookup...", "source": "pageindex"}

        if not self.client:
            yield {"status": "error", "message": "PageIndex API client not available.", "source": "pageindex"}
            return

        documents = _load_user_mapping(user_id, self.config)
        if not documents:
            yield {
                "status": "completed",
                "message": "No documents indexed for this user.",
                "results": [],
                "source": "pageindex",
            }
            return

        all_results = []
        loop = asyncio.get_event_loop()
        any_doc_ready = False
        retrieval_ran = False

        for i, doc_entry in enumerate(documents):
            api_doc_id = doc_entry.get("api_doc_id")
            logical_id = doc_entry.get("document_id", api_doc_id)
            if not api_doc_id:
                continue

            def _is_ready():
                return self.client.is_retrieval_ready(api_doc_id)

            try:
                ready = await loop.run_in_executor(None, _is_ready)
            except Exception as e:
                logger.warning("is_retrieval_ready failed for %s: %s", api_doc_id, e)
                continue

            if not ready:
                logger.debug("PageIndex doc %s (%s) not retrieval_ready yet", logical_id, api_doc_id)
                continue
            any_doc_ready = True

            def _submit_query():
                return self.client.submit_query(api_doc_id, query, thinking=False)

            try:
                sub = await loop.run_in_executor(None, _submit_query)
            except Exception as e:
                logger.warning("submit_query failed for %s: %s", api_doc_id, e)
                continue

            retrieval_id = sub.get("retrieval_id")
            if not retrieval_id:
                continue
            retrieval_ran = True

            # Poll for completion
            for _ in range(60):
                def _get_retrieval():
                    return self.client.get_retrieval(retrieval_id)

                res = await loop.run_in_executor(None, _get_retrieval)
                # API may return a dict with status/retrieved_nodes or (legacy) a list of nodes
                if isinstance(res, list):
                    nodes = res
                    status = "completed"
                else:
                    status = res.get("status") if isinstance(res, dict) else None
                    nodes = res.get("retrieved_nodes", []) if isinstance(res, dict) else []
                if status == "completed":
                    chunk = _map_retrieved_nodes_to_results(
                        api_doc_id, logical_id, nodes, top_k
                    )
                    if not chunk and nodes:
                        first = nodes[0] if nodes else {}
                        keys = list(first.keys()) if isinstance(first, dict) else type(first).__name__
                        rc0 = first.get("relevant_contents", []) if isinstance(first, dict) else []
                        rc0_keys = list(rc0[0].keys()) if (rc0 and isinstance(rc0[0], dict)) else ("empty or non-dict",)
                        logger.info(
                            "PageIndex retrieval completed but mapped to 0 results for doc %s; node keys=%s, first relevant_content keys=%s",
                            logical_id, keys, rc0_keys,
                        )
                    all_results.extend(chunk)
                    break
                if status == "error":
                    break
                await asyncio.sleep(0.5)

        # Cap total and sort by relevance (here we just take first top_k)
        merged = all_results[:top_k]
        if not merged and documents:
            if not any_doc_ready:
                msg = (
                    "Your documents are still being processed by PageIndex (tree build can take a few minutes). Try again in 2–5 minutes."
                )
                logger.info("PageIndex: user has %d doc(s) but none retrieval_ready yet", len(documents))
            elif retrieval_ran:
                msg = "No matching sections for this query in your documents. Try different keywords or a longer phrase."
            else:
                msg = "No results for this query. Your documents may still be processing."
        else:
            msg = f"Found {len(merged)} results via PageIndex"

        yield {
            "status": "completed",
            "message": msg,
            "results": merged,
            "source": "pageindex",
        }

    async def delete_document_tree(self, user_id: str, document_id: str) -> bool:
        """
        Resolve api_doc_id from mapping, call client.delete_document(api_doc_id), remove from mapping.
        """
        if not self.client:
            return False
        documents = _load_user_mapping(user_id, self.config)
        for i, doc in enumerate(documents):
            if doc.get("document_id") == document_id:
                api_doc_id = doc.get("api_doc_id")
                if api_doc_id:
                    def _delete():
                        try:
                            self.client.delete_document(api_doc_id)
                        except Exception as e:
                            # Remove from mapping even if API returns 404 or error
                            logger.warning("delete_document failed for %s: %s", api_doc_id, e)

                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, _delete)
                documents.pop(i)
                _save_user_mapping(user_id, documents, self.config)
                return True
        return False

    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Return document count and list of logical document_ids for the user."""
        documents = _load_user_mapping(user_id, self.config)
        return {
            "user_id": user_id,
            "document_count": len(documents),
            "total_nodes": 0,  # API does not expose per-doc node count
            "document_ids": [d.get("document_id") for d in documents if d.get("document_id")],
            "documents": documents,
        }

    async def ask_chat(self, user_id: str, message: str) -> Optional[Dict[str, Any]]:
        """
        Q&A over all user documents using PageIndex Chat API.
        Sends all user api_doc_ids; if the API fails with "Document not found"
        for a specific id, we remove that doc from the mapping and retry with the rest.
        Returns {"answer": str, "citations": list} or None if no docs or error.
        """
        if not self.client:
            return None
        documents = _load_user_mapping(user_id, self.config)
        api_doc_ids = [d.get("api_doc_id") for d in documents if d.get("api_doc_id")]
        if not api_doc_ids:
            return None

        def _chat(doc_ids: List[str]):
            doc_id_arg = doc_ids[0] if len(doc_ids) == 1 else doc_ids
            return self.client.chat_completions(
                messages=[{"role": "user", "content": message}],
                doc_id=doc_id_arg,
                stream=False,
                enable_citations=True,
            )

        loop = asyncio.get_event_loop()
        current_ids = list(api_doc_ids)
        last_error = None
        while current_ids:
            try:
                response = await loop.run_in_executor(None, lambda ids=current_ids: _chat(ids))
            except Exception as e:
                last_error = e
                err_str = str(e)
                # e.g. "Document not found or access denied: pi-cmlcdsv9x005609r2hzari18t"
                bad_id = None
                if "Document not found" in err_str or "access denied" in err_str.lower():
                    m = re.search(r"pi-[a-z0-9]+", err_str)
                    if m:
                        bad_id = m.group(0)
                if bad_id and bad_id in current_ids:
                    logger.info(
                        "Removing doc %s from user mapping (not found on PageIndex), retrying with remaining docs.",
                        bad_id,
                    )
                    current_ids = [x for x in current_ids if x != bad_id]
                    # Remove from persisted mapping so we don't send it again
                    documents = _load_user_mapping(user_id, self.config)
                    documents = [d for d in documents if d.get("api_doc_id") != bad_id]
                    _save_user_mapping(user_id, documents, self.config)
                    continue
                logger.warning("PageIndex chat_completions failed: %s", e)
                return None
            break

        if last_error and not current_ids:
            logger.warning("PageIndex chat_completions failed (no docs left): %s", last_error)
            return None

        if not response or "choices" not in response or not response["choices"]:
            return None
        msg = response["choices"][0].get("message", {})
        raw_content = msg.get("content", "")
        # Parse inline <doc=...;page=...> from raw text so we can show them in References
        inline_citations = _parse_inline_citations(raw_content)
        raw_citations = msg.get("citations") if isinstance(msg.get("citations"), list) else []
        citations = [_normalize_chat_citation(c) for c in raw_citations]
        # Add inline refs not already present (by document_id + page_number)
        seen = {(c.get("document_id"), c.get("page_number")) for c in citations}
        for c in inline_citations:
            key = (c.get("document_id"), c.get("page_number"))
            if key not in seen:
                seen.add(key)
                citations.append(c)
        content = _clean_chat_answer(raw_content)
        return {"answer": content, "citations": citations}


def is_pageindex_available() -> bool:
    """True when PAGEINDEX_API_KEY is set and PageIndexClient can be imported."""
    if not PAGEINDEX_AVAILABLE or not PageIndexClient:
        return False
    config = get_pageindex_config()
    return bool(config.api_key and config.enabled)
