"""
Result Merger Service

Uses LLM to intelligently merge and rank results from RAGFlow (vector-based)
and PageIndex (reasoning-based) retrieval systems.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Check if OpenAI is available
OPENAI_AVAILABLE = False
OpenAI = None

try:
    from openai import OpenAI as OpenAIClient
    OpenAI = OpenAIClient
    OPENAI_AVAILABLE = True
    logger.info("OpenAI library loaded for ResultMergerService")
except ImportError:
    logger.warning("OpenAI not available for ResultMergerService")


class MergeStrategy:
    """Merge strategy constants."""
    LLM = "llm"  # Use LLM to rank and deduplicate
    WEIGHTED = "weighted"  # Simple weighted scoring
    PAGEINDEX_FIRST = "pageindex_first"  # Prefer PageIndex, then RAGFlow
    RAGFLOW_FIRST = "ragflow_first"  # Prefer RAGFlow, then PageIndex


class ResultMergerConfig:
    """Configuration for result merger."""
    
    def __init__(self):
        self.strategy: str = os.getenv('MERGE_STRATEGY', MergeStrategy.LLM)
        self.model: str = os.getenv('MERGER_MODEL', 'gpt-4o-mini')
        self.api_key: str = os.getenv('OPENAI_API_KEY', '')
        self.temperature: float = float(os.getenv('MERGER_TEMPERATURE', '0.3'))
        self.max_tokens: int = int(os.getenv('MERGER_MAX_TOKENS', '2000'))
        
        # Weights for weighted merge strategy
        self.ragflow_weight: float = float(os.getenv('RAGFLOW_WEIGHT', '0.4'))
        self.pageindex_weight: float = float(os.getenv('PAGEINDEX_WEIGHT', '0.6'))


class MergedResult(BaseModel):
    """Model for a merged document result."""
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
    rank: int = 0
    rank_reason: Optional[str] = None
    metadata: Optional[Dict] = {}


class ResultMergerService:
    """
    Service for merging results from multiple retrieval systems.
    
    Supports multiple merge strategies:
    - LLM: Uses GPT to intelligently rank and deduplicate
    - Weighted: Simple weighted scoring based on confidence
    - PageIndex First: Prioritizes reasoning-based results
    - RAGFlow First: Prioritizes vector-based results
    """

    def __init__(self, config: ResultMergerConfig = None):
        self.config = config or ResultMergerConfig()
        self.client = None
        
        if OPENAI_AVAILABLE and self.config.api_key:
            try:
                self.client = OpenAI(api_key=self.config.api_key)
                logger.info(f"ResultMergerService initialized with LLM: {self.config.model}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        else:
            logger.warning("LLM not available for ResultMergerService, will use fallback strategies")

    async def merge(
        self,
        query: str,
        ragflow_results: List[Dict],
        pageindex_results: List[Dict],
        top_k: int = 5
    ) -> List[MergedResult]:
        """
        Merge results from RAGFlow and PageIndex.
        
        Args:
            query: Original search query
            ragflow_results: Results from RAGFlow (vector-based)
            pageindex_results: Results from PageIndex (reasoning-based)
            top_k: Maximum number of results to return
            
        Returns:
            List of merged and ranked results
        """
        # Normalize results to common format
        normalized_ragflow = self._normalize_ragflow_results(ragflow_results)
        normalized_pageindex = self._normalize_pageindex_results(pageindex_results)
        
        # Choose merge strategy
        if self.config.strategy == MergeStrategy.LLM and self.client:
            return await self._llm_merge(query, normalized_ragflow, normalized_pageindex, top_k)
        elif self.config.strategy == MergeStrategy.PAGEINDEX_FIRST:
            return self._priority_merge(normalized_pageindex, normalized_ragflow, top_k)
        elif self.config.strategy == MergeStrategy.RAGFLOW_FIRST:
            return self._priority_merge(normalized_ragflow, normalized_pageindex, top_k)
        else:
            # Default to weighted merge
            return self._weighted_merge(normalized_ragflow, normalized_pageindex, top_k)

    def _normalize_ragflow_results(self, results: List[Dict]) -> List[Dict]:
        """Normalize RAGFlow results to common format."""
        normalized = []
        
        for r in results:
            normalized.append({
                "document_id": r.get("document_id", "unknown"),
                "content": r.get("content", ""),
                "page_number": r.get("page_number"),
                "section_title": None,
                "confidence_score": r.get("confidence_score", r.get("score", 0.0)),
                "source_file": r.get("source_file", r.get("filename")),
                "source_system": "ragflow",
                "retrieval_method": "vector",
                "reasoning_path": None,
                "bbox_coordinates": r.get("bbox_coordinates"),
                "metadata": r.get("metadata", {})
            })
        
        return normalized

    def _normalize_pageindex_results(self, results: List[Dict]) -> List[Dict]:
        """Normalize PageIndex results to common format."""
        normalized = []
        
        for r in results:
            normalized.append({
                "document_id": r.get("document_id", "unknown"),
                "content": r.get("content", ""),
                "page_number": r.get("page_number"),
                "section_title": r.get("section_title"),
                "confidence_score": r.get("confidence_score", 0.0),
                "source_file": r.get("source_file"),
                "source_system": "pageindex",
                "retrieval_method": "reasoning",
                "reasoning_path": r.get("section_path", r.get("reasoning_path")),
                "bbox_coordinates": None,
                "metadata": r.get("metadata", {})
            })
        
        return normalized

    async def _llm_merge(
        self,
        query: str,
        ragflow_results: List[Dict],
        pageindex_results: List[Dict],
        top_k: int
    ) -> List[MergedResult]:
        """Use LLM to intelligently merge and rank results."""
        
        if not ragflow_results and not pageindex_results:
            return []
        
        # Build the prompt
        prompt = self._build_merge_prompt(query, ragflow_results, pageindex_results, top_k)
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert document retrieval ranker. Your job is to merge and rank 
search results from two different retrieval systems:

1. RAGFlow (vector-based): Uses semantic similarity to find relevant content
2. PageIndex (reasoning-based): Uses LLM reasoning through document structure to find relevant sections

Your task is to:
1. Rank all results by relevance to the query
2. Identify and merge duplicate content (same information from both systems)
3. Prefer PageIndex results when they have explicit page/section references
4. Consider confidence scores but also use your judgment about content relevance

Return a JSON array of ranked results."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"}
            )
            
            # Parse LLM response
            response_text = response.choices[0].message.content
            parsed = json.loads(response_text)
            
            ranked_results = parsed.get("ranked_results", [])
            
            # Convert to MergedResult objects
            merged = []
            for i, r in enumerate(ranked_results[:top_k]):
                merged.append(MergedResult(
                    document_id=r.get("document_id", "unknown"),
                    content=r.get("content", ""),
                    page_number=r.get("page_number"),
                    section_title=r.get("section_title"),
                    confidence_score=r.get("confidence_score", 0.0),
                    source_file=r.get("source_file"),
                    source_system=r.get("source_system", "unknown"),
                    retrieval_method=r.get("retrieval_method", "hybrid"),
                    reasoning_path=r.get("reasoning_path"),
                    bbox_coordinates=r.get("bbox_coordinates"),
                    rank=i + 1,
                    rank_reason=r.get("rank_reason", ""),
                    metadata=r.get("metadata", {})
                ))
            
            logger.info(f"LLM merge completed: {len(merged)} results")
            return merged
            
        except Exception as e:
            logger.error(f"LLM merge failed: {e}, falling back to weighted merge")
            return self._weighted_merge(ragflow_results, pageindex_results, top_k)

    def _build_merge_prompt(
        self,
        query: str,
        ragflow_results: List[Dict],
        pageindex_results: List[Dict],
        top_k: int
    ) -> str:
        """Build the prompt for LLM merge."""
        
        prompt_parts = [
            f"## Query\n{query}\n",
            f"## Instructions\nMerge and rank the following results. Return top {top_k} results.\n",
        ]
        
        # Add RAGFlow results
        if ragflow_results:
            prompt_parts.append("## RAGFlow Results (Vector-based)\n")
            for i, r in enumerate(ragflow_results, 1):
                prompt_parts.append(f"""
### Result R{i}
- Document: {r.get('document_id', 'unknown')}
- Confidence: {r.get('confidence_score', 0):.2%}
- Page: {r.get('page_number', 'N/A')}
- Content: {r.get('content', '')[:500]}{'...' if len(r.get('content', '')) > 500 else ''}
""")
        
        # Add PageIndex results
        if pageindex_results:
            prompt_parts.append("\n## PageIndex Results (Reasoning-based)\n")
            for i, r in enumerate(pageindex_results, 1):
                reasoning_path = " > ".join(r.get('reasoning_path', [])) if r.get('reasoning_path') else "N/A"
                prompt_parts.append(f"""
### Result P{i}
- Document: {r.get('document_id', 'unknown')}
- Confidence: {r.get('confidence_score', 0):.2%}
- Page: {r.get('page_number', 'N/A')}
- Section: {r.get('section_title', 'N/A')}
- Reasoning Path: {reasoning_path}
- Content: {r.get('content', '')[:500]}{'...' if len(r.get('content', '')) > 500 else ''}
""")
        
        prompt_parts.append("""
## Output Format
Return a JSON object with this structure:
{
  "ranked_results": [
    {
      "document_id": "...",
      "content": "...",
      "page_number": 1,
      "section_title": "...",
      "confidence_score": 0.95,
      "source_file": "...",
      "source_system": "ragflow|pageindex|both",
      "retrieval_method": "vector|reasoning|hybrid",
      "reasoning_path": ["Section1", "Subsection"],
      "rank_reason": "Brief explanation of why this result is ranked here"
    }
  ]
}

Important:
- If the same content appears in both systems, merge them with source_system="both" and retrieval_method="hybrid"
- Prefer results with explicit page/section references
- Consider both confidence scores and actual content relevance
""")
        
        return "\n".join(prompt_parts)

    def _weighted_merge(
        self,
        ragflow_results: List[Dict],
        pageindex_results: List[Dict],
        top_k: int
    ) -> List[MergedResult]:
        """Simple weighted merge based on confidence scores."""
        
        all_results = []
        
        # Apply weights to RAGFlow results
        for r in ragflow_results:
            score = r.get("confidence_score", 0) * self.config.ragflow_weight
            all_results.append({
                **r,
                "weighted_score": score,
                "source_system": "ragflow",
                "retrieval_method": "vector"
            })
        
        # Apply weights to PageIndex results (with bonus for having section info)
        for r in pageindex_results:
            base_score = r.get("confidence_score", 0) * self.config.pageindex_weight
            # Bonus for having reasoning path
            if r.get("reasoning_path"):
                base_score += 0.1
            all_results.append({
                **r,
                "weighted_score": base_score,
                "source_system": "pageindex",
                "retrieval_method": "reasoning"
            })
        
        # Sort by weighted score
        all_results.sort(key=lambda x: x.get("weighted_score", 0), reverse=True)
        
        # Deduplicate by content similarity (simple approach)
        seen_content = set()
        deduplicated = []
        
        for r in all_results:
            content_key = r.get("content", "")[:100].lower().strip()
            if content_key and content_key not in seen_content:
                seen_content.add(content_key)
                deduplicated.append(r)
        
        # Convert to MergedResult
        merged = []
        for i, r in enumerate(deduplicated[:top_k]):
            merged.append(MergedResult(
                document_id=r.get("document_id", "unknown"),
                content=r.get("content", ""),
                page_number=r.get("page_number"),
                section_title=r.get("section_title"),
                confidence_score=r.get("weighted_score", r.get("confidence_score", 0)),
                source_file=r.get("source_file"),
                source_system=r.get("source_system", "unknown"),
                retrieval_method=r.get("retrieval_method", "unknown"),
                reasoning_path=r.get("reasoning_path"),
                bbox_coordinates=r.get("bbox_coordinates"),
                rank=i + 1,
                rank_reason="Weighted score ranking",
                metadata=r.get("metadata", {})
            ))
        
        return merged

    def _priority_merge(
        self,
        primary_results: List[Dict],
        secondary_results: List[Dict],
        top_k: int
    ) -> List[MergedResult]:
        """Priority merge: take from primary first, then fill with secondary."""
        
        merged = []
        seen_docs = set()
        
        # Add primary results first
        for i, r in enumerate(primary_results):
            if len(merged) >= top_k:
                break
            
            doc_key = f"{r.get('document_id')}:{r.get('page_number')}"
            if doc_key not in seen_docs:
                seen_docs.add(doc_key)
                merged.append(MergedResult(
                    document_id=r.get("document_id", "unknown"),
                    content=r.get("content", ""),
                    page_number=r.get("page_number"),
                    section_title=r.get("section_title"),
                    confidence_score=r.get("confidence_score", 0),
                    source_file=r.get("source_file"),
                    source_system=r.get("source_system", "unknown"),
                    retrieval_method=r.get("retrieval_method", "unknown"),
                    reasoning_path=r.get("reasoning_path"),
                    bbox_coordinates=r.get("bbox_coordinates"),
                    rank=len(merged) + 1,
                    rank_reason="Primary source priority",
                    metadata=r.get("metadata", {})
                ))
        
        # Fill with secondary results
        for r in secondary_results:
            if len(merged) >= top_k:
                break
            
            doc_key = f"{r.get('document_id')}:{r.get('page_number')}"
            if doc_key not in seen_docs:
                seen_docs.add(doc_key)
                merged.append(MergedResult(
                    document_id=r.get("document_id", "unknown"),
                    content=r.get("content", ""),
                    page_number=r.get("page_number"),
                    section_title=r.get("section_title"),
                    confidence_score=r.get("confidence_score", 0),
                    source_file=r.get("source_file"),
                    source_system=r.get("source_system", "unknown"),
                    retrieval_method=r.get("retrieval_method", "unknown"),
                    reasoning_path=r.get("reasoning_path"),
                    bbox_coordinates=r.get("bbox_coordinates"),
                    rank=len(merged) + 1,
                    rank_reason="Secondary source fill",
                    metadata=r.get("metadata", {})
                ))
        
        return merged
