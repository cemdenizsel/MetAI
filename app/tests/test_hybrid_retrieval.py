"""
Test Cases for Hybrid RAGFlow + PageIndex Retrieval System

Tests the complete hybrid retrieval system including:
- PageIndex service
- Hybrid retrieval service
- Result merger service
- Meeting controller endpoints
"""

import pytest
import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

# Set test environment
os.environ['PAGEINDEX_ENABLED'] = 'true'
os.environ['HYBRID_ENABLED'] = 'true'
os.environ['MERGE_STRATEGY'] = 'weighted'  # Use weighted for testing without LLM

from main import app
from services.pageindex_service import (
    PageIndexService, 
    PageIndexLookupRequest, 
    PageIndexDocumentResult,
    PageIndexConfig,
    is_pageindex_available
)
from services.hybrid_retrieval_service import (
    HybridRetrievalService, 
    HybridLookupRequest,
    HybridDocumentResult
)
from services.result_merger_service import (
    ResultMergerService,
    MergeStrategy,
    MergedResult,
    ResultMergerConfig
)
from config.pageindex_config import PageIndexConfig, HybridRetrievalConfig


# ============================================================================
# Configuration Tests
# ============================================================================

class TestPageIndexConfig:
    """Tests for PageIndex configuration."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = PageIndexConfig()
        
        assert config.enabled == True
        assert config.model == 'gpt-4o'
        assert config.storage_path == './pageindex_trees'
        assert config.max_pages_per_node == 10
        assert config.max_tokens_per_node == 20000
    
    def test_config_from_env(self):
        """Test configuration from environment variables."""
        os.environ['PAGEINDEX_MODEL'] = 'gpt-4-turbo'
        os.environ['PAGEINDEX_ENABLED'] = 'false'
        
        config = PageIndexConfig()
        
        assert config.model == 'gpt-4-turbo'
        assert config.enabled == False
        
        # Reset
        os.environ['PAGEINDEX_MODEL'] = 'gpt-4o'
        os.environ['PAGEINDEX_ENABLED'] = 'true'


class TestHybridRetrievalConfig:
    """Tests for hybrid retrieval configuration."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = HybridRetrievalConfig()
        
        assert config.enabled == True
        assert config.merge_strategy in ['llm', 'weighted', 'pageindex_first', 'ragflow_first']
        assert 0 <= config.ragflow_weight <= 1
        assert 0 <= config.pageindex_weight <= 1
    
    def test_invalid_strategy_raises(self):
        """Test that invalid merge strategy raises error."""
        os.environ['MERGE_STRATEGY'] = 'invalid_strategy'
        
        with pytest.raises(ValueError):
            HybridRetrievalConfig()
        
        # Reset
        os.environ['MERGE_STRATEGY'] = 'weighted'


# ============================================================================
# PageIndex Service Tests
# ============================================================================

class TestPageIndexService:
    """Tests for PageIndex service."""
    
    @pytest.fixture
    def service(self, tmp_path):
        """Create a PageIndex service with temporary storage."""
        os.environ['PAGEINDEX_STORAGE'] = str(tmp_path)
        return PageIndexService()
    
    def test_service_initialization(self, service):
        """Test service initializes correctly."""
        assert service.storage_path.exists()
        assert service.config is not None
    
    def test_get_user_storage_path(self, service):
        """Test user storage path creation."""
        user_id = "test_user_123"
        path = service._get_user_storage_path(user_id)
        
        assert path.exists()
        assert f"kb_user_{user_id}" in str(path)
    
    @pytest.mark.asyncio
    async def test_get_user_trees_empty(self, service):
        """Test getting trees for user with no documents."""
        trees = await service.get_user_trees("nonexistent_user")
        
        assert trees == {}
    
    @pytest.mark.asyncio
    async def test_get_user_stats(self, service):
        """Test getting user statistics."""
        stats = await service.get_user_stats("test_user")
        
        assert "user_id" in stats
        assert "document_count" in stats
        assert "total_nodes" in stats
        assert stats["document_count"] == 0
    
    @pytest.mark.asyncio
    async def test_search_no_documents(self, service):
        """Test search returns empty when no documents indexed."""
        results = await service.search(
            query="test query",
            user_id="test_user",
            top_k=5
        )
        
        assert results == []
    
    @pytest.mark.asyncio
    async def test_lookup_with_progress_no_documents(self, service):
        """Test lookup with progress when no documents."""
        request = PageIndexLookupRequest(
            query="test query",
            user_id="test_user",
            top_k=5
        )
        
        results = []
        async for update in service.lookup_with_progress(request):
            results.append(update)
        
        # Should have processing, completed
        assert len(results) >= 2
        assert results[-1]["status"] == "completed"
        assert results[-1]["results"] == []


# ============================================================================
# Result Merger Service Tests
# ============================================================================

class TestResultMergerService:
    """Tests for result merger service."""
    
    @pytest.fixture
    def merger(self):
        """Create a result merger with weighted strategy."""
        os.environ['MERGE_STRATEGY'] = 'weighted'
        config = ResultMergerConfig()
        return ResultMergerService(config)
    
    @pytest.fixture
    def sample_ragflow_results(self):
        """Sample RAGFlow results."""
        return [
            {
                "document_id": "doc1",
                "content": "RAGFlow result 1 about quarterly earnings",
                "page_number": 5,
                "confidence_score": 0.85,
                "source_file": "report.pdf"
            },
            {
                "document_id": "doc2",
                "content": "RAGFlow result 2 about revenue",
                "page_number": 12,
                "confidence_score": 0.72,
                "source_file": "report.pdf"
            }
        ]
    
    @pytest.fixture
    def sample_pageindex_results(self):
        """Sample PageIndex results."""
        return [
            {
                "document_id": "doc1",
                "content": "PageIndex result about quarterly earnings from Section 3",
                "page_number": 5,
                "section_title": "Financial Overview",
                "section_path": ["Annual Report", "Financial Data", "Q3 Results"],
                "confidence_score": 0.90,
                "source_file": "report.pdf"
            }
        ]
    
    @pytest.mark.asyncio
    async def test_weighted_merge(self, merger, sample_ragflow_results, sample_pageindex_results):
        """Test weighted merge strategy."""
        results = await merger.merge(
            query="quarterly earnings",
            ragflow_results=sample_ragflow_results,
            pageindex_results=sample_pageindex_results,
            top_k=5
        )
        
        assert len(results) > 0
        assert all(isinstance(r, MergedResult) for r in results)
        # PageIndex result should be ranked higher due to reasoning path bonus
        assert results[0].source_system == "pageindex" or results[0].reasoning_path is not None
    
    @pytest.mark.asyncio
    async def test_merge_empty_results(self, merger):
        """Test merge with empty results."""
        results = await merger.merge(
            query="test",
            ragflow_results=[],
            pageindex_results=[],
            top_k=5
        )
        
        assert results == []
    
    @pytest.mark.asyncio
    async def test_merge_ragflow_only(self, merger, sample_ragflow_results):
        """Test merge with only RAGFlow results."""
        results = await merger.merge(
            query="test",
            ragflow_results=sample_ragflow_results,
            pageindex_results=[],
            top_k=5
        )
        
        assert len(results) == len(sample_ragflow_results)
        assert all(r.source_system == "ragflow" for r in results)
    
    @pytest.mark.asyncio
    async def test_merge_pageindex_only(self, merger, sample_pageindex_results):
        """Test merge with only PageIndex results."""
        results = await merger.merge(
            query="test",
            ragflow_results=[],
            pageindex_results=sample_pageindex_results,
            top_k=5
        )
        
        assert len(results) == len(sample_pageindex_results)
        assert all(r.source_system == "pageindex" for r in results)
    
    def test_priority_merge_pageindex_first(self, merger, sample_ragflow_results, sample_pageindex_results):
        """Test priority merge with PageIndex first."""
        normalized_ragflow = merger._normalize_ragflow_results(sample_ragflow_results)
        normalized_pageindex = merger._normalize_pageindex_results(sample_pageindex_results)
        
        results = merger._priority_merge(
            normalized_pageindex,
            normalized_ragflow,
            top_k=5
        )
        
        # First result should be from PageIndex
        assert results[0].source_system == "pageindex"


# ============================================================================
# Hybrid Retrieval Service Tests
# ============================================================================

class TestHybridRetrievalService:
    """Tests for hybrid retrieval service."""
    
    @pytest.fixture
    def mock_ragflow_service(self):
        """Create mock RAGFlow service."""
        service = MagicMock()
        service.lookup_with_progress = AsyncMock(return_value=iter([
            {"status": "completed", "results": [
                {"document_id": "doc1", "content": "test", "confidence_score": 0.8}
            ]}
        ]))
        return service
    
    @pytest.fixture
    def mock_pageindex_service(self):
        """Create mock PageIndex service."""
        service = MagicMock()
        service.lookup_with_progress = AsyncMock(return_value=iter([
            {"status": "completed", "results": []}
        ]))
        return service
    
    @pytest.fixture
    def hybrid_service(self, mock_ragflow_service, mock_pageindex_service):
        """Create hybrid service with mocks."""
        return HybridRetrievalService(
            ragflow_service=mock_ragflow_service,
            pageindex_service=mock_pageindex_service
        )
    
    def test_service_initialization(self, hybrid_service):
        """Test service initializes correctly."""
        assert hybrid_service.ragflow_service is not None
        assert hybrid_service.pageindex_service is not None
    
    def test_fallback_merge(self, hybrid_service):
        """Test fallback merge when LLM is unavailable."""
        ragflow_results = [
            {"content": "test1", "confidence_score": 0.8, "document_id": "doc1"}
        ]
        pageindex_results = [
            {"content": "test2", "confidence_score": 0.9, "document_id": "doc2"}
        ]
        
        merged = hybrid_service._fallback_merge(ragflow_results, pageindex_results, top_k=5)
        
        assert len(merged) == 2
        # Should be sorted by confidence
        assert merged[0]["confidence_score"] >= merged[1]["confidence_score"]


# ============================================================================
# API Endpoint Tests
# ============================================================================

class TestMeetingControllerEndpoints:
    """Tests for meeting controller API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self, client):
        """Create auth headers with a test token.
        
        Note: For integration tests that need real auth, 
        use test_with_live_auth tests below.
        """
        import jwt
        from datetime import datetime, timedelta
        
        # Create a test token (won't work for DB operations but good for endpoint testing)
        secret = os.environ.get("JWT_SECRET_KEY", "test-secret-key")
        payload = {
            "sub": "test@example.com",
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        token = jwt.encode(payload, secret, algorithm="HS256")
        
        return {"Authorization": f"Bearer {token}"}
    
    def test_retrieval_status_endpoint(self, client):
        """Test /meeting/retrieval-status endpoint."""
        response = client.get("/meeting/retrieval-status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "hybrid_retrieval" in data
        assert "ragflow" in data
        assert "pageindex" in data
        assert "enabled" in data["hybrid_retrieval"]
        assert "merge_strategy" in data["hybrid_retrieval"]
    
    def test_user_dataset_info_requires_auth(self, client):
        """Test that /meeting/user-dataset-info requires authentication."""
        response = client.get("/meeting/user-dataset-info")
        
        assert response.status_code == 401
    
    @pytest.mark.skip(reason="Requires MongoDB connection - run as integration test")
    def test_user_dataset_info_with_auth(self, client, auth_headers):
        """Test /meeting/user-dataset-info with authentication.
        
        Note: This test requires a running MongoDB instance.
        For unit tests, the auth fixture creates a mock token.
        """
        response = client.get(
            "/meeting/user-dataset-info",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "user_id" in data
        assert "dataset_id" in data
        assert "ragflow" in data
        assert "pageindex" in data
    
    def test_document_lookup_requires_auth(self, client):
        """Test that /meeting/document-lookup requires authentication."""
        response = client.post(
            "/meeting/document-lookup",
            json={
                "meeting_transcript": "test transcript",
                "topic_keywords": ["test"],
                "top_k": 5
            }
        )
        
        assert response.status_code == 401
    
    def test_index_document_requires_auth(self, client):
        """Test that /meeting/index-document requires authentication."""
        response = client.post(
            "/meeting/index-document",
            files={"document": ("test.pdf", b"fake pdf content", "application/pdf")}
        )
        
        assert response.status_code == 401
    
    @pytest.mark.skip(reason="Requires MongoDB connection - run as integration test")
    def test_index_document_validates_file_type(self, client, auth_headers):
        """Test that /meeting/index-document validates file type.
        
        Note: This test requires a running MongoDB instance.
        """
        response = client.post(
            "/meeting/index-document",
            files={"document": ("test.txt", b"text content", "text/plain")},
            headers=auth_headers
        )
        
        # Should reject non-PDF files
        assert response.status_code in [400, 503]  # 400 for wrong type, 503 if PageIndex not available


# ============================================================================
# Integration Tests
# ============================================================================

class TestHybridRetrievalIntegration:
    """Integration tests for the complete hybrid retrieval flow."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def auth_token(self, client):
        """Get auth token for testing."""
        response = client.post(
            "/auth/register",
            json={
                "username": f"integration_test_{os.urandom(4).hex()}",
                "email": f"integration_{os.urandom(4).hex()}@example.com",
                "password": "TestPassword123!"
            }
        )
        
        if response.status_code == 200:
            return response.json()["access_token"]
        return None
    
    @pytest.mark.asyncio
    async def test_full_lookup_flow(self, client, auth_token):
        """Test complete document lookup flow."""
        if not auth_token:
            pytest.skip("Could not get auth token")
        
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # 1. Check retrieval status
        status_response = client.get("/meeting/retrieval-status")
        assert status_response.status_code == 200
        
        # 2. Check user dataset info
        info_response = client.get(
            "/meeting/user-dataset-info",
            headers=headers
        )
        assert info_response.status_code == 200
        
        # 3. Perform document lookup (will use SSE)
        # Note: TestClient doesn't handle SSE well, so we check it starts correctly
        with client.stream(
            "POST",
            "/meeting/document-lookup",
            json={
                "meeting_transcript": "Let's discuss the quarterly results",
                "topic_keywords": ["quarterly", "results"],
                "top_k": 5,
                "use_hybrid": True
            },
            headers=headers
        ) as response:
            assert response.status_code == 200
            # Just verify we can read some content
            for line in response.iter_lines():
                if line:
                    break  # Got at least one SSE event


# ============================================================================
# Model Tests
# ============================================================================

class TestDataModels:
    """Tests for data models."""
    
    def test_pageindex_document_result(self):
        """Test PageIndexDocumentResult model."""
        result = PageIndexDocumentResult(
            document_id="doc1",
            content="Test content",
            page_number=5,
            section_title="Test Section",
            section_path=["Root", "Chapter 1", "Section 1.1"],
            confidence_score=0.95,
            source_file="test.pdf"
        )
        
        assert result.document_id == "doc1"
        assert result.retrieval_method == "reasoning"
        assert len(result.section_path) == 3
    
    def test_hybrid_lookup_request(self):
        """Test HybridLookupRequest model."""
        request = HybridLookupRequest(
            query="test query",
            user_id="user123",
            dataset_id="kb_user_user123",
            topic_keywords=["keyword1", "keyword2"],
            top_k=10,
            use_ragflow=True,
            use_pageindex=True
        )
        
        assert request.query == "test query"
        assert request.use_ragflow == True
        assert request.use_pageindex == True
        assert len(request.topic_keywords) == 2
    
    def test_merged_result(self):
        """Test MergedResult model."""
        result = MergedResult(
            document_id="doc1",
            content="Merged content",
            page_number=10,
            section_title="Financial Overview",
            confidence_score=0.88,
            source_file="annual_report.pdf",
            source_system="both",
            retrieval_method="hybrid",
            reasoning_path=["Reports", "Financial", "Overview"],
            rank=1,
            rank_reason="High confidence from both systems"
        )
        
        assert result.source_system == "both"
        assert result.retrieval_method == "hybrid"
        assert result.rank == 1


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
