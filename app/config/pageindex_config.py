"""
PageIndex Configuration

Configuration settings for PageIndex integration including model settings,
storage paths, and merge strategy options.
"""

import os
from typing import Optional


class PageIndexConfig:
    """
    Configuration class for PageIndex service settings.
    
    When using PageIndex cloud API (0.2.x):
    - PAGEINDEX_API_KEY: Required; get from https://dash.pageindex.ai
    - PAGEINDEX_API_DOCS_DIR: Dir for user->doc mapping (default: ./pageindex_api_docs)
    
    Legacy/local-only (unused when using API):
    - PAGEINDEX_ENABLED, PAGEINDEX_STORAGE, PAGEINDEX_MODEL, etc.
    """
    
    def __init__(self):
        # Enable/disable PageIndex
        self.enabled: bool = os.getenv('PAGEINDEX_ENABLED', 'true').lower() == 'true'
        
        # PageIndex Cloud API (required for API-based integration)
        self.api_key: str = os.getenv('PAGEINDEX_API_KEY', '')
        self.api_docs_dir: str = os.getenv('PAGEINDEX_API_DOCS_DIR', './pageindex_api_docs')
        
        # LLM Model Configuration (local SDK; unused when using API)
        self.model: str = os.getenv('PAGEINDEX_MODEL', 'gpt-4o')
        
        # Storage Configuration (local trees; unused when using API)
        self.storage_path: str = os.getenv('PAGEINDEX_STORAGE', './pageindex_trees')
        
        # Tree Building Configuration (local; unused when using API)
        self.max_pages_per_node: int = int(os.getenv('PAGEINDEX_MAX_PAGES_PER_NODE', '10'))
        self.max_tokens_per_node: int = int(os.getenv('PAGEINDEX_MAX_TOKENS_PER_NODE', '20000'))
        self.toc_check_pages: int = int(os.getenv('PAGEINDEX_TOC_CHECK_PAGES', '20'))
        
        # Search Configuration
        self.search_model: str = os.getenv('PAGEINDEX_SEARCH_MODEL', self.model)
        self.search_timeout: int = int(os.getenv('PAGEINDEX_SEARCH_TIMEOUT', '60'))
        
        # OpenAI API Key (shared with main app; used by local SDK only)
        self.openai_api_key: str = os.getenv('OPENAI_API_KEY', '')
        
        # Validation (only for local tree params)
        if self.max_pages_per_node <= 0:
            raise ValueError("PAGEINDEX_MAX_PAGES_PER_NODE must be positive")
        if self.max_tokens_per_node <= 0:
            raise ValueError("PAGEINDEX_MAX_TOKENS_PER_NODE must be positive")


class HybridRetrievalConfig:
    """
    Configuration for document retrieval (PageIndex only).
    
    Environment Variables:
        HYBRID_ENABLED: Enable retrieval (default: true)
        MERGE_STRATEGY: Merge strategy (llm, weighted, pageindex_first)
        MERGER_MODEL: LLM model for result merging (default: gpt-4o-mini)
        MERGER_TEMPERATURE: LLM temperature for merging (default: 0.3)
        MERGER_MAX_TOKENS: Max tokens for merge response (default: 2000)
        PAGEINDEX_WEIGHT: Weight for PageIndex results in weighted merge (default: 1.0)
    """
    
    def __init__(self):
        self.enabled: bool = os.getenv('HYBRID_ENABLED', 'true').lower() == 'true'
        self.use_pageindex: bool = os.getenv('USE_PAGEINDEX', 'true').lower() == 'true'
        
        self.merge_strategy: str = os.getenv('MERGE_STRATEGY', 'llm')
        self.merger_model: str = os.getenv('MERGER_MODEL', 'gpt-4o-mini')
        self.merger_temperature: float = float(os.getenv('MERGER_TEMPERATURE', '0.3'))
        self.merger_max_tokens: int = int(os.getenv('MERGER_MAX_TOKENS', '2000'))
        self.pageindex_weight: float = float(os.getenv('PAGEINDEX_WEIGHT', '1.0'))
        self.api_key: str = os.getenv('OPENAI_API_KEY', '')
        
        valid_strategies = ['llm', 'weighted', 'pageindex_first']
        if self.merge_strategy not in valid_strategies:
            raise ValueError(f"MERGE_STRATEGY must be one of: {valid_strategies}")
        if not (0 <= self.pageindex_weight <= 1):
            raise ValueError("PAGEINDEX_WEIGHT must be between 0 and 1")


def get_pageindex_config() -> PageIndexConfig:
    """Get PageIndex configuration singleton."""
    return PageIndexConfig()


def get_hybrid_config() -> HybridRetrievalConfig:
    """Get hybrid retrieval configuration singleton."""
    return HybridRetrievalConfig()
