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
    
    Environment Variables:
        PAGEINDEX_ENABLED: Enable/disable PageIndex (default: true)
        PAGEINDEX_MODEL: LLM model for tree building and search (default: gpt-4o)
        PAGEINDEX_STORAGE: Storage path for tree indices (default: ./pageindex_trees)
        PAGEINDEX_MAX_PAGES_PER_NODE: Max pages per tree node (default: 10)
        PAGEINDEX_MAX_TOKENS_PER_NODE: Max tokens per tree node (default: 20000)
        PAGEINDEX_TOC_CHECK_PAGES: Pages to check for TOC (default: 20)
    """
    
    def __init__(self):
        # Enable/disable PageIndex
        self.enabled: bool = os.getenv('PAGEINDEX_ENABLED', 'true').lower() == 'true'
        
        # LLM Model Configuration
        self.model: str = os.getenv('PAGEINDEX_MODEL', 'gpt-4o')
        
        # Storage Configuration
        self.storage_path: str = os.getenv('PAGEINDEX_STORAGE', './pageindex_trees')
        
        # Tree Building Configuration
        self.max_pages_per_node: int = int(os.getenv('PAGEINDEX_MAX_PAGES_PER_NODE', '10'))
        self.max_tokens_per_node: int = int(os.getenv('PAGEINDEX_MAX_TOKENS_PER_NODE', '20000'))
        self.toc_check_pages: int = int(os.getenv('PAGEINDEX_TOC_CHECK_PAGES', '20'))
        
        # Search Configuration
        self.search_model: str = os.getenv('PAGEINDEX_SEARCH_MODEL', self.model)
        self.search_timeout: int = int(os.getenv('PAGEINDEX_SEARCH_TIMEOUT', '60'))
        
        # OpenAI API Key (shared with main app)
        self.api_key: str = os.getenv('OPENAI_API_KEY', '')
        
        # Validation
        if self.max_pages_per_node <= 0:
            raise ValueError("PAGEINDEX_MAX_PAGES_PER_NODE must be positive")
        if self.max_tokens_per_node <= 0:
            raise ValueError("PAGEINDEX_MAX_TOKENS_PER_NODE must be positive")


class HybridRetrievalConfig:
    """
    Configuration class for hybrid retrieval settings.
    
    Environment Variables:
        HYBRID_ENABLED: Enable hybrid retrieval (default: true)
        MERGE_STRATEGY: Merge strategy (llm, weighted, pageindex_first, ragflow_first)
        MERGER_MODEL: LLM model for result merging (default: gpt-4o-mini)
        MERGER_TEMPERATURE: LLM temperature for merging (default: 0.3)
        MERGER_MAX_TOKENS: Max tokens for merge response (default: 2000)
        RAGFLOW_WEIGHT: Weight for RAGFlow results in weighted merge (default: 0.4)
        PAGEINDEX_WEIGHT: Weight for PageIndex results in weighted merge (default: 0.6)
    """
    
    def __init__(self):
        # Enable/disable hybrid retrieval
        self.enabled: bool = os.getenv('HYBRID_ENABLED', 'true').lower() == 'true'
        
        # Default retrieval system preferences
        self.use_ragflow: bool = os.getenv('USE_RAGFLOW', 'true').lower() == 'true'
        self.use_pageindex: bool = os.getenv('USE_PAGEINDEX', 'true').lower() == 'true'
        
        # Merge Strategy
        self.merge_strategy: str = os.getenv('MERGE_STRATEGY', 'llm')
        
        # LLM Merge Configuration
        self.merger_model: str = os.getenv('MERGER_MODEL', 'gpt-4o-mini')
        self.merger_temperature: float = float(os.getenv('MERGER_TEMPERATURE', '0.3'))
        self.merger_max_tokens: int = int(os.getenv('MERGER_MAX_TOKENS', '2000'))
        
        # Weighted Merge Configuration
        self.ragflow_weight: float = float(os.getenv('RAGFLOW_WEIGHT', '0.4'))
        self.pageindex_weight: float = float(os.getenv('PAGEINDEX_WEIGHT', '0.6'))
        
        # OpenAI API Key
        self.api_key: str = os.getenv('OPENAI_API_KEY', '')
        
        # Validation
        valid_strategies = ['llm', 'weighted', 'pageindex_first', 'ragflow_first']
        if self.merge_strategy not in valid_strategies:
            raise ValueError(f"MERGE_STRATEGY must be one of: {valid_strategies}")
        
        if not (0 <= self.ragflow_weight <= 1):
            raise ValueError("RAGFLOW_WEIGHT must be between 0 and 1")
        if not (0 <= self.pageindex_weight <= 1):
            raise ValueError("PAGEINDEX_WEIGHT must be between 0 and 1")


def get_pageindex_config() -> PageIndexConfig:
    """Get PageIndex configuration singleton."""
    return PageIndexConfig()


def get_hybrid_config() -> HybridRetrievalConfig:
    """Get hybrid retrieval configuration singleton."""
    return HybridRetrievalConfig()
