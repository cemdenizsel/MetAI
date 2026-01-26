"""
RAGFlow Configuration

Configuration settings for RAGFlow integration including timeouts and retry policies.
"""

import os
from typing import Optional


class RAGFlowConfig:
    """
    Configuration class for RAGFlow service settings.
    """
    
    def __init__(self):
        # API Configuration
        self.api_url: str = os.getenv('RAGFLOW_API_URL', 'http://localhost:9380')
        self.api_key: str = os.getenv('RAGFLOW_API_KEY', 'your_api_key_here')
        
        # Timeout Configuration (in seconds)
        self.request_timeout: int = int(os.getenv('RAGFLOW_REQUEST_TIMEOUT', '300'))  # 5 minutes
        self.connect_timeout: int = int(os.getenv('RAGFLOW_CONNECT_TIMEOUT', '30'))   # 30 seconds
        self.total_timeout: int = int(os.getenv('RAGFLOW_TOTAL_TIMEOUT', '600'))     # 10 minutes for long operations
        
        # Retry Configuration
        self.max_retries: int = int(os.getenv('RAGFLOW_MAX_RETRIES', '3'))
        self.retry_multiplier: int = int(os.getenv('RAGFLOW_RETRY_MULTIPLIER', '1'))
        self.retry_min_wait: int = int(os.getenv('RAGFLOW_RETRY_MIN_WAIT', '4'))
        self.retry_max_wait: int = int(os.getenv('RAGFLOW_RETRY_MAX_WAIT', '10'))
        
        # Lookup Configuration
        self.default_top_k: int = int(os.getenv('RAGFLOW_DEFAULT_TOP_K', '5'))
        self.default_similarity_threshold: float = float(os.getenv('RAGFLOW_DEFAULT_SIMILARITY_THRESHOLD', '0.5'))
        
        # Validation
        if self.request_timeout <= 0:
            raise ValueError("RAGFLOW_REQUEST_TIMEOUT must be greater than 0")
        if self.max_retries < 0:
            raise ValueError("RAGFLOW_MAX_RETRIES cannot be negative")
        if not (0.0 <= self.default_similarity_threshold <= 1.0):
            raise ValueError("RAGFLOW_DEFAULT_SIMILARITY_THRESHOLD must be between 0.0 and 1.0")
    
    @property
    def headers(self) -> dict:
        """
        Get the headers for RAGFlow API requests.
        """
        return {
            'Authorization': f'Api-Key {self.api_key}',
            'Content-Type': 'application/json'
        }