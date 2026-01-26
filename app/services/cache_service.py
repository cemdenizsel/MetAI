"""
Result Caching Service

Service for caching emotion analysis results using Redis.
"""

import os
import hashlib
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from services.redis_service import redis_service

logger = logging.getLogger(__name__)


class ResultCacheService:
    """
    Service for caching emotion analysis results.
    
    Uses video file hashing (SHA-256) and Redis storage with compression.
    """
    
    def __init__(self):
        """Initialize cache service."""
        self.cache_enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"
        self.default_ttl_days = int(os.getenv("CACHE_TTL_DAYS", "30"))
        self.max_cache_size_gb = int(os.getenv("CACHE_MAX_SIZE_GB", "10"))
        self.cache_prefix = "emotion_result:"
        self.hit_count = 0
        self.miss_count = 0
        
        logger.info(f"ResultCacheService initialized: enabled={self.cache_enabled}, ttl={self.default_ttl_days}days")
    
    def compute_video_hash(self, video_path: str) -> str:
        """
        Compute SHA-256 hash of video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            SHA-256 hash as hex string
        """
        try:
            sha256 = hashlib.sha256()
            
            with open(video_path, 'rb') as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    sha256.update(chunk)
            
            video_hash = sha256.hexdigest()
            logger.debug(f"Computed video hash: {video_hash[:16]}...")
            
            return video_hash
            
        except Exception as e:
            logger.error(f"Error computing video hash: {e}")
            raise
    
    async def get_cached_result(self, video_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached result for video hash.
        
        Args:
            video_hash: Video file hash
            
        Returns:
            Cached result dictionary if found, None otherwise
        """
        if not self.cache_enabled:
            return None
        
        try:
            cache_key = f"{self.cache_prefix}{video_hash}"
            cached_data = await redis_service.get_json(cache_key)
            
            if cached_data:
                self.hit_count += 1
                logger.info(f"Cache HIT for video hash: {video_hash[:16]}...")
                
                cached_data['from_cache'] = True
                cached_data['cache_hit_time'] = datetime.now().isoformat()
                
                return cached_data
            else:
                self.miss_count += 1
                logger.info(f"Cache MISS for video hash: {video_hash[:16]}...")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving cached result: {e}")
            return None
    
    async def cache_result(
        self,
        video_hash: str,
        result: Dict[str, Any],
        ttl_days: Optional[int] = None
    ) -> bool:
        """
        Cache analysis result.
        
        Args:
            video_hash: Video file hash
            result: Analysis result dictionary
            ttl_days: Time-to-live in days (None = use default)
            
        Returns:
            True if cached successfully, False otherwise
        """
        if not self.cache_enabled:
            return False
        
        try:
            cache_key = f"{self.cache_prefix}{video_hash}"
            
            cache_data = {
                **result,
                'cached_at': datetime.now().isoformat(),
                'video_hash': video_hash,
            }
            
            ttl_days = ttl_days or self.default_ttl_days
            expire_seconds = ttl_days * 24 * 60 * 60
            
            success = await redis_service.set_json(cache_key, cache_data, expire_seconds)
            
            if success:
                logger.info(f"Cached result for video hash: {video_hash[:16]}... (TTL: {ttl_days}days)")
            
            return success
            
        except Exception as e:
            logger.error(f"Error caching result: {e}")
            return False
    
    async def invalidate_cache(self, video_hash: str) -> bool:
        """
        Remove cached result for video hash.
        
        Args:
            video_hash: Video file hash
            
        Returns:
            True if cache was invalidated, False otherwise
        """
        try:
            cache_key = f"{self.cache_prefix}{video_hash}"
            
            success = await redis_service.delete_cache(cache_key)
            
            if success:
                logger.info(f"Invalidated cache for video hash: {video_hash[:16]}...")
            
            return success
            
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
            return False
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Statistics dictionary
        """
        try:
            redis_health = await redis_service.health_check()
            
            total_requests = self.hit_count + self.miss_count
            hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0.0
            
            cache_keys = await redis_service.get_cache_keys(f"{self.cache_prefix}*")
            cache_count = len(cache_keys) if cache_keys else 0
            
            stats = {
                'enabled': self.cache_enabled,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'total_requests': total_requests,
                'hit_rate_percent': round(hit_rate, 2),
                'cached_results': cache_count,
                'ttl_days': self.default_ttl_days,
                'max_size_gb': self.max_cache_size_gb,
                'redis_health': redis_health.get('status', 'unknown'),
                'redis_memory': redis_health.get('used_memory', 'unknown'),
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {
                'enabled': self.cache_enabled,
                'error': str(e)
            }
    
    async def flush_all_cache(self) -> bool:
        """
        Flush all cached results (admin only).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            cache_keys = await redis_service.get_cache_keys(f"{self.cache_prefix}*")
            
            if not cache_keys:
                logger.info("No cache keys to flush")
                return True
            
            deleted_count = 0
            for key in cache_keys:
                if await redis_service.delete_cache(key):
                    deleted_count += 1
            
            logger.info(f"Flushed {deleted_count} cached results")
            
            self.hit_count = 0
            self.miss_count = 0
            
            return True
            
        except Exception as e:
            logger.error(f"Error flushing cache: {e}")
            return False
    
    def get_cache_key(self, video_hash: str) -> str:
        """Get Redis cache key for video hash."""
        return f"{self.cache_prefix}{video_hash}"


_cache_service = None


def get_cache_service() -> ResultCacheService:
    """Get the global cache service instance."""
    global _cache_service
    if _cache_service is None:
        _cache_service = ResultCacheService()
    return _cache_service

