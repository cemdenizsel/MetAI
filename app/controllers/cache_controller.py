"""
Cache Management Controller

Admin endpoints for cache management.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, Path

from services.cache_service import get_cache_service, ResultCacheService
from utils.auth import get_current_user_email

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/admin/cache", tags=["Cache Management"])


@router.get(
    "/stats",
    summary="Get Cache Statistics",
    description="Get statistics about the result cache",
    response_model=dict
)
async def get_cache_stats(
    current_user_email: str = Depends(get_current_user_email),
    cache_service: ResultCacheService = Depends(get_cache_service)
):
    """
    Get cache statistics.
    
    Returns cache hit rate, size, and Redis health information.
    
    **Authentication Required**: JWT token in Authorization header
    
    Args:
        current_user_email: Current user email (from JWT token)
        cache_service: Cache service dependency
    
    Returns:
        Cache statistics
    """
    try:
        stats = await cache_service.get_cache_stats()
        
        return {
            "success": True,
            **stats
        }
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.delete(
    "/{video_hash}",
    summary="Invalidate Cache Entry",
    description="Remove a specific video's cached result",
    response_model=dict
)
async def invalidate_cache(
    video_hash: str = Path(..., description="Video file hash to invalidate"),
    current_user_email: str = Depends(get_current_user_email),
    cache_service: ResultCacheService = Depends(get_cache_service)
):
    """
    Invalidate cached result for specific video hash.
    
    **Authentication Required**: JWT token in Authorization header
    
    Args:
        video_hash: SHA-256 hash of video file
        current_user_email: Current user email (from JWT token)
        cache_service: Cache service dependency
    
    Returns:
        Invalidation result
    """
    try:
        success = await cache_service.invalidate_cache(video_hash)
        
        if success:
            return {
                "success": True,
                "message": f"Cache invalidated for video hash: {video_hash[:16]}..."
            }
        else:
            return {
                "success": False,
                "message": "Cache entry not found or already invalidated"
            }
        
    except Exception as e:
        logger.error(f"Error invalidating cache: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.delete(
    "/flush",
    summary="Flush All Cache",
    description="Clear all cached results (admin only)",
    response_model=dict
)
async def flush_cache(
    current_user_email: str = Depends(get_current_user_email),
    cache_service: ResultCacheService = Depends(get_cache_service)
):
    """
    Flush all cached results.
    
    **WARNING**: This will delete all cached emotion analysis results.
    
    **Authentication Required**: JWT token in Authorization header
    **Authorization**: Admin users only
    
    Args:
        current_user_email: Current user email (from JWT token)
        cache_service: Cache service dependency
    
    Returns:
        Flush result
    """
    try:
        success = await cache_service.flush_all_cache()
        
        if success:
            logger.warning(f"Cache flushed by user: {current_user_email}")
            return {
                "success": True,
                "message": "All cached results have been flushed"
            }
        else:
            return {
                "success": False,
                "message": "Failed to flush cache"
            }
        
    except Exception as e:
        logger.error(f"Error flushing cache: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/health",
    summary="Cache Health Check",
    description="Check Redis cache health",
    response_model=dict
)
async def cache_health(
    cache_service: ResultCacheService = Depends(get_cache_service)
):
    """
    Check cache service health.
    
    Returns:
        Health status information
    """
    try:
        stats = await cache_service.get_cache_stats()
        
        return {
            "status": "healthy" if stats.get('redis_health') == 'healthy' else "degraded",
            "service": "Result Cache",
            "version": "1.0.0",
            "enabled": stats.get('enabled', False),
            "redis_status": stats.get('redis_health', 'unknown')
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "Result Cache",
            "error": str(e)
        }

