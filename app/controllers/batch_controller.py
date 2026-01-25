"""
Batch Processing Controller

REST API endpoints for batch video processing.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form, Query

from services.batch_service import get_batch_service, BatchProcessingService
from services.subs_service import check_batch_quota, deduct_quota
from utils.auth import get_current_user_email
from services.user_service import get_user_service, UserService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/batch", tags=["Batch Processing"])


# Configuration
MAX_FILES_PER_BATCH = 10
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
MAX_TOTAL_SIZE = 3 * 1024 * 1024 * 1024  # 3GB
ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.webm', '.mkv', '.flv'}


def validate_batch_files(files: List[UploadFile]):
    """Validate batch upload files."""
    import os
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if len(files) > MAX_FILES_PER_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum is {MAX_FILES_PER_BATCH} videos per batch"
        )
    
    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="File without filename provided")
        
        ext = os.path.splitext(file.filename.lower())[1]
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type '{ext}' for {file.filename}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )


@router.post(
    "/analyze",
    summary="Submit Batch for Analysis",
    description="Upload multiple videos for batch emotion analysis",
    response_model=dict,
    responses={
        200: {"description": "Batch submitted successfully"},
        400: {"description": "Invalid request or files"},
        403: {"description": "No active subscription or insufficient quota"},
        500: {"description": "Server error"},
    }
)
async def submit_batch(
    videos: List[UploadFile] = File(..., description="List of video files to analyze (max 10)"),
    fusion_strategy: Optional[str] = Form(None, description="Fusion strategy (optional)"),
    run_ai_analysis: Optional[bool] = Form(True, description="Whether to run AI agent analysis"),
    llm_provider: Optional[str] = Form("cloud", description="LLM provider: 'cloud' or 'local'"),
    use_cache: Optional[bool] = Form(True, description="Whether to use result caching"),
    current_user_email: str = Depends(get_current_user_email),
    user_service: UserService = Depends(get_user_service),
    batch_service: BatchProcessingService = Depends(get_batch_service)
):
    """
    Submit multiple videos for batch analysis.
    
    This endpoint accepts multiple video files and processes them in parallel.
    All videos are queued for asynchronous processing by background workers.
    
    **Authentication Required**: JWT token in Authorization header
    
    **Subscription Required**: User must have an active subscription
    
    **Quota Required**: User must have sufficient monthly quota
    
    Args:
        videos: List of video files (max 10, each max 500MB)
        fusion_strategy: Optional fusion strategy override
        run_ai_analysis: Whether to run AI agent analysis (default: True)
        llm_provider: LLM provider for AI analysis: 'cloud' (OpenAI) or 'local' (Ollama/LM Studio)
        use_cache: Whether to use result caching (default: True)
        current_user_email: Current user email (from JWT token)
        user_service: User service dependency
        batch_service: Batch service dependency
    
    Returns:
        Batch submission result with batch_id
    
    Raises:
        HTTPException: Various error conditions
    """
    logger.info(f"Batch submission request from user: {current_user_email}, videos={len(videos)}")
    
    try:
        # Validate files
        validate_batch_files(videos)
        
        # Get user info
        user_doc = await user_service.get_user_by_email(current_user_email)
        if not user_doc:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_id = str(user_doc['_id'])
        
        # Check quota
        quota_check = await check_batch_quota(user_id, len(videos))
        if not quota_check.get('allowed', False):
            raise HTTPException(
                status_code=403,
                detail=quota_check.get('message', 'Insufficient quota for batch processing')
            )
        
        # Prepare options
        options = {
            'run_ai_analysis': run_ai_analysis,
            'llm_provider': llm_provider,
            'use_cache': use_cache,
        }
        
        if fusion_strategy:
            options['fusion_strategy'] = fusion_strategy
        
        # Submit batch
        result = await batch_service.submit_batch(videos, user_id, options)
        
        # Deduct quota
        await deduct_quota(user_id, len(videos))
        
        logger.info(f"Batch submitted: batch_id={result['batch_id']}, user={current_user_email}")
        
        return {
            "success": True,
            **result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting batch: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/{batch_id}/status",
    summary="Get Batch Status",
    description="Check the status and progress of a batch",
    response_model=dict
)
async def get_batch_status(
    batch_id: str,
    current_user_email: str = Depends(get_current_user_email),
    batch_service: BatchProcessingService = Depends(get_batch_service)
):
    """
    Get batch status and progress.
    
    Returns current status, progress, and individual video statuses.
    
    **Authentication Required**: JWT token in Authorization header
    
    Args:
        batch_id: Batch identifier
        current_user_email: Current user email (from JWT token)
        batch_service: Batch service dependency
    
    Returns:
        Batch status information
    """
    try:
        status = await batch_service.get_batch_status(batch_id)
        
        if status.get('status') == 'not_found':
            raise HTTPException(status_code=404, detail="Batch not found")
        
        return {
            "success": True,
            **status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batch status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/{batch_id}/results",
    summary="Get Batch Results",
    description="Retrieve results of a completed batch",
    response_model=dict
)
async def get_batch_results(
    batch_id: str,
    current_user_email: str = Depends(get_current_user_email),
    batch_service: BatchProcessingService = Depends(get_batch_service)
):
    """
    Get batch results if completed.
    
    Returns the complete analysis results for all videos in the batch.
    
    **Authentication Required**: JWT token in Authorization header
    
    Args:
        batch_id: Batch identifier
        current_user_email: Current user email (from JWT token)
        batch_service: Batch service dependency
    
    Returns:
        Batch results including all video analyses
    """
    try:
        results = await batch_service.get_batch_results(batch_id)
        
        if results.get('status') == 'not_found':
            raise HTTPException(status_code=404, detail="Batch not found")
        
        return {
            "success": True,
            **results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batch results: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.delete(
    "/{batch_id}",
    summary="Cancel Batch",
    description="Cancel a running batch",
    response_model=dict
)
async def cancel_batch(
    batch_id: str,
    current_user_email: str = Depends(get_current_user_email),
    user_service: UserService = Depends(get_user_service),
    batch_service: BatchProcessingService = Depends(get_batch_service)
):
    """
    Cancel a running or pending batch.
    
    Terminates all running jobs and marks batch as cancelled.
    Users can only cancel their own batches.
    
    **Authentication Required**: JWT token in Authorization header
    
    Args:
        batch_id: Batch ID to cancel
        current_user_email: Current user email (from JWT token)
        user_service: User service dependency
        batch_service: Batch service dependency
    
    Returns:
        Cancellation result
    """
    try:
        # Get user info
        user_doc = await user_service.get_user_by_email(current_user_email)
        if not user_doc:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_id = str(user_doc['_id'])
        
        # Cancel batch
        result = await batch_service.cancel_batch(batch_id, user_id)
        
        if not result.get('success'):
            if 'Unauthorized' in result.get('error', ''):
                raise HTTPException(status_code=403, detail=result.get('error'))
            elif 'not found' in result.get('error', '').lower():
                raise HTTPException(status_code=404, detail=result.get('error'))
            else:
                raise HTTPException(status_code=400, detail=result.get('error'))
        
        logger.info(f"Batch cancelled: batch_id={batch_id}, user={current_user_email}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling batch: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/my-batches",
    summary="List User Batches",
    description="Get a list of batches submitted by the current user",
    response_model=dict
)
async def list_my_batches(
    limit: int = Query(20, ge=1, le=100, description="Maximum number of batches to return"),
    skip: int = Query(0, ge=0, description="Number of batches to skip (pagination)"),
    current_user_email: str = Depends(get_current_user_email),
    user_service: UserService = Depends(get_user_service),
    batch_service: BatchProcessingService = Depends(get_batch_service)
):
    """
    List batches for current user.
    
    Returns a paginated list of batches submitted by the authenticated user,
    ordered by creation time (most recent first).
    
    **Authentication Required**: JWT token in Authorization header
    
    Args:
        limit: Maximum number of batches to return (1-100, default: 20)
        skip: Number of batches to skip for pagination (default: 0)
        current_user_email: Current user email (from JWT token)
        user_service: User service dependency
        batch_service: Batch service dependency
    
    Returns:
        List of batch information
    """
    try:
        # Get user info
        user_doc = await user_service.get_user_by_email(current_user_email)
        if not user_doc:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_id = str(user_doc['_id'])
        
        # Get batches
        batches = await batch_service.list_user_batches(user_id, limit, skip)
        
        return {
            "success": True,
            "count": len(batches),
            "limit": limit,
            "skip": skip,
            "batches": batches
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing batches: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/health",
    summary="Batch Service Health Check",
    description="Check if the batch processing service is operational",
    response_model=dict
)
async def batch_health():
    """
    Health check for batch service.
    
    Returns:
        Health status information
    """
    return {
        "status": "healthy",
        "service": "Batch Processing",
        "version": "1.0.0",
        "max_videos_per_batch": MAX_FILES_PER_BATCH
    }

