"""
Job Controller

REST API endpoints for async job management.
"""

import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form, Query

from services.job_service import get_job_service, JobService
from utils.auth import get_current_user_email
from services.user_service import get_user_service, UserService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/jobs", tags=["Async Jobs"])


# File validation
ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.webm', '.mkv', '.flv'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB


def validate_video_file(file: UploadFile):
    """Validate uploaded video file."""
    import os
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    if file.content_type and not file.content_type.startswith('video/'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type: {file.content_type}. Must be a video file."
        )


@router.post(
    "/submit",
    summary="Submit Video for Async Analysis",
    description="Upload a video file for asynchronous emotion analysis",
    response_model=dict,
    responses={
        200: {"description": "Job submitted successfully"},
        400: {"description": "Invalid request or file"},
        403: {"description": "No active subscription"},
        500: {"description": "Server error"},
    }
)
async def submit_job(
    video: UploadFile = File(..., description="Video file to analyze"),
    fusion_strategy: Optional[str] = Form(None, description="Fusion strategy (optional)"),
    run_ai_analysis: Optional[bool] = Form(True, description="Whether to run AI agent analysis"),
    llm_provider: Optional[str] = Form("cloud", description="LLM provider: 'cloud' or 'local'"),
    current_user_email: str = Depends(get_current_user_email),
    user_service: UserService = Depends(get_user_service),
    job_service: JobService = Depends(get_job_service)
):
    """
    Submit a video for async analysis.
    
    This endpoint queues a video file for analysis and returns immediately with a job ID.
    The analysis will be processed by background workers.
    
    **Authentication Required**: JWT token in Authorization header
    
    **Subscription Required**: User must have an active subscription
    
    Args:
        video: Video file (MP4, AVI, MOV, WebM, MKV, FLV)
        fusion_strategy: Optional fusion strategy override
        run_ai_analysis: Whether to run AI agent analysis (default: True)
        llm_provider: LLM provider for AI analysis: 'cloud' (OpenAI) or 'local' (Ollama/LM Studio)
        current_user_email: Current user email (from JWT token)
        user_service: User service dependency
        job_service: Job service dependency
    
    Returns:
        Job submission result with job_id
    
    Raises:
        HTTPException: Various error conditions
    """
    logger.info(f"Job submission request from user: {current_user_email}")
    
    try:
        # Validate file
        validate_video_file(video)
        
        # Get user info
        user_doc = await user_service.get_user_by_email(current_user_email)
        if not user_doc:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_id = str(user_doc['_id'])
        
        # Prepare options
        options = {
            'run_ai_analysis': run_ai_analysis,
            'llm_provider': llm_provider,
            'filename': video.filename,
        }
        
        if fusion_strategy:
            options['fusion_strategy'] = fusion_strategy
        
        # Submit job
        result = await job_service.submit_job(video, user_id, options)
        
        logger.info(f"Job submitted: job_id={result['job_id']}, user={current_user_email}")
        
        return {
            "success": True,
            "message": "Job submitted successfully",
            **result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting job: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/{job_id}/status",
    summary="Get Job Status",
    description="Check the status and progress of an async job",
    response_model=dict
)
async def get_job_status(
    job_id: str,
    current_user_email: str = Depends(get_current_user_email),
    job_service: JobService = Depends(get_job_service)
):
    """
    Get job status and progress.
    
    Returns current status, progress, and stage information for the job.
    
    **Authentication Required**: JWT token in Authorization header
    
    Args:
        job_id: Job/Task ID
        current_user_email: Current user email (from JWT token)
        job_service: Job service dependency
    
    Returns:
        Job status information
    """
    try:
        status = await job_service.get_job_status(job_id)
        
        if status.get('status') == 'NOT_FOUND':
            raise HTTPException(status_code=404, detail="Job not found")
        
        return {
            "success": True,
            **status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/{job_id}/result",
    summary="Get Job Result",
    description="Retrieve the result of a completed job",
    response_model=dict
)
async def get_job_result(
    job_id: str,
    current_user_email: str = Depends(get_current_user_email),
    job_service: JobService = Depends(get_job_service)
):
    """
    Get job result if completed.
    
    Returns the complete analysis result if the job has finished successfully.
    
    **Authentication Required**: JWT token in Authorization header
    
    Args:
        job_id: Job/Task ID
        current_user_email: Current user email (from JWT token)
        job_service: Job service dependency
    
    Returns:
        Job result including emotion analysis
    """
    try:
        result = await job_service.get_job_result(job_id)
        
        if result.get('status') == 'ERROR':
            raise HTTPException(status_code=500, detail=result.get('error'))
        
        if result.get('status') != 'SUCCESS':
            raise HTTPException(
                status_code=400,
                detail=f"Job is not complete. Current status: {result.get('status')}"
            )
        
        return {
            "success": True,
            **result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job result: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.delete(
    "/{job_id}",
    summary="Cancel Job",
    description="Cancel a running job",
    response_model=dict
)
async def cancel_job(
    job_id: str,
    current_user_email: str = Depends(get_current_user_email),
    user_service: UserService = Depends(get_user_service),
    job_service: JobService = Depends(get_job_service)
):
    """
    Cancel a running or pending job.
    
    Terminates the job execution and marks it as cancelled.
    Users can only cancel their own jobs.
    
    **Authentication Required**: JWT token in Authorization header
    
    Args:
        job_id: Job/Task ID to cancel
        current_user_email: Current user email (from JWT token)
        user_service: User service dependency
        job_service: Job service dependency
    
    Returns:
        Cancellation result
    """
    try:
        # Get user info
        user_doc = await user_service.get_user_by_email(current_user_email)
        if not user_doc:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_id = str(user_doc['_id'])
        
        # Cancel job
        result = await job_service.cancel_job(job_id, user_id)
        
        if not result.get('success'):
            if 'Unauthorized' in result.get('error', ''):
                raise HTTPException(status_code=403, detail=result.get('error'))
            elif 'not found' in result.get('error', '').lower():
                raise HTTPException(status_code=404, detail=result.get('error'))
            else:
                raise HTTPException(status_code=400, detail=result.get('error'))
        
        logger.info(f"Job cancelled: job_id={job_id}, user={current_user_email}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/my-jobs",
    summary="List User Jobs",
    description="Get a list of jobs submitted by the current user",
    response_model=dict
)
async def list_my_jobs(
    limit: int = Query(20, ge=1, le=100, description="Maximum number of jobs to return"),
    skip: int = Query(0, ge=0, description="Number of jobs to skip (pagination)"),
    current_user_email: str = Depends(get_current_user_email),
    user_service: UserService = Depends(get_user_service),
    job_service: JobService = Depends(get_job_service)
):
    """
    List jobs for current user.
    
    Returns a paginated list of jobs submitted by the authenticated user,
    ordered by creation time (most recent first).
    
    **Authentication Required**: JWT token in Authorization header
    
    Args:
        limit: Maximum number of jobs to return (1-100, default: 20)
        skip: Number of jobs to skip for pagination (default: 0)
        current_user_email: Current user email (from JWT token)
        user_service: User service dependency
        job_service: Job service dependency
    
    Returns:
        List of job information
    """
    try:
        # Get user info
        user_doc = await user_service.get_user_by_email(current_user_email)
        if not user_doc:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_id = str(user_doc['_id'])
        
        # Get jobs
        jobs = await job_service.list_user_jobs(user_id, limit, skip)
        
        return {
            "success": True,
            "count": len(jobs),
            "limit": limit,
            "skip": skip,
            "jobs": jobs
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing jobs: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/health",
    summary="Job Service Health Check",
    description="Check if the job service is operational",
    response_model=dict
)
async def job_health():
    """
    Health check for job service.
    
    Returns:
        Health status information
    """
    return {
        "status": "healthy",
        "service": "Async Job Management",
        "version": "1.0.0"
    }

