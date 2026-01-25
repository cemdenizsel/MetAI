"""
Emotion Analysis Controller

FastAPI endpoints for emotion analysis from video files.
"""

import logging

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form, Request
from typing import Optional

from services.emotion_service import EmotionAnalysisService
from models.response_models import MultiModelResponse
from utils.auth import get_current_user_email
from services.user_service import get_user_service, UserService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/emotion", tags=["Emotion Analysis"])


# Allowed video file extensions
ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.webm', '.mkv', '.flv'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB


def validate_video_file(file: UploadFile):
    """
    Validate uploaded video file.
    
    Args:
        file: Uploaded file
    
    Raises:
        HTTPException: If validation fails
    """
    # Check filename
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Check file extension
    import os
    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check content type
    if file.content_type and not file.content_type.startswith('video/'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type: {file.content_type}. Must be a video file."
        )


@router.get(
    "/health",
    summary="Health Check",
    description="Check if the emotion analysis service is running",
    response_model=dict
)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Status information
    """
    return {
        "status": "healthy",
        "service": "Emotion Analysis API",
        "version": "1.0.0"
    }


@router.post(
    "/analyze",
    summary="Analyze Video for Emotions",
    description="Upload a video file and get comprehensive emotion analysis",
    response_model=MultiModelResponse,
    responses={
        200: {"description": "Video analyzed successfully"},
        400: {"description": "Invalid request or file"},
        403: {"description": "No active subscription"},
        500: {"description": "Server error"},
    }
)
async def analyze_video(
    request: Request,
    video: UploadFile = File(..., description="Video file to analyze"),
    fusion_strategy: Optional[str] = Form(None, description="Fusion strategy (optional)"),
    run_ai_analysis: Optional[bool] = Form(True, description="Whether to run AI agent analysis"),
    llm_provider: Optional[str] = Form("cloud", description="LLM provider: 'cloud' or 'local'"),
    use_cache: Optional[bool] = Form(True, description="Whether to use cache for results"),
    cache_ttl_days: Optional[int] = Form(30, description="Cache TTL in days (default: 30)"),
    current_user_email: str = Depends(get_current_user_email),
    user_service: UserService = Depends(get_user_service)
):
    """
    Analyze a video file for emotions.
    
    This endpoint accepts a video file and returns comprehensive emotion analysis
    including:
    - Overall emotion prediction
    - Temporal emotion timeline
    - Mental health indicators
    - Transcription
    - AI-generated insights (if enabled)
    
    **Authentication Required**: JWT token in Authorization header
    
    **Subscription Required**: User must have an active subscription
    
    Args:
        video: Video file (MP4, AVI, MOV, WebM, MKV, FLV)
        fusion_strategy: Optional fusion strategy override
        run_ai_analysis: Whether to run AI agent analysis (default: True)
        llm_provider: LLM provider for AI analysis: 'cloud' (OpenAI) or 'local' (Ollama/LM Studio)
        current_user_email: Current user email (from JWT token)
        user_service: User service dependency
    
    Returns:
        MultiModelResponse with analysis results
    
    Raises:
        HTTPException: Various error conditions
    """
    logger.info(f"Emotion analysis request from user: {current_user_email}")
    
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
            'use_cache': use_cache,
            'cache_ttl_days': cache_ttl_days,
        }
        
        if fusion_strategy:
            options['fusion_strategy'] = fusion_strategy
        
        # Process video
        service = EmotionAnalysisService()
        result = await service.analyze_video(video, user_id, options, request = request)
        
        logger.info(f"Analysis completed successfully for user: {current_user_email}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in analyze_video: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post(
    "/analyze-realtime",
    summary="Real-time Emotion Analysis (Future)",
    description="Analyze video chunks for real-time emotion detection",
    response_model=dict,
    responses={
        501: {"description": "Not implemented yet"},
    }
)
async def analyze_realtime(
    video_chunk: UploadFile = File(..., description="Video chunk (4 seconds)"),
    session_id: Optional[str] = Form(None, description="Session ID for continuity"),
    timestamp: Optional[float] = Form(0.0, description="Timestamp in video"),
    current_user_email: str = Depends(get_current_user_email),
    user_service: UserService = Depends(get_user_service)
):
    """
    Real-time emotion analysis endpoint (Future feature).
    
    This endpoint will support real-time emotion analysis by accepting
    video chunks (4-second windows with 1-second stride).
    
    **Status**: Not yet implemented. This is a skeleton for future development.
    
    Args:
        video_chunk: Video chunk to analyze
        session_id: Session ID to maintain state across chunks
        timestamp: Timestamp of this chunk in the original video
        current_user_email: Current user email (from JWT token)
        user_service: User service dependency
    
    Returns:
        Analysis results for the chunk
    """
    logger.info(f"Real-time analysis request from user: {current_user_email}")
    
    raise HTTPException(
        status_code=501,
        detail="Real-time analysis is available via WebSocket at /ws/analyze. Use that endpoint for streaming analysis."
    )


@router.get(
    "/models",
    summary="List Available Models",
    description="Get information about available emotion recognition ai_models",
    response_model=dict
)
async def list_models():
    """
    List available emotion recognition ai_models.
    
    Returns:
        Information about available ai_models
    """
    models = [
        {
            "name": "Hybrid (Best)",
            "type": "hybrid",
            "description": "Ensemble combining RFRBoost + Deep Learning + Attention",
            "features": [
                "Modality weights",
                "Model agreement tracking",
                "Highest accuracy"
            ]
        },
        {
            "name": "RFRBoost Only",
            "type": "rfrboost",
            "description": "Random Feature Representation Boosting",
            "features": [
                "Fast processing",
                "Robust to noise",
                "Good with tabular features"
            ]
        },
        {
            "name": "Maelfabien Multimodal",
            "type": "maelfabien",
            "description": "Specialized ai_models: Text CNN-LSTM + Audio Time-CNN + Video XCeption",
            "features": [
                "Individual modality predictions",
                "Specialized architectures",
                "High interpretability"
            ]
        },
        {
            "name": "Emotion-LLaMA",
            "type": "emotion_llama",
            "description": "Transformer-based with reasoning and temporal awareness",
            "features": [
                "Natural language reasoning",
                "Emotion intensity scores",
                "Temporal context modeling"
            ]
        },
        {
            "name": "Simple Concatenation",
            "type": "baseline",
            "description": "Baseline approach with feature concatenation",
            "features": [
                "Fast processing",
                "Simple architecture",
                "Good baseline"
            ]
        }
    ]
    
    return {
        "success": True,
        "ai_models": models
    }

