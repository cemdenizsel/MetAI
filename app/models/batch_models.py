"""
Batch Processing Models

Pydantic ai_models for batch video processing.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class BatchVideoItem(BaseModel):
    """Single video item in a batch."""
    filename: str = Field(..., description="Video filename")
    status: str = Field(..., description="Processing status (pending, processing, completed, failed)")
    job_id: Optional[str] = Field(None, description="Celery task ID")
    progress: Optional[float] = Field(None, ge=0, le=1, description="Processing progress (0-1)")
    result_id: Optional[str] = Field(None, description="MongoDB result document ID")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")


class BatchSubmitRequest(BaseModel):
    """Request model for batch submission (not used with multipart/form-data_model)."""
    fusion_strategy: Optional[str] = Field(None, description="Fusion strategy override")
    run_ai_analysis: Optional[bool] = Field(True, description="Whether to run AI analysis")
    llm_provider: Optional[str] = Field("cloud", description="LLM provider: 'cloud' or 'local'")
    use_cache: Optional[bool] = Field(True, description="Whether to use result caching")


class BatchSubmitResponse(BaseModel):
    """Response model after batch submission."""
    success: bool = Field(..., description="Whether batch was submitted successfully")
    batch_id: str = Field(..., description="Unique batch identifier")
    total_videos: int = Field(..., description="Total number of videos in batch")
    message: str = Field(..., description="Status message")
    estimated_completion: Optional[str] = Field(None, description="Estimated completion time (ISO format)")
    created_at: str = Field(..., description="Batch creation time (ISO format)")


class BatchStatus(BaseModel):
    """Overall batch processing status."""
    batch_id: str = Field(..., description="Batch identifier")
    user_id: str = Field(..., description="User identifier")
    status: str = Field(..., description="Overall batch status")
    total_videos: int = Field(..., description="Total videos in batch")
    completed: int = Field(..., description="Number of completed videos")
    processing: int = Field(..., description="Number of videos currently processing")
    failed: int = Field(..., description="Number of failed videos")
    pending: int = Field(..., description="Number of pending videos")
    progress: float = Field(..., ge=0, le=1, description="Overall progress (0-1)")
    created_at: str = Field(..., description="Batch creation time")
    estimated_completion: Optional[str] = Field(None, description="Estimated completion time")
    videos: List[BatchVideoItem] = Field(..., description="List of videos with their statuses")


class BatchResult(BaseModel):
    """Complete batch processing result."""
    batch_id: str = Field(..., description="Batch identifier")
    user_id: str = Field(..., description="User identifier")
    status: str = Field(..., description="Overall batch status (completed, failed, partial)")
    total_videos: int = Field(..., description="Total videos")
    completed: int = Field(..., description="Successfully completed")
    failed: int = Field(..., description="Failed videos")
    total_processing_time: float = Field(..., description="Total processing time for all videos")
    created_at: str = Field(..., description="Batch creation time")
    completed_at: Optional[str] = Field(None, description="Batch completion time")
    results: List[Dict[str, Any]] = Field(..., description="Analysis results for each video")


class BatchListItem(BaseModel):
    """Batch item for list view."""
    batch_id: str = Field(..., description="Batch identifier")
    total_videos: int = Field(..., description="Total videos")
    status: str = Field(..., description="Batch status")
    completed: int = Field(..., description="Completed videos")
    failed: int = Field(..., description="Failed videos")
    created_at: str = Field(..., description="Creation time")
    completed_at: Optional[str] = Field(None, description="Completion time")


class BatchListResponse(BaseModel):
    """Response for batch list endpoint."""
    success: bool = Field(..., description="Success status")
    count: int = Field(..., description="Number of batches returned")
    total: Optional[int] = Field(None, description="Total number of batches for user")
    batches: List[BatchListItem] = Field(..., description="List of batch items")

