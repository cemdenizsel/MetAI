"""
Batch Processing Service

Service for processing multiple videos in parallel.
"""

import os
import uuid
import logging
import tempfile
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import UploadFile

from data.mongodb_config import get_database
from tasks.emotion_tasks import batch_analyze_videos

logger = logging.getLogger(__name__)


class BatchProcessingService:
    """
    Service for batch video processing.
    
    Handles multi-file uploads, parallel processing, and result aggregation.
    """
    
    def __init__(self):
        """Initialize batch processing service."""
        self.db = get_database()
        self.batches_collection = self.db['emotion_batches']
        
        # Configuration
        self.max_videos = int(os.getenv("BATCH_MAX_VIDEOS", "10"))
        self.max_concurrent = int(os.getenv("BATCH_MAX_CONCURRENT", "3"))
        self.max_size_gb = int(os.getenv("BATCH_MAX_SIZE_GB", "3"))
        self.max_size_bytes = self.max_size_gb * 1024 * 1024 * 1024
        
        logger.info(f"BatchProcessingService initialized: max_videos={self.max_videos}, max_concurrent={self.max_concurrent}")
    
    async def submit_batch(
        self,
        video_files: List[UploadFile],
        user_id: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Submit a batch of videos for processing.
        
        Args:
            video_files: List of uploaded video files
            user_id: User ID
            options: Processing options
            
        Returns:
            Batch submission result
            
        Raises:
            ValueError: If validation fails
        """
        self._validate_batch(video_files)
        
        batch_id = str(uuid.uuid4())
        video_paths = []
        video_info = []
        
        try:
            for video_file in video_files:
                temp_path = await self._save_temp_file(video_file)
                video_paths.append(temp_path)
                video_info.append({
                    'filename': video_file.filename,
                    'status': 'pending',
                    'video_path': temp_path,
                })
            
            batch_doc = {
                'batch_id': batch_id,
                'user_id': user_id,
                'created_at': datetime.now(),
                'status': 'pending',
                'total_videos': len(video_files),
                'completed': 0,
                'processing': 0,
                'failed': 0,
                'pending': len(video_files),
                'videos': video_info,
                'options': options or {}
            }
            
            self.batches_collection.insert_one(batch_doc)
            
            task = batch_analyze_videos.apply_async(
                args=[batch_id, video_paths, user_id, options],
                queue='batch_processing'
            )
            
            self.batches_collection.update_one(
                {'batch_id': batch_id},
                {'$set': {'task_id': task.id}}
            )
            
            logger.info(f"Batch submitted: batch_id={batch_id}, videos={len(video_files)}, user={user_id}")
            
            return {
                'batch_id': batch_id,
                'total_videos': len(video_files),
                'message': 'Batch submitted successfully',
                'created_at': batch_doc['created_at'].isoformat(),
                'estimated_completion': None
            }
            
        except Exception as e:
            for path in video_paths:
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                except:
                    pass
            raise
    
    async def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Get batch processing status.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            Batch status dictionary
        """
        batch_doc = self.batches_collection.find_one({'batch_id': batch_id})
        
        if not batch_doc:
            return {
                'batch_id': batch_id,
                'status': 'not_found',
                'error': 'Batch not found'
            }
        
        total = batch_doc.get('total_videos', 0)
        completed = batch_doc.get('completed', 0)
        processing = batch_doc.get('processing', 0)
        failed = batch_doc.get('failed', 0)
        pending = batch_doc.get('pending', 0)
        
        progress = completed / total if total > 0 else 0.0
        
        videos_info = []
        for video in batch_doc.get('videos', []):
            videos_info.append({
                'filename': video.get('filename'),
                'status': video.get('status'),
                'progress': video.get('progress'),
                'error': video.get('error')
            })
        
        return {
            'batch_id': batch_id,
            'user_id': batch_doc.get('user_id'),
            'status': batch_doc.get('status'),
            'total_videos': total,
            'completed': completed,
            'processing': processing,
            'failed': failed,
            'pending': pending,
            'progress': progress,
            'created_at': batch_doc.get('created_at', '').isoformat() if batch_doc.get('created_at') else None,
            'estimated_completion': None,
            'videos': videos_info
        }
    
    async def get_batch_results(self, batch_id: str) -> Dict[str, Any]:
        """
        Get batch processing results.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            Batch results dictionary
        """
        batch_doc = self.batches_collection.find_one({'batch_id': batch_id})
        
        if not batch_doc:
            return {
                'batch_id': batch_id,
                'status': 'not_found',
                'error': 'Batch not found'
            }
        
        status = batch_doc.get('status')
        if status not in ['completed', 'partial', 'failed']:
            return {
                'batch_id': batch_id,
                'status': status,
                'message': f'Batch is not complete. Current status: {status}',
                'progress': batch_doc.get('completed', 0) / batch_doc.get('total_videos', 1)
            }
        
        results = batch_doc.get('results', [])
        
        return {
            'batch_id': batch_id,
            'user_id': batch_doc.get('user_id'),
            'status': status,
            'total_videos': batch_doc.get('total_videos'),
            'completed': batch_doc.get('completed'),
            'failed': batch_doc.get('failed'),
            'total_processing_time': batch_doc.get('total_processing_time', 0),
            'created_at': batch_doc.get('created_at', '').isoformat() if batch_doc.get('created_at') else None,
            'completed_at': batch_doc.get('completed_at', '').isoformat() if batch_doc.get('completed_at') else None,
            'results': results
        }
    
    async def cancel_batch(self, batch_id: str, user_id: str) -> Dict[str, Any]:
        """
        Cancel batch processing.
        
        Args:
            batch_id: Batch identifier
            user_id: User ID (for authorization)
            
        Returns:
            Cancellation result
        """
        batch_doc = self.batches_collection.find_one({'batch_id': batch_id})
        
        if not batch_doc:
            return {
                'success': False,
                'error': 'Batch not found'
            }
        
        if batch_doc.get('user_id') != user_id:
            return {
                'success': False,
                'error': 'Unauthorized: Batch belongs to another user'
            }
        
        task_id = batch_doc.get('task_id')
        if task_id:
            from celery_app import celery_app
            celery_app.control.revoke(task_id, terminate=True, signal='SIGKILL')
        
        self.batches_collection.update_one(
            {'batch_id': batch_id},
            {
                '$set': {
                    'status': 'cancelled',
                    'cancelled_at': datetime.now()
                }
            }
        )
        
        logger.info(f"Batch cancelled: batch_id={batch_id}, user={user_id}")
        
        return {
            'success': True,
            'message': 'Batch cancelled successfully'
        }
    
    async def list_user_batches(
        self,
        user_id: str,
        limit: int = 20,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List batches for a user.
        
        Args:
            user_id: User ID
            limit: Maximum number of batches to return
            skip: Number to skip (pagination)
            
        Returns:
            List of batch information
        """
        batches_cursor = self.batches_collection.find(
            {'user_id': user_id}
        ).sort('created_at', -1).skip(skip).limit(limit)
        
        batches = []
        
        for batch_doc in batches_cursor:
            batches.append({
                'batch_id': batch_doc.get('batch_id'),
                'total_videos': batch_doc.get('total_videos'),
                'status': batch_doc.get('status'),
                'completed': batch_doc.get('completed'),
                'failed': batch_doc.get('failed'),
                'created_at': batch_doc.get('created_at', '').isoformat() if batch_doc.get('created_at') else None,
                'completed_at': batch_doc.get('completed_at', '').isoformat() if batch_doc.get('completed_at') else None
            })
        
        return batches
    
    def _validate_batch(self, video_files: List[UploadFile]):
        """
        Validate batch submission.
        
        Args:
            video_files: List of video files
            
        Raises:
            ValueError: If validation fails
        """
        if len(video_files) == 0:
            raise ValueError("No video files provided")
        
        if len(video_files) > self.max_videos:
            raise ValueError(f"Too many videos. Maximum is {self.max_videos}")
        
        total_size = 0
        for video_file in video_files:
            if hasattr(video_file, 'size') and video_file.size:
                total_size += video_file.size
        
        if total_size > self.max_size_bytes:
            raise ValueError(f"Total batch size exceeds maximum of {self.max_size_gb}GB")
    
    async def _save_temp_file(self, video_file: UploadFile) -> str:
        """
        Save uploaded file to temporary location.
        
        Args:
            video_file: Uploaded video file
            
        Returns:
            Path to temporary file
        """
        filename = video_file.filename or "video.mp4"
        ext = os.path.splitext(filename)[1] or ".mp4"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext, prefix="batch_") as tmp_file:
            chunk_size = 1024 * 1024
            while True:
                chunk = await video_file.read(chunk_size)
                if not chunk:
                    break
                tmp_file.write(chunk)
            
            temp_path = tmp_file.name
        
        logger.debug(f"Saved batch video to {temp_path}")
        return temp_path


_batch_service = None


def get_batch_service() -> BatchProcessingService:
    """Get the global batch service instance."""
    global _batch_service
    if _batch_service is None:
        _batch_service = BatchProcessingService()
    return _batch_service

