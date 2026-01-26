"""
Job Management Service

Service for managing async emotion analysis jobs.
"""

import os
import logging
import tempfile
from typing import Dict, Any, Optional, List
from datetime import datetime
from fastapi import UploadFile

from celery_app import celery_app
from tasks.emotion_tasks import analyze_video_async
from data.mongodb_config import get_sync_mongo_client, mongo_config

logger = logging.getLogger(__name__)


class JobService:
    """
    Service for managing async emotion analysis jobs.

    Handles job submission, status checking, result retrieval, and cancellation.
    """

    def __init__(self):
        """Initialize job service."""
        client = get_sync_mongo_client()
        if client is None:
            raise RuntimeError("Failed to connect to MongoDB")
        self.db = client[mongo_config.database_name]
        self.jobs_collection = self.db['emotion_jobs']
        logger.info("JobService initialized")
    
    async def submit_job(
        self,
        video_file: UploadFile,
        user_id: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Submit a video for async analysis.
        
        Args:
            video_file: Uploaded video file
            user_id: User ID
            options: Analysis options
            
        Returns:
            Job information dictionary
        """
        try:
            temp_path = await self._save_temp_file(video_file)
            
            if options is None:
                options = {}
            options['filename'] = video_file.filename
            
            task = analyze_video_async.apply_async(
                args=[temp_path, user_id, options],
                queue='emotion_analysis'
            )
            
            job_doc = {
                'task_id': task.id,
                'user_id': user_id,
                'filename': video_file.filename,
                'status': 'PENDING',
                'progress': 0.0,
                'stage': 'queued',
                'created_at': datetime.now(),
                'video_path': temp_path,
                'options': options
            }
            
            self.jobs_collection.insert_one(job_doc)
            
            logger.info(f"Job submitted: task_id={task.id}, user={user_id}")
            
            return {
                'job_id': task.id,
                'status': 'PENDING',
                'message': 'Job submitted successfully',
                'created_at': job_doc['created_at'].isoformat(),
                'estimated_completion': None
            }
            
        except Exception as e:
            logger.error(f"Error submitting job: {e}", exc_info=True)
            raise
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get job status and progress.
        
        Args:
            job_id: Job/Task ID
            
        Returns:
            Status information dictionary
        """
        try:
            task_result = celery_app.AsyncResult(job_id)
            job_doc = self.jobs_collection.find_one({'task_id': job_id})
            
            if not job_doc:
                return {
                    'job_id': job_id,
                    'status': 'NOT_FOUND',
                    'error': 'Job not found'
                }
            
            status = {
                'job_id': job_id,
                'status': task_result.state,
                'created_at': job_doc.get('created_at', '').isoformat() if job_doc.get('created_at') else None,
            }
            
            if task_result.state == 'PENDING':
                status['progress'] = 0.0
                status['stage'] = 'queued'
                status['message'] = 'Job is queued and waiting to be processed'
                
            elif task_result.state == 'PROCESSING':
                info = task_result.info or {}
                status['progress'] = info.get('progress', 0.0)
                status['stage'] = info.get('stage', 'processing')
                status['message'] = info.get('message', 'Processing...')
                status['current'] = info.get('current', 0)
                status['total'] = info.get('total', 100)
                
            elif task_result.state == 'SUCCESS':
                status['progress'] = 1.0
                status['stage'] = 'completed'
                status['message'] = 'Analysis completed successfully'
                status['completed_at'] = job_doc.get('completed_at', '').isoformat() if job_doc.get('completed_at') else None
                status['processing_time'] = job_doc.get('processing_time', 0)
                
            elif task_result.state == 'FAILURE':
                info = task_result.info or {}
                status['progress'] = 0.0
                status['stage'] = 'failed'
                status['error'] = str(info) if not isinstance(info, dict) else info.get('error', str(task_result.info))
                status['message'] = 'Analysis failed'
                
            else:
                status['progress'] = 0.0
                status['stage'] = task_result.state.lower()
                status['message'] = f'Job status: {task_result.state}'
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting job status: {e}", exc_info=True)
            return {
                'job_id': job_id,
                'status': 'ERROR',
                'error': str(e)
            }
    
    async def get_job_result(self, job_id: str) -> Dict[str, Any]:
        """
        Get job result if completed.
        
        Args:
            job_id: Job/Task ID
            
        Returns:
            Result dictionary
        """
        try:
            task_result = celery_app.AsyncResult(job_id)
            
            if task_result.state != 'SUCCESS':
                return {
                    'job_id': job_id,
                    'status': task_result.state,
                    'message': f'Job is not complete. Current status: {task_result.state}',
                    'result': None
                }
            
            job_doc = self.jobs_collection.find_one({'task_id': job_id})
            
            if job_doc and 'result' in job_doc:
                result = job_doc['result']
            else:
                result = task_result.result
            
            return {
                'job_id': job_id,
                'status': 'SUCCESS',
                'message': 'Job completed successfully',
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Error getting job result: {e}", exc_info=True)
            return {
                'job_id': job_id,
                'status': 'ERROR',
                'error': str(e)
            }
    
    async def cancel_job(self, job_id: str, user_id: str) -> Dict[str, Any]:
        """
        Cancel a running job.
        
        Args:
            job_id: Job/Task ID
            user_id: User ID (for authorization)
            
        Returns:
            Cancellation result dictionary
        """
        try:
            job_doc = self.jobs_collection.find_one({'task_id': job_id})
            
            if not job_doc:
                return {
                    'job_id': job_id,
                    'success': False,
                    'error': 'Job not found'
                }
            
            if job_doc.get('user_id') != user_id:
                return {
                    'job_id': job_id,
                    'success': False,
                    'error': 'Unauthorized: Job belongs to another user'
                }
            
            celery_app.control.revoke(job_id, terminate=True, signal='SIGKILL')
            
            self.jobs_collection.update_one(
                {'task_id': job_id},
                {
                    '$set': {
                        'status': 'CANCELLED',
                        'cancelled_at': datetime.now()
                    }
                }
            )
            
            logger.info(f"Job cancelled: job_id={job_id}, user={user_id}")
            
            return {
                'job_id': job_id,
                'success': True,
                'message': 'Job cancelled successfully'
            }
            
        except Exception as e:
            logger.error(f"Error cancelling job: {e}", exc_info=True)
            return {
                'job_id': job_id,
                'success': False,
                'error': str(e)
            }
    
    async def list_user_jobs(
        self,
        user_id: str,
        limit: int = 20,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List jobs for a user.
        
        Args:
            user_id: User ID
            limit: Maximum number of jobs to return
            skip: Number of jobs to skip (pagination)
            
        Returns:
            List of job information dictionaries
        """
        try:
            jobs_cursor = self.jobs_collection.find(
                {'user_id': user_id}
            ).sort('created_at', -1).skip(skip).limit(limit)
            
            jobs = []
            
            for job_doc in jobs_cursor:
                task_id = job_doc.get('task_id')
                task_result = celery_app.AsyncResult(task_id)
                
                job_info = {
                    'job_id': task_id,
                    'filename': job_doc.get('filename'),
                    'status': task_result.state,
                    'created_at': job_doc.get('created_at', '').isoformat() if job_doc.get('created_at') else None,
                }
                
                if 'completed_at' in job_doc:
                    job_info['completed_at'] = job_doc['completed_at'].isoformat()
                
                if 'processing_time' in job_doc:
                    job_info['processing_time'] = job_doc['processing_time']
                
                jobs.append(job_info)
            
            return jobs
            
        except Exception as e:
            logger.error(f"Error listing user jobs: {e}", exc_info=True)
            return []
    
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
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext, prefix="emotion_") as tmp_file:
            chunk_size = 1024 * 1024
            while True:
                chunk = await video_file.read(chunk_size)
                if not chunk:
                    break
                tmp_file.write(chunk)
            
            temp_path = tmp_file.name
        
        logger.info(f"Saved uploaded file to {temp_path}")
        return temp_path


_job_service = None


def get_job_service() -> JobService:
    """Get the global job service instance."""
    global _job_service
    if _job_service is None:
        _job_service = JobService()
    return _job_service

