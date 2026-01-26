"""
Emotion Analysis Celery Tasks

Async tasks for video emotion analysis processing.
"""

import os
import sys
import time
import glob
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from celery_app import celery_app
from services.emotion_service import EmotionAnalysisService
from data.mongodb_config import get_database

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="tasks.emotion_tasks.analyze_video_async")
def analyze_video_async(
    self,
    video_path: str,
    user_id: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Async task to analyze a video file for emotions.
    
    Args:
        self: Celery task instance (bound)
        video_path: Path to video file
        user_id: User ID
        options: Analysis options
        
    Returns:
        Analysis result dictionary
    """
    task_id = self.request.id
    logger.info(f"Starting async video analysis: task={task_id}, user={user_id}, video={video_path}")
    
    try:
        self.update_state(
            state='PROCESSING',
            meta={
                'progress': 0.0,
                'stage': 'initializing',
                'message': 'Starting video analysis...',
                'current': 0,
                'total': 100
            }
        )
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        with open(video_path, 'rb') as f:
            video_data = f.read()
        
        from io import BytesIO
        from fastapi import UploadFile
        
        video_file = UploadFile(
            file=BytesIO(video_data),
            filename=options.get('filename', os.path.basename(video_path))
        )
        
        service = EmotionAnalysisService()
        
        def progress_callback(message: str, progress: float):
            """Update task progress."""
            stage = "processing"
            if "Stage 1" in message or "input" in message.lower():
                stage = "extracting"
            elif "Stage 2" in message or "feature" in message.lower():
                stage = "features"
            elif "Stage 3" in message or "fusion" in message.lower():
                stage = "analyzing"
            elif "Stage 4" in message or "advanced" in message.lower():
                stage = "ai_analysis"
            
            self.update_state(
                state='PROCESSING',
                meta={
                    'progress': progress,
                    'stage': stage,
                    'message': message,
                    'current': int(progress * 100),
                    'total': 100
                }
            )
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                service.analyze_video(video_file, user_id, options)
            )
        finally:
            loop.close()
        
        self.update_state(
            state='SUCCESS',
            meta={
                'progress': 1.0,
                'stage': 'completed',
                'message': 'Analysis complete',
                'current': 100,
                'total': 100,
                'result': result.dict() if hasattr(result, 'dict') else result
            }
        )
        
        db = get_database()
        jobs_collection = db['emotion_jobs']
        
        jobs_collection.update_one(
            {'task_id': task_id},
            {
                '$set': {
                    'status': 'SUCCESS',
                    'result': result.dict() if hasattr(result, 'dict') else result,
                    'completed_at': datetime.now(),
                    'processing_time': result.total_processing_time if hasattr(result, 'total_processing_time') else 0
                }
            },
            upsert=True
        )
        
        logger.info(f"Video analysis completed: task={task_id}")
        
        return result.dict() if hasattr(result, 'dict') else result
        
    except Exception as e:
        logger.error(f"Error in async video analysis: {e}", exc_info=True)
        
        # Update task state to FAILURE
        self.update_state(
            state='FAILURE',
            meta={
                'progress': 0.0,
                'stage': 'failed',
                'message': f'Error: {str(e)}',
                'error': str(e)
            }
        )
        
        # Store error in MongoDB
        try:
            db = get_database()
            jobs_collection = db['emotion_jobs']
            
            jobs_collection.update_one(
                {'task_id': task_id},
                {
                    '$set': {
                        'status': 'FAILURE',
                        'error': str(e),
                        'completed_at': datetime.now()
                    }
                },
                upsert=True
            )
        except:
            pass
        
        raise


@celery_app.task(bind=True, name="tasks.emotion_tasks.batch_analyze_videos")
def batch_analyze_videos(
    self,
    batch_id: str,
    video_paths: List[str],
    user_id: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Async task to analyze multiple videos in batch.
    
    Args:
        self: Celery task instance (bound)
        batch_id: Batch identifier
        video_paths: List of video file paths
        user_id: User ID
        options: Analysis options
        
    Returns:
        Batch result dictionary
    """
    from celery import group
    
    task_id = self.request.id
    logger.info(f"Starting batch analysis: task={task_id}, batch={batch_id}, videos={len(video_paths)}")
    
    try:
        self.update_state(
            state='PROCESSING',
            meta={
                'progress': 0.0,
                'stage': 'queuing',
                'message': 'Queuing video analysis tasks...',
                'total_videos': len(video_paths),
                'completed': 0
            }
        )
        
        job = group(
            analyze_video_async.s(path, user_id, options)
            for path in video_paths
        )
        
        result = job.apply_async()
        
        total = len(video_paths)
        completed = 0
        results = []
        
        while not result.ready():
            completed = sum(1 for r in result.results if r.ready())
            progress = completed / total if total > 0 else 0
            
            self.update_state(
                state='PROCESSING',
                meta={
                    'progress': progress,
                    'stage': 'processing',
                    'message': f'Processing videos: {completed}/{total}',
                    'total_videos': total,
                    'completed': completed
                }
            )
            
            time.sleep(2)
        
        results = result.get()
        
        db = get_database()
        batches_collection = db['emotion_batches']
        
        batches_collection.update_one(
            {'batch_id': batch_id},
            {
                '$set': {
                    'status': 'completed',
                    'completed': len(results),
                    'completed_at': datetime.now(),
                    'results': results
                }
            }
        )
        
        logger.info(f"Batch analysis completed: batch={batch_id}, completed={len(results)}")
        
        return {
            'batch_id': batch_id,
            'total_videos': total,
            'completed': len(results),
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}", exc_info=True)
        
        # Update batch status
        try:
            db = get_database()
            batches_collection = db['emotion_batches']
            
            batches_collection.update_one(
                {'batch_id': batch_id},
                {
                    '$set': {
                        'status': 'failed',
                        'error': str(e),
                        'completed_at': datetime.now()
                    }
                }
            )
        except:
            pass
        
        raise


@celery_app.task(name="tasks.emotion_tasks.cleanup_temp_files")
def cleanup_temp_files(
    file_patterns: List[str],
    age_hours: int = 24
) -> Dict[str, Any]:
    """
    Periodic task to cleanup old temporary files.
    
    Args:
        file_patterns: List of file patterns to match (e.g., ["/tmp/emotion_*"])
        age_hours: Delete files older than this many hours
        
    Returns:
        Cleanup result dictionary
    """
    logger.info(f"Starting temp file cleanup: patterns={file_patterns}, age={age_hours}h")
    
    try:
        cutoff_time = datetime.now() - timedelta(hours=age_hours)
        deleted_count = 0
        deleted_size = 0
            
        for pattern in file_patterns:
            files = glob.glob(pattern)
                
            for file_path in files:
                try:
                    if os.path.isfile(file_path):
                        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                            
                        if file_time < cutoff_time:
                            file_size = os.path.getsize(file_path)
                            os.unlink(file_path)
                            deleted_count += 1
                            deleted_size += file_size
                            logger.debug(f"Deleted temp file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete {file_path}: {e}")
        
        logger.info(f"Cleanup completed: deleted={deleted_count}, size={deleted_size / 1024 / 1024:.2f}MB")
        
        return {
            'deleted_count': deleted_count,
            'deleted_size_mb': deleted_size / 1024 / 1024,
            'patterns': file_patterns,
            'age_hours': age_hours
        }
        
    except Exception as e:
        logger.error(f"Error in cleanup task: {e}", exc_info=True)
        raise

