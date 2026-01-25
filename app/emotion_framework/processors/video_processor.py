"""
Video processor wrapper.

Wraps the existing VideoProcessor and ASRModule to provide a clean interface.
"""

import os
import sys
import logging
from typing import Dict, Any, Tuple, List
import numpy as np

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from modules.stage1_input import VideoProcessor, ASRModule
from emotion_framework.models.result_models import VideoMetadata, TranscriptionResult

logger = logging.getLogger(__name__)


class VideoProcessorWrapper:
    """
    Wraps VideoProcessor and ASRModule to provide a unified interface.
    """
    
    def __init__(self, video_path: str, config: Dict[str, Any]):
        """
        Initialize the video processor wrapper.
        
        Args:
            video_path: Path to video file
            config: Configuration dictionary
        """
        self.video_path = video_path
        self.config = config
        self.processor = VideoProcessor(video_path, output_dir="data_model/processed")
    
    def process(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process video: extract audio, frames, and transcription.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            Dictionary with all extracted data_model
        """
        results = {}
        
        # Extract audio
        try:
            audio, sr, audio_path = self.processor.extract_audio(
                sample_rate=config['modalities']['audio']['sample_rate']
            )
            results['audio'] = audio
            results['sample_rate'] = sr
            results['audio_path'] = audio_path
            
            if len(audio) == 0:
                logger.warning(f"Audio extraction returned empty array for {self.video_path}")
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            results['audio'] = np.array([])
            results['sample_rate'] = 16000
            results['audio_path'] = None
        
        # Extract frames
        try:
            frames, timestamps = self.processor.extract_frames(
                fps=config['modalities']['visual']['fps']
            )
            results['frames'] = frames
            results['timestamps'] = timestamps
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            results['frames'] = []
            results['timestamps'] = []
        
        # Extract frames to files (LlamaIndex approach: 1 per 5 seconds)
        try:
            frames_folder = os.path.join("data_model/processed", "frames")
            frame_paths = self.processor.extract_frames_to_files(
                output_folder=frames_folder,
                fps=0.2  # 1 frame every 5 seconds
            )
            results['frame_paths'] = frame_paths
            results['frames_folder'] = frames_folder
        except Exception as e:
            logger.error(f"Frame extraction to files failed: {e}")
            results['frame_paths'] = []
            results['frames_folder'] = None
        
        # Get metadata
        try:
            metadata_dict = self.processor.get_video_metadata()
            results['metadata'] = VideoMetadata(
                duration=metadata_dict.get('duration', 0.0),
                fps=metadata_dict.get('fps', 0.0),
                width=metadata_dict.get('width', 0),
                height=metadata_dict.get('height', 0),
                frame_count=metadata_dict.get('frame_count', 0),
                filename=os.path.basename(self.video_path)
            )
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            results['metadata'] = VideoMetadata(
                duration=0.0, fps=0.0, width=0, height=0, frame_count=0,
                filename=os.path.basename(self.video_path)
            )
        
        # Transcribe audio
        if config['modalities']['text']['enabled'] and len(results.get('audio', [])) > 0:
            try:
                asr = ASRModule(model_name="base")
                transcription_dict = asr.transcribe(results['audio'], results['sample_rate'])
                
                text = transcription_dict.get('text', '')
                results['transcription'] = TranscriptionResult(
                    text=text,
                    word_count=len(text.split()) if text else 0,
                    language=transcription_dict.get('language', 'en'),
                    segments=transcription_dict.get('segments', [])
                )
                
                if not text.strip():
                    logger.info("Transcription returned empty text")
            except Exception as e:
                logger.error(f"Transcription failed: {e}")
                results['transcription'] = TranscriptionResult(
                    text='', word_count=0, language='en', segments=[]
                )
        else:
            results['transcription'] = TranscriptionResult(
                text='', word_count=0, language='en', segments=[]
            )
        
        return results

