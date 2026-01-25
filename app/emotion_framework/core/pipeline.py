"""
Main emotion analysis pipeline.

This module contains the EmotionAnalysisPipeline class that orchestrates
the entire video emotion analysis process.
"""

import os
import sys
import time
import logging
import traceback
from typing import Dict, Any, Optional, Callable
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from emotion_framework.models.result_models import (
    EmotionAnalysisResult,
    VideoMetadata,
    EmotionPrediction,
    TemporalPrediction,
    MentalHealthAnalysis,
    TranscriptionResult,
    FeatureInfo,
    AIAnalysisResult,
)
from emotion_framework.core.config_loader import load_framework_config, merge_config

logger = logging.getLogger(__name__)


class EmotionAnalysisPipeline:
    """
    Main pipeline for emotion analysis from video files.
    
    This class orchestrates the entire process:
    1. Video/audio extraction
    2. Feature extraction (audio, visual, text)
    3. Emotion prediction and fusion
    4. FER analysis
    5. AI agent analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None):
        """
        Initialize the emotion analysis pipeline.
        
        Args:
            config: Configuration dictionary. If None, loads from config_path or defaults.
            config_path: Path to configuration YAML file.
        """
        if config is None:
            config = load_framework_config(config_path)
        
        self.config = config
        self.emotion_labels = config.get('emotions', {}).get('labels', 
                                                              ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise'])
        
        # Initialize processors (lazy loading)
        self._video_processor = None
        self._feature_extractor = None
        self._fusion_engine = None
        self._fer_analyzer = None
        self._ai_agent = None
        
        logger.info("EmotionAnalysisPipeline initialized")
    
    def analyze_video(
        self,
        video_path: str,
        config_overrides: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> EmotionAnalysisResult:
        """
        Analyze a video file for emotions.
        
        Args:
            video_path: Path to video file
            config_overrides: Optional configuration overrides
            options: Optional processing options
            progress_callback: Optional callback function(message, progress) for progress updates
        
        Returns:
            EmotionAnalysisResult with all analysis data_model
        """
        start_time = time.time()
        
        # Merge configuration
        config = merge_config(self.config, config_overrides or {})
        options = options or {}
        
        try:
            # Stage 1: Input Processing
            self._update_progress(progress_callback, "Stage 1: Processing video input...", 0.05)
            stage1_result = self._process_stage1_input(video_path, config)
            
            # Stage 2: Feature Extraction
            self._update_progress(progress_callback, "Stage 2: Extracting features...", 0.25)
            stage2_result = self._process_stage2_features(stage1_result, config)
            
            # Stage 3: Fusion & Prediction
            self._update_progress(progress_callback, "Stage 3: Analyzing emotions...", 0.60)
            stage3_result = self._process_stage3_fusion(stage2_result, config)
            
            # Stage 4: Advanced Analysis (FER, AI Agent)
            self._update_progress(progress_callback, "Stage 4: Running advanced analysis...", 0.85)
            stage4_result = self._process_stage4_analysis(stage1_result, stage2_result, stage3_result, config, options)
            
            # Combine all results
            processing_time = time.time() - start_time
            
            result = EmotionAnalysisResult(
                metadata=stage1_result['metadata'],
                prediction=stage3_result.get('prediction'),
                temporal_predictions=stage4_result.get('temporal_predictions', []),
                mental_health_analysis=stage4_result.get('mental_health_analysis'),
                transcription=stage1_result.get('transcription'),
                ai_analysis=stage4_result.get('ai_analysis'),
                features=stage2_result['features'],
                frame_paths=stage1_result.get('frame_paths', []),
                frames_folder=stage1_result.get('frames_folder'),
                audio_path=stage1_result.get('audio_path'),
                processing_time=processing_time,
                audio=stage1_result.get('audio'),
                sample_rate=stage1_result.get('sample_rate'),
                frames=stage1_result.get('frames'),
                timestamps=stage1_result.get('timestamps'),
                combined_features=stage3_result.get('combined_features'),
            )
            
            self._update_progress(progress_callback, "Processing complete!", 1.0)
            logger.info(f"Video analysis completed in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing video: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _process_stage1_input(self, video_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 1: Process video input - extract audio, frames, and transcription.
        
        Args:
            video_path: Path to video file
            config: Configuration dictionary
        
        Returns:
            Dictionary with extracted data_model
        """
        from emotion_framework.processors import VideoProcessorWrapper
        
        processor = VideoProcessorWrapper(video_path, config)
        return processor.process(config)
    
    def _process_stage2_features(self, stage1_result: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 2: Extract features from audio, visual, and text modalities.
        
        Args:
            stage1_result: Results from stage 1
            config: Configuration dictionary
        
        Returns:
            Dictionary with extracted features
        """
        from emotion_framework.processors import FeatureExtractorOrchestrator
        
        extractor = FeatureExtractorOrchestrator(config)
        return extractor.extract_features(stage1_result)
    
    def _process_stage3_fusion(self, stage2_result: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 3: Fuse modalities and predict emotions.
        
        Args:
            stage2_result: Results from stage 2
            config: Configuration dictionary
        
        Returns:
            Dictionary with emotion predictions
        """
        from emotion_framework.processors import FusionEngine
        
        fusion_engine = FusionEngine(config, self.emotion_labels)
        return fusion_engine.predict(stage2_result)
    
    def _process_stage4_analysis(
        self,
        stage1_result: Dict[str, Any],
        stage2_result: Dict[str, Any],
        stage3_result: Dict[str, Any],
        config: Dict[str, Any],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Stage 4: Advanced analysis - FER and AI agent.
        
        Args:
            stage1_result: Results from stage 1
            stage2_result: Results from stage 2
            stage3_result: Results from stage 3
            config: Configuration dictionary
            options: Processing options
        
        Returns:
            Dictionary with advanced analysis results
        """
        results = {}
        
        # FER Analysis
        if stage1_result.get('frame_paths') and len(stage1_result['frame_paths']) > 0:
            try:
                from emotion_framework.analyzers import FERAnalyzer
                
                fer = FERAnalyzer(model_type='custom_cnn')
                
                # Analyze frame sequence
                temporal_predictions = fer.analyze_frame_sequence(
                    frame_paths=stage1_result['frame_paths'],
                    timestamps=[i * 5.0 for i in range(len(stage1_result['frame_paths']))]
                )
                
                results['temporal_predictions'] = [
                    TemporalPrediction(
                        timestamp=pred['timestamp'],
                        emotion=pred['emotion'],
                        confidences=pred['confidences']
                    )
                    for pred in temporal_predictions
                ]
                
                # Calculate mental health score
                if temporal_predictions:
                    mh_data = fer.calculate_mental_health_score(temporal_predictions)
                    results['mental_health_analysis'] = MentalHealthAnalysis(**mh_data)
                    
            except Exception as e:
                logger.warning(f"FER analysis failed: {e}")
                results['temporal_predictions'] = []
        
        # AI Agent Analysis
        if options.get('run_ai_analysis', True):
            try:
                from emotion_framework.analyzers import MeetingAnalysisAgent
                
                # Get LLM provider from options
                provider = options.get('llm_provider', 'cloud')
                
                agent = MeetingAnalysisAgent(provider=provider)
                
                # Prepare data_model for agent
                emotion_data = {
                    'overall_prediction': stage3_result.get('prediction'),
                    'temporal_predictions': results.get('temporal_predictions', []),
                    'mental_health_analysis': results.get('mental_health_analysis')
                }
                
                video_meta = {
                    'filename': options.get('filename', 'video.mp4'),
                    'duration': stage1_result.get('metadata', {}).get('duration', 0),
                    'upload_date': options.get('upload_date', 'Unknown')
                }
                
                transcription = stage1_result.get('transcription', {}).get('text', '')
                context_query = options.get('context_query', f"meeting analysis video call {transcription[:200] if transcription else ''}")
                
                # Generate AI analysis
                ai_result = agent.analyze_meeting(
                    emotion_results=emotion_data,
                    video_metadata=video_meta,
                    transcription=transcription,
                    context_query=context_query
                )
                
                results['ai_analysis'] = AIAnalysisResult(**ai_result) if ai_result else None
                
            except Exception as e:
                logger.warning(f"AI agent analysis failed: {e}")
                results['ai_analysis'] = AIAnalysisResult(
                    agent_available=False,
                    error=str(e)
                )
        
        return results
    
    def _update_progress(self, callback: Optional[Callable], message: str, progress: float):
        """Update progress via callback if provided."""
        if callback:
            try:
                callback(message, progress)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

