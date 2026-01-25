"""
Feature extraction orchestrator.

Wraps the existing feature extractors to provide a unified interface.
"""

import os
import sys
import logging
from typing import Dict, Any
import numpy as np

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from modules.stage2_unimodal import AudioFeatureExtractor, VisualFeatureExtractor, TextFeatureExtractor
from emotion_framework.models.result_models import FeatureInfo

logger = logging.getLogger(__name__)


class FeatureExtractorOrchestrator:
    """
    Orchestrates feature extraction from all modalities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feature extractor orchestrator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.extractors = {}
        
        # Initialize extractors based on config
        if config['modalities']['audio']['enabled']:
            self.extractors['audio'] = AudioFeatureExtractor(config['modalities']['audio'])
        
        if config['modalities']['visual']['enabled']:
            self.extractors['visual'] = VisualFeatureExtractor(config['modalities']['visual'])
        
        if config['modalities']['text']['enabled']:
            self.extractors['text'] = TextFeatureExtractor(config['modalities']['text'])
        
        logger.info(f"Initialized {len(self.extractors)} feature extractors")
    
    def extract_features(self, stage1_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from all modalities.
        
        Args:
            stage1_result: Results from stage 1 (video processing)
        
        Returns:
            Dictionary with extracted features
        """
        features_dict = {}
        
        # Audio features
        if 'audio' in self.extractors and len(stage1_result.get('audio', [])) > 0:
            try:
                audio_features = self.extractors['audio'].extract_all_features(
                    stage1_result['audio'],
                    stage1_result['sample_rate'],
                    stage1_result.get('audio_path')
                )
                features_dict['audio'] = audio_features
                logger.info(f"Extracted {len(audio_features)} audio features")
            except Exception as e:
                logger.error(f"Audio feature extraction failed: {e}")
                features_dict['audio'] = None
        else:
            features_dict['audio'] = None
        
        # Visual features
        if 'visual' in self.extractors and len(stage1_result.get('frames', [])) > 0:
            try:
                visual_features = self.extractors['visual'].extract_video_features(
                    stage1_result['frames']
                )
                features_dict['visual'] = visual_features
                logger.info(f"Extracted {len(visual_features)} visual features")
            except Exception as e:
                logger.error(f"Visual feature extraction failed: {e}")
                features_dict['visual'] = None
        else:
            features_dict['visual'] = None
        
        # Text features
        if 'text' in self.extractors and stage1_result.get('transcription'):
            try:
                text = stage1_result['transcription'].text
                if text:
                    text_features = self.extractors['text'].extract_all_features(text)
                    features_dict['text'] = text_features
                    logger.info(f"Extracted {len(text_features)} text features")
                else:
                    features_dict['text'] = None
            except Exception as e:
                logger.error(f"Text feature extraction failed: {e}")
                features_dict['text'] = None
        else:
            features_dict['text'] = None
        
        # Create FeatureInfo object
        features = FeatureInfo(
            audio=features_dict.get('audio'),
            visual=features_dict.get('visual'),
            text=features_dict.get('text')
        )
        
        return {
            'features': features,
            'features_dict': features_dict  # Keep dict for backward compatibility
        }

