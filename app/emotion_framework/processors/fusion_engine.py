"""
Fusion engine for multimodal emotion prediction.

Handles different fusion strategies and returns emotion predictions.
"""

import os
import sys
import logging
from typing import Dict, Any, List
import numpy as np

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from emotion_framework.models.result_models import EmotionPrediction

logger = logging.getLogger(__name__)


class FusionEngine:
    """
    Handles multimodal fusion and emotion prediction.
    """
    
    def __init__(self, config: Dict[str, Any], emotion_labels: List[str]):
        """
        Initialize the fusion engine.
        
        Args:
            config: Configuration dictionary
            emotion_labels: List of emotion label strings
        """
        self.config = config
        self.emotion_labels = emotion_labels
        self.fusion_strategy = config.get('fusion_strategy', 'Hybrid (Best)')
    
    def predict(self, stage2_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform emotion prediction using multimodal fusion.
        
        Args:
            stage2_result: Results from stage 2 (feature extraction)
        
        Returns:
            Dictionary with prediction results
        """
        features_dict = stage2_result.get('features_dict', {})
        audio_feat = features_dict.get('audio')
        visual_feat = features_dict.get('visual')
        text_feat = features_dict.get('text')
        
        # Check if we have any features
        if audio_feat is None and visual_feat is None and text_feat is None:
            logger.warning("No features available for prediction")
            return {'prediction': None, 'combined_features': None}
        
        # Handle missing modalities with zero vectors
        if audio_feat is None:
            audio_feat = np.zeros(1)
        if visual_feat is None:
            visual_feat = np.zeros(1)
        if text_feat is None:
            text_feat = np.zeros(1)
        
        # Concatenate features
        combined_features = np.concatenate([audio_feat, visual_feat, text_feat])
        
        # Generate prediction based on fusion strategy
        # NOTE: In production, this would use trained ai_models
        # For now, we create a demo prediction
        prediction = self._generate_demo_prediction(
            audio_feat, visual_feat, text_feat,
            self.fusion_strategy
        )
        
        return {
            'prediction': prediction,
            'combined_features': combined_features
        }
    
    def _generate_demo_prediction(
        self,
        audio_feat: np.ndarray,
        visual_feat: np.ndarray,
        text_feat: np.ndarray,
        fusion_strategy: str
    ) -> EmotionPrediction:
        """
        Generate a demo prediction.
        
        In production, this would use trained ai_models. For now, generates
        random predictions for demonstration.
        
        Args:
            audio_feat: Audio features
            visual_feat: Visual features
            text_feat: Text features
            fusion_strategy: Fusion strategy name
        
        Returns:
            EmotionPrediction object
        """
        # Generate random emotion prediction
        predicted_emotion = np.random.choice(self.emotion_labels)
        predicted_label = self.emotion_labels.index(predicted_emotion)
        
        # Generate random confidences
        all_confidences = {
            label: float(np.random.uniform(0.05, 0.95))
            for label in self.emotion_labels
        }
        
        # Normalize confidences
        total = sum(all_confidences.values())
        all_confidences = {k: v/total for k, v in all_confidences.items()}
        
        # Get confidence for predicted emotion
        confidence = all_confidences[predicted_emotion]
        
        # Create base prediction
        prediction_dict = {
            'predicted_emotion': predicted_emotion,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'all_confidences': all_confidences,
            'fusion_method': fusion_strategy,
        }
        
        # Add strategy-specific fields
        if fusion_strategy == "Hybrid (Best)":
            # Hybrid model with modality weights and model agreement
            modality_weights = {
                'audio': float(np.random.uniform(0.2, 0.4)),
                'visual': float(np.random.uniform(0.3, 0.5)),
                'text': float(np.random.uniform(0.2, 0.4))
            }
            # Normalize weights
            total_weight = sum(modality_weights.values())
            modality_weights = {k: v/total_weight for k, v in modality_weights.items()}
            
            prediction_dict['modality_weights'] = modality_weights
            prediction_dict['individual_models'] = {
                'rfrboost': np.random.choice(self.emotion_labels),
                'attention_deep': np.random.choice(self.emotion_labels),
                'mlp_baseline': np.random.choice(self.emotion_labels)
            }
            
        elif fusion_strategy == "Maelfabien Multimodal":
            # Maelfabien approach with individual model predictions
            prediction_dict['individual_models'] = {
                'text_cnn_lstm': np.random.choice(self.emotion_labels),
                'audio_time_cnn': np.random.choice(self.emotion_labels),
                'video_xception': np.random.choice(self.emotion_labels)
            }
            
        elif fusion_strategy == "Emotion-LLaMA":
            # Emotion-LLaMA with reasoning
            prediction_dict['intensity'] = float(np.random.uniform(0.5, 0.9))
            prediction_dict['reasoning'] = (
                f"The person appears {predicted_emotion}. Analysis based on multimodal cues "
                "including facial expressions, voice tone, and semantic content."
            )
        
        return EmotionPrediction(**prediction_dict)

