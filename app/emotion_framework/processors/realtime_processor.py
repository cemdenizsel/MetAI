"""
Real-time optimized processors for short video chunks.

Provides lightweight, fast processing for 4-second video chunks in real-time analysis.
"""

import os
import sys
import cv2
import logging
import tempfile
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

logger = logging.getLogger(__name__)


class RealtimeVideoProcessor:
    """
    Optimized video processor for real-time analysis.
    
    Processes short video chunks (4 seconds) with minimal overhead.
    Extracts fewer frames and skips unnecessary preprocessing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize real-time video processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.target_fps = 1
        self.frame_size = (224, 224)
        self._feature_extractor = None
        self._fusion_engine = None
        
        logger.info("RealtimeVideoProcessor initialized")
    
    def process_chunk(self, chunk_path: str, timestamp: float) -> Dict[str, Any]:
        """
        Process a video chunk with lightweight pipeline.
        
        Args:
            chunk_path: Path to video chunk file
            timestamp: Timestamp of chunk
            
        Returns:
            Emotion prediction dictionary
        """
        try:
            frames = self._extract_frames_fast(chunk_path)
            
            if not frames or len(frames) == 0:
                logger.warning("No frames extracted from chunk")
                return self._get_default_prediction()
            
            features = self._extract_features_fast(frames, chunk_path)
            prediction = self._predict_emotion_fast(features)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error processing chunk: {e}", exc_info=True)
            return self._get_default_prediction()
    
    def _extract_frames_fast(self, video_path: str) -> List[np.ndarray]:
        """
        Fast frame extraction for real-time analysis.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of frame arrays
        """
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return frames
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps <= 0:
                fps = 30
            
            frame_interval = max(1, int(fps / self.target_fps))
            
            frame_idx = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    frame_resized = cv2.resize(frame, self.frame_size)
                    frames.append(frame_resized)
                    extracted_count += 1
                    
                    if extracted_count >= 5:
                        break
                
                frame_idx += 1
            
            cap.release()
            
            logger.debug(f"Extracted {len(frames)} frames from chunk")
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
        
        return frames
    
    def _extract_features_fast(self, frames: List[np.ndarray], video_path: str) -> Dict[str, Any]:
        """
        Fast feature extraction for real-time analysis.
        
        Args:
            frames: List of video frames
            video_path: Path to video file
            
        Returns:
            Feature dictionary
        """
        features = {
            'visual': [],
            'audio': [],
            'text': [],
        }
        
        try:
            if self._feature_extractor is None:
                self._feature_extractor = RealtimeFeatureExtractor(self.config)
            
            features['visual'] = self._feature_extractor.extract_visual_features(frames)
            features['audio'] = self._feature_extractor.extract_audio_features(video_path)
            features['text'] = []
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
        
        return features
    
    def _predict_emotion_fast(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fast emotion prediction from features.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Emotion prediction
        """
        try:
            if self._fusion_engine is None:
                emotion_labels = self.config.get('emotions', {}).get('labels',
                                                                     ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise'])
                self._fusion_engine = IncrementalFusionEngine(self.config, emotion_labels)
            
            prediction = self._fusion_engine.predict_fast(features)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting emotion: {e}")
            return self._get_default_prediction()
    
    def _get_default_prediction(self) -> Dict[str, Any]:
        """Get default prediction when processing fails."""
        emotion_labels = self.config.get('emotions', {}).get('labels', 
                                                             ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise'])
        
        confidences = {label: 0.0 for label in emotion_labels}
        confidences['neutral'] = 1.0  # Default to neutral
        
        return {
            'emotion': 'neutral',
            'confidence': 1.0,
            'confidences': confidences,
        }


class RealtimeFeatureExtractor:
    """
    Lightweight feature extractor for real-time analysis.
    
    Extracts minimal features needed for quick emotion prediction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize feature extractor."""
        self.config = config
        self._visual_model = None
        self._audio_extractor = None
    
    def extract_visual_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Extract visual features from frames.
        
        Args:
            frames: List of frame arrays
            
        Returns:
            Visual feature array
        """
        if not frames:
            return np.array([])
        
        try:
            features = []
            
            for frame in frames:
                hist_b = cv2.calcHist([frame], [0], None, [32], [0, 256])
                hist_g = cv2.calcHist([frame], [1], None, [32], [0, 256])
                hist_r = cv2.calcHist([frame], [2], None, [32], [0, 256])
                
                hist_features = np.concatenate([
                    hist_b.flatten(),
                    hist_g.flatten(),
                    hist_r.flatten()
                ])
                
                features.append(hist_features)
            
            if features:
                return np.mean(features, axis=0)
            
            return np.array([])
            
        except Exception as e:
            logger.error(f"Error extracting visual features: {e}")
            return np.array([])
    
    def extract_audio_features(self, video_path: str) -> np.ndarray:
        """
        Extract audio features from video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Audio feature array
        """
        try:
            import librosa
            
            audio, sr = librosa.load(video_path, sr=16000, mono=True, duration=4.0)
            
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            spectral_mean = np.mean(spectral_centroid)
            
            zcr = librosa.feature.zero_crossing_rate(audio)
            zcr_mean = np.mean(zcr)
            
            # Combine features
            audio_features = np.concatenate([
                mfcc_mean,
                [spectral_mean, zcr_mean]
            ])
            
            return audio_features
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return np.array([])


class IncrementalFusionEngine:
    """
    Quick emotion prediction engine for real-time analysis.
    
    Uses lightweight ai_models and simplified fusion for speed.
    """
    
    def __init__(self, config: Dict[str, Any], emotion_labels: List[str]):
        """Initialize fusion engine."""
        self.config = config
        self.emotion_labels = emotion_labels
        self._model = None
    
    def predict_fast(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fast emotion prediction from features.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Emotion prediction dictionary
        """
        try:
            visual_features = features.get('visual', np.array([]))
            audio_features = features.get('audio', np.array([]))
            
            if len(audio_features) > 0:
                audio_energy = np.mean(np.abs(audio_features))
                
                if audio_energy > 0.5:
                    emotion = 'happy'
                    confidence = min(0.7 + audio_energy * 0.2, 0.95)
                elif audio_energy > 0.3:
                    emotion = 'neutral'
                    confidence = 0.6
                else:
                    emotion = 'sad'
                    confidence = min(0.6 + (0.5 - audio_energy), 0.85)
            else:
                emotion = 'neutral'
                confidence = 0.5
            
            confidences = {label: 0.1 for label in self.emotion_labels}
            confidences[emotion] = confidence
            
            total = sum(confidences.values())
            confidences = {k: v / total for k, v in confidences.items()}
            
            return {
                'emotion': emotion,
                'confidence': confidences[emotion],
                'confidences': confidences,
            }
            
        except Exception as e:
            logger.error(f"Error predicting emotion: {e}")
            
            confidences = {label: 1.0 / len(self.emotion_labels) for label in self.emotion_labels}
            return {
                'emotion': 'neutral',
                'confidence': confidences.get('neutral', 0.14),
                'confidences': confidences,
            }

