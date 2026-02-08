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
        self._fer_analyzer = None
        
        logger.info("RealtimeVideoProcessor initialized")
    
    def process_chunk(self, chunk_path: str, timestamp: float) -> Dict[str, Any]:
        """
        Process a video chunk with lightweight pipeline.

        Strategy:
        1. Try FER (Facial Expression Recognition) first - most accurate for faces
        2. Fall back to multimodal fusion (audio + visual features)
        3. Use default neutral if all methods fail

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

            # Strategy 1: Try FER first if faces are detected (most accurate)
            fer_prediction = self._predict_emotion_from_visual(frames)
            if fer_prediction is not None and fer_prediction.get('confidence', 0) > 0.3:
                logger.info(f"✓ Using FER prediction: {fer_prediction['emotion']} (confidence: {fer_prediction['confidence']:.2f})")
                return fer_prediction

            # Strategy 2: Fall back to multimodal fusion (audio + visual features)
            features = self._extract_features_fast(frames, chunk_path)

            # Check if we have any meaningful features
            has_audio = len(features.get('audio', [])) > 0
            has_visual = len(features.get('visual', [])) > 0

            if has_audio or has_visual:
                logger.info(f"→ Using multimodal fusion (audio={has_audio}, visual={has_visual})")
                prediction = self._predict_emotion_fast(features)
                logger.info(f"→ Multimodal result: {prediction.get('emotion')} (confidence: {prediction.get('confidence', 0):.2f})")
                return prediction

            # Strategy 3: Last resort - default prediction
            logger.warning("No features extracted, using default prediction")
            return self._get_default_prediction()

        except Exception as e:
            logger.error(f"Error processing chunk: {e}", exc_info=True)
            return self._get_default_prediction()
    
    def _predict_emotion_from_visual(self, frames: List[np.ndarray]) -> Optional[Dict[str, Any]]:
        """
        Predict emotion from frames using FER (Facial Expression Recognition).

        Uses the middle frame for stability and averages predictions from
        multiple frames if needed for better accuracy.

        Returns:
            Emotion prediction dict, or None if FER unavailable/fails
        """
        try:
            if self._fer_analyzer is None:
                from emotion_framework.analyzers.fer_analyzer import FERAnalyzer
                logger.info("Initializing pre-trained FER analyzer for realtime processing")
                self._fer_analyzer = FERAnalyzer(model_type='pretrained')

            if not frames or len(frames) == 0:
                return None

            # Try multiple frames for robustness, weight middle frame more
            predictions = []
            frame_indices = [0, len(frames) // 2, -1] if len(frames) >= 3 else [len(frames) // 2]

            for idx in frame_indices:
                frame = frames[idx]
                if frame is None or frame.size == 0:
                    continue

                # FER expects RGB; OpenCV frames are BGR
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self._fer_analyzer.predict_emotion(frame_rgb)

                if result and result.get('confidence', 0) > 0:
                    predictions.append(result)

            if not predictions:
                logger.debug("FER: No faces detected in frames")
                return None

            # Average predictions if we have multiple
            if len(predictions) == 1:
                result = predictions[0]
            else:
                # Weighted average (middle frame gets 50%, others split remaining)
                weights = [0.25, 0.5, 0.25] if len(predictions) == 3 else [0.5, 0.5]
                weights = weights[:len(predictions)]

                # Average confidences across all emotion labels
                all_labels = set()
                for pred in predictions:
                    all_labels.update(pred.get('all_confidences', pred.get('confidences', {})).keys())

                avg_confidences = {}
                for label in all_labels:
                    weighted_sum = sum(
                        pred.get('all_confidences', pred.get('confidences', {})).get(label, 0) * weight
                        for pred, weight in zip(predictions, weights)
                    )
                    avg_confidences[label] = weighted_sum

                # Get dominant emotion
                top_emotion = max(avg_confidences.items(), key=lambda x: x[1])

                result = {
                    'emotion': top_emotion[0],
                    'confidence': top_emotion[1],
                    'all_confidences': avg_confidences,
                }

            # Log successful FER detection
            logger.info(f"✓ FER detected face: {result['emotion']} (confidence: {result['confidence']:.2f})")

            return {
                'emotion': result.get('emotion', 'neutral'),
                'confidence': result.get('confidence', 0.0),
                'confidences': result.get('all_confidences', result.get('confidences', {})),
            }

        except Exception as e:
            logger.info(f"✗ FER failed (no face detected or error): {e}")
            return None
    
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

            logger.info(f"  Audio extracted: {len(audio)} samples, sample_rate={sr}, duration={len(audio)/sr:.2f}s")

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
            # Realtime webcam chunks often have no audio or unsupported codec; we fall back to FER.
            logger.debug("Audio extraction skipped (%s): %s", type(e).__name__, repr(e), exc_info=True)
            return np.array([])


class IncrementalFusionEngine:
    """
    Quick emotion prediction engine for real-time analysis.

    Uses lightweight models and simplified fusion for speed.
    Now includes proper multimodal fusion using both audio and visual features.
    """

    def __init__(self, config: Dict[str, Any], emotion_labels: List[str]):
        """Initialize fusion engine."""
        self.config = config
        self.emotion_labels = emotion_labels
        self._model = None
        self._fer_analyzer = None

    def predict_fast(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fast emotion prediction from features using multimodal fusion.

        Args:
            features: Feature dictionary with 'visual' and 'audio' keys

        Returns:
            Emotion prediction dictionary
        """
        try:
            visual_features = features.get('visual', np.array([]))
            audio_features = features.get('audio', np.array([]))

            has_visual = len(visual_features) > 0
            has_audio = len(audio_features) > 0

            # Strategy: Use audio + visual features for robust prediction
            emotion_scores = {label: 0.0 for label in self.emotion_labels}

            # Audio-based emotion scoring (if available)
            if has_audio:
                audio_scores = self._score_from_audio(audio_features)
                for label, score in audio_scores.items():
                    emotion_scores[label] += score * 0.6  # 60% weight to audio

            # Visual-based emotion scoring (if available)
            if has_visual:
                visual_scores = self._score_from_visual(visual_features)
                for label, score in visual_scores.items():
                    emotion_scores[label] += score * 0.4  # 40% weight to visual

            # If neither modality available, return neutral
            if not has_audio and not has_visual:
                return self._get_neutral_prediction()

            # Normalize scores to probabilities
            total_score = sum(emotion_scores.values())
            if total_score > 0:
                confidences = {k: v / total_score for k, v in emotion_scores.items()}
            else:
                confidences = {label: 1.0 / len(self.emotion_labels) for label in self.emotion_labels}

            # Get top emotion
            emotion = max(confidences.items(), key=lambda x: x[1])[0]
            confidence = confidences[emotion]

            # Log top 3 final combined scores
            top_combined = sorted(confidences.items(), key=lambda x: x[1], reverse=True)[:3]
            logger.info(f"  Final combined: {', '.join([f'{k}={v:.2f}' for k, v in top_combined])}")

            return {
                'emotion': emotion,
                'confidence': confidence,
                'confidences': confidences,
            }

        except Exception as e:
            logger.error(f"Error predicting emotion: {e}", exc_info=True)
            return self._get_neutral_prediction()

    def _score_from_audio(self, audio_features: np.ndarray) -> Dict[str, float]:
        """
        Score emotions from audio features (MFCCs + spectral).

        Args:
            audio_features: Audio feature vector [mfcc_13, spectral_centroid, zcr]

        Returns:
            Dictionary of emotion scores
        """
        scores = {label: 0.1 for label in self.emotion_labels}  # Base score

        try:
            # Extract components
            mfcc_features = audio_features[:13] if len(audio_features) >= 13 else audio_features
            spectral_centroid = audio_features[13] if len(audio_features) > 13 else 0.0
            zcr = audio_features[14] if len(audio_features) > 14 else 0.0

            # MFCC-based features
            mfcc_energy = np.sqrt(np.mean(mfcc_features ** 2))  # RMS energy
            mfcc_variance = np.var(mfcc_features)

            logger.info(f"  Audio features: energy={mfcc_energy:.2f}, variance={mfcc_variance:.2f}, spectral={spectral_centroid:.2f}, zcr={zcr:.4f}")

            # Emotion heuristics based on audio characteristics:
            # High energy + high variance = excited emotions (happy, angry, surprise)
            # Low energy + low variance = calm emotions (sad, neutral)
            # High ZCR = fear, surprise

            # Recalibrated thresholds based on actual MFCC scales:
            # Energy typically ranges: 150-250
            # Variance typically ranges: 25,000-45,000
            # Spectral centroid: 700-2000 Hz
            # ZCR: 0.03-0.15

            # Happy: High energy, moderate-high variance
            if mfcc_energy > 195 and mfcc_variance > 35000:
                scores['happy'] += 0.5

            # Sad: Low energy, low variance
            if mfcc_energy < 180 and mfcc_variance < 32000:
                scores['sad'] += 0.5

            # Angry: Very high energy, very high variance, high spectral
            if mfcc_energy > 205 and mfcc_variance > 40000:
                scores['angry'] += 0.4

            # Fear: High ZCR (trembling voice), moderate energy
            if zcr > 0.10:
                scores['fear'] += 0.4
                scores['surprise'] += 0.3

            # Surprise: High energy spike, high variance
            if mfcc_energy > 200 and mfcc_variance > 38000:
                scores['surprise'] += 0.3

            # Neutral: Moderate energy, moderate variance
            if 180 <= mfcc_energy <= 195 and 32000 <= mfcc_variance <= 38000:
                scores['neutral'] += 0.6

            # Disgust: Higher spectral patterns
            if spectral_centroid > 1400:
                scores['disgust'] += 0.2

        except Exception as e:
            logger.warning(f"Error scoring audio features: {e}")
            # Return uniform scores on error
            scores = {label: 1.0 / len(self.emotion_labels) for label in self.emotion_labels}

        # Log top 3 audio scores
        top_audio = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        logger.info(f"  Top audio scores: {', '.join([f'{k}={v:.2f}' for k, v in top_audio])}")

        return scores

    def _score_from_visual(self, visual_features: np.ndarray) -> Dict[str, float]:
        """
        Score emotions from visual features (color histograms).

        Args:
            visual_features: Visual feature vector (RGB histograms)

        Returns:
            Dictionary of emotion scores
        """
        scores = {label: 0.1 for label in self.emotion_labels}  # Base score

        try:
            # Split RGB histograms (32 bins each = 96 total features)
            if len(visual_features) >= 96:
                hist_b = visual_features[:32]
                hist_g = visual_features[32:64]
                hist_r = visual_features[64:96]

                # Normalize histogram values to 0-1 range
                # Histograms are in range 0-N (where N is number of pixels per bin)
                # We need to normalize them to get meaningful brightness/contrast values
                max_hist_val = np.max(visual_features) + 1e-6
                norm_hist_b = hist_b / max_hist_val
                norm_hist_g = hist_g / max_hist_val
                norm_hist_r = hist_r / max_hist_val

                # Calculate color statistics (now in 0-1 range)
                brightness = np.mean([np.mean(norm_hist_b), np.mean(norm_hist_g), np.mean(norm_hist_r)])
                contrast = np.std([np.std(norm_hist_b), np.std(norm_hist_g), np.std(norm_hist_r)])

                # Red dominance (may indicate anger, excitement)
                # Compare red channel mean to overall mean
                overall_mean = np.mean(visual_features)
                red_mean = np.mean(hist_r)
                green_mean = np.mean(hist_g)
                blue_mean = np.mean(hist_b)

                red_ratio = red_mean / (overall_mean + 1e-6)
                green_ratio = green_mean / (overall_mean + 1e-6)
                blue_ratio = blue_mean / (overall_mean + 1e-6)

                logger.info(f"  Visual features: brightness={brightness:.2f}, contrast={contrast:.2f}, red_ratio={red_ratio:.2f}, green_ratio={green_ratio:.2f}, blue_ratio={blue_ratio:.2f}")

                # Blue dominance (may indicate sadness, calmness)
                blue_ratio = np.mean(hist_b) / (np.mean(visual_features) + 1e-6)

                # Color-based emotion heuristics:
                # Bright scenes = positive emotions
                # Dark scenes = negative emotions
                # High contrast = strong emotions

                # Happy: Bright, warm colors (red dominant)
                if brightness > 0.15 and red_ratio > 0.34:
                    scores['happy'] += 0.4

                # Sad: Dark, cool colors (blue dominant)
                if brightness < 0.12 and blue_ratio > 0.34:
                    scores['sad'] += 0.4

                # Angry: High contrast, red dominant
                if contrast > 0.08 and red_ratio > 0.35:
                    scores['angry'] += 0.3

                # Fear: Low brightness, high contrast
                if brightness < 0.13 and contrast > 0.09:
                    scores['fear'] += 0.3

                # Surprise: High brightness, high contrast
                if brightness > 0.18 and contrast > 0.08:
                    scores['surprise'] += 0.3

                # Neutral: Moderate brightness and contrast
                if 0.12 <= brightness <= 0.16 and 0.04 <= contrast <= 0.09:
                    scores['neutral'] += 0.5

                # Disgust: Greenish tint
                if green_ratio > 0.34:
                    scores['disgust'] += 0.2

        except Exception as e:
            logger.warning(f"Error scoring visual features: {e}")
            # Return uniform scores on error
            scores = {label: 1.0 / len(self.emotion_labels) for label in self.emotion_labels}

        # Log top 3 visual scores
        top_visual = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        logger.info(f"  Top visual scores: {', '.join([f'{k}={v:.2f}' for k, v in top_visual])}")

        return scores

    def _get_neutral_prediction(self) -> Dict[str, Any]:
        """Get neutral prediction when no features available."""
        confidences = {label: 0.1 for label in self.emotion_labels}
        confidences['neutral'] = 0.5

        total = sum(confidences.values())
        confidences = {k: v / total for k, v in confidences.items()}

        return {
            'emotion': 'neutral',
            'confidence': confidences['neutral'],
            'confidences': confidences,
        }

