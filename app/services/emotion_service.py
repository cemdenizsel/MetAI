"""
Emotion Analysis Service

Business logic for emotion analysis API endpoints.
Handles subscription validation, file processing, and result formatting.
"""

import os
import sys
import logging
import tempfile
import time
from typing import Dict, Any, Optional
from fastapi import UploadFile, HTTPException, Request

from utils.general import is_localhost


from emotion_framework import EmotionAnalysisPipeline, EmotionAnalysisResult
from emotion_framework.core.config_loader import load_framework_config
from services.subs_service import get_user_subscription
from services.cache_service import get_cache_service
from models.response_models import (
    MultiModelResponse,
    ModelResults,
    OverallPrediction,
    TemporalPrediction,
    MentalHealthAnalysis,
    VideoMetadata,
    Transcription,
    ModalityFeatures,
    EmotionConfidence,
    ModalityWeights,
    ModelAgreement,
    MaelfabienPredictions,
    EmotionLLaMaDetails,
)

logger = logging.getLogger(__name__)


class EmotionAnalysisService:
    """
    Service for analyzing videos for emotions.

    Handles:
    - Subscription validation
    - Video processing using emotion framework
    - Result formatting for API responses
    """

    def __init__(self):
        """Initialize the emotion analysis service."""
        self.config = load_framework_config()
        logger.info("EmotionAnalysisService initialized")

    async def analyze_video(
        self,
        video_file: UploadFile,
        user_id: str,
        options: Optional[Dict[str, Any]] = None,
        request: Optional[Request] = None,
    ) -> MultiModelResponse:
        """
        Analyze a video file for emotions.

        Args:
            video_file: Uploaded video file
            user_id: User ID making the request
            options: Optional processing options
            request: Optional request

        Returns:
            MultiModelResponse with analysis results

        Raises:
            HTTPException: If validation fails or processing errors occur
        """
        start_time = time.time()

        # Validate subscription
        await self._validate_subscription(user_id, request=request)

        # Save uploaded file temporarily
        temp_file_path = None
        try:
            temp_file_path = await self._save_temp_file(video_file)
            
            # Process options
            options = options or {}
            options['filename'] = video_file.filename
            use_cache = options.get('use_cache', True)
            
            # Check cache if enabled
            cache_service = get_cache_service()
            video_hash = None
            cached_result = None
            
            if use_cache:
                video_hash = cache_service.compute_video_hash(temp_file_path)
                cached_result = await cache_service.get_cached_result(video_hash)
                
                if cached_result:
                    logger.info(f"Returning cached result for video hash: {video_hash[:16]}...")
                    total_time = time.time() - start_time
                    
                    # Convert cached result to MultiModelResponse if needed
                    if isinstance(cached_result, dict):
                        cached_result['total_processing_time'] = total_time
                        return MultiModelResponse(**cached_result)
                    return cached_result

            # Process video using framework
            results = await self._process_video(temp_file_path, options)

            # Format API response
            api_response = self._format_api_response(results, video_file.filename)

            total_time = time.time() - start_time
            api_response.total_processing_time = total_time
            
            # Cache result if enabled
            if use_cache and video_hash:
                cache_ttl_days = options.get('cache_ttl_days', None)
                await cache_service.cache_result(
                    video_hash,
                    api_response.dict() if hasattr(api_response, 'dict') else api_response,
                    ttl_days=cache_ttl_days
                )

            logger.info(f"Video analysis completed in {total_time:.2f}s for user {user_id}")

            return api_response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error analyzing video: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error analyzing video: {str(e)}"
            )
        finally:
            # Cleanup temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file {temp_file_path}: {e}")

    async def _validate_subscription(self, user_id: str, request: Optional[Request] = None):
        """
        Validate user has active subscription.

        Args:
            user_id: User ID to validate

        Raises:
            HTTPException: If user doesn't have active subscription
        """
        try:
            # Skip subscription validation for local/dev environments (useful for development)
            if self._is_dev_or_local(request):
                logger.info("Local/dev request: skipping subscription validation for user %s", user_id)
                return

            subscription = await get_user_subscription(user_id)

            if not subscription:
                raise HTTPException(
                    status_code=403,
                    detail="No active subscription found. Please subscribe to use emotion analysis."
                )

            # Check if subscription is active
            status = subscription.get('status', '')
            if status not in ['active', 'trialing']:
                raise HTTPException(
                    status_code=403,
                    detail=f"Subscription status is '{status}'. Active subscription required."
                )

            logger.info(f"Subscription validated for user {user_id}")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error validating subscription: {e}")
            raise HTTPException(
                status_code=500,
                detail="Error validating subscription"
            )

    async def _save_temp_file(self, video_file: UploadFile) -> str:
        """
        Save uploaded file to temporary location.

        Args:
            video_file: Uploaded video file

        Returns:
            Path to temporary file
        """
        # Get file extension
        filename = video_file.filename or "video.mp4"
        ext = os.path.splitext(filename)[1] or ".mp4"

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            # Read and write in chunks to handle large files
            chunk_size = 1024 * 1024  # 1MB chunks
            while True:
                chunk = await video_file.read(chunk_size)
                if not chunk:
                    break
                tmp_file.write(chunk)

            temp_path = tmp_file.name

        logger.info(f"Saved uploaded file to {temp_path}")
        return temp_path

    async def _process_video(
        self,
        video_path: str,
        options: Dict[str, Any]
    ) -> EmotionAnalysisResult:
        """
        Process video using emotion framework.

        Args:
            video_path: Path to video file
            options: Processing options

        Returns:
            EmotionAnalysisResult
        """
        # Initialize pipeline
        pipeline = EmotionAnalysisPipeline(self.config)

        # Process video
        # Run in executor to avoid blocking (framework operations are CPU-intensive)
        import asyncio
        loop = asyncio.get_event_loop()

        result = await loop.run_in_executor(
            None,
            pipeline.analyze_video,
            video_path,
            None,  # config_overrides
            options
        )

        return result

    def _format_api_response(
        self,
        result: EmotionAnalysisResult,
        filename: str
    ) -> MultiModelResponse:
        """
        Format framework result as API response.

        Args:
            result: EmotionAnalysisResult from framework
            filename: Original filename

        Returns:
            MultiModelResponse
        """
        # Convert framework result to API ai_models
        model_results = self._convert_to_model_results(result, filename)

        return MultiModelResponse(
            success=True,
            message="Successfully analyzed video",
            results=[model_results],
            total_processing_time=result.processing_time
        )

    def _convert_to_model_results(
        self,
        result: EmotionAnalysisResult,
        filename: str
    ) -> ModelResults:
        """
        Convert EmotionAnalysisResult to ModelResults.

        Args:
            result: Framework result
            filename: Original filename

        Returns:
            ModelResults object
        """
        # Overall prediction
        overall_pred = None
        if result.prediction:
            overall_pred = OverallPrediction(
                predicted_emotion=result.prediction.predicted_emotion,
                confidence=result.prediction.confidence,
                all_confidences=EmotionConfidence(**result.prediction.all_confidences)
            )

        # Temporal predictions
        temporal_preds = []
        for tp in result.temporal_predictions:
            temporal_preds.append(TemporalPrediction(
                timestamp=tp.timestamp,
                emotion=tp.emotion,
                confidence=tp.confidences.get(tp.emotion, 0.0),
                all_confidences=EmotionConfidence(**tp.confidences)
            ))

        # Mental health analysis
        mental_health = None
        if result.mental_health_analysis:
            mh = result.mental_health_analysis
            mental_health = MentalHealthAnalysis(
                mental_health_score=mh.mental_health_score,
                avg_confidence=mh.avg_confidence,
                num_frames=mh.num_frames,
                dominant_emotion=mh.dominant_emotion,
                positive_percentage=mh.positive_percentage,
                negative_percentage=mh.negative_percentage,
                emotion_distribution=mh.emotion_distribution,
                status=mh.status,
                recommendation=mh.recommendation
            )

        # Video metadata
        video_meta = VideoMetadata(
            filename=filename,
            duration=result.metadata.duration,
            fps=result.metadata.fps,
            width=result.metadata.width,
            height=result.metadata.height,
            frame_count=result.metadata.frame_count
        )

        # Transcription
        transcription = None
        if result.transcription and result.transcription.text:
            transcription = Transcription(
                text=result.transcription.text,
                word_count=result.transcription.word_count,
                language=result.transcription.language
            )

        # Features
        feature_counts = result.features.get_counts()
        features = ModalityFeatures(
            audio_features=feature_counts.get('audio', 0),
            visual_features=feature_counts.get('visual', 0),
            text_features=feature_counts.get('text', 0)
        )

        # Model-specific data_model
        modality_weights = None
        model_agreement = None
        maelfabien_preds = None
        emotion_llama = None

        if result.prediction:
            # Modality weights (Hybrid model)
            if result.prediction.modality_weights:
                modality_weights = ModalityWeights(**result.prediction.modality_weights)

            # Model agreement (Hybrid model)
            if result.prediction.individual_models and 'rfrboost' in result.prediction.individual_models:
                indiv = result.prediction.individual_models
                agreement_status = self._determine_agreement_status(
                    indiv, result.prediction.predicted_emotion
                )
                model_agreement = ModelAgreement(
                    rfrboost=indiv.get('rfrboost', 'unknown'),
                    attention_deep=indiv.get('attention_deep', 'unknown'),
                    mlp_baseline=indiv.get('mlp_baseline', 'unknown'),
                    agreement_status=agreement_status
                )

            # Maelfabien predictions
            if result.prediction.individual_models and 'text_cnn_lstm' in result.prediction.individual_models:
                indiv = result.prediction.individual_models
                maelfabien_preds = MaelfabienPredictions(
                    text_cnn_lstm=indiv.get('text_cnn_lstm', 'unknown'),
                    audio_time_cnn=indiv.get('audio_time_cnn', 'unknown'),
                    video_xception=indiv.get('video_xception', 'unknown')
                )

            # Emotion-LLaMA details
            if result.prediction.intensity is not None and result.prediction.reasoning:
                emotion_llama = EmotionLLaMaDetails(
                    intensity=result.prediction.intensity,
                    reasoning=result.prediction.reasoning
                )

        # Create ModelResults
        return ModelResults(
            model_name=result.prediction.fusion_method if result.prediction else "Unknown",
            fusion_strategy=result.prediction.fusion_method if result.prediction else "Unknown",
            processing_time=result.processing_time,
            overall_prediction=overall_pred,
            temporal_predictions=temporal_preds,
            mental_health_analysis=mental_health,
            video_metadata=video_meta,
            transcription=transcription,
            features=features,
            modality_weights=modality_weights,
            model_agreement=model_agreement,
            maelfabien_predictions=maelfabien_preds,
            emotion_llama_details=emotion_llama
        )

    def _determine_agreement_status(self, models: Dict[str, str], predicted: str) -> str:
        """
        Determine model agreement status.

        Args:
            models: Dictionary of model predictions
            predicted: Overall predicted emotion

        Returns:
            Agreement status string
        """
        predictions = list(models.values())

        # All agree
        if len(set(predictions)) == 1:
            return "all_agree"

        # Majority agree with final prediction
        if predictions.count(predicted) >= 2:
            return "majority_agree"

        # Disagree
        return "disagree"


    def _is_dev_or_local(self, request: Optional[Request] = None) -> bool:
        """Return True if running in a dev-like environment or the incoming request is localhost."""
        app_env = os.getenv("APP_ENV", "").lower()
        if app_env in {"dev", "development", "local", "test", "testing"}:
            return True

        if request is None:
            return False

        # `utils.general.is_localhost` might accept a Request, but keep backwards-compatibility
        try:
            return bool(is_localhost(request))
        except TypeError:
            return bool(is_localhost())