"""
RAGFlow Service

Handles integration with RAGFlow API for document lookup and retrieval.
Manages user-specific dataset isolation and provides progress updates via async generators.
"""

import asyncio
import json
import logging
import time
from typing import AsyncGenerator, Dict, List, Optional
from pydantic import BaseModel
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from config.ragflow_config import RAGFlowConfig


logger = logging.getLogger(__name__)


class RAGFlowLookupRequest(BaseModel):
    """
    Request model for RAGFlow lookup operations.
    """
    query: str
    dataset_ids: List[str]
    topic_keywords: Optional[List[str]] = []
    context_window: Optional[int] = 500
    top_k: Optional[int] = 5


class RAGFlowDocumentResult(BaseModel):
    """
    Model for individual document results from RAGFlow.
    """
    document_id: str
    content: str
    page_number: Optional[int] = None
    bbox_coordinates: Optional[List[float]] = None  # [x, y, width, height]
    source_file: Optional[str] = None
    confidence_score: float
    metadata: Optional[Dict] = {}


class RAGFlowService:
    """
    Service class for interacting with RAGFlow API.
    """

    def __init__(self, config: RAGFlowConfig = None):
        self.config = config or RAGFlowConfig()
        self.timeout = aiohttp.ClientTimeout(total=self.config.total_timeout)  # Configurable timeout for long operations

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(
            multiplier=1,
            min=4,
            max=10
        )
    )
    async def _make_request(self, method: str, endpoint: str, data: Optional[dict] = None):
        """
        Make a request to the RAGFlow API with retry logic.
        """
        url = f"{self.config.api_url}{endpoint}"

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                if method.upper() == 'GET':
                    async with session.get(url, headers=self.config.headers) as response:
                        result = await response.json()
                elif method.upper() == 'POST':
                    async with session.post(url, headers=self.config.headers, json=data) as response:
                        result = await response.json()
                elif method.upper() == 'PUT':
                    async with session.put(url, headers=self.config.headers, json=data) as response:
                        result = await response.json()
                elif method.upper() == 'DELETE':
                    async with session.delete(url, headers=self.config.headers) as response:
                        result = await response.json()
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                # Raise exception for bad status codes
                response.raise_for_status()
                return result
        except aiohttp.ServerTimeoutError:
            logger.error(f"RAGFlow API request timed out for {method} {endpoint}")
            raise TimeoutError(f"Request to {endpoint} timed out after {self.timeout.total} seconds")
        except aiohttp.ClientResponseError as e:
            logger.error(f"RAGFlow API request failed: {e.status} - {e.message}")
            # Provide more specific error messages based on status code
            if e.status == 401:
                raise PermissionError("Unauthorized: Invalid RAGFlow API key")
            elif e.status == 403:
                raise PermissionError("Forbidden: Access denied to RAGFlow resource")
            elif e.status == 404:
                raise FileNotFoundError(f"RAGFlow resource not found: {endpoint}")
            elif e.status >= 500:
                raise ConnectionError(f"RAGFlow server error: {e.status}")
            else:
                raise
        except aiohttp.ClientConnectorError as e:
            logger.error(f"Unable to connect to RAGFlow API: {str(e)}")
            raise ConnectionError(f"Cannot connect to RAGFlow API at {self.config.api_url}")
        except asyncio.TimeoutError:
            logger.error(f"Async operation timed out for {method} {endpoint}")
            raise TimeoutError(f"Operation timed out after {self.timeout.total} seconds")
        except Exception as e:
            logger.error(f"Unexpected error in RAGFlow API request: {str(e)}")
            raise

    async def lookup_with_progress(self, request: RAGFlowLookupRequest, user_email: str = None) -> AsyncGenerator[Dict, None]:
        """
        Perform document lookup with progress updates via async generator.

        Args:
            request: RAGFlow lookup request with query and parameters
            user_email: The authenticated user's email for dataset validation

        Yields:
            Dictionary with progress updates and final results
        """
        start_time = time.time()

        # Validate user access to requested datasets if user_email is provided
        if user_email:
            try:
                validated_datasets = await self._validate_and_prepare_datasets(user_email, request.dataset_ids)
                # Update the request with validated datasets
                request.dataset_ids = validated_datasets
            except PermissionError as e:
                logger.error(f"Permission error: {str(e)}")
                yield {
                    "status": "error",
                    "message": f"Permission denied: {str(e)}",
                    "error_details": str(e),
                    "timestamp": time.time()
                }
                return

        # Send initial progress update
        yield {
            "status": "processing",
            "message": "Preparing to search documents...",
            "progress": 10,
            "timestamp": start_time
        }

        try:
            # Perform the actual lookup
            results = await self._perform_lookup(request)

            # Send intermediate progress update
            mid_time = time.time()
            yield {
                "status": "retrieving",
                "message": f"Found {len(results)} relevant documents, retrieving details...",
                "progress": 60,
                "timestamp": mid_time
            }

            # Process results to extract metadata for PDF highlighting
            processed_results = []
            for result in results:
                processed_result = await self._process_result(result)
                processed_results.append(processed_result)

            # Send final results
            end_time = time.time()
            yield {
                "status": "completed",
                "message": "Document lookup completed successfully",
                "progress": 100,
                "results": processed_results,
                "query_duration_ms": int((end_time - start_time) * 1000),
                "timestamp": end_time
            }

        except Exception as e:
            logger.error(f"Error during RAGFlow lookup: {str(e)}", exc_info=True)
            yield {
                "status": "error",
                "message": f"Lookup failed: {str(e)}",
                "error_details": str(e),
                "timestamp": time.time()
            }

    async def _perform_lookup(self, request: RAGFlowLookupRequest) -> List[Dict]:
        """
        Perform the actual lookup operation against RAGFlow API.
        """
        try:
            # Prepare the payload for RAGFlow chat completion API
            payload = {
                "conversation_id": "",
                "dataset_ids": request.dataset_ids,
                "query": request.query,
                "stream": False,  # We're not using streaming from RAGFlow, just getting results
                "top_k": request.top_k
            }

            # Add any additional parameters based on the request
            if hasattr(request, 'temperature'):
                payload['temperature'] = getattr(request, 'temperature', 0.7)

            # Make the API call to RAGFlow
            result = await self._make_request('POST', '/api/v1/chat/completion', payload)

            # Extract candidates from the response
            candidates = result.get('candidates', [])

            return candidates
        except TimeoutError as e:
            logger.error(f"RAGFlow lookup timed out: {str(e)}")
            raise
        except PermissionError as e:
            logger.error(f"RAGFlow lookup permission error: {str(e)}")
            raise
        except ConnectionError as e:
            logger.error(f"RAGFlow lookup connection error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during RAGFlow lookup: {str(e)}", exc_info=True)
            raise RuntimeError(f"Lookup failed due to an internal error: {str(e)}")

    async def _process_result(self, result: Dict) -> RAGFlowDocumentResult:
        """
        Process a single result from RAGFlow to extract metadata for PDF highlighting.
        """
        # Extract content
        content = result.get('content', '')

        # Extract metadata
        meta = result.get('meta', {})

        # Extract page number (could be in different fields depending on RAGFlow version)
        page_number = (
            meta.get('page_number') or
            meta.get('page_idx') or
            meta.get('page', None)
        )

        # Extract bounding box coordinates (if available)
        # RAGFlow typically stores this as [x0, y0, x1, y1] format
        bbox_raw = (
            meta.get('bbox') or
            meta.get('bounding_box') or
            meta.get('coordinates', None)
        )

        # Normalize bounding box format if needed
        bbox_coordinates = self._normalize_bbox(bbox_raw)

        # Extract source file information
        source_file = (
            meta.get('source_file') or
            meta.get('filename') or
            meta.get('file_name', None)
        )

        # Extract confidence score (if available)
        confidence_score = result.get('score', result.get('similarity', 0.0))  # Default to 0 if not available

        # Extract document ID
        document_id = result.get('document_id', result.get('id', 'unknown'))

        # Extract additional metadata that might be useful for highlighting
        additional_meta = {
            'char_start': meta.get('char_start'),
            'char_end': meta.get('char_end'),
            'paragraph_id': meta.get('paragraph_id'),
            'section_title': meta.get('section_title'),
            'original_page_text': meta.get('original_page_text'),
        }

        # Filter out None values
        additional_meta = {k: v for k, v in additional_meta.items() if v is not None}

        # Create and return the processed result
        return RAGFlowDocumentResult(
            document_id=document_id,
            content=content,
            page_number=page_number,
            bbox_coordinates=bbox_coordinates,
            source_file=source_file,
            confidence_score=confidence_score,
            metadata={**meta, **additional_meta}
        )

    def _normalize_bbox(self, bbox_raw) -> Optional[List[float]]:
        """
        Normalize bounding box coordinates to a standard format [x, y, width, height].

        Args:
            bbox_raw: Raw bounding box data from RAGFlow (might be [x0,y0,x1,y1] or other format)

        Returns:
            Normalized bounding box as [x, y, width, height] or None if invalid
        """
        if not bbox_raw:
            return None

        try:
            # If bbox is in [x0, y0, x1, y1] format, convert to [x, y, width, height]
            if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
                x0, y0, x1, y1 = bbox_raw
                x = min(x0, x1)
                y = min(y0, y1)
                width = abs(x1 - x0)
                height = abs(y1 - y0)
                return [x, y, width, height]
            elif isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 2:
                # If it's in [x, y] format with no dimensions, return as is with zeros
                x, y = bbox_raw
                return [x, y, 0.0, 0.0]
            else:
                # Return as is if it's already in the expected format
                return list(bbox_raw)
        except (ValueError, TypeError):
            # If conversion fails, return as is
            return bbox_raw if isinstance(bbox_raw, list) else None

    async def get_dataset_info(self, dataset_id: str) -> Dict:
        """
        Get information about a specific dataset in RAGFlow.
        """
        try:
            # This endpoint might vary depending on RAGFlow version
            # Using a generic approach - might need adjustment based on actual API
            result = await self._make_request('GET', f'/api/v1/dataset/{dataset_id}')
            return result
        except Exception as e:
            # If specific dataset info endpoint doesn't exist, return basic info
            logger.warning(f"Could not get detailed dataset info: {str(e)}, returning basic info")
            return {
                "dataset_id": dataset_id,
                "status": "available",
                "documents_count": "unknown"
            }

    def _generate_user_dataset_id(self, user_email: str) -> str:
        """
        Generate the dataset ID for a specific user following the naming convention.

        Args:
            user_email: The user's email address

        Returns:
            Dataset ID in the format 'kb_user_<user_id>'
        """
        # Sanitize the email to create a valid dataset ID
        # Replace special characters that might not be valid in dataset names
        sanitized_email = user_email.replace('@', '_at_').replace('.', '_dot_').replace('+', '_plus_')
        return f"kb_user_{sanitized_email}"

    async def verify_user_access(self, user_email: str, requested_dataset_id: str) -> bool:
        """
        Verify that a user has access to a specific dataset.
        Enforces user isolation by ensuring users can only access their own datasets.

        Args:
            user_email: The authenticated user's email
            requested_dataset_id: The dataset ID the user is requesting access to

        Returns:
            Boolean indicating if the user has access to the dataset
        """
        expected_dataset_id = self._generate_user_dataset_id(user_email)
        return requested_dataset_id == expected_dataset_id

    async def _validate_and_prepare_datasets(self, user_email: str, dataset_ids: List[str]) -> List[str]:
        """
        Validate that the user has access to all requested datasets.

        Args:
            user_email: The authenticated user's email
            dataset_ids: List of dataset IDs to validate

        Returns:
            List of validated dataset IDs
        """
        validated_datasets = []

        for dataset_id in dataset_ids:
            has_access = await self.verify_user_access(user_email, dataset_id)
            if not has_access:
                raise PermissionError(f"User does not have access to dataset: {dataset_id}")
            validated_datasets.append(dataset_id)

        return validated_datasets