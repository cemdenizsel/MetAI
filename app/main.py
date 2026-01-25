"""
Emotion Recognition API

FastAPI application for multimodal emotion recognition.
"""

import logging
import sys
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for the application."""
    # Startup
    logger.info("Starting Emotion Recognition API")
    yield
    # Shutdown
    logger.info("Shutting down Emotion Recognition API")


# Initialize FastAPI app
app = FastAPI(
    title="Emotion Recognition API",
    description="""
    ## Multimodal Emotion Recognition & Knowledge Base API
    
    This API provides advanced emotion recognition from video files using multiple
    state-of-the-art ai_models running in parallel, plus a multimodal knowledge base
    for document management and intelligent retrieval.
    
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "ValidationError",
            "message": "Request validation failed",
            "details": exc.errors()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "details": {"error": str(exc)}
        }
    )


# Include routers
from controllers.auth_controller import router as auth_router
from controllers.emotion_controller import router as emotion_router
from controllers.websocket_controller import router as websocket_router
from controllers.job_controller import router as job_router
from controllers.cache_controller import router as cache_router
from controllers.batch_controller import router as batch_router
from controllers.meeting_controller import router as meeting_router

app.include_router(auth_router)
app.include_router(emotion_router)
app.include_router(websocket_router)
app.include_router(job_router)
app.include_router(cache_router)
app.include_router(batch_router)
app.include_router(meeting_router)


# Root endpoint
@app.get(
    "/",
    tags=["Root"],
    summary="API Root",
    description="Get API information"
)
async def root():
    """
    Root endpoint providing API information.
    
    Returns:
        API information and available endpoints
    """
    return {
        "service": "Emotion Recognition & Knowledge Base API",
        "version": "1.0.0",
        "description": "Multimodal emotion recognition + RAG knowledge base + AI agent analysis",
    }


if __name__ == "__main__":
    import uvicorn
    from os import getenv

    port = int(getenv("PORT", "8084"))
    uvicorn.run(app, host="0.0.0.0", port=port)
