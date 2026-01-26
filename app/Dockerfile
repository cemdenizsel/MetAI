FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies (required by emotion-framework)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libhdf5-dev \
    pkg-config \
    git \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    meson \
    ninja-build \
    pybind11-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code/api

# Install build tools first (required for emotion-framework)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install numpy first (required by many packages including h5py)
# Use a version compatible with the requirements
RUN pip install --no-cache-dir "numpy>=1.26.0,<2.0"

# Install h5py with build dependencies available
# Disable build isolation so setuptools is available
RUN pip install --no-cache-dir --no-build-isolation "h5py>=2.7.0"

# Install numexpr separately (has build requirements, needs compatible numpy)
RUN pip install --no-cache-dir --no-build-isolation numexpr || \
    pip install --no-cache-dir numexpr

# Install scipy early as it's a dependency for many packages
RUN pip install --no-cache-dir --no-build-isolation "scipy>=1.11.0"

# Install PyTorch and torchvision first to avoid conflicts with other packages
RUN pip install --no-cache-dir torch==2.2.0 torchvision>=0.17.0

# Install opencv-python before mediapipe to avoid conflicts
RUN pip install --no-cache-dir opencv-python>=4.8.0

# Install additional build dependencies for packages that need compilation
RUN pip install --no-cache-dir meson-python cython pybind11

# Copy requirements file
COPY requirements.txt /code/api/requirements.txt

# Install dependencies in stages to handle problematic packages
# First, install core dependencies
RUN pip install --no-cache-dir --upgrade --no-build-isolation fastapi uvicorn python-multipart pydantic pydantic-settings

# Install database and utility dependencies
RUN pip install --no-cache-dir --upgrade --no-build-isolation motor pymongo python-dotenv PyJWT passlib python-jose redis

# Install async task processing dependencies
RUN pip install --no-cache-dir --upgrade --no-build-isolation celery kombu flower

# Install testing dependencies
RUN pip install --no-cache-dir --upgrade --no-build-isolation pytest pytest-asyncio httpx

# Install core ML dependencies
RUN pip install --no-cache-dir --upgrade --no-build-isolation scikit-learn pytorch-minimize optuna xgboost

# Install web UI dependencies
RUN pip install --no-cache-dir --upgrade --no-build-isolation streamlit

# Install audio processing dependencies (with specific versions to avoid conflicts)
RUN pip install --no-cache-dir --upgrade --no-build-isolation librosa==0.10.2.post1 pydub webrtcvad soundfile

# Install visual processing dependencies with compatible versions
RUN pip install --no-cache-dir --upgrade --no-build-isolation pillow
RUN pip install --no-cache-dir --upgrade --no-build-isolation opencv-contrib-python
RUN pip install --no-cache-dir --upgrade --no-build-isolation scikit-image matplotlib==3.10.8
RUN pip install --no-cache-dir --upgrade --no-build-isolation mediapipe
RUN pip install --no-cache-dir --upgrade --no-build-isolation face-recognition

# Note: py-feat is skipped due to numpy compatibility issues
# RUN pip install --no-cache-dir --upgrade --no-build-isolation py-feat

# Install text processing dependencies
RUN pip install --no-cache-dir --upgrade --no-build-isolation transformers sentence-transformers nltk spacy textblob vaderSentiment openai

# Install video processing dependencies
RUN pip install --no-cache-dir --upgrade --no-build-isolation moviepy ffmpeg-python

# Install data processing dependencies
RUN pip install --no-cache-dir --upgrade --no-build-isolation pandas matplotlib seaborn plotly tqdm

# Install utility dependencies
RUN pip install --no-cache-dir --upgrade --no-build-isolation PyYAML

# Install TTS dependencies
RUN pip install --no-cache-dir --upgrade --no-build-isolation numba yacs g2p-en jieba pypinyin pypinyin-dict

# Install remaining dependencies individually to isolate issues
RUN pip install --no-cache-dir --upgrade --no-build-isolation opensmile
RUN pip install --no-cache-dir --upgrade --no-build-isolation pyAudioAnalysis
RUN pip install --no-cache-dir --upgrade --no-build-isolation openai-whisper
RUN pip install 'pydantic[email]'
# Copy API code
COPY . /code/api

# Expose port
EXPOSE 8084

# Run the API
CMD ["python", "main.py"]