#!/bin/bash
# One-time setup: dependencies, NLTK data, EmotiVoice.
# Run with:  ./setup_all.sh   or   bash setup_all.sh   (not with python)
# Then start the API with:
#   python -m uvicorn main:app --host 0.0.0.0 --port 8084 --reload

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "MetAI app – full setup"
echo "======================"
echo ""

# Python version
echo "Checking Python..."
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &>/dev/null; then
  PYTHON=python
fi
if ! "$PYTHON" -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
  echo "Python 3.8+ required. Activate your venv and run this script again."
  exit 1
fi
echo "Using: $($PYTHON --version)"
echo ""

# Pip install
echo "Installing dependencies (pip install -r requirements.txt)..."
"$PYTHON" -m pip install --upgrade pip -q
"$PYTHON" -m pip install -r requirements.txt -q
echo "Dependencies installed."
echo ""

# Directories
echo "Creating directories..."
mkdir -p data_model/raw data_model/processed data_model/results pretrained notebooks
touch data_model/raw/.gitkeep data_model/processed/.gitkeep data_model/results/.gitkeep pretrained/.gitkeep 2>/dev/null || true
echo "Directories ready."
echo ""

# NLTK data (full set)
echo "Downloading NLTK data..."
if "$PYTHON" utils/download_nltk_data.py; then
  echo "NLTK data ready."
else
  echo "NLTK download had warnings; continuing."
fi
echo ""

# EmotiVoice (TTS)
echo "Setting up EmotiVoice (TTS)..."
if "$PYTHON" tts/setup_emotivoice.py; then
  echo "EmotiVoice ready."
else
  echo "EmotiVoice setup failed or skipped; TTS may be unavailable."
fi
echo ""

# Optional: ffmpeg
if command -v ffmpeg &>/dev/null || command -v ffprobe &>/dev/null; then
  echo "ffmpeg/ffprobe found."
else
  echo "Note: ffmpeg not found. For video/audio processing install it:"
  echo "  macOS: brew install ffmpeg"
  echo "  Linux: sudo apt-get install ffmpeg  # or sudo yum install ffmpeg"
fi
echo ""

echo "======================"
echo "Setup finished."
echo ""
echo "Start the API with:"
echo "  python -m uvicorn main:app --host 0.0.0.0 --port 8084 --reload"
echo ""
