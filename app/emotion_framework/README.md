# Emotion Framework

> A comprehensive multimodal emotion recognition framework for video analysis powered by deep learning.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![PyPI version](https://badge.fury.io/py/emotion-framework.svg)](https://badge.fury.io/py/emotion-framework)

## 🎯 Features

- **Multimodal Analysis**: Combines audio, visual, and text features for robust emotion recognition
- **Multiple Fusion Strategies**: Choose from various fusion approaches (early, late, hybrid)
- **Pre-trained Models**: Includes state-of-the-art models (RFRBoost, Attention-Deep, MLP Baseline)
- **Real-time Support**: Process video streams in real-time with configurable window sizes
- **AI-Powered Insights**: Optional LLM-based analysis for meeting insights
- **Mental Health Scoring**: Comprehensive emotion-based mental health assessment
- **Easy Integration**: Simple API for quick integration into your applications

## 📦 Installation

```bash
pip install emotion-framework
```

### System Dependencies

The framework requires some system-level dependencies:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libgl1-mesa-glx libglib2.0-0
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
- Install [ffmpeg](https://ffmpeg.org/download.html) and add to PATH

## 🚀 Quick Start

```python
from emotion_framework import EmotionAnalysisPipeline
from emotion_framework.core.config_loader import load_framework_config

# Initialize the pipeline
config = load_framework_config()
pipeline = EmotionAnalysisPipeline(config)

# Analyze a video
result = pipeline.analyze_video("path/to/video.mp4")

# Access results
print(f"Predicted Emotion: {result.prediction.predicted_emotion}")
print(f"Confidence: {result.prediction.confidence:.2f}")
print(f"Processing Time: {result.processing_time:.2f}s")

# Get temporal predictions
for temporal_pred in result.temporal_predictions:
    print(f"Time: {temporal_pred.timestamp}s - Emotion: {temporal_pred.emotion}")

# Mental health analysis
if result.mental_health_analysis:
    mh = result.mental_health_analysis
    print(f"Mental Health Score: {mh.mental_health_score}/100")
    print(f"Status: {mh.status}")
    print(f"Recommendation: {mh.recommendation}")
```

## 📊 Advanced Usage

### Custom Configuration

```python
from emotion_framework import EmotionAnalysisPipeline

# Create custom config
config = {
    "fusion_strategy": "hybrid",  # early, late, or hybrid
    "extract_audio": True,
    "extract_visual": True,
    "extract_text": True,
    "fps_for_analysis": 1,  # Extract 1 frame per second
}

pipeline = EmotionAnalysisPipeline(config)

# Analyze with options
options = {
    "fusion_strategy": "late",
    "run_ai_analysis": True,
    "llm_provider": "openai"
}

result = pipeline.analyze_video("video.mp4", options)
```

### Real-time Analysis

```python
from emotion_framework.core.realtime_pipeline import RealtimeEmotionAnalyzer

# Initialize real-time analyzer
analyzer = RealtimeEmotionAnalyzer(
    window_size=4.0,  # 4-second windows
    stride=1.0,       # 1-second stride
)

# Process video stream
for chunk_result in analyzer.analyze_stream("rtsp://camera-url"):
    print(f"Real-time emotion: {chunk_result.emotion}")
```

### AI-Powered Meeting Analysis

```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"

options = {
    "run_ai_analysis": True,
    "llm_provider": "openai",
    "llm_model": "gpt-4"
}

result = pipeline.analyze_video("meeting.mp4", options)

if result.ai_analysis:
    print(f"Summary: {result.ai_analysis.summary}")
    print(f"Key Insights: {result.ai_analysis.key_insights}")
    print(f"Recommendations: {result.ai_analysis.recommendations}")
```

## 📖 API Reference

### EmotionAnalysisPipeline

Main class for emotion analysis.

**Methods:**
- `analyze_video(video_path: str, options: dict = None) -> EmotionAnalysisResult`

### EmotionAnalysisResult

Contains all analysis results.

**Attributes:**
- `prediction`: Overall emotion prediction
- `temporal_predictions`: Frame-by-frame predictions
- `mental_health_analysis`: Mental health assessment
- `transcription`: Speech-to-text results
- `ai_analysis`: AI-generated insights
- `metadata`: Video metadata
- `features`: Extracted features
- `processing_time`: Total processing time

## 🎨 Supported Emotions

- **Happy**: Joy, contentment, positive emotions
- **Sad**: Sorrow, disappointment, low mood
- **Angry**: Frustration, irritation, rage
- **Fear**: Anxiety, worry, nervousness
- **Surprise**: Shock, amazement, unexpected reactions
- **Disgust**: Aversion, repulsion, distaste
- **Neutral**: Calm, balanced, no strong emotion

## 🧠 Models & Architecture

The framework uses a hierarchical approach:

1. **Feature Extraction**
   - Audio: librosa, openSMILE, pyAudioAnalysis
   - Visual: OpenCV, MediaPipe, py-feat
   - Text: Transformers, BERT, sentence-transformers

2. **Fusion Strategies**
   - Early Fusion: Combine features before classification
   - Late Fusion: Combine predictions after classification
   - Hybrid Fusion: Adaptive combination based on modality confidence

3. **Classification Models**
   - RFRBoost: Random Feature Representation with Boosting
   - Attention-Deep: Deep learning with attention mechanisms
   - MLP Baseline: Multi-layer perceptron baseline

## 🔧 Configuration

Create a `config.yaml` file:

```yaml
# Feature Extraction
extract_audio: true
extract_visual: true
extract_text: true
fps_for_analysis: 1

# Fusion Strategy
fusion_strategy: "hybrid"  # early, late, hybrid

# AI Analysis
enable_ai_analysis: false
llm_provider: "openai"  # or "local"
llm_model: "gpt-4"

# Paths (optional)
pretrained_models_path: "./pretrained"
temp_directory: "./temp"
```

Load it:

```python
from emotion_framework.core.config_loader import load_framework_config

config = load_framework_config("path/to/config.yaml")
pipeline = EmotionAnalysisPipeline(config)
```

## 🛠️ Development

### Installation for Development

```bash
git clone https://github.com/DogukanGun/MetAI.git
cd emotion-framework
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/
```

## 📋 Requirements

- Python 3.8+
- PyTorch 2.2+
- OpenCV 4.8+
- librosa 0.10+
- transformers 4.30+
- ffmpeg (system dependency)

See `setup.py` for complete list.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with PyTorch, transformers, and OpenCV
- Inspired by state-of-the-art multimodal emotion recognition research
- Thanks to the open-source ML community

## 📧 Contact

For questions, issues, or contributions:
- GitHub Issues: [https://github.com/DogukanGun/MetAI/issues](https://github.com/DogukanGun/MetAI/issues)

## 🗺️ Roadmap

- [ ] GPU acceleration optimization
- [ ] Additional fusion strategies
- [ ] More pre-trained models
- [ ] Web UI for demo
- [ ] Cloud deployment support
- [ ] Mobile SDK

---

**Made with ❤️ by the Emotion Analysis Team**
