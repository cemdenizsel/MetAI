"""
Configuration loading utilities for the emotion framework.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def load_framework_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file or use defaults.
    
    Args:
        config_path: Path to config YAML file. If None, uses default path.
    
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Try to find config in default locations
        possible_paths = [
            "config/config.yaml",
            "../config/config.yaml",
            "../../config/config.yaml",
            os.path.join(os.path.dirname(__file__), "../../config/config.yaml"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return get_default_config()
    else:
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'system': {
            'name': 'Multimodal Emotion Recognition System',
            'version': '1.0.1'
        },
        'modalities': {
            'audio': {
                'enabled': True,
                'sample_rate': 16000,
                'window_size': 3.0,
                'hop_length': 1.5,
                'n_mfcc': 40,
                'extract_opensmile': True
            },
            'visual': {
                'enabled': True,
                'fps': 5,
                'face_detection': 'mediapipe',
                'extract_action_units': True,
                'extract_landmarks': True
            },
            'text': {
                'enabled': True,
                'asr_model': 'openai/whisper-base',
                'embedding_model': 'all-MiniLM-L6-v2',
                'extract_sentiment': True
            }
        },
        'fusion': {
            'strategy': 'early',
            'model': 'rfrboost'
        },
        'rfrboost': {
            'n_layers': 6,
            'hidden_dim': 256,
            'randfeat_xt_dim': 512,
            'randfeat_x0_dim': 512,
            'boost_lr': 0.5,
            'feature_type': 'SWIM',
            'upscale_type': 'SWIM',
            'activation': 'tanh',
            'use_batchnorm': True,
            'do_linesearch': True,
            'l2_cls': 0.001,
            'l2_ghat': 0.001
        },
        'emotions': {
            'labels': ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise'],
            'colors': {
                'neutral': '#95a5a6',
                'happy': '#f1c40f',
                'sad': '#3498db',
                'angry': '#e74c3c',
                'fear': '#9b59b6',
                'disgust': '#16a085',
                'surprise': '#e67e22'
            }
        },
        'output': {
            'save_json': True,
            'save_csv': True,
            'save_video': False,
            'save_report': True
        },
        'paths': {
            'raw_data': 'data_model/raw',
            'processed_data': 'data_model/processed',
            'results': 'data_model/results',
            'pretrained': 'pretrained'
        }
    }


def merge_config(base_config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge configuration overrides into base configuration.
    
    Args:
        base_config: Base configuration dictionary
        overrides: Configuration overrides
    
    Returns:
        Merged configuration
    """
    import copy
    result = copy.deepcopy(base_config)
    
    def deep_merge(base: Dict, override: Dict) -> Dict:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    return deep_merge(result, overrides)

