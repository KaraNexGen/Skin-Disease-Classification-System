"""
Configuration settings for the skin disease classification system.
"""

import os
from typing import Dict, List

# Model Configuration
MODEL_CONFIG = {
    "model_path": os.getenv("MODEL_PATH", "model.h5"),
    "input_shape": (224, 224, 3),
    "batch_size": 1,
    "confidence_threshold": 0.5
}

# Disease Classes (HAM10000 dataset)
DISEASE_CLASSES = [
    'Actinic keratoses',
    'Basal cell carcinoma', 
    'Benign keratosis-like lesions',
    'Dermatofibroma',
    'Melanoma',
    'Melanocytic nevi',
    'Vascular lesions'
]

# Disease Class Mappings
DISEASE_MAPPINGS = {
    'akiec': 'Actinic keratoses',
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis-like lesions',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic nevi',
    'vasc': 'Vascular lesions'
}

# API Configuration
API_CONFIG = {
    "title": "Skin Disease Classification API",
    "description": "AI-powered skin disease classification using CNN",
    "version": "1.0.0",
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "allowed_extensions": ['.jpg', '.jpeg', '.png', '.gif', '.bmp'],
    "cors_origins": [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001"
    ]
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "app.log"
}

# Image Processing Configuration
IMAGE_CONFIG = {
    "min_size": (50, 50),
    "max_size": (4096, 4096),
    "target_size": (224, 224),
    "normalization": "standard"  # or "minmax"
}

# Model Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    "high_confidence": 0.8,
    "medium_confidence": 0.6,
    "low_confidence": 0.4,
    "uncertainty_threshold": 0.3
}

def get_config() -> Dict:
    """Get complete configuration dictionary."""
    return {
        "model": MODEL_CONFIG,
        "disease_classes": DISEASE_CLASSES,
        "disease_mappings": DISEASE_MAPPINGS,
        "api": API_CONFIG,
        "logging": LOGGING_CONFIG,
        "image": IMAGE_CONFIG,
        "performance": PERFORMANCE_THRESHOLDS
    }
