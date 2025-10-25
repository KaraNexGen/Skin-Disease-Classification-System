"""
Utility functions for model management and validation.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Tuple
import tensorflow as tf

logger = logging.getLogger(__name__)

def validate_model_file(model_path: str) -> bool:
    """
    Validate if the model file exists and is accessible.
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        bool: True if model file is valid, False otherwise
    """
    try:
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            return False
        
        if not model_path.endswith(('.h5', '.hdf5', '.keras')):
            logger.warning(f"Unsupported model format: {model_path}")
            return False
        
        # Try to load the model to validate it
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model validation successful: {model_path}")
        return True
        
    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}")
        return False

def get_model_architecture_info(model_path: str) -> Dict:
    """
    Get architecture information about the model.
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        dict: Model architecture information
    """
    try:
        model = tf.keras.models.load_model(model_path)
        
        return {
            "input_shape": model.input_shape,
            "output_shape": model.output_shape,
            "num_layers": len(model.layers),
            "total_params": model.count_params(),
            "trainable_params": sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
            "non_trainable_params": sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
        }
        
    except Exception as e:
        logger.error(f"Error getting model architecture: {str(e)}")
        return {"error": str(e)}

def preprocess_image_for_model(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess image for model input.
    
    Args:
        image (np.ndarray): Input image
        target_size (tuple): Target size for resizing
        
    Returns:
        np.ndarray: Preprocessed image
    """
    try:
        # Resize image
        resized = tf.image.resize(image, target_size)
        
        # Normalize to [0, 1]
        normalized = resized / 255.0
        
        # Add batch dimension
        batched = tf.expand_dims(normalized, 0)
        
        return batched.numpy()
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def calculate_prediction_metrics(predictions: np.ndarray, threshold: float = 0.5) -> Dict:
    """
    Calculate prediction metrics and confidence scores.
    
    Args:
        predictions (np.ndarray): Model predictions
        threshold (float): Confidence threshold
        
    Returns:
        dict: Prediction metrics
    """
    try:
        # Get top prediction
        top_pred_idx = np.argmax(predictions)
        top_confidence = float(np.max(predictions))
        
        # Calculate entropy (uncertainty measure)
        entropy = -np.sum(predictions * np.log(predictions + 1e-8))
        
        # Count predictions above threshold
        high_confidence_count = np.sum(predictions > threshold)
        
        return {
            "top_prediction_index": int(top_pred_idx),
            "top_confidence": top_confidence,
            "entropy": float(entropy),
            "high_confidence_predictions": int(high_confidence_count),
            "prediction_distribution": predictions.tolist()
        }
        
    except Exception as e:
        logger.error(f"Error calculating prediction metrics: {str(e)}")
        return {"error": str(e)}

def create_model_summary(model_path: str) -> str:
    """
    Create a text summary of the model architecture.
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        str: Model summary
    """
    try:
        model = tf.keras.models.load_model(model_path)
        
        # Create string buffer for summary
        import io
        string_buffer = io.StringIO()
        model.summary(print_fn=lambda x: string_buffer.write(x + '\n'))
        summary = string_buffer.getvalue()
        string_buffer.close()
        
        return summary
        
    except Exception as e:
        logger.error(f"Error creating model summary: {str(e)}")
        return f"Error creating model summary: {str(e)}"
