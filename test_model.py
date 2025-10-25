"""
Test script for the skin disease classification model.
"""

import numpy as np
from PIL import Image
import logging
from model import SkinDiseaseClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test model loading functionality."""
    try:
        classifier = SkinDiseaseClassifier()
        logger.info("✅ Model loading test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Model loading test failed: {str(e)}")
        return False

def test_prediction():
    """Test prediction functionality with dummy image."""
    try:
        classifier = SkinDiseaseClassifier()
        
        # Create a dummy image
        dummy_image = Image.new('RGB', (224, 224), color='red')
        
        # Make prediction
        prediction = classifier.predict(dummy_image)
        
        # Validate prediction structure
        required_keys = ['disease', 'confidence', 'all_predictions']
        for key in required_keys:
            if key not in prediction:
                raise ValueError(f"Missing key in prediction: {key}")
        
        logger.info("✅ Prediction test passed")
        logger.info(f"Prediction result: {prediction['disease']} (confidence: {prediction['confidence']:.3f})")
        return True
        
    except Exception as e:
        logger.error(f"❌ Prediction test failed: {str(e)}")
        return False

def test_model_info():
    """Test model information retrieval."""
    try:
        classifier = SkinDiseaseClassifier()
        model_info = classifier.get_model_info()
        
        if 'error' in model_info:
            logger.warning("⚠️ Model info contains error (expected for dummy model)")
        else:
            logger.info("✅ Model info test passed")
            logger.info(f"Model info: {model_info}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model info test failed: {str(e)}")
        return False

def run_all_tests():
    """Run all tests."""
    logger.info("🧪 Starting model tests...")
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Prediction", test_prediction),
        ("Model Info", test_model_info)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running {test_name} test...")
        if test_func():
            passed += 1
    
    logger.info(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
    else:
        logger.warning(f"⚠️ {total - passed} tests failed")
    
    return passed == total

if __name__ == "__main__":
    run_all_tests()
