from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import logging
from model import SkinDiseaseClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Skin Disease Classification API",
    description="AI-powered skin disease classification using CNN",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://192.168.31.227:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize model
try:
    classifier = SkinDiseaseClassifier()
    logger.info("Skin disease classifier initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize classifier: {str(e)}")
    classifier = None

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Skin Disease Classification API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "model_info": "/model-info",
            "docs": "/docs"
        }
    }

@app.post("/predict")
async def predict_skin_disease(file: UploadFile = File(...)):
    """
    Predict skin disease from uploaded image.
    
    Args:
        file: Uploaded image file
        
    Returns:
        dict: Prediction results with disease name, confidence, and all predictions
    """
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Check file size (max 10MB)
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=400, detail="File size too large (max 10MB)")
        
        # Check if classifier is available
        if classifier is None:
            raise HTTPException(status_code=500, detail="Model not available")
        
        # Read and process image
        image = Image.open(io.BytesIO(contents))
        
        # Validate image dimensions
        if image.size[0] < 50 or image.size[1] < 50:
            raise HTTPException(status_code=400, detail="Image too small (minimum 50x50 pixels)")
        
        logger.info(f"Processing image: {file.filename}, size: {image.size}")
        
        # Make prediction
        prediction = classifier.predict(image)
        
        logger.info(f"Prediction completed for {file.filename}")
        
        return {
            "success": True,
            "filename": file.filename,
            "disease": prediction["disease"],
            "confidence": prediction["confidence"],
            "all_predictions": prediction["all_predictions"]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": classifier is not None,
        "timestamp": "2024-01-01T00:00:00Z"
    }

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint for debugging."""
    return {
        "message": "Backend is working!",
        "cors_enabled": True,
        "model_status": "loaded" if classifier else "not_loaded"
    }

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model."""
    if classifier is None:
        raise HTTPException(status_code=500, detail="Model not available")
    
    try:
        model_info = classifier.get_model_info()
        return {
            "success": True,
            "model_info": model_info
        }
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )
