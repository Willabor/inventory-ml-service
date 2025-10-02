"""FastAPI service for ML predictions."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
from datetime import datetime

from config import settings
from models.transfer_predictor import TransferPredictor

# Initialize FastAPI app
app = FastAPI(
    title="Inventory ML Service",
    description="Machine learning predictions for inventory management",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance (loaded on startup)
transfer_model: Optional[TransferPredictor] = None


# Pydantic models for API
class TransferPrediction(BaseModel):
    style_number: str
    item_name: str
    category: Optional[str]
    from_store: str
    to_store: str
    from_store_qty: float
    to_store_qty: float
    from_store_daily_sales: float
    to_store_daily_sales: float
    success_probability: float
    recommended_qty: int
    ml_priority: str
    ml_priority_score: float
    margin_percent: float
    model_version: str


class TransferPredictionResponse(BaseModel):
    success: bool
    count: int
    model_version: str
    generated_at: str
    predictions: List[TransferPrediction]


class TrainingRequest(BaseModel):
    days_back: int = 90


class TrainingResponse(BaseModel):
    success: bool
    model_version: str
    metrics: Dict
    training_date: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str]
    service_uptime: str


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load the latest model on service startup."""
    global transfer_model

    print("=" * 60)
    print("STARTING ML SERVICE")
    print("=" * 60)
    print(f"Port: {settings.port}")
    print(f"Host: {settings.ml_service_host}")
    print(f"Database URL: {settings.database_url[:30]}..." if settings.database_url else "Database URL: NOT SET")
    print(f"Model Cache Dir: {settings.model_cache_dir}")
    print("=" * 60)

    try:
        transfer_model = TransferPredictor.load_latest()
        print("✓ Model loaded successfully")
        print(f"  Version: {transfer_model.model_version}")
    except FileNotFoundError:
        print("⚠ No trained model found. Train a model using /api/ml/train endpoint")
        transfer_model = None
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        transfer_model = None

    print("=" * 60)
    print("ML SERVICE READY")
    print("=" * 60)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint - service info."""
    return {
        "service": "Inventory ML Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "train": "/api/ml/train",
            "predict": "/api/ml/predict-transfers",
            "model_info": "/api/ml/model-info"
        }
    }


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": transfer_model is not None,
        "model_version": transfer_model.model_version if transfer_model else None,
        "service_uptime": "running"
    }


# Training endpoint
@app.post("/api/ml/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    """
    Train a new transfer prediction model.

    This endpoint trains a TabPFN model on historical data.
    It should be called periodically (e.g., weekly) to update the model.
    """
    global transfer_model

    try:
        print(f"Training request received: {request.days_back} days of data")

        # Create new predictor
        predictor = TransferPredictor()

        # Train
        metrics = predictor.train(days_back=request.days_back)

        # Save model
        predictor.save_model()

        # Update global model
        transfer_model = predictor

        return {
            "success": True,
            "model_version": predictor.model_version,
            "metrics": metrics,
            "training_date": predictor.training_date
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


# Prediction endpoint
@app.post("/api/ml/predict-transfers", response_model=TransferPredictionResponse)
async def predict_transfers(limit: int = 20):
    """
    Generate ML-powered transfer recommendations.

    This endpoint uses the trained TabPFN model to predict which
    inventory transfers are most likely to result in sales.
    """

    if transfer_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model first using /api/ml/train"
        )

    try:
        print(f"Prediction request received (limit={limit})")

        # Generate predictions
        results = transfer_model.predict_transfers(limit=limit)

        if len(results) == 0:
            return {
                "success": True,
                "count": 0,
                "model_version": transfer_model.model_version,
                "generated_at": datetime.now().isoformat(),
                "predictions": []
            }

        # Convert to response format
        predictions = []
        for _, row in results.iterrows():
            predictions.append(TransferPrediction(
                style_number=row['style_number'],
                item_name=row['item_name'],
                category=row.get('category'),
                from_store=row['from_store'],
                to_store=row['to_store'],
                from_store_qty=float(row['from_store_qty']),
                to_store_qty=float(row['to_store_qty']),
                from_store_daily_sales=float(row['from_store_daily_sales']),
                to_store_daily_sales=float(row['to_store_daily_sales']),
                success_probability=float(row['success_probability']),
                recommended_qty=int(row['recommended_qty']),
                ml_priority=row['ml_priority'],
                ml_priority_score=float(row['ml_priority_score']),
                margin_percent=float(row['margin_percent']),
                model_version=row['model_version']
            ))

        return {
            "success": True,
            "count": len(predictions),
            "model_version": transfer_model.model_version,
            "generated_at": datetime.now().isoformat(),
            "predictions": predictions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Model info endpoint
@app.get("/api/ml/model-info")
async def get_model_info():
    """Get information about the currently loaded model."""

    if transfer_model is None:
        return {
            "loaded": False,
            "message": "No model loaded"
        }

    return {
        "loaded": True,
        "model_version": transfer_model.model_version,
        "training_date": transfer_model.training_date,
        "metrics": transfer_model.metrics,
        "feature_count": len(transfer_model.feature_columns) if transfer_model.feature_columns else 0
    }


# Main entry point (for local development only)
# In production, use: uvicorn main:app --host 0.0.0.0 --port $PORT
if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

