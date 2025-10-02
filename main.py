"""FastAPI service for ML predictions."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
from datetime import datetime

from config import settings
from models.transfer_predictor import TransferPredictor
from models.segmentation_predictor import SegmentationPredictor

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

# Global model instances (loaded on startup)
transfer_model: Optional[TransferPredictor] = None
segmentation_model: Optional[SegmentationPredictor] = None


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
    new_arrivals_days: Optional[int] = 60
    best_seller_threshold: Optional[int] = 50
    core_high_threshold: Optional[int] = 40
    core_medium_threshold: Optional[int] = 20
    core_low_threshold: Optional[int] = 6
    clearance_days: Optional[int] = 180


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
    """Load the latest models on service startup."""
    global transfer_model, segmentation_model

    print("=" * 60)
    print("STARTING ML SERVICE")
    print("=" * 60)
    print(f"Port: {settings.port}")
    print(f"Host: {settings.ml_service_host}")
    print(f"Database URL: {settings.database_url[:30]}..." if settings.database_url else "Database URL: NOT SET")
    print(f"Model Cache Dir: {settings.model_cache_dir}")
    print("=" * 60)

    # Load transfer prediction model
    try:
        transfer_model = TransferPredictor.load_latest()
        print("✓ Transfer model loaded successfully")
        print(f"  Version: {transfer_model.model_version}")
    except FileNotFoundError:
        print("⚠ No trained transfer model found. Train using /api/ml/train endpoint")
        transfer_model = None
    except Exception as e:
        print(f"✗ Error loading transfer model: {e}")

    # Load segmentation model
    try:
        segmentation_model = SegmentationPredictor.load_latest()
        print("✓ Segmentation model loaded successfully")
        print(f"  Version: {segmentation_model.model_version}")
    except FileNotFoundError:
        print("⚠ No trained segmentation model found. Train using /api/ml/train-segmentation endpoint")
        segmentation_model = None
    except Exception as e:
        print(f"✗ Error loading segmentation model: {e}")
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


# ==========================
# PRODUCT SEGMENTATION ENDPOINTS
# ==========================

@app.post("/api/ml/train-segmentation")
async def train_segmentation_model(request: TrainingRequest):
    """Train the product segmentation model with custom parameters."""
    global segmentation_model

    try:
        # Initialize new model with custom thresholds
        segmentation_model = SegmentationPredictor(
            new_arrivals_days=request.new_arrivals_days,
            best_seller_threshold=request.best_seller_threshold,
            core_high_threshold=request.core_high_threshold,
            core_medium_threshold=request.core_medium_threshold,
            core_low_threshold=request.core_low_threshold,
            clearance_days=request.clearance_days
        )

        # Train the model
        metrics = segmentation_model.train(days_back=request.days_back)

        # Save model
        segmentation_model.save_model()

        return TrainingResponse(
            success=True,
            model_version=segmentation_model.model_version,
            metrics=metrics,
            training_date=segmentation_model.training_date.isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/api/ml/product-segmentation")
async def predict_product_segmentation():
    """Generate ML-powered product segmentation for Google Marketing."""
    from utils.database import db

    try:
        if segmentation_model is None:
            raise HTTPException(
                status_code=503,
                detail="Segmentation model not loaded. Train a model first using /api/ml/train-segmentation"
            )

        # Fetch product data from database
        query = """
            WITH style_metrics AS (
                SELECT
                    i.style_number,
                    MAX(i.item_name) as item_name,
                    MAX(i.category) as category,
                    MAX(i.vendor_name) as vendor_name,
                    MAX(i.gender) as gender,
                    SUM(COALESCE(i.gm_qty, 0) + COALESCE(i.hm_qty, 0) + COALESCE(i.nm_qty, 0) + COALESCE(i.lm_qty, 0) + COALESCE(i.hq_qty, 0)) as total_active_qty,
                    AVG(i.order_cost::numeric) as avg_order_cost,
                    AVG(i.selling_price::numeric) as avg_selling_price,
                    AVG(CASE
                        WHEN i.selling_price::numeric > 0
                        THEN ((i.selling_price::numeric - i.order_cost::numeric) / i.selling_price::numeric * 100)
                        ELSE 0
                    END) as avg_margin_percent,
                    SUM((COALESCE(i.gm_qty, 0) + COALESCE(i.hm_qty, 0) + COALESCE(i.nm_qty, 0) + COALESCE(i.lm_qty, 0) + COALESCE(i.hq_qty, 0)) * i.order_cost::numeric) as inventory_value,
                    MAX(i.last_rcvd) as last_received,
                    COUNT(DISTINCT i.item_number) as receive_count,
                    MAX(i.style_number_2) as classification,
                    'All-Season' as seasonal_pattern,
                    'Normal' as stock_status
                FROM item_list i
                WHERE i.style_number IS NOT NULL
                GROUP BY i.style_number
            ),
            style_sales AS (
                SELECT
                    i.style_number,
                    COUNT(*) FILTER (WHERE s.date >= CURRENT_DATE - INTERVAL '30 days') as units_sold_30d,
                    COUNT(*) FILTER (WHERE s.date >= CURRENT_DATE - INTERVAL '90 days') as units_sold_90d,
                    MAX(s.date) as last_sale_date
                FROM sales_transactions s
                JOIN item_list i ON s.sku = i.item_number
                WHERE i.style_number IS NOT NULL
                GROUP BY i.style_number
            )
            SELECT
                sm.*,
                COALESCE(ss.units_sold_30d, 0) as units_sold_30d,
                COALESCE(ss.units_sold_90d, 0) as units_sold_90d,
                COALESCE(ss.last_sale_date, NULL) as last_sale_date,
                CASE
                    WHEN ss.last_sale_date IS NOT NULL
                    THEN CURRENT_DATE - ss.last_sale_date
                    ELSE NULL
                END as days_since_last_sale,
                CASE
                    WHEN sm.last_received IS NOT NULL
                    THEN CURRENT_DATE - sm.last_received
                    ELSE NULL
                END as days_since_last_receive,
                CASE
                    WHEN sm.total_active_qty > 0
                    THEN COALESCE(ss.units_sold_30d, 0)::numeric / 30.0
                    ELSE 0
                END as sales_velocity,
                CASE
                    WHEN sm.avg_selling_price > 0
                    THEN sm.avg_selling_price - sm.avg_order_cost
                    ELSE 0
                END as margin_per_unit
            FROM style_metrics sm
            LEFT JOIN style_sales ss ON sm.style_number = ss.style_number
            WHERE sm.total_active_qty > 0
        """

        data = db.execute_query(query)

        if data.empty:
            return {
                "metadata": {
                    "generatedDate": datetime.now().isoformat(),
                    "totalStyles": 0,
                    "totalActiveInventoryValue": 0,
                    "analysisDateRange": "No data",
                    "modelVersion": segmentation_model.model_version,
                    "mlPowered": True
                },
                "segments": {
                    "bestSellers": [],
                    "coreHighFrequency": [],
                    "coreMediumFrequency": [],
                    "coreLowFrequency": [],
                    "nonCoreRepeat": [],
                    "oneTimePurchase": [],
                    "newArrivals": [],
                    "summerItems": [],
                    "winterItems": [],
                    "clearanceCandidates": []
                }
            }

        # Predict segments using ML model
        results = segmentation_model.predict(data)

        # Enrich with marketing data
        def enrich_product(row):
            # Generate Google-optimized product title
            title_parts = []
            if row.get('vendor_name'):
                title_parts.append(row['vendor_name'])
            title_parts.append(row['item_name'])
            if row.get('category'):
                title_parts.append(f"- {row['category']}")
            product_title = " ".join(title_parts)[:150]  # Google limit

            # Generate keywords
            keywords = []
            if row.get('vendor_name'):
                keywords.append(row['vendor_name'].lower())
            if row.get('category'):
                keywords.append(row['category'].lower())
            if row.get('gender'):
                keywords.append(row['gender'].lower())
            keywords.extend(row['item_name'].lower().split()[:5])

            # Determine budget tier based on ML confidence + margin
            confidence = row.get('ml_confidence', 0)
            margin = row.get('avg_margin_percent', 0)
            score = (confidence * 60) + (margin * 0.4)

            if score >= 65:
                budget_tier = 'High'
                priority = 5
            elif score >= 45:
                budget_tier = 'Medium'
                priority = 3
            else:
                budget_tier = 'Low'
                priority = 1

            # Map segment to Google category
            category_map = {
                'Apparel & Accessories': row.get('category', 'Apparel & Accessories'),
                'default': 'Apparel & Accessories > Clothing'
            }
            google_category = category_map.get(row.get('category', ''), category_map['default'])

            import pandas as pd
            import numpy as np

            # Helper to safely convert to int, handling NaN
            def safe_int(value, default=0):
                if pd.isna(value) or value is None:
                    return None if default is None else default
                try:
                    return int(value)
                except (ValueError, TypeError):
                    return None if default is None else default

            # Helper to safely convert to float
            def safe_float(value, default=0.0):
                if pd.isna(value) or value is None:
                    return default
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default

            return {
                'styleNumber': row['style_number'],
                'itemName': row['item_name'],
                'category': row.get('category'),
                'vendorName': row.get('vendor_name'),
                'gender': row.get('gender'),
                'totalActiveQty': safe_int(row.get('total_active_qty'), 0),
                'avgOrderCost': safe_float(row.get('avg_order_cost'), 0),
                'avgSellingPrice': safe_float(row.get('avg_selling_price'), 0),
                'avgMarginPercent': safe_float(row.get('avg_margin_percent'), 0),
                'inventoryValue': safe_float(row.get('inventory_value'), 0),
                'classification': row.get('classification', 'Unknown'),
                'seasonalPattern': row.get('seasonal_pattern', 'All-Season'),
                'lastReceived': row.get('last_received').isoformat() if row.get('last_received') and not pd.isna(row.get('last_received')) else None,
                'daysSinceLastReceive': safe_int(row.get('days_since_last_receive'), None),
                'receiveCount': safe_int(row.get('receive_count'), 0),
                'stockStatus': row.get('stock_status', 'Normal'),
                'unitsSold30d': safe_int(row.get('units_sold_30d'), 0),
                'unitsSold90d': safe_int(row.get('units_sold_90d'), 0),
                'salesVelocity': safe_float(row.get('sales_velocity'), 0),
                'lastSaleDate': row.get('last_sale_date').isoformat() if row.get('last_sale_date') and not pd.isna(row.get('last_sale_date')) else None,
                'productTitle': product_title,
                'keywords': keywords,
                'googleCategory': google_category,
                'priority': priority,
                'budgetTier': budget_tier,
                'segment': row['ml_segment'],
                'marginPerUnit': safe_float(row.get('margin_per_unit'), 0),
            }

        # Organize by ML-predicted segments
        enriched_products = results.apply(enrich_product, axis=1).tolist()

        # Group by segments
        segments = {
            'bestSellers': [p for p in enriched_products if p['segment'] == 'Best Seller'],
            'coreHighFrequency': [p for p in enriched_products if p['segment'] == 'Core High'],
            'coreMediumFrequency': [p for p in enriched_products if p['segment'] == 'Core Medium'],
            'coreLowFrequency': [p for p in enriched_products if p['segment'] == 'Core Low'],
            'nonCoreRepeat': [p for p in enriched_products if p['segment'] == 'Non-Core Repeat'],
            'oneTimePurchase': [p for p in enriched_products if p['segment'] == 'One-Time'],
            'newArrivals': [p for p in enriched_products if p['segment'] == 'New Arrival'],
            'summerItems': [p for p in enriched_products if p['seasonalPattern'] == 'Summer'],
            'winterItems': [p for p in enriched_products if p['seasonalPattern'] == 'Winter'],
            'clearanceCandidates': [p for p in enriched_products if p['segment'] == 'Clearance'],
        }

        # Calculate metadata
        total_value = sum(p['inventoryValue'] for p in enriched_products)

        return {
            "metadata": {
                "generatedDate": datetime.now().isoformat(),
                "totalStyles": len(enriched_products),
                "totalActiveInventoryValue": total_value,
                "analysisDateRange": "Last 90 days",
                "modelVersion": segmentation_model.model_version,
                "mlPowered": True
            },
            "segments": segments,
            "mlInsights": {
                "segmentConfidence": {
                    seg: float(results[results['ml_segment'] == seg]['ml_confidence'].mean())
                    for seg in results['ml_segment'].unique()
                },
                "recommendedActions": [
                    f"Focus High budget tier on {len(segments['bestSellers'])} Best Sellers",
                    f"Launch campaigns for {len(segments['newArrivals'])} New Arrivals",
                    f"Clear {len(segments['clearanceCandidates'])} products with deep discounts"
                ]
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


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

