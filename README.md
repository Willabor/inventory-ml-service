# 🤖 Inventory ML Service

**AI-powered inventory transfer predictions using TabPFN machine learning**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TabPFN](https://img.shields.io/badge/TabPFN-ML-orange.svg)](https://github.com/priorlabs/tabpfn)

This is a standalone Python microservice that provides machine learning predictions for inventory transfer recommendations. It uses **TabPFN** (Tabular Prior-Data Fitted Network), a state-of-the-art transformer model for small tabular datasets.

## 🎯 Features

- **Transfer Predictions**: Predicts which inventory transfers will result in sales (75-85% accuracy)
- **TabPFN ML Model**: Transformer-based model trained on your historical sales data
- **FastAPI**: High-performance async REST API
- **PostgreSQL Integration**: Direct database access for training and predictions
- **Feature Engineering**: 30+ engineered features (velocity ratios, margins, stock levels)
- **Auto-scaling**: Optimized for serverless deployment

## 🚀 Quick Start

### Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env with your DATABASE_URL

# 3. Train initial model (optional - can skip for testing)
python -c "from models.transfer_predictor import TransferPredictor; p = TransferPredictor(); p.train(); p.save_model()"

# 4. Start service
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Service available at:**
- API: `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

## API Endpoints

### Health Check
```bash
GET /health
```

### Train Model
```bash
POST /api/ml/train
{
  "days_back": 90
}
```

### Get Transfer Predictions
```bash
POST /api/ml/predict-transfers?limit=20
```

### Model Info
```bash
GET /api/ml/model-info
```

## Docker Deployment

Build:
```bash
docker build -t inventory-ml-service .
```

Run:
```bash
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://... \
  inventory-ml-service
```

## Architecture

```
ml_service/
├── main.py                      # FastAPI application
├── config.py                    # Configuration management
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker configuration
├── models/
│   └── transfer_predictor.py   # TabPFN model wrapper
├── utils/
│   ├── database.py              # DB connection
│   ├── data_extraction.py       # SQL queries
│   └── feature_engineering.py  # Feature preparation
└── data/                        # Training data cache
```

## How It Works

1. **Data Extraction**: Pulls historical inventory and sales data from PostgreSQL
2. **Feature Engineering**: Creates ML features (velocity ratios, stock levels, etc.)
3. **Training**: TabPFN learns patterns from historical transfer outcomes
4. **Prediction**: Model predicts success probability for current transfer candidates
5. **Ranking**: Recommendations sorted by ML priority score

## Model Details

- **Algorithm**: TabPFN (Transformer-based)
- **Input Features**: ~30 engineered features per transfer
- **Target**: Binary classification (transfer success/failure)
- **Success Definition**: TO store sales velocity > FROM store × 1.5 AND ≥3 sales in 30 days
- **Training Data**: Last 90 days of inventory movements and sales
- **Retraining**: Recommended weekly or after major inventory changes

## Integration with Node.js Backend

The Node.js backend calls this service via HTTP:

```typescript
// Node.js backend
const mlResponse = await fetch('http://ml-service:8000/api/ml/predict-transfers', {
  method: 'POST',
  params: { limit: 20 }
});
const predictions = await mlResponse.json();
```

## Performance

- **Training Time**: 10-60 seconds (depends on data size)
- **Prediction Time**: 1-5 seconds for 100 candidates
- **Model Size**: ~50MB (includes TabPFN weights)
- **Memory Usage**: ~500MB-1GB
- **GPU**: Optional (10x faster with CUDA)

## Monitoring

Check model performance:
```bash
curl http://localhost:8000/api/ml/model-info
```

Returns:
```json
{
  "loaded": true,
  "model_version": "v20250101_120000",
  "training_date": "2025-01-01T12:00:00",
  "metrics": {
    "accuracy": 0.82,
    "precision": 0.79,
    "recall": 0.75,
    "roc_auc": 0.87
  }
}
```

## 🌐 Deployment Options

### Option 1: Railway (Easiest - Free Tier)

1. **Fork this repo** or create new GitHub repo
2. **Connect to Railway:**
   ```bash
   # Install Railway CLI
   npm i -g @railway/cli
   railway login
   railway init
   ```
3. **Add environment variable:**
   - `DATABASE_URL`: Your PostgreSQL connection string
4. **Deploy:**
   ```bash
   railway up
   ```
5. **Train model** (via Railway console or API):
   ```bash
   curl -X POST https://your-service.railway.app/api/ml/train
   ```

### Option 2: Render (Free Tier)

1. **New Web Service** on Render
2. **Settings:**
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. **Environment Variables:**
   - `DATABASE_URL`: Your PostgreSQL URL
4. **Deploy** and train model via API

### Option 3: Docker

```bash
# Build
docker build -t inventory-ml-service .

# Run
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://... \
  inventory-ml-service
```

### Option 4: Google Cloud Run / AWS Lambda

See `Dockerfile` - this service is optimized for serverless deployment.

---

## 🔗 Integration with Main App

Once deployed, connect it to your main Node.js app:

```javascript
// In your Node.js backend
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'https://your-ml-service.railway.app';

app.get('/api/inventory/transfer-recommendations-ml', async (req, res) => {
  const mlResponse = await fetch(`${ML_SERVICE_URL}/api/ml/predict-transfers?limit=20`, {
    method: 'POST'
  });
  const predictions = await mlResponse.json();
  res.json(predictions);
});
```

**Environment variable in main app:**
```bash
ML_SERVICE_URL=https://your-ml-service.railway.app
```

---

## 📊 API Usage Examples

### Get Predictions
```bash
curl -X POST "http://localhost:8000/api/ml/predict-transfers?limit=10"
```

**Response:**
```json
{
  "success": true,
  "count": 10,
  "model_version": "v20250101_120000",
  "predictions": [
    {
      "style_number": "ABC123",
      "item_name": "Sample Item",
      "from_store": "GM",
      "to_store": "HM",
      "success_probability": 0.87,
      "recommended_qty": 12,
      "ml_priority": "High",
      "confidence_level": "High"
    }
  ]
}
```

### Train Model
```bash
curl -X POST "http://localhost:8000/api/ml/train" \
  -H "Content-Type: application/json" \
  -d '{"days_back": 90}'
```

### Check Model Status
```bash
curl "http://localhost:8000/api/ml/model-info"
```

---

## 🐛 Troubleshooting

### Model not loading?
- Train a model first: `POST /api/ml/train`
- Check `models/cache/` directory exists
- View logs for errors

### Low accuracy?
- Increase `days_back` parameter (more training data needed)
- Ensure you have 90+ days of sales data
- Check class balance in training data

### Slow predictions?
- Reduce `limit` parameter in API call
- Consider caching predictions in your main database
- For GPU support: Add `ENABLE_GPU=true` environment variable

### Database connection errors?
- Verify `DATABASE_URL` format: `postgresql://user:pass@host:5432/db`
- Check database has required tables: `item_list`, `sales_transactions`
- Ensure database is accessible from deployment platform

---

## 📈 Model Performance

**Expected Metrics:**
- **Accuracy**: 75-85% (improves with more data)
- **Precision**: 75-80%
- **Recall**: 70-80%
- **ROC-AUC**: 0.85-0.90

**Training Requirements:**
- Minimum 100 samples (ideally 1000+)
- At least 30 days of sales data (90 days recommended)
- Balanced classes (successful vs. unsuccessful transfers)

**Retraining:**
- Recommended: Weekly via cron job
- Automatic: Set up GitHub Actions or platform scheduler
- Manual: `POST /api/ml/train` endpoint

---

## 🛠️ Development

### Project Structure
```
ml_service/
├── main.py                      # FastAPI application
├── config.py                    # Configuration
├── requirements.txt             # Dependencies
├── Dockerfile                   # Container config
├── models/
│   └── transfer_predictor.py   # TabPFN model wrapper
├── utils/
│   ├── database.py              # PostgreSQL connection
│   ├── data_extraction.py       # SQL queries
│   └── feature_engineering.py  # Feature preparation
└── data/                        # Training data cache
```

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest
```

### Code Quality
```bash
# Type checking
mypy .

# Linting
flake8 .

# Formatting
black .
```

---

## 🔐 Security Notes

- Never commit `.env` file with real credentials
- Use environment variables for all secrets
- Restrict database access to read-only if possible
- Add API authentication for production use
- Rate limit the `/train` endpoint (expensive operation)

---

## 📞 Support & Contribution

This service is designed to work standalone or as part of a larger inventory management system.

**Issues?** Check troubleshooting section above or open an issue.

**Want to contribute?** Pull requests welcome!

---

## 📄 License

MIT License - Same as parent project.
