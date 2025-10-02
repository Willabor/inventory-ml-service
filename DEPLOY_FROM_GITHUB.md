# 🚀 Deploy ML Service from GitHub

This guide shows how to deploy the ML service after pushing it to GitHub.

---

## 📦 Step 1: Create GitHub Repository

### From Replit Shell:

```bash
# Navigate to ML service directory
cd ml_service

# Initialize git (if not already done)
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial ML service for inventory predictions"

# Create repo on GitHub and push
# Option A: Using GitHub CLI (if installed)
gh repo create inventory-ml-service --public --source=. --remote=origin --push

# Option B: Manual (create repo on github.com first, then:)
git remote add origin https://github.com/YOUR_USERNAME/inventory-ml-service.git
git branch -M main
git push -u origin main
```

**Your ML service is now on GitHub!** 🎉

---

## 🌐 Step 2: Deploy to Platform

### Option A: Railway (Recommended - Easiest)

1. **Go to [railway.app](https://railway.app)**
2. **Click "New Project"**
3. **Select "Deploy from GitHub repo"**
4. **Choose your `inventory-ml-service` repo**
5. **Railway auto-detects Python** and uses:
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. **Add environment variable:**
   - Variable: `DATABASE_URL`
   - Value: `postgresql://your_user:pass@host:5432/db`
7. **Deploy** (automatic)
8. **Get your URL:** `https://your-service.railway.app`

**Train model:**
```bash
curl -X POST https://your-service.railway.app/api/ml/train
```

---

### Option B: Render

1. **Go to [render.com](https://render.com)**
2. **New → Web Service**
3. **Connect your GitHub** `inventory-ml-service` repo
4. **Configure:**
   - **Name**: `inventory-ml-service`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. **Add environment variable:**
   - `DATABASE_URL`: Your PostgreSQL connection string
6. **Create Web Service**
7. **Get your URL:** `https://inventory-ml-service.onrender.com`

---

### Option C: Vercel (Serverless)

1. **Install Vercel CLI:**
   ```bash
   npm i -g vercel
   ```

2. **Create `vercel.json` in ml_service:**
   ```json
   {
     "builds": [
       {
         "src": "main.py",
         "use": "@vercel/python"
       }
     ],
     "routes": [
       {
         "src": "/(.*)",
         "dest": "main.py"
       }
     ]
   }
   ```

3. **Deploy:**
   ```bash
   cd ml_service
   vercel --prod
   ```

4. **Add environment variables** in Vercel dashboard

---

## 🔗 Step 3: Connect to Your Main App

Once deployed, update your main Replit app:

### In Replit Secrets (🔒 icon):

Add secret:
- **Key**: `ML_SERVICE_URL`
- **Value**: `https://your-ml-service.railway.app` (your deployed URL)

### Or in `.env` file:

```bash
ML_SERVICE_URL=https://your-ml-service.railway.app
```

**That's it!** Your main app will now use the deployed ML service.

---

## ✅ Step 4: Test Integration

### Test 1: ML Service Health
```bash
curl https://your-ml-service.railway.app/health
```

Should return:
```json
{"status": "healthy", "model_loaded": false}
```

### Test 2: Train Model
```bash
curl -X POST https://your-ml-service.railway.app/api/ml/train \
  -H "Content-Type: application/json" \
  -d '{"days_back": 90}'
```

Wait ~60 seconds. Should return training metrics.

### Test 3: Get Predictions
```bash
curl -X POST https://your-ml-service.railway.app/api/ml/predict-transfers?limit=5
```

Should return ML predictions!

### Test 4: Frontend Toggle

1. Open your Replit app
2. Go to **Insights → Inventory Turnover**
3. Toggle **"Use AI"** ON
4. Should see confidence scores! 🎉

---

## 🔄 Continuous Deployment

Every time you push to GitHub, the platform auto-deploys:

```bash
cd ml_service

# Make changes to code
vim models/transfer_predictor.py

# Commit and push
git add .
git commit -m "Improve ML model accuracy"
git push

# Platform auto-deploys in ~2 minutes
```

**Railway/Render will automatically:**
1. Pull latest code
2. Install dependencies
3. Restart service
4. Your main app uses new version instantly!

---

## 📊 Monitor Deployments

### Railway
- Dashboard: https://railway.app/project/your-project
- Logs: Click service → "Deployments" → "Logs"
- Metrics: CPU, Memory, Requests

### Render
- Dashboard: https://dashboard.render.com
- Logs: Service → "Logs" tab
- Metrics: "Metrics" tab

---

## 🔧 Troubleshooting Deployment

### Build fails on Railway/Render

**Check Python version:**
- Railway auto-detects from `runtime.txt` (if you create one)
- Create `runtime.txt`:
  ```
  python-3.11
  ```

### Model training times out

**Increase timeout** (platform settings):
- Railway: Project Settings → "Timeouts"
- Render: Service Settings → "Health Check Path" (set to `/health`)

### Database connection fails

**Whitelist platform IPs:**
- If using Neon/Supabase, check IP whitelist
- Most platforms use dynamic IPs → use "Allow all" for now
- Or use connection pooling

### "Module not found" errors

**Check requirements.txt:**
```bash
cd ml_service
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Update dependencies"
git push
```

---

## 💰 Cost Estimates

### Free Tier Limits

| Platform | Free Tier | Limits |
|----------|-----------|--------|
| **Railway** | $5/month credit | 500 hours, sleeps after inactivity |
| **Render** | Free forever | Sleeps after 15 min, slow cold starts |
| **Vercel** | Hobby plan | 100GB bandwidth, serverless |

**Recommendation:**
- **Development**: Render (truly free)
- **Production**: Railway ($5-10/month, always-on)

---

## 🎯 Next Steps After Deployment

1. ✅ **Train initial model** (via API or platform console)
2. ✅ **Set up weekly retraining** (cron job or GitHub Actions)
3. ✅ **Monitor performance** (check model metrics)
4. ✅ **Scale if needed** (upgrade plan when traffic increases)

---

## 📞 Quick Reference

**GitHub repo structure:**
```
inventory-ml-service/
├── .gitignore
├── README.md
├── requirements.txt
├── Dockerfile
├── main.py
├── config.py
├── models/
├── utils/
└── .env.example
```

**Environment variables needed:**
- `DATABASE_URL` - PostgreSQL connection string
- `ML_SERVICE_PORT` - 8000 (auto-set by platforms)

**Deploy commands:**
- Railway: `railway up` or connect GitHub
- Render: Connect GitHub repo
- Vercel: `vercel --prod`

**Deployed URL goes to:**
- Replit Secrets: `ML_SERVICE_URL`
- Or `.env`: `ML_SERVICE_URL=https://...`

---

**You're ready to deploy! 🚀**

Choose Railway or Render, follow the steps above, and your ML service will be live in ~5 minutes!
