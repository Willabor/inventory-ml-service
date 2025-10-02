"""Minimal FastAPI server for debugging Railway deployment."""
from fastapi import FastAPI
import os

app = FastAPI()

@app.get("/")
def root():
    return {
        "status": "running",
        "message": "Minimal test server is working",
        "port": os.getenv("PORT", "not set"),
        "host": "0.0.0.0"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"Starting minimal server on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
