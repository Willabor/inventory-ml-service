#!/usr/bin/env python3
"""
Auto-train ML models on Railway startup
This script trains both Transfer and Segmentation models
to solve the ephemeral storage issue
"""

import os
import sys
import time
import requests
from datetime import datetime

ML_SERVICE_URL = os.getenv('ML_SERVICE_URL', 'http://localhost:8080')
MAX_RETRIES = 30  # 30 retries = ~5 minutes with 10 second intervals
RETRY_INTERVAL = 10  # seconds

def wait_for_service():
    """Wait for the ML service to be ready"""
    print("⏳ Waiting for ML service to be ready...")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(f"{ML_SERVICE_URL}/health", timeout=5)
            if response.status_code == 200:
                print(f"✓ ML service is ready (attempt {attempt}/{MAX_RETRIES})")
                return True
        except requests.exceptions.RequestException as e:
            print(f"⌛ Attempt {attempt}/{MAX_RETRIES}: Service not ready yet...")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_INTERVAL)

    print(f"❌ Service did not become ready after {MAX_RETRIES} attempts")
    return False

def train_model(endpoint: str, model_name: str, days_back: int = 90) -> bool:
    """Train a specific model"""
    print(f"\n🧠 Training {model_name} model...")
    print(f"   Endpoint: {endpoint}")
    print(f"   Days back: {days_back}")

    try:
        start_time = time.time()
        response = requests.post(
            f"{ML_SERVICE_URL}{endpoint}",
            json={"days_back": days_back},
            timeout=300  # 5 minutes timeout for training
        )
        elapsed = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            print(f"✓ {model_name} model trained successfully!")
            print(f"   Time taken: {elapsed:.1f}s")
            print(f"   Model version: {result.get('model_version', 'unknown')}")

            if 'test_accuracy' in result:
                print(f"   Test accuracy: {result['test_accuracy']:.2%}")
            if 'samples_trained' in result:
                print(f"   Samples trained: {result['samples_trained']:,}")

            return True
        else:
            print(f"❌ {model_name} training failed!")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print(f"⚠️  {model_name} training timed out (>5 minutes)")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ {model_name} training error: {e}")
        return False

def verify_models():
    """Verify both models are loaded"""
    print("\n🔍 Verifying models are loaded...")

    try:
        response = requests.get(f"{ML_SERVICE_URL}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()

            if health.get('model_loaded'):
                print("✓ Models are loaded and ready!")
                print(f"   Status: {health.get('status')}")
                print(f"   Model version: {health.get('model_version')}")
                return True
            else:
                print("⚠️  Models are NOT loaded")
                return False
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Health check error: {e}")
        return False

def main():
    """Main auto-training sequence"""
    print("=" * 60)
    print("🤖 ML MODEL AUTO-TRAINING")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target: {ML_SERVICE_URL}")
    print()

    # Wait for service to be ready
    if not wait_for_service():
        print("\n❌ Auto-training aborted: Service not ready")
        sys.exit(1)

    # Train Transfer Recommendations model
    transfer_success = train_model(
        endpoint="/api/ml/train",
        model_name="Transfer Recommendations",
        days_back=90
    )

    # Train Product Segmentation model
    segmentation_success = train_model(
        endpoint="/api/ml/train-segmentation",
        model_name="Product Segmentation",
        days_back=90
    )

    # Verify models are loaded
    models_verified = verify_models()

    # Summary
    print("\n" + "=" * 60)
    print("📊 AUTO-TRAINING SUMMARY")
    print("=" * 60)
    print(f"Transfer Model:     {'✓ Success' if transfer_success else '❌ Failed'}")
    print(f"Segmentation Model: {'✓ Success' if segmentation_success else '❌ Failed'}")
    print(f"Models Verified:    {'✓ Yes' if models_verified else '❌ No'}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Exit with appropriate code
    if transfer_success and segmentation_success and models_verified:
        print("\n✅ All models trained successfully!")
        sys.exit(0)
    else:
        print("\n⚠️  Some models failed to train")
        sys.exit(1)

if __name__ == "__main__":
    main()
