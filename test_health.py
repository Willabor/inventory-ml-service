"""Simple health check script for debugging Railway deployment."""
import sys

print("=" * 60)
print("HEALTH CHECK TEST")
print("=" * 60)

try:
    # Test 1: Import modules
    print("Test 1: Importing FastAPI...")
    from fastapi import FastAPI
    print("✓ FastAPI imported")

    print("Test 2: Importing config...")
    from config import settings
    print(f"✓ Config loaded - Port: {settings.port}")

    print("Test 3: Importing models...")
    from models.transfer_predictor import TransferPredictor
    print("✓ TransferPredictor imported")

    print("Test 4: Creating FastAPI app...")
    app = FastAPI()
    print("✓ App created")

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    sys.exit(0)

except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    print("=" * 60)
    sys.exit(1)
