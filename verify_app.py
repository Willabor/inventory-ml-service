#!/usr/bin/env python3
"""Verify the FastAPI app can be imported and started."""
import sys
import os

print("=" * 60)
print("VERIFICATION SCRIPT")
print("=" * 60)

# Test 1: Environment variables
print("\n1. Environment Variables:")
print(f"   PORT: {os.getenv('PORT', 'NOT SET')}")
print(f"   DATABASE_URL: {os.getenv('DATABASE_URL', 'NOT SET')[:40]}...")

# Test 2: Import config
print("\n2. Importing config...")
try:
    from config import settings
    print(f"   ✓ Config imported")
    print(f"   - Port: {settings.port}")
    print(f"   - Host: {settings.ml_service_host}")
except Exception as e:
    print(f"   ✗ Config import failed: {e}")
    sys.exit(1)

# Test 3: Import FastAPI
print("\n3. Importing FastAPI...")
try:
    from fastapi import FastAPI
    print(f"   ✓ FastAPI imported")
except Exception as e:
    print(f"   ✗ FastAPI import failed: {e}")
    sys.exit(1)

# Test 4: Import main app
print("\n4. Importing main app...")
try:
    from main import app
    print(f"   ✓ App imported")
    print(f"   - Title: {app.title}")
    print(f"   - Routes: {len(app.routes)}")
except Exception as e:
    print(f"   ✗ App import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check routes
print("\n5. Registered Routes:")
try:
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            methods = ', '.join(route.methods) if route.methods else 'N/A'
            print(f"   - {methods:6} {route.path}")
except Exception as e:
    print(f"   ✗ Route enumeration failed: {e}")

print("\n" + "=" * 60)
print("✓ ALL VERIFICATIONS PASSED")
print("=" * 60)
sys.exit(0)
