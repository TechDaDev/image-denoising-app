#!/usr/bin/env python3
"""
Test script to verify the Streamlit app works correctly
"""

import sys
import subprocess
import time
import requests
import threading
from pathlib import Path

def test_app_startup():
    """Test if the Streamlit app can start without errors"""
    print("Testing Streamlit app startup...")
    
    # Start the app in a subprocess
    process = subprocess.Popen([
        'streamlit', 'run', 'streamlit_app.py', 
        '--server.headless', 'true',
        '--server.port', '8503'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Wait a bit for the app to start
    time.sleep(5)
    
    # Check if the process is still running (no immediate crash)
    if process.poll() is None:
        print("✅ App started successfully")
        
        # Try to access the app
        try:
            response = requests.get('http://localhost:8503', timeout=10)
            if response.status_code == 200:
                print("✅ App is accessible via HTTP")
            else:
                print(f"⚠️ App returned status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Could not access app via HTTP: {e}")
        
        # Terminate the process
        process.terminate()
        process.wait()
        print("✅ App terminated cleanly")
        return True
    else:
        # Process crashed, get the error output
        stdout, stderr = process.communicate()
        print("❌ App failed to start")
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        return False

def test_dependencies():
    """Test if all required dependencies can be imported"""
    print("Testing dependencies...")
    
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__}")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__}")
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import skimage
        print(f"✅ scikit-image {skimage.__version__}")
    except ImportError as e:
        print(f"❌ scikit-image import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"✅ Pillow")
    except ImportError as e:
        print(f"❌ Pillow import failed: {e}")
        return False
    
    try:
        import pydicom
        print(f"✅ pydicom {pydicom.__version__}")
    except ImportError as e:
        print(f"❌ pydicom import failed: {e}")
        return False
    
    return True

def test_models():
    """Test if model files exist and can be loaded"""
    print("Testing models...")
    
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ Models directory does not exist")
        return False
    
    model_files = list(models_dir.glob("*.h5"))
    if not model_files:
        print("❌ No model files found")
        return False
    
    print(f"✅ Found {len(model_files)} model files")
    
    # Test loading one model
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(str(model_files[0]), compile=False)
        model.compile(optimizer='adam', loss='mse')
        print(f"✅ Successfully loaded model: {model_files[0].name}")
        return True
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running Streamlit App Tests")
    print("=" * 50)
    
    all_passed = True
    
    # Test dependencies
    if not test_dependencies():
        all_passed = False
    
    print()
    
    # Test models
    if not test_models():
        all_passed = False
    
    print()
    
    # Test app startup
    if not test_app_startup():
        all_passed = False
    
    print()
    print("=" * 50)
    if all_passed:
        print("🎉 All tests passed! The app should work correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())