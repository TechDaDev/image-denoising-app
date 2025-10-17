#!/usr/bin/env python3
"""
Test script to verify all dependencies can be imported successfully.
"""

import sys

def test_imports():
    """Test importing all required packages."""
    try:
        print("Testing imports...")
        
        # Core dependencies
        import numpy as np
        print(f"✓ numpy {np.__version__}")
        
        from PIL import Image
        print(f"✓ Pillow")
        
        import scipy
        print(f"✓ scipy {scipy.__version__}")
        
        # Image processing
        import cv2
        print(f"✓ opencv-python-headless {cv2.__version__}")
        
        import pywt
        print(f"✓ pywavelets {pywt.__version__}")
        
        import skimage
        print(f"✓ scikit-image {skimage.__version__}")
        
        # Deep learning
        import tensorflow as tf
        print(f"✓ tensorflow {tf.__version__}")
        
        # Web framework
        import streamlit as st
        print(f"✓ streamlit {st.__version__}")
        
        # Data handling
        import pandas as pd
        print(f"✓ pandas {pd.__version__}")
        
        import matplotlib
        print(f"✓ matplotlib {matplotlib.__version__}")
        
        # Medical imaging
        import pydicom
        print(f"✓ pydicom {pydicom.__version__}")
        
        # Test protobuf explicitly
        import google.protobuf
        print(f"✓ protobuf {google.protobuf.__version__}")
        
        print("\n✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        return False

def test_tensorflow():
    """Test basic TensorFlow functionality."""
    try:
        print("\nTesting TensorFlow...")
        import tensorflow as tf
        
        # Test basic operations
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = tf.add(a, b)
        print(f"✓ TensorFlow basic operations work")
        
        # Test Keras (used in the app)
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(1,))
        ])
        print(f"✓ Keras model creation works")
        
        return True
        
    except Exception as e:
        print(f"❌ TensorFlow test failed: {e}")
        return False

def test_streamlit():
    """Test basic Streamlit functionality."""
    try:
        print("\nTesting Streamlit...")
        import streamlit as st
        
        # Test basic streamlit functions (without running the server)
        print(f"✓ Streamlit import successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Streamlit test failed: {e}")
        return False

if __name__ == "__main__":
    print("Dependency Test Script")
    print("=" * 40)
    
    success = True
    success &= test_imports()
    success &= test_tensorflow()
    success &= test_streamlit()
    
    if success:
        print("\n🎉 All tests passed! Dependencies are working correctly.")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed. Check the error messages above.")
        sys.exit(1)