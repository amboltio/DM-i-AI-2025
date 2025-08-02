#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

def test_imports():
    """Test all required package imports"""
    print("🧪 Testing package imports...")
    
    test_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("sentence_transformers", "Sentence Transformers"),
        ("sklearn", "Scikit-learn"),
        ("numpy", "NumPy"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("tqdm", "TQDM"),
        ("datasets", "Datasets"),
        ("accelerate", "Accelerate"),
        ("pydantic", "Pydantic"),
        ("loguru", "Loguru")
    ]
    
    all_good = True
    for import_name, display_name in test_packages:
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {display_name} - Version: {version}")
        except ImportError as e:
            print(f"❌ Failed to import {display_name}: {e}")
            all_good = False
    
    return all_good

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n🎉 All imports successful! You're ready to run the competition.")
    else:
        print("\n❌ Some imports failed. Please check your installation.") 