#!/usr/bin/env python3
"""
Setup script for DM-i-AI 2025 Emergency Healthcare RAG Competition
"""

import subprocess
import sys
import os
import platform

def print_banner():
    print("=" * 60)
    print("🏥 DM-i-AI 2025 Emergency Healthcare RAG Setup")
    print("=" * 60)
    print("🔧 Installing dependencies and preparing competition environment")
    print("=" * 60)

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_packages():
    """Install required packages"""
    print("\n📦 Installing Python packages...")
    
    # Try simple requirements first
    try:
        print("Attempting to install with simplified requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_simple.txt"])
        print("✅ Packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install with simple requirements: {e}")
        
        # Try individual packages
        print("\nTrying to install packages individually...")
        packages = [
            "torch", "transformers", "sentence-transformers", 
            "scikit-learn", "numpy", "tqdm", "datasets", "accelerate",
            "fastapi", "uvicorn", "pydantic", "loguru"
        ]
        
        for package in packages:
            try:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ {package} installed")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
                return False
        
        return True

def test_imports():
    """Test if all required packages can be imported"""
    print("\n🧪 Testing package imports...")
    
    test_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("sentence_transformers", "Sentence Transformers"),
        ("sklearn", "Scikit-learn"),
        ("numpy", "NumPy"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn")
    ]
    
    all_good = True
    for package, name in test_packages:
        try:
            __import__(package)
            print(f"✅ {name} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {name}: {e}")
            all_good = False
    
    return all_good

def check_data():
    """Check if training data is available"""
    print("\n📁 Checking data availability...")
    
    required_paths = [
        "data/train/statements",
        "data/train/answers", 
        "data/topics",
        "data/topics.json"
    ]
    
    all_good = True
    for path in required_paths:
        if os.path.exists(path):
            print(f"✅ {path}")
        else:
            print(f"❌ {path} - MISSING")
            all_good = False
    
    return all_good

def create_directories():
    """Create necessary directories"""
    print("\n📂 Creating directories...")
    
    directories = [
        "fine_tuned_medical_model",
        "logs"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created {directory}/")
        else:
            print(f"✅ {directory}/ already exists")

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install packages
    if not install_packages():
        print("\n❌ Package installation failed!")
        print("Please try the manual installation steps in fix_rust_installation.md")
        return
    
    # Test imports
    if not test_imports():
        print("\n❌ Some packages failed to import!")
        print("Please check the installation and try again.")
        return
    
    # Check data
    if not check_data():
        print("\n❌ Some data files are missing!")
        print("Please ensure all data files are in place.")
        return
    
    # Create directories
    create_directories()
    
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\n🚀 You can now run the competition:")
    print("   python run_competition.py")
    print("\n🏆 Or start training directly:")
    print("   python optimized_training.py")
    print("\n📖 For troubleshooting, see: fix_rust_installation.md")
    print("=" * 60)

if __name__ == "__main__":
    main() 