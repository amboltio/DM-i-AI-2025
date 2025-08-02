#!/usr/bin/env python3
"""
Main script to run the DM-i-AI 2025 Emergency Healthcare RAG Competition
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def print_banner():
    """Print competition banner"""
    print("=" * 60)
    print("🏥 DM-i-AI 2025 Emergency Healthcare RAG Competition")
    print("=" * 60)
    print("🏆 Optimized Solution for Best Performance")
    print("=" * 60)

def check_dependencies():
    """Check if all dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        ('torch', 'torch'),
        ('transformers', 'transformers'), 
        ('sentence_transformers', 'sentence-transformers'),  # Note: import name vs package name
        ('sklearn', 'scikit-learn'),  # Note: import name vs package name
        ('numpy', 'numpy'),
        ('fastapi', 'fastapi'),
        ('uvicorn', 'uvicorn')
    ]
    
    missing_packages = []
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print(f"  ✅ {package_name}")
        except ImportError:
            missing_packages.append(package_name)
            print(f"  ❌ {package_name}")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed!")
    return True

def check_data():
    """Check if training data is available"""
    print("\n📁 Checking data availability...")
    
    required_paths = [
        'data/train/statements',
        'data/train/answers', 
        'data/topics',
        'data/topics.json'
    ]
    
    missing_data = []
    for path in required_paths:
        if os.path.exists(path):
            print(f"  ✅ {path}")
        else:
            missing_data.append(path)
            print(f"  ❌ {path}")
    
    if missing_data:
        print(f"\n❌ Missing data: {', '.join(missing_data)}")
        print("Please ensure all data files are in place.")
        return False
    
    print("✅ All data available!")
    return True

def train_model():
    """Train the optimized model"""
    print("\n🚀 Starting model training...")
    print("This will take 30-60 minutes depending on your GPU.")
    
    try:
        # Run training script
        result = subprocess.run([
            sys.executable, 'optimized_training.py'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
            return True
        else:
            print(f"❌ Training failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False

def test_model():
    """Test the trained model"""
    print("\n🧪 Testing model performance...")
    
    try:
        # Run test script
        result = subprocess.run([
            sys.executable, 'test_optimized_model.py'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Model testing completed!")
            print(result.stdout)
            return True
        else:
            print(f"❌ Testing failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Testing error: {e}")
        return False

def start_api():
    """Start the API server"""
    print("\n🌐 Starting API server...")
    print("API will be available at: http://localhost:8000")
    print("Press Ctrl+C to stop the server")
    
    try:
        # Start API server
        subprocess.run([
            sys.executable, 'api.py'
        ])
    except KeyboardInterrupt:
        print("\n🛑 API server stopped.")
    except Exception as e:
        print(f"❌ API error: {e}")

def main():
    """Main competition runner"""
    print_banner()
    
    # Step 1: Check dependencies
    if not check_dependencies():
        return
    
    # Step 2: Check data
    if not check_data():
        return
    
    # Step 3: Show menu
    while True:
        print("\n" + "=" * 40)
        print("🎯 COMPETITION MENU")
        print("=" * 40)
        print("1. 🚀 Train Optimized Model")
        print("2. 🧪 Test Model Performance") 
        print("3. 🌐 Start API Server")
        print("4. 📊 View Competition Info")
        print("5. 🏆 Run Full Competition Pipeline")
        print("6. ❌ Exit")
        print("=" * 40)
        
        choice = input("\nSelect an option (1-6): ").strip()
        
        if choice == "1":
            if train_model():
                print("\n✅ Model training completed! You can now test or deploy.")
            else:
                print("\n❌ Training failed. Check the error messages above.")
                
        elif choice == "2":
            if test_model():
                print("\n✅ Model testing completed!")
            else:
                print("\n❌ Testing failed. Check the error messages above.")
                
        elif choice == "3":
            start_api()
            
        elif choice == "4":
            show_competition_info()
            
        elif choice == "5":
            run_full_pipeline()
            
        elif choice == "6":
            print("\n👋 Good luck with the competition! 🏆")
            break
            
        else:
            print("❌ Invalid choice. Please select 1-6.")

def show_competition_info():
    """Show competition information"""
    print("\n" + "=" * 50)
    print("📊 COMPETITION INFORMATION")
    print("=" * 50)
    print("🏥 Task: Emergency Healthcare RAG")
    print("📝 Goal: Classify medical statements (true/false + topic)")
    print("📊 Data: 200 training + 200 validation + 749 evaluation")
    print("⏱️  Constraint: <5 seconds per prediction")
    print("🔒 Privacy: Must run offline")
    print("💾 Memory: Max 24GB VRAM")
    print("🎯 Evaluation: Accuracy on both classifications")
    print("=" * 50)
    
    print("\n🏆 OUR OPTIMIZED SOLUTION:")
    print("• PubMedBERT (medical-specific model)")
    print("• Enhanced RAG with medical context")
    print("• Ensemble approach for best accuracy")
    print("• Expected accuracy: 82-87%")
    print("• Expected speed: 0.5-1.5 seconds")
    
    print("\n📋 NEXT STEPS:")
    print("1. Train the model (30-60 minutes)")
    print("2. Test performance on validation set")
    print("3. Deploy API for evaluation")
    print("4. Submit to competition (ONCE ONLY)")

def run_full_pipeline():
    """Run the complete competition pipeline"""
    print("\n🏆 RUNNING FULL COMPETITION PIPELINE")
    print("=" * 50)
    
    # Step 1: Train
    print("Step 1/4: Training model...")
    if not train_model():
        print("❌ Pipeline failed at training step")
        return
    
    # Step 2: Test
    print("\nStep 2/4: Testing model...")
    if not test_model():
        print("❌ Pipeline failed at testing step")
        return
    
    # Step 3: Show deployment info
    print("\nStep 3/4: Deployment ready!")
    print("✅ Model trained and tested successfully")
    print("🌐 Ready to deploy API server")
    
    # Step 4: Competition submission info
    print("\nStep 4/4: Competition submission")
    print("📋 To submit to competition:")
    print("1. Deploy your API (local or cloud)")
    print("2. Test on validation set multiple times")
    print("3. Submit to evaluation set (ONCE ONLY)")
    print("4. Monitor leaderboard performance")
    
    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("You're ready for the competition! 🏆")

if __name__ == "__main__":
    main() 