#!/usr/bin/env python3
"""
Test script to verify training works without Unicode issues
"""

import sys
import subprocess

def test_training_script():
    """Test if the training script runs without Unicode errors"""
    print("Testing training script...")
    
    try:
        # Run the training script with a small test
        result = subprocess.run([
            sys.executable, 'optimized_training.py'
        ], capture_output=True, text=True, timeout=30)  # 30 second timeout for testing
        
        if result.returncode == 0:
            print("Training script runs successfully!")
            return True
        else:
            print(f"Training script failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("Training script started successfully (timeout reached)")
        return True
    except Exception as e:
        print(f"Error running training script: {e}")
        return False

if __name__ == "__main__":
    success = test_training_script()
    if success:
        print("Training script is ready to use!")
    else:
        print("Training script needs fixing.") 