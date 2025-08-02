#!/usr/bin/env python3
"""
Run the full training process for the optimized medical model
"""

import sys
import os
import time

def main():
    print("=" * 60)
    print("🏥 DM-i-AI 2025 Emergency Healthcare RAG Training")
    print("=" * 60)
    print("Training the optimized medical statement classifier...")
    print("=" * 60)
    
    try:
        # Import and run training
        from optimized_training import train_optimized_model
        
        print("Starting training process...")
        start_time = time.time()
        
        # Train the model with optimized parameters
        model, tokenizer = train_optimized_model(
            model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
            batch_size=4,  # Reduced for memory constraints
            epochs=3,      # Reduced for faster training
            learning_rate=3e-5,
            warmup_steps=50,
            save_path="fine_tuned_medical_model",
            use_class_weights=True
        )
        
        training_time = time.time() - start_time
        
        print("=" * 60)
        print("✅ Training completed successfully!")
        print(f"⏱️  Total training time: {training_time/60:.1f} minutes")
        print("📁 Model saved to: fine_tuned_medical_model/")
        print("=" * 60)
        print("🚀 You can now use the trained model in your API!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("Please check the error message above and try again.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("🎉 Training completed successfully!")
    else:
        print("💥 Training failed. Please check the error messages above.")
        sys.exit(1) 