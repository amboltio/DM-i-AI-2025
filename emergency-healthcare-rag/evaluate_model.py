#!/usr/bin/env python3
"""
Evaluate the trained medical model
"""

import json
import os
import torch
from transformers import AutoTokenizer
from optimized_model import OptimizedMedicalClassifier
from utils import load_statement_sample

def load_topics():
    """Load topic mapping"""
    with open('data/topics.json', 'r') as f:
        topics = json.load(f)
    return topics

def evaluate_model():
    """Evaluate the trained model on test samples"""
    print("=" * 60)
    print("🔍 Evaluating Trained Medical Model")
    print("=" * 60)
    
    # Check if model exists
    model_path = "fine_tuned_medical_model"
    if not os.path.exists(model_path):
        print("❌ Model not found! Please train the model first.")
        print("Run: python run_training.py")
        return
    
    try:
        # Load model and tokenizer
        print("Loading trained model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = OptimizedMedicalClassifier.from_pretrained(model_path)
        
        # Load topics
        topics = load_topics()
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully on {device}")
        
        # Test on sample data
        test_samples = ["0000", "0001", "0002", "0003", "0004"]
        
        correct_truth = 0
        correct_topic = 0
        total = 0
        
        print("\nTesting on sample data:")
        print("-" * 60)
        
        for sample_id in test_samples:
            try:
                # Load sample
                statement, true_answer = load_statement_sample(sample_id)
                
                # Tokenize
                inputs = tokenizer(
                    statement,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(device)
                
                # Predict
                with torch.no_grad():
                    truth_logits, topic_logits = model(**inputs)
                    
                    truth_pred = torch.argmax(truth_logits, dim=1).item()
                    topic_pred = torch.argmax(topic_logits, dim=1).item()
                
                # Get topic names
                true_topic_name = topics[str(true_answer['statement_topic'])]
                pred_topic_name = topics[str(topic_pred)]
                
                # Check accuracy
                truth_correct = truth_pred == true_answer['statement_is_true']
                topic_correct = topic_pred == true_answer['statement_topic']
                
                if truth_correct:
                    correct_truth += 1
                if topic_correct:
                    correct_topic += 1
                total += 1
                
                # Print results
                print(f"Sample {sample_id}:")
                print(f"  Statement: {statement[:100]}...")
                print(f"  Truth: Predicted={truth_pred}, Actual={true_answer['statement_is_true']} {'✅' if truth_correct else '❌'}")
                print(f"  Topic: Predicted={pred_topic_name}, Actual={true_topic_name} {'✅' if topic_correct else '❌'}")
                print()
                
            except Exception as e:
                print(f"Error processing sample {sample_id}: {e}")
                continue
        
        # Calculate accuracy
        truth_accuracy = correct_truth / total if total > 0 else 0
        topic_accuracy = correct_topic / total if total > 0 else 0
        combined_accuracy = 0.4 * truth_accuracy + 0.6 * topic_accuracy
        
        print("=" * 60)
        print("📊 Evaluation Results:")
        print(f"  Truth Accuracy: {truth_accuracy:.2%} ({correct_truth}/{total})")
        print(f"  Topic Accuracy: {topic_accuracy:.2%} ({correct_topic}/{total})")
        print(f"  Combined Accuracy: {combined_accuracy:.2%}")
        print("=" * 60)
        
        if combined_accuracy > 0.7:
            print("🎉 Excellent performance!")
        elif combined_accuracy > 0.5:
            print("👍 Good performance!")
        else:
            print("⚠️  Performance needs improvement. Consider training longer or adjusting parameters.")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        print("Please check if the model was trained correctly.")

if __name__ == "__main__":
    evaluate_model() 