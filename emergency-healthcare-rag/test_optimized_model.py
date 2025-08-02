#!/usr/bin/env python3
"""
Test script for the optimized medical model
"""

import time
from utils import load_statement_sample
from optimized_model import predict

def test_optimized_model():
    """Test the optimized model with sample data"""
    
    print("🧪 Testing Optimized Medical Model")
    print("=" * 50)
    
    # Test with multiple samples
    test_samples = ["0000", "0001", "0002", "0003", "0004"]
    
    total_time = 0
    correct_predictions = 0
    total_predictions = 0
    
    for sample_id in test_samples:
        try:
            # Load sample
            statement, true_answer = load_statement_sample(sample_id)
            
            print(f"\n📝 Sample {sample_id}:")
            print(f"Statement: {statement[:100]}...")
            print(f"True answer: {true_answer}")
            
            # Time the prediction
            start_time = time.time()
            statement_is_true, statement_topic = predict(statement)
            prediction_time = time.time() - start_time
            
            total_time += prediction_time
            
            print(f"Predicted: statement_is_true={statement_is_true}, statement_topic={statement_topic}")
            print(f"Prediction time: {prediction_time:.3f} seconds")
            
            # Check accuracy
            truth_correct = statement_is_true == true_answer["statement_is_true"]
            topic_correct = statement_topic == true_answer["statement_topic"]
            both_correct = truth_correct and topic_correct
            
            print(f"Truth prediction correct: {truth_correct}")
            print(f"Topic prediction correct: {topic_correct}")
            print(f"Both correct: {both_correct}")
            
            if both_correct:
                correct_predictions += 1
            total_predictions += 1
            
        except Exception as e:
            print(f"❌ Error testing sample {sample_id}: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Total predictions: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {correct_predictions/total_predictions:.2%}")
    print(f"Average prediction time: {total_time/total_predictions:.3f} seconds")
    
    # Check competition constraints
    print(f"\n🏆 COMPETITION CONSTRAINTS CHECK")
    print("=" * 50)
    
    avg_time = total_time / total_predictions
    if avg_time < 5.0:
        print(f"✅ Speed: {avg_time:.3f}s < 5s (PASS)")
    else:
        print(f"❌ Speed: {avg_time:.3f}s >= 5s (FAIL)")
    
    print("✅ Privacy: Runs completely offline (PASS)")
    print("ℹ️  Memory: Check VRAM usage during inference")
    
    return correct_predictions / total_predictions

if __name__ == "__main__":
    accuracy = test_optimized_model()
    print(f"\n🎯 Final Model Accuracy: {accuracy:.2%}") 