#!/usr/bin/env python3
"""
Local test harness for rapid iteration - mirrors eval.sh functionality
"""

import json
import subprocess
import sys
from pathlib import Path

def load_test_cases(filename="public_cases.json"):
    """Load test cases from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def run_prediction(trip_days, miles, receipts):
    """Run the prediction using run.sh"""
    try:
        result = subprocess.run(
            ['./run.sh', str(trip_days), str(miles), str(receipts)],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            print(f"Error running prediction: {result.stderr}")
            return None
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Exception running prediction: {e}")
        return None

def evaluate_predictions(test_cases, max_cases=None):
    """Evaluate predictions against test cases"""
    if max_cases:
        test_cases = test_cases[:max_cases]
    
    exact_matches = 0
    close_matches = 0
    total_error = 0.0
    errors = []
    
    for i, case in enumerate(test_cases):
        input_data = case['input']
        expected = case['expected_output']
        
        predicted = run_prediction(
            input_data['trip_duration_days'],
            input_data['miles_traveled'],
            input_data['total_receipts_amount']
        )
        
        if predicted is None:
            print(f"Failed to get prediction for case {i}")
            continue
            
        error = abs(predicted - expected)
        errors.append(error)
        total_error += error
        
        if error <= 0.01:
            exact_matches += 1
        if error <= 1.00:
            close_matches += 1
            
        if i < 10 or error > 10:  # Show first 10 cases and large errors
            print(f"Case {i}: Days={input_data['trip_duration_days']}, "
                  f"Miles={input_data['miles_traveled']}, "
                  f"Receipts=${input_data['total_receipts_amount']:.2f} -> "
                  f"Expected=${expected:.2f}, Predicted=${predicted:.2f}, "
                  f"Error=${error:.2f}")
    
    num_cases = len(test_cases)
    avg_error = total_error / num_cases if num_cases > 0 else 0
    
    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Total cases: {num_cases}")
    print(f"Exact matches (±$0.01): {exact_matches} ({exact_matches/num_cases*100:.1f}%)")
    print(f"Close matches (±$1.00): {close_matches} ({close_matches/num_cases*100:.1f}%)")
    print(f"Average error: ${avg_error:.2f}")
    print(f"Max error: ${max(errors):.2f}" if errors else "No errors calculated")
    
    return {
        'exact_matches': exact_matches,
        'close_matches': close_matches,
        'avg_error': avg_error,
        'max_error': max(errors) if errors else 0
    }

def main():
    """Main evaluation function"""
    # Check if run.sh exists
    if not Path('run.sh').exists():
        print("Error: run.sh not found. Make sure you're in the right directory.")
        sys.exit(1)
    
    # Load test cases
    print("Loading test cases...")
    test_cases = load_test_cases()
    print(f"Loaded {len(test_cases)} test cases")
    
    # Run evaluation on a subset first for quick feedback
    print("\n=== Quick test (first 50 cases) ===")
    evaluate_predictions(test_cases, max_cases=50)
    
    # Ask if user wants to run full evaluation
    response = input("\nRun full evaluation on all cases? (y/n): ")
    if response.lower().startswith('y'):
        print("\n=== Full evaluation ===")
        evaluate_predictions(test_cases)

if __name__ == "__main__":
    main() 