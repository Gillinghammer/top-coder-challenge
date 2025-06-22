#!/usr/bin/env python3
import sys
sys.path.insert(0, 'venv/lib/python3.13/site-packages')
"""
Production prediction script for the legacy reimbursement system.
Takes 3 command line arguments and outputs a reimbursement amount.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from feature_engineering import create_all_features

# Global model variable to load only once
_model = None
_feature_names = None

def load_model():
    """Load the single, final model from the root directory."""
    global _model, _feature_names
    if _model is None:
        try:
            with open('reimbursement_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            _model = model_data['model']
            _feature_names = model_data['feature_names']
        except Exception as e:
            print(f"Error loading model: {e}", file=sys.stderr)
            sys.exit(1)
    return _model, _feature_names

def predict_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    """Predicts reimbursement using the single best model."""
    model, feature_names = load_model()
    
    input_df = pd.DataFrame([{'trip_duration_days': trip_duration_days, 'miles_traveled': miles_traveled, 'total_receipts_amount': total_receipts_amount}])
    
    # create_all_features no longer needs a params argument
    features = create_all_features(input_df)
    
    features = features[feature_names]
    
    prediction = model.predict(features)[0]
    return float(prediction)

def main():
    """Main entry point to be called by run.sh"""
    if len(sys.argv) != 4:
        print("Usage: python predict.py <days> <miles> <receipts>", file=sys.stderr)
        sys.exit(1)
    
    try:
        days = int(sys.argv[1])
        miles = float(sys.argv[2])
        receipts = float(sys.argv[3])
        
        prediction = predict_reimbursement(days, miles, receipts)
        print(f"{prediction:.2f}")
        
    except ValueError as e:
        print(f"Error parsing arguments: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 