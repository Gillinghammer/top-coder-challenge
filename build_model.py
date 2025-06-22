#!/usr/bin/env python3
"""
Model building script based on EDA findings
"""

import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle
import sys
from feature_engineering import create_all_features

def main():
    """Main training pipeline for the single, best model."""
    print("🚀 Training Final Model (Champion Version)...")

    with open('public_cases.json', 'r') as f:
        cases = json.load(f)
    df = pd.DataFrame([case['input'] for case in cases])
    df['expected_output'] = [case['expected_output'] for case in cases]

    X = create_all_features(df)
    y = df['expected_output']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    params = {
        'objective': 'reg:absoluteerror', 'eval_metric': 'mae',
        'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 1000,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42,
        'early_stopping_rounds': 50
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    val_pred = model.predict(X_val)
    print(f"\nValidation MAE for final model: ${mean_absolute_error(y_val, val_pred):.2f}")

    model_path = "reimbursement_model.pkl"
    model_data = {'model': model, 'feature_names': list(X.columns)}
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"✅ Final champion model saved to {model_path}")

if __name__ == "__main__":
    main() 