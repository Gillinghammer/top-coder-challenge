import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle
import sys
import json
import optuna
from feature_engineering import create_all_features

# Global variable to cache data loading
DF_CACHE = None

def load_data():
    """Loads and caches the public cases data."""
    global DF_CACHE
    if DF_CACHE is None:
        with open('public_cases.json', 'r') as f:
            cases = json.load(f)
        df = pd.DataFrame([case['input'] for case in cases])
        df['expected_output'] = [case['expected_output'] for case in cases]
        DF_CACHE = df
    return DF_CACHE

def objective(trial):
    """
    The Optuna objective function. Optuna will try to minimize the value returned by this function.
    """
    df = load_data()

    # 1. Define the search space for our feature parameters
    params = {
        'high_mileage_threshold': trial.suggest_int('high_mileage_threshold', 800, 1200),
        'high_mileage_bonus_multiplier': trial.suggest_float('high_mileage_bonus_multiplier', 0.1, 0.5),
        'low_receipt_threshold': trial.suggest_int('low_receipt_threshold', 40, 60),
        'low_receipt_penalty_multiplier': trial.suggest_float('low_receipt_penalty_multiplier', 1.0, 3.0),
        'efficiency_min': trial.suggest_int('efficiency_min', 150, 200),
        'efficiency_max': trial.suggest_int('efficiency_max', 210, 250),
        'modest_spend_5_day_max': trial.suggest_int('modest_spend_5_day_max', 80, 120),
        'long_trip_days_threshold': trial.suggest_int('long_trip_days_threshold', 7, 9),
        'long_trip_spend_threshold': trial.suggest_int('long_trip_spend_threshold', 80, 100),
        'efficiency_sweet_spot': trial.suggest_int('efficiency_sweet_spot', 180, 220),
        'efficiency_width': trial.suggest_int('efficiency_width', 40, 60),
        'short_trip_spend_cap': trial.suggest_int('short_trip_spend_cap', 60, 90),
        'medium_trip_spend_cap': trial.suggest_int('medium_trip_spend_cap', 110, 130),
        'long_trip_spend_cap_2': trial.suggest_int('long_trip_spend_cap_2', 80, 100),
    }

    # 2. Create features using the suggested parameters
    X = create_all_features(df, params)
    y = df['expected_output']

    # 3. Train and evaluate the model (using a simple, fast validation split)
    # We use a fixed random_state to ensure fair comparison between trials
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

    model_params = {
        'objective': 'reg:absoluteerror',
        'eval_metric': 'mae',
        'n_estimators': 500, # Fewer estimators for faster optimization runs
        'learning_rate': 0.05,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
    }
    
    model = xgb.XGBRegressor(**model_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, predictions)

    return mae

def main():
    """Main optimization function."""
    print("🤖 Starting feature parameter optimization with Optuna...")

    # We tell Optuna to minimize the MAE returned by the objective function
    study = optuna.create_study(direction='minimize')
    
    # Run the optimization for a set number of trials
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    print("\n" + "="*50)
    print("🏆 Optimization Finished!")
    print(f"  Best MAE: ${study.best_value:.2f}")
    print("  Best parameters found:")
    for key, value in study.best_params.items():
        print(f"    - {key}: {value}")
    print("="*50)
    
    print("\n💡 Recommendation: Update these values in feature_engineering.py and run build_model.py to create the final model.")

if __name__ == "__main__":
    main() 