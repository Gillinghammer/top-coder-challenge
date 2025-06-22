"""
Shared feature engineering logic for the reimbursement model.
"""
import numpy as np
import pandas as pd

def create_all_features(df_in):
    """
    Creates all engineered features for the model. (Reverted to Best - Iteration 6)
    """
    features = df_in[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']].copy()

    epsilon = 1e-6
    features['miles_per_day'] = features['miles_traveled'] / (features['trip_duration_days'] + epsilon)
    features['spend_per_day'] = features['total_receipts_amount'] / (features['trip_duration_days'] + epsilon)

    # Dampened High-Mileage Bonus (from Iteration 6)
    features['bonus_high_mileage'] = np.where(
        features['miles_traveled'] > 1000,
        (features['miles_traveled'] - 1000) * 0.25 * np.log1p(features['total_receipts_amount']),
        0
    )

    # Refined Low-Receipts Penalty (from Iteration 4)
    features['penalty_low_receipts_amount'] = np.where(
        (features['total_receipts_amount'] > 0) & (features['total_receipts_amount'] < 50),
        (50 - features['total_receipts_amount']) * 1.5,
        0
    )
    
    # --- Base Features (from Iteration 2) ---
    is_5_days = features['trip_duration_days'] == 5
    is_efficient_5_day = (features['miles_per_day'] >= 180) & (features['miles_per_day'] <= 220)
    is_modest_spending_5_day = features['spend_per_day'] <= 100
    features['bonus_5_day_sweet_spot'] = (is_5_days & is_efficient_5_day & is_modest_spending_5_day).astype(int)

    is_long_trip_penalty = features['trip_duration_days'] >= 8
    is_high_spending_long_trip = features['spend_per_day'] > 90
    features['penalty_vacation'] = (is_long_trip_penalty & is_high_spending_long_trip).astype(int)
    
    efficiency_sweet_spot = 200 
    efficiency_width = 50
    features['efficiency_score'] = np.exp(-((features['miles_per_day'] - efficiency_sweet_spot)**2) / (2 * efficiency_width**2))

    cents = np.round(features['total_receipts_amount'] * 100) % 100
    features['bonus_rounding_bug'] = ((cents == 49) | (cents == 99)).astype(int)
    
    short_trip_overspend = (features['trip_duration_days'] <= 3) & (features['spend_per_day'] > 75)
    medium_trip_overspend = (features['trip_duration_days'].between(4, 7, inclusive='both')) & (features['spend_per_day'] > 120)
    long_trip_overspend = (features['trip_duration_days'] >= 8) & (features['spend_per_day'] > 90)
    features['penalty_overspending'] = (short_trip_overspend | medium_trip_overspend | long_trip_overspend).astype(int)

    # Original useful features
    features['log_miles'] = np.log1p(features['miles_traveled'])
    features['log_receipts'] = np.log1p(features['total_receipts_amount'])
    features['days_squared'] = features['trip_duration_days'] ** 2
    features['miles_squared'] = features['miles_traveled'] ** 2
    features['receipts_squared'] = features['total_receipts_amount'] ** 2
    features['days_x_miles'] = features['trip_duration_days'] * features['miles_traveled']
    features['days_x_receipts'] = features['trip_duration_days'] * features['total_receipts_amount']
    features['miles_x_receipts'] = features['miles_traveled'] * features['total_receipts_amount']

    for days in range(1, 15):
        features[f'is_{days}_days'] = (features['trip_duration_days'] == days).astype(int)
    
    return features 