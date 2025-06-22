#!/usr/bin/env python3
"""
Exploratory Data Analysis for the Legacy Reimbursement System
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data():
    """Load and prepare the data"""
    with open('public_cases.json', 'r') as f:
        cases = json.load(f)
    
    # Convert to DataFrame
    data = []
    for case in cases:
        row = case['input'].copy()
        row['reimbursement'] = case['expected_output']
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Compute derived features
    df['miles_per_day'] = df['miles_traveled'] / df['trip_duration_days']
    df['spend_per_day'] = df['total_receipts_amount'] / df['trip_duration_days']
    df['log_miles'] = np.log1p(df['miles_traveled'])
    df['log_receipts'] = np.log1p(df['total_receipts_amount'])
    df['is_5_days'] = (df['trip_duration_days'] == 5).astype(int)
    df['is_short_trip'] = (df['trip_duration_days'] <= 3).astype(int)
    df['is_long_trip'] = (df['trip_duration_days'] >= 8).astype(int)
    df['low_receipts'] = (df['total_receipts_amount'] < 50).astype(int)
    df['high_receipts'] = (df['total_receipts_amount'] > 800).astype(int)
    
    return df

def basic_statistics(df):
    """Print basic statistics"""
    print("=== BASIC STATISTICS ===")
    print(f"Total cases: {len(df)}")
    print(f"\nTrip Duration:")
    print(df['trip_duration_days'].describe())
    print(f"\nMiles Traveled:")
    print(df['miles_traveled'].describe())
    print(f"\nReceipts Amount:")
    print(df['total_receipts_amount'].describe())
    print(f"\nReimbursement:")
    print(df['reimbursement'].describe())
    
    print(f"\n=== TRIP DURATION DISTRIBUTION ===")
    print(df['trip_duration_days'].value_counts().sort_index())

def analyze_per_diem_pattern(df):
    """Analyze per-diem patterns"""
    print("\n=== PER-DIEM ANALYSIS ===")
    
    # Group by trip duration and compute average reimbursement per day
    per_day_stats = df.groupby('trip_duration_days').agg({
        'reimbursement': ['mean', 'std', 'count'],
        'miles_traveled': 'mean',
        'total_receipts_amount': 'mean'
    }).round(2)
    
    per_day_stats.columns = ['avg_reimbursement', 'std_reimbursement', 'count', 'avg_miles', 'avg_receipts']
    per_day_stats['reimbursement_per_day'] = (per_day_stats['avg_reimbursement'] / per_day_stats.index).round(2)
    
    print(per_day_stats)
    
    # Check for 5-day bonus
    print(f"\n=== 5-DAY BONUS CHECK ===")
    five_day_avg = df[df['trip_duration_days'] == 5]['reimbursement'].mean() / 5
    four_day_avg = df[df['trip_duration_days'] == 4]['reimbursement'].mean() / 4
    six_day_avg = df[df['trip_duration_days'] == 6]['reimbursement'].mean() / 6
    
    print(f"4-day avg per day: ${four_day_avg:.2f}")
    print(f"5-day avg per day: ${five_day_avg:.2f}")
    print(f"6-day avg per day: ${six_day_avg:.2f}")
    print(f"5-day bonus vs 4-day: ${five_day_avg - four_day_avg:.2f}")
    print(f"5-day bonus vs 6-day: ${five_day_avg - six_day_avg:.2f}")

def analyze_mileage_patterns(df):
    """Analyze mileage reimbursement patterns"""
    print("\n=== MILEAGE ANALYSIS ===")
    
    # Create mileage bins
    df['mileage_bin'] = pd.cut(df['miles_traveled'], 
                              bins=[0, 100, 200, 300, 500, 800, 1200], 
                              labels=['0-100', '100-200', '200-300', '300-500', '500-800', '800+'])
    
    mileage_stats = df.groupby('mileage_bin').agg({
        'reimbursement': ['mean', 'count'],
        'miles_traveled': 'mean',
        'trip_duration_days': 'mean'
    }).round(2)
    
    print("Mileage bin analysis:")
    print(mileage_stats)
    
    # Calculate effective rate per mile
    df['effective_rate_per_mile'] = df['reimbursement'] / df['miles_traveled']
    df['effective_rate_per_mile'] = df['effective_rate_per_mile'].replace([np.inf, -np.inf], np.nan)
    
    print(f"\n=== EFFECTIVE MILEAGE RATES ===")
    rate_by_mileage = df.groupby('mileage_bin')['effective_rate_per_mile'].agg(['mean', 'std', 'count']).round(3)
    print(rate_by_mileage)

def analyze_receipts_patterns(df):
    """Analyze receipt reimbursement patterns"""
    print("\n=== RECEIPTS ANALYSIS ===")
    
    # Create receipt bins
    df['receipt_bin'] = pd.cut(df['total_receipts_amount'], 
                              bins=[0, 50, 200, 500, 1000, 2000], 
                              labels=['0-50', '50-200', '200-500', '500-1000', '1000+'])
    
    receipt_stats = df.groupby('receipt_bin').agg({
        'reimbursement': ['mean', 'count'],
        'total_receipts_amount': 'mean',
        'trip_duration_days': 'mean'
    }).round(2)
    
    print("Receipt bin analysis:")
    print(receipt_stats)
    
    # Calculate effective reimbursement rate for receipts
    df['receipt_reimbursement_rate'] = df['reimbursement'] / df['total_receipts_amount']
    df['receipt_reimbursement_rate'] = df['receipt_reimbursement_rate'].replace([np.inf, -np.inf], np.nan)
    
    print(f"\n=== RECEIPT REIMBURSEMENT RATES ===")
    rate_by_receipts = df.groupby('receipt_bin')['receipt_reimbursement_rate'].agg(['mean', 'std', 'count']).round(3)
    print(rate_by_receipts)

def analyze_efficiency_bonus(df):
    """Analyze efficiency bonus (miles per day)"""
    print("\n=== EFFICIENCY BONUS ANALYSIS ===")
    
    # Create efficiency bins
    df['efficiency_bin'] = pd.cut(df['miles_per_day'], 
                                 bins=[0, 50, 100, 150, 200, 250, 1000], 
                                 labels=['0-50', '50-100', '100-150', '150-200', '200-250', '250+'])
    
    efficiency_stats = df.groupby('efficiency_bin').agg({
        'reimbursement': ['mean', 'count'],
        'miles_per_day': 'mean',
        'trip_duration_days': 'mean'
    }).round(2)
    
    print("Efficiency bin analysis:")
    print(efficiency_stats)
    
    # Look specifically at the 180-220 range mentioned in interviews
    sweet_spot = df[(df['miles_per_day'] >= 180) & (df['miles_per_day'] <= 220)]
    other_trips = df[(df['miles_per_day'] < 180) | (df['miles_per_day'] > 220)]
    
    print(f"\n=== SWEET SPOT ANALYSIS (180-220 miles/day) ===")
    print(f"Sweet spot trips: {len(sweet_spot)}")
    print(f"Sweet spot avg reimbursement: ${sweet_spot['reimbursement'].mean():.2f}")
    print(f"Other trips avg reimbursement: ${other_trips['reimbursement'].mean():.2f}")
    print(f"Difference: ${sweet_spot['reimbursement'].mean() - other_trips['reimbursement'].mean():.2f}")

def check_for_duplicates(df):
    """Check for identical inputs with different outputs (randomness)"""
    print("\n=== DUPLICATE INPUT ANALYSIS ===")
    
    # Group by input parameters
    input_cols = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount']
    duplicates = df.groupby(input_cols).agg({
        'reimbursement': ['count', 'nunique', 'std']
    }).round(2)
    
    duplicates.columns = ['count', 'unique_outputs', 'std_reimbursement']
    duplicates = duplicates[duplicates['count'] > 1]
    
    if len(duplicates) > 0:
        print(f"Found {len(duplicates)} sets of duplicate inputs:")
        print(duplicates.head(10))
        
        # Show examples of duplicates with different outputs
        varied_outputs = duplicates[duplicates['unique_outputs'] > 1]
        if len(varied_outputs) > 0:
            print(f"\nFound {len(varied_outputs)} input combinations with varying outputs!")
            print(varied_outputs.head())
        else:
            print("\nNo identical inputs with different outputs found - system appears deterministic")
    else:
        print("No duplicate inputs found")

def create_visualizations(df):
    """Create key visualizations"""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Reimbursement vs Trip Duration
    axes[0,0].scatter(df['trip_duration_days'], df['reimbursement'], alpha=0.6)
    axes[0,0].set_xlabel('Trip Duration (days)')
    axes[0,0].set_ylabel('Reimbursement ($)')
    axes[0,0].set_title('Reimbursement vs Trip Duration')
    
    # 2. Reimbursement vs Miles
    axes[0,1].scatter(df['miles_traveled'], df['reimbursement'], alpha=0.6)
    axes[0,1].set_xlabel('Miles Traveled')
    axes[0,1].set_ylabel('Reimbursement ($)')
    axes[0,1].set_title('Reimbursement vs Miles Traveled')
    
    # 3. Reimbursement vs Receipts
    axes[0,2].scatter(df['total_receipts_amount'], df['reimbursement'], alpha=0.6)
    axes[0,2].set_xlabel('Total Receipts ($)')
    axes[0,2].set_ylabel('Reimbursement ($)')
    axes[0,2].set_title('Reimbursement vs Receipts')
    
    # 4. Miles per day distribution
    axes[1,0].hist(df['miles_per_day'], bins=50, alpha=0.7)
    axes[1,0].axvline(180, color='red', linestyle='--', label='Sweet spot start')
    axes[1,0].axvline(220, color='red', linestyle='--', label='Sweet spot end')
    axes[1,0].set_xlabel('Miles per Day')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Miles per Day Distribution')
    axes[1,0].legend()
    
    # 5. Reimbursement per day by trip length
    per_day_data = df.groupby('trip_duration_days')['reimbursement'].apply(lambda x: x / df.loc[x.index, 'trip_duration_days']).reset_index()
    per_day_summary = df.groupby('trip_duration_days').apply(lambda x: (x['reimbursement'] / x['trip_duration_days']).mean()).reset_index()
    per_day_summary.columns = ['trip_duration_days', 'reimbursement_per_day']
    
    axes[1,1].bar(per_day_summary['trip_duration_days'], per_day_summary['reimbursement_per_day'])
    axes[1,1].axhline(100, color='red', linestyle='--', label='$100 baseline')
    axes[1,1].set_xlabel('Trip Duration (days)')
    axes[1,1].set_ylabel('Avg Reimbursement per Day ($)')
    axes[1,1].set_title('Average Reimbursement per Day by Trip Length')
    axes[1,1].legend()
    
    # 6. Heatmap of Miles vs Receipts
    # Create bins for heatmap
    miles_bins = pd.cut(df['miles_traveled'], bins=10)
    receipt_bins = pd.cut(df['total_receipts_amount'], bins=10)
    heatmap_data = df.groupby([miles_bins, receipt_bins])['reimbursement'].mean().unstack()
    
    sns.heatmap(heatmap_data, ax=axes[1,2], cmap='viridis', cbar_kws={'label': 'Avg Reimbursement ($)'})
    axes[1,2].set_xlabel('Receipt Bins')
    axes[1,2].set_ylabel('Miles Bins')
    axes[1,2].set_title('Reimbursement Heatmap: Miles vs Receipts')
    
    plt.tight_layout()
    plt.savefig('eda_visualizations.png', dpi=150, bbox_inches='tight')
    plt.show()

def naive_baseline_analysis(df):
    """Fit a naive baseline and analyze residuals"""
    print("\n=== NAIVE BASELINE ANALYSIS ===")
    
    # Simple baseline: $100 per day + $0.58 per mile + 80% of receipts
    df['baseline_prediction'] = (100 * df['trip_duration_days'] + 
                                0.58 * df['miles_traveled'] + 
                                0.8 * df['total_receipts_amount'])
    
    df['residual'] = df['reimbursement'] - df['baseline_prediction']
    
    print(f"Baseline MAE: ${abs(df['residual']).mean():.2f}")
    print(f"Baseline RMSE: ${np.sqrt((df['residual']**2).mean()):.2f}")
    
    # Analyze residuals by different factors
    print(f"\n=== RESIDUAL ANALYSIS ===")
    print("Residuals by trip duration:")
    residual_by_days = df.groupby('trip_duration_days')['residual'].agg(['mean', 'std', 'count']).round(2)
    print(residual_by_days)
    
    print("\nResiduals by efficiency:")
    residual_by_efficiency = df.groupby('efficiency_bin')['residual'].agg(['mean', 'std', 'count']).round(2)
    print(residual_by_efficiency)
    
    return df

def main():
    """Main analysis function"""
    print("Loading data...")
    df = load_data()
    
    # Run all analyses
    basic_statistics(df)
    analyze_per_diem_pattern(df)
    analyze_mileage_patterns(df)
    analyze_receipts_patterns(df)
    analyze_efficiency_bonus(df)
    check_for_duplicates(df)
    df = naive_baseline_analysis(df)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(df)
    
    # Save processed data
    df.to_csv('processed_data.csv', index=False)
    print(f"\nProcessed data saved to processed_data.csv")
    print(f"Visualizations saved to eda_visualizations.png")
    
    return df

if __name__ == "__main__":
    df = main() 