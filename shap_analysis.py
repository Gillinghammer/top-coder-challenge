"""
SHAP Analysis Script

This script uses the SHAP library to explain the predictions of our model
on specific high-error cases, providing deep insight into which features
are driving the prediction for individual trips.
"""
import pandas as pd
import numpy as np
import pickle
import sys
import json
import shap
import matplotlib.pyplot as plt
from feature_engineering import create_all_features

def load_data_and_model():
    """Loads the public cases, the model, and identifies high-error cases."""
    # Load public cases
    with open('public_cases.json', 'r') as f:
        cases = json.load(f)
    
    data = []
    for i, case in enumerate(cases):
        row = case['input'].copy()
        row['expected_output'] = case['expected_output']
        row['case_id'] = i
        data.append(row)
    df = pd.DataFrame(data)
    
    # Load the trained model
    try:
        with open('reimbursement_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        model = model_data['model']
        feature_names = model_data['feature_names']
    except FileNotFoundError:
        print("❌ Error: reimbursement_model.pkl not found. Run build_model.py first.", file=sys.stderr)
        sys.exit(1)
        
    return df, model, feature_names

def main():
    """Main SHAP analysis function."""
    print("🔬 Starting SHAP Analysis on the Best Model...")
    
    df, model, feature_names = load_data_and_model()
    
    # Create features and make predictions
    df_features_in = df.drop(columns=['expected_output', 'case_id'])
    features = create_all_features(df_features_in)
    
    # Align columns to match model's training order
    features = features[feature_names]
    
    predictions = np.round(model.predict(features), 2)
    df['predicted'] = predictions
    df['error'] = df['predicted'] - df['expected_output']
    
    # --- Select Cases for Analysis ---
    worst_under_pred = df.loc[df['error'].idxmin()]
    worst_over_pred = df.loc[df['error'].idxmax()]
    
    cases_to_analyze = {
        "Worst Under-Prediction": worst_under_pred.name,
        "Worst Over-Prediction": worst_over_pred.name,
    }
    
    print("\nCases selected for analysis:")
    for name, idx in cases_to_analyze.items():
        print(f"  - {name} (Case ID: {df.loc[idx, 'case_id']}): Error = ${df.loc[idx, 'error']:.2f}")

    # --- SHAP Analysis ---
    explainer = shap.Explainer(model, features)
    
    print("\nCalculating SHAP values for the dataset...")
    shap_values = explainer(features)
    print("✅ SHAP values calculated.")

    # Generate and save force plots for our selected cases
    for name, idx in cases_to_analyze.items():
        print(f"\nGenerating plot for: {name}...")
        
        plt.figure()
        shap.force_plot(
            explainer.expected_value, 
            shap_values.values[idx,:], 
            features.iloc[idx,:],
            matplotlib=True,
            show=False,
            text_rotation=15
        )
        
        filename = f"shap_force_plot_{name.replace(' ', '_')}.png"
        plt.title(f"{name} (Case ID: {df.loc[idx, 'case_id']}) | Expected: ${df.loc[idx, 'expected_output']:.2f}, Predicted: ${df.loc[idx, 'predicted']:.2f}")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  -> Plot saved to {filename}")

    # Generate and save a summary plot
    print("\nGenerating SHAP summary plot...")
    plt.figure()
    shap.summary_plot(shap_values, features, show=False)
    plt.title("SHAP Feature Importance Summary")
    plt.savefig("shap_summary_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> Summary plot saved to shap_summary_plot.png")

if __name__ == "__main__":
    main() 