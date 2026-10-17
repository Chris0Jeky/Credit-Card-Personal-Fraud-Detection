# scripts/integrate_results.py
"""
Integrates results from rules and ML models into a combined risk score.
"""

import pandas as pd
import sys
from pathlib import Path
import numpy as np
from scripts.prediction_inputs import load_prediction_inputs, write_provenance

# Import configuration
try:
    import config
except ModuleNotFoundError:
    print("Error: config.py not found. Make sure it's in the project root.")
    sys.exit(1)


def calculate_risk_score(row):
    """Calculate a weighted risk score based on different signals."""
    score = 0

    # Rule component (Binary: 1 if triggered, 0 otherwise)
    rule_signal = float(row.get('rule_triggered', 0))
    score += config.WEIGHT_RULE * rule_signal

    # PCA component (Use negative score, as lower score = more anomalous)
    # Normalize score roughly? Assume scores range roughly -0.1 to 0.02 based on viz?
    # Simple clamping and scaling: scale(-score) from [0, 0.1] to [0, 1]
    pca_raw_score = row.get('anomaly_score', 0)
    pca_signal = max(0, min(1, -pca_raw_score / 0.1))  # Crude normalization
    score += config.WEIGHT_PCA_SCORE * pca_signal

    # RF component (Use probability directly)
    rf_signal = float(row.get('fraud_probability', 0))
    score += config.WEIGHT_RF_PROB * rf_signal

    # Ensure score is between 0 and 1 (due to weights summing to 1 and signals >= 0)
    return max(0, min(1, score))


def categorize_risk(score):
    """Categorize risk score into Low, Medium, High."""
    if score >= config.RISK_THRESHOLD_HIGH:
        return "High"
    elif score >= config.RISK_THRESHOLD_MEDIUM:
        return "Medium"
    else:
        return "Low"


def main():
    print("\n--- Starting Results Integration ---")

    # --- Load Data ---
    print("Loading rule flags and ML predictions...")
    try:
        flags_df, pca_df, rf_df, source_digest = load_prediction_inputs()

        # --- Merge Data ---
        print("Merging results...")
        # Start with base transaction info + rules
        integrated_df = flags_df[
            ['trans_num', 'trans_date_trans_time', 'amt', 'merchant', 'category', 'is_fraud', 'rule_flags',
             'rule_triggered']].copy()

        # Merge PCA results
        integrated_df = integrated_df.merge(
            pca_df[['trans_num', 'anomaly', 'anomaly_score']], on='trans_num', how='left', validate='one_to_one'
        )

        # Merge RF results
        integrated_df = integrated_df.merge(
            rf_df[['trans_num', 'predicted_fraud', 'fraud_probability']], on='trans_num', how='left', validate='one_to_one'
        )

        print(f"   Successfully merged data for {len(integrated_df)} transactions.")

    except FileNotFoundError as e:
        print(f"Error: Required input file not found: {e.filename}", file=sys.stderr)
        print("Ensure simulation, rules, and ML predictions have run.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading or merging data: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Calculate Integrated Score and Level ---
    print("Calculating integrated risk score...")
    integrated_df['risk_score'] = integrated_df.apply(calculate_risk_score, axis=1)
    integrated_df['risk_level'] = integrated_df['risk_score'].apply(categorize_risk)

    print("   Risk Score Statistics:")
    print(integrated_df['risk_score'].describe())
    print("\n   Risk Level Distribution:")
    print(integrated_df['risk_level'].value_counts(normalize=True) * 100)

    # --- Save Results ---
    print(f"\nSaving integrated assessment to {config.INTEGRATED_ASSESSMENT_FILE}...")
    try:
        integrated_df.to_csv(config.INTEGRATED_ASSESSMENT_FILE, index=False, float_format='%.4f')
        write_provenance(config.INTEGRATED_ASSESSMENT_FILE, source_digest)
        print("   Integrated assessment saved.")
    except Exception as e:
        print(f"Error saving integrated assessment: {e}", file=sys.stderr)
        return 1

    print("--- Results Integration Complete ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
