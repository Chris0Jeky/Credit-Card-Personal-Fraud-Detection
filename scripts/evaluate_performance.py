# scripts/evaluate_performance.py
"""
Evaluates the performance of different fraud detection components:
- Rule-based system
- PCA/Isolation Forest model
- Random Forest model
- Integrated risk score (optional, if integration script is run first)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    precision_recall_curve,
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from scripts.prediction_inputs import load_prediction_inputs, validate_rows, verify_provenance

# Import configuration
try:
    import config
except ModuleNotFoundError:
    print("Error: config.py not found. Make sure it's in the project root.")
    sys.exit(1)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate Fraud Detection Performance")
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate and save PR curve plots."
    )
    return parser.parse_args()

def calculate_metrics(y_true, y_pred, y_score=None):
    """Calculate standard classification metrics."""
    metrics = {}
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics['confusion_matrix'] = {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)

    if y_score is not None:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        metrics['auprc'] = auc(recall, precision)
        # Store data for plotting PR curve
        metrics['pr_curve_data'] = {'precision': precision.tolist(), 'recall': recall.tolist()}
    else:
        metrics['auprc'] = None
        metrics['pr_curve_data'] = None

    # Round metrics for readability
    for key in ['accuracy', 'precision', 'recall', 'f1_score', 'auprc']:
        if metrics[key] is not None:
            metrics[key] = round(metrics[key], 4)

    return metrics

def plot_pr_curve(pr_data, model_name, output_dir):
    """Plots and saves a Precision-Recall curve."""
    if not pr_data:
        return
    plt.figure(figsize=(8, 6))
    plt.plot(pr_data['recall'], pr_data['precision'], marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.grid(True)
    plt.tight_layout()
    plot_file = output_dir / f"pr_curve_{model_name.lower().replace(' ', '_')}.{config.DEFAULT_VIZ_FORMAT}"
    plt.savefig(plot_file)
    plt.close()
    print(f"   Saved PR curve plot: {plot_file.name}")

def main():
    args = parse_arguments()
    print("\n--- Starting Performance Evaluation ---")

    evaluation_results = {}

    # --- Load Data ---
    print("Loading prediction files...")
    try:
        flags_df, pca_df, rf_df, source_digest = load_prediction_inputs()
        # Optional: Load integrated results if they exist
        integrated_df = None
        if config.INTEGRATED_ASSESSMENT_FILE.exists():
            integrated_df = pd.read_csv(config.INTEGRATED_ASSESSMENT_FILE, dtype={'trans_num': str})
            validate_rows(integrated_df, 'Integrated', set(flags_df['trans_num']), ('risk_score',))
            if 'risk_level' not in integrated_df.columns or not integrated_df['risk_level'].isin(['Low', 'Medium', 'High']).all():
                raise ValueError('Integrated: missing or invalid risk_level.')
            verify_provenance(config.INTEGRATED_ASSESSMENT_FILE, source_digest)
            print("   Loaded integrated assessment results.")

        # --- Merge Data ---
        print("Merging results for evaluation...")

        # Start with the core flagged data (contains ground truth)
        eval_df = flags_df[['trans_num', 'is_fraud', 'rule_triggered']].copy()

        eval_df = eval_df.merge(
            pca_df[['trans_num', 'anomaly', 'anomaly_score']],
            on='trans_num', how='left', validate='one_to_one'
        )
        eval_df = eval_df.merge(
            rf_df[['trans_num', 'predicted_fraud', 'fraud_probability']],
            on='trans_num', how='left', validate='one_to_one'
        )

        # Left merge Integrated results if available
        if integrated_df is not None:
            eval_df = eval_df.merge(
                integrated_df[['trans_num', 'risk_score', 'risk_level']],
                on='trans_num', how='left', validate='one_to_one'
            )
            print(f"   Merged Integrated results. Shape after merge: {eval_df.shape}")
        else:
            print("   Integrated results unavailable. Skipping integrated evaluation.")

        print(f"   Successfully loaded and merged data for {len(eval_df)} transactions.")

    except FileNotFoundError as e:
        print(f"Error: Required prediction file not found: {e.filename}", file=sys.stderr)
        print("Please ensure simulation, rule checks, and ML predictions have been run.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading or merging data: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Evaluate Components ---
    # Crucial Check: Ensure y_true has valid values AFTER merge
    eval_df.dropna(subset=['is_fraud'], inplace=True) # Drop rows where ground truth is missing
    y_true = eval_df['is_fraud'].astype(int) # Ensure integer type

    if eval_df.empty or y_true.empty:
        print("Error: No valid transactions remaining after merging and cleaning for evaluation.", file=sys.stderr)
        sys.exit(1)
    if not np.all(np.isin(y_true.unique(), [0, 1])):
        print(f"Error: Unexpected values found in y_true (is_fraud): {y_true.unique()}. Expected 0 or 1.", file=sys.stderr)
        sys.exit(1)

    print(f"   Final evaluation data ready for {len(eval_df)} transactions.")

    # 1. Rule-Based System
    print("\nEvaluating Rule-Based System...")
    if 'rule_triggered' not in eval_df.columns:
        print("Error: 'rule_triggered' column not found in flagged data. Cannot evaluate rules.", file=sys.stderr)
    else:
        y_pred_rules = eval_df['rule_triggered'].astype(int)  # Ensure integer type
        # Rules don't have a score, so AUPRC is not applicable directly
        evaluation_results['rules'] = calculate_metrics(y_true, y_pred_rules)
        print(f"   F1 Score: {evaluation_results['rules']['f1_score']}")

    # 2. PCA/Isolation Forest Model
    print("\nEvaluating PCA/Isolation Forest Model...")
    if 'anomaly' in eval_df.columns and 'anomaly_score' in eval_df.columns:
        y_pred_pca = eval_df['anomaly'].astype(int)  # Ensure integer type
        # Use negative anomaly score for PR curve (lower score = more anomalous)
        y_score_pca = -eval_df['anomaly_score']
        evaluation_results['pca_isolation_forest'] = calculate_metrics(y_true, y_pred_pca, y_score_pca)
        print(f"   F1 Score: {evaluation_results['pca_isolation_forest']['f1_score']}")
        print(f"   AUPRC: {evaluation_results['pca_isolation_forest']['auprc']}")
        if args.plot and evaluation_results['pca_isolation_forest'].get('pr_curve_data'):
            plot_pr_curve(evaluation_results['pca_isolation_forest']['pr_curve_data'],
                        "PCA Isolation Forest", config.VISUALIZATIONS_DIR)
    else:
        print("   Skipping PCA evaluation: Required columns ('anomaly', 'anomaly_score') not found after merge.")
        evaluation_results['pca_isolation_forest'] = {'error': 'Missing columns'}

    # 3. Random Forest Model
    print("\nEvaluating Random Forest Model...")
    if 'predicted_fraud' in eval_df.columns and 'fraud_probability' in eval_df.columns:
        y_pred_rf = eval_df['predicted_fraud'].astype(int)  # Ensure integer type
        y_score_rf = eval_df['fraud_probability']
        evaluation_results['random_forest'] = calculate_metrics(y_true, y_pred_rf, y_score_rf)
        print(f"   F1 Score: {evaluation_results['random_forest']['f1_score']}")
        print(f"   AUPRC: {evaluation_results['random_forest']['auprc']}")
        if args.plot and evaluation_results['random_forest'].get('pr_curve_data'):
            plot_pr_curve(evaluation_results['random_forest']['pr_curve_data'],
                        "Random Forest", config.VISUALIZATIONS_DIR)
    else:
        print("   Skipping RF evaluation: Required columns ('predicted_fraud', 'fraud_probability') not found after merge.")
        evaluation_results['random_forest'] = {'error': 'Missing columns'}

    # 4. Integrated Risk Score (if available)
    if 'risk_score' in eval_df.columns and 'risk_level' in eval_df.columns:
        print("\nEvaluating Integrated Risk Score...")
        try:
            # Define prediction based on 'High' risk level for standard metrics
            y_pred_integrated = (eval_df['risk_level'] == 'High').astype(int)
            y_score_integrated = eval_df['risk_score']
            evaluation_results['integrated_score'] = calculate_metrics(y_true, y_pred_integrated, y_score_integrated)
            print(f"   F1 Score (High Risk Threshold): {evaluation_results['integrated_score']['f1_score']}")
            print(f"   AUPRC: {evaluation_results['integrated_score']['auprc']}")
            if args.plot and evaluation_results['integrated_score'].get('pr_curve_data'):
                plot_pr_curve(evaluation_results['integrated_score']['pr_curve_data'],
                            "Integrated Score", config.VISUALIZATIONS_DIR)
        except Exception as e:
            print(f"   Error evaluating integrated score: {e}")
            evaluation_results['integrated_score'] = {'error': str(e)}

    # --- Save Results ---
    print(f"\nSaving evaluation report to {config.EVALUATION_REPORT_FILE}...")
    try:
        # Remove bulky PR curve data before saving JSON
        for model in evaluation_results:
             if evaluation_results[model] and 'pr_curve_data' in evaluation_results[model]:
                 del evaluation_results[model]['pr_curve_data']

        with open(config.EVALUATION_REPORT_FILE, 'w') as f:
            json.dump(evaluation_results, f, indent=4)
        print("   Evaluation report saved.")
    except Exception as e:
        print(f"Error saving evaluation report: {e}", file=sys.stderr)

    print("--- Performance Evaluation Complete ---")
    return 0

if __name__ == "__main__":
    sys.exit(main())
