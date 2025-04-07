# scripts/evaluate_performance.py
"""
Evaluates the performance of different fraud detection components:
- Rule-based system
- PCA/Isolation Forest model
- Random Forest model
- Integrated risk score (optional, if integration script is run first)
"""

import pandas as pd
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
        flags_df = pd.read_csv(config.FLAGGED_TRANSACTIONS_FILE)
        pca_df = pd.read_csv(config.PCA_PREDICTIONS_FILE)
        rf_df = pd.read_csv(config.RF_PREDICTIONS_FILE)
        # Optional: Load integrated results if they exist
        integrated_df = None
        if config.INTEGRATED_ASSESSMENT_FILE.exists():
            integrated_df = pd.read_csv(config.INTEGRATED_ASSESSMENT_FILE)
            print("   Loaded integrated assessment results.")

        # Basic check: Ensure ground truth 'is_fraud' exists
        if 'is_fraud' not in flags_df.columns:
            raise ValueError("'is_fraud' column missing in flagged transactions.")

        # Merge data - Use transaction number as key
        if 'trans_num' not in flags_df.columns or 'trans_num' not in pca_df.columns or 'trans_num' not in rf_df.columns:
             print("Warning: 'trans_num' missing in one or more files. Merging by index.")
             # Potential issue if rows aren't perfectly aligned across files
             eval_df = pd.concat([
                flags_df[['is_fraud', 'rule_triggered']].reset_index(drop=True), # Ensure 'rule_triggered' exists
                pca_df[['anomaly', 'anomaly_score']].reset_index(drop=True),
                rf_df[['predicted_fraud', 'fraud_probability']].reset_index(drop=True)
             ], axis=1)
        else:
             eval_df = flags_df[['trans_num', 'is_fraud', 'rule_triggered']].merge(
                pca_df[['trans_num', 'anomaly', 'anomaly_score']], on='trans_num', how='inner'
             ).merge(
                rf_df[['trans_num', 'predicted_fraud', 'fraud_probability']], on='trans_num', how='inner'
             )
             if integrated_df is not None and 'trans_num' in integrated_df.columns:
                 eval_df = eval_df.merge(
                     integrated_df[['trans_num', 'risk_score', 'risk_level']], on='trans_num', how='inner'
                 )

        print(f"   Successfully loaded and merged data for {len(eval_df)} transactions.")

    except FileNotFoundError as e:
        print(f"Error: Required prediction file not found: {e.filename}", file=sys.stderr)
        print("Please ensure simulation, rule checks, and ML predictions have been run.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading or merging data: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Evaluate Components ---
    y_true = eval_df['is_fraud']

    # 1. Rule-Based System
    print("\nEvaluating Rule-Based System...")
    if 'rule_triggered' not in eval_df.columns:
        print("Error: 'rule_triggered' column not found in flagged data. Cannot evaluate rules.", file=sys.stderr)
    else:
        y_pred_rules = eval_df['rule_triggered']
        # Rules don't have a score, so AUPRC is not applicable directly
        evaluation_results['rules'] = calculate_metrics(y_true, y_pred_rules)
        print(f"   F1 Score: {evaluation_results['rules']['f1_score']}")

    # 2. PCA/Isolation Forest Model
    print("\nEvaluating PCA/Isolation Forest Model...")
    y_pred_pca = eval_df['anomaly']
    # Use negative anomaly score for PR curve (lower score = more anomalous)
    y_score_pca = -eval_df['anomaly_score']
    evaluation_results['pca_isolation_forest'] = calculate_metrics(y_true, y_pred_pca, y_score_pca)
    print(f"   F1 Score: {evaluation_results['pca_isolation_forest']['f1_score']}")
    print(f"   AUPRC: {evaluation_results['pca_isolation_forest']['auprc']}")
    if args.plot:
        plot_pr_curve(evaluation_results['pca_isolation_forest']['pr_curve_data'],
                      "PCA Isolation Forest", config.VISUALIZATIONS_DIR)

    # 3. Random Forest Model
    print("\nEvaluating Random Forest Model...")
    y_pred_rf = eval_df['predicted_fraud']
    y_score_rf = eval_df['fraud_probability']
    evaluation_results['random_forest'] = calculate_metrics(y_true, y_pred_rf, y_score_rf)
    print(f"   F1 Score: {evaluation_results['random_forest']['f1_score']}")
    print(f"   AUPRC: {evaluation_results['random_forest']['auprc']}")
    if args.plot:
         plot_pr_curve(evaluation_results['random_forest']['pr_curve_data'],
                       "Random Forest", config.VISUALIZATIONS_DIR)

    # 4. Integrated Risk Score (if available)
    if 'risk_score' in eval_df.columns and 'risk_level' in eval_df.columns:
        print("\nEvaluating Integrated Risk Score...")
        # Define prediction based on 'High' risk level for standard metrics
        y_pred_integrated = (eval_df['risk_level'] == 'High').astype(int)
        y_score_integrated = eval_df['risk_score']
        evaluation_results['integrated_score'] = calculate_metrics(y_true, y_pred_integrated, y_score_integrated)
        print(f"   F1 Score (High Risk Threshold): {evaluation_results['integrated_score']['f1_score']}")
        print(f"   AUPRC: {evaluation_results['integrated_score']['auprc']}")
        if args.plot:
            plot_pr_curve(evaluation_results['integrated_score']['pr_curve_data'],
                          "Integrated Score", config.VISUALIZATIONS_DIR)

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