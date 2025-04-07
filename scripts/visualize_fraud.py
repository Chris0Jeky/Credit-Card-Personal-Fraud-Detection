# visualize_fraud.py
"""
Generates visualizations from transaction data, ML results, evaluation metrics,
and integrated scores to help identify patterns and assess performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import argparse
import json
import os
from datetime import datetime

# Import configuration
try:
    import config
except ModuleNotFoundError:
    print("Error: config.py not found. Make sure it's in the project root.")
    sys.exit(1)

# --- Configuration ---
OUTPUT_DIR = config.VISUALIZATIONS_DIR # From config
DEFAULT_FORMAT = config.DEFAULT_VIZ_FORMAT # From config

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate fraud detection visualizations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Input file paths passed from runner.py
    parser.add_argument("--transactions", type=str, help="Path to the raw transactions CSV file")
    parser.add_argument("--flagged", type=str, help="Path to the flagged transactions CSV file")
    parser.add_argument("--pca", type=str, help="Path to the PCA model results file")
    parser.add_argument("--simulated", type=str, help="Path to the simulated features (RF) model results file")
    parser.add_argument("--integrated", type=str, help="Path to the integrated assessment file")
    parser.add_argument("--evaluation", type=str, help="Path to the evaluation report JSON file")

    parser.add_argument(
        "--output", type=str, default=str(OUTPUT_DIR),
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--format", type=str, default=DEFAULT_FORMAT,
        choices=["png", "svg", "pdf", "jpg"],
        help="Output file format"
    )
    return parser.parse_args()

def load_data_files(args):
    """Load all available data files passed as arguments."""
    data = {}
    files_to_load = {
        'transactions': args.transactions,
        'flagged': args.flagged,
        'pca': args.pca,
        'simulated': args.simulated, # RF results
        'integrated': args.integrated,
        'evaluation': args.evaluation # JSON file
    }

    for key, filepath in files_to_load.items():
        if filepath and Path(filepath).exists():
            try:
                if Path(filepath).suffix == '.csv':
                    data[key] = pd.read_csv(filepath)
                    print(f"Loaded {key} data: {len(data[key])} rows.")
                elif Path(filepath).suffix == '.json':
                    with open(filepath, 'r') as f:
                        data[key] = json.load(f)
                    print(f"Loaded {key} data (JSON).")
                else:
                    print(f"Warning: Skipping unknown file type for {key}: {filepath}")
            except Exception as e:
                print(f"Error loading {key} data from {filepath}: {e}")
        elif filepath:
            print(f"Warning: {key} file not found at {filepath}")

    return data

def setup_visualizations(output_dir, style="whitegrid"):
    """Set up the visualization environment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style(style)
    sns.set_context("talk")
    plt.rcParams["figure.figsize"] = (12, 8) # Default size
    return output_dir

# --- Visualization Functions (Keep existing ones like amount, rules, time, distance) ---
# (Include visualize_transaction_amounts, visualize_rule_violations,
#  visualize_time_patterns, visualize_distance_patterns from previous version,
#  adapting them to check if data['key'] exists before using it)
# ... previous visualization functions go here ...
# Make sure they use `data.get('key')` or check `if 'key' in data:`

# --- Updated/New Visualization Functions ---

def visualize_ml_results(data, output_dir, format="png"):
    """Create visualizations from ML model results (PCA & RF)."""
    pca_df = data.get('pca')
    sim_df = data.get('simulated') # RF results

    if pca_df is None and sim_df is None:
        print("Skipping ML visualizations: No PCA or RF data loaded.")
        return

    # PCA anomaly detection results
    if pca_df is not None and 'anomaly_score' in pca_df.columns and 'anomaly' in pca_df.columns:
        print("Creating PCA/IF model visualizations...")
        # Anomaly score distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(pca_df['anomaly_score'], bins=30, kde=True)
        # Anomaly threshold in IF is implicitly defined by contamination, score=0 is rough boundary
        plt.axvline(x=0, color='red', linestyle='--', label='Approx. Threshold')
        plt.title('PCA/IF Anomaly Score Distribution')
        plt.xlabel('Anomaly Score (Lower = More Anomalous)')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"pca_anomaly_score_distribution.{format}")
        plt.close()

        # Anomaly score vs Amount (if available)
        if 'amt' in pca_df.columns:
            plt.figure(figsize=(12, 8))
            sns.scatterplot(x='anomaly_score', y='amt',
                            hue='anomaly', data=pca_df,
                            palette={0: 'blue', 1: 'red'}, alpha=0.7)
            plt.title('Transaction Amount vs PCA/IF Anomaly Score')
            plt.xlabel('Anomaly Score')
            plt.ylabel('Amount ($)')
            plt.grid(True, alpha=0.3)
            plt.legend(title='Predicted Anomaly')
            plt.tight_layout()
            plt.savefig(output_dir / f"pca_score_vs_amount.{format}")
            plt.close()

    # Random Forest model results
    if sim_df is not None and 'fraud_probability' in sim_df.columns and 'predicted_fraud' in sim_df.columns:
        print("Creating Random Forest model visualizations...")
        # Fraud probability distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(sim_df['fraud_probability'], bins=30, kde=True)
        plt.axvline(x=config.RF_PREDICTION_THRESHOLD, color='red', linestyle='--',
                    label=f'Threshold ({config.RF_PREDICTION_THRESHOLD})')
        plt.title('RF Fraud Probability Distribution')
        plt.xlabel('Predicted Fraud Probability')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"rf_fraud_probability_distribution.{format}")
        plt.close()

        # Fraud probability vs Amount (if available)
        if 'amt' in sim_df.columns:
            plt.figure(figsize=(12, 8))
            sns.scatterplot(x='fraud_probability', y='amt',
                            hue='predicted_fraud', data=sim_df,
                            palette={0: 'blue', 1: 'red'}, alpha=0.7)
            plt.title('Transaction Amount vs RF Fraud Probability')
            plt.xlabel('Fraud Probability')
            plt.ylabel('Amount ($)')
            plt.grid(True, alpha=0.3)
            plt.legend(title='Predicted Fraud')
            plt.tight_layout()
            plt.savefig(output_dir / f"rf_probability_vs_amount.{format}")
            plt.close()

    # Compare models if both are available and mergeable
    if pca_df is not None and sim_df is not None and 'trans_num' in pca_df.columns and 'trans_num' in sim_df.columns:
        print("Creating model comparison visualizations...")
        try:
            merged_df = pd.merge(
                pca_df[['trans_num', 'anomaly', 'anomaly_score']],
                sim_df[['trans_num', 'predicted_fraud', 'fraud_probability']],
                on='trans_num',
                how='inner'
            )

            if not merged_df.empty:
                # Model agreement heatmap
                agreement = pd.crosstab(merged_df['anomaly'], merged_df['predicted_fraud'])
                plt.figure(figsize=(8, 6))
                sns.heatmap(agreement, annot=True, fmt='d', cmap='Blues', cbar=True)
                plt.title('Model Prediction Agreement (Counts)')
                plt.xlabel('Random Forest Prediction (1=Fraud)')
                plt.ylabel('PCA/IF Prediction (1=Anomaly)')
                plt.tight_layout()
                plt.savefig(output_dir / f"model_agreement_counts.{format}")
                plt.close()

                # Scatter plot comparing scores/probabilities
                merged_df['models_disagree'] = merged_df['anomaly'] != merged_df['predicted_fraud']
                plt.figure(figsize=(12, 8))
                sns.scatterplot(
                    x='anomaly_score',
                    y='fraud_probability',
                    hue='models_disagree', # Highlight disagreements
                    data=merged_df,
                    palette={False: 'blue', True: 'red'}, alpha=0.8
                )
                plt.axhline(y=config.RF_PREDICTION_THRESHOLD, color='grey', linestyle='--')
                plt.axvline(x=0, color='grey', linestyle='--') # Approx IF threshold
                plt.title('PCA/IF Anomaly Score vs RF Fraud Probability')
                plt.xlabel('PCA/IF Anomaly Score (Lower = More Anomalous)')
                plt.ylabel('RF Fraud Probability')
                plt.legend(title='Models Disagree')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / f"model_score_comparison.{format}")
                plt.close()
        except Exception as e:
            print(f"Warning: Could not create model comparison plots: {e}")

    print("Finished ML model visualizations.")


def visualize_evaluation_results(data, output_dir, format="png"):
    """Create visualizations for performance evaluation metrics."""
    eval_data = data.get('evaluation')
    if not eval_data:
        print("Skipping evaluation visualizations: No evaluation data loaded.")
        return

    print("Creating evaluation visualizations...")
    metrics_to_plot = ['precision', 'recall', 'f1_score', 'auprc']
    model_names = list(eval_data.keys())
    plot_data = {metric: [] for metric in metrics_to_plot}

    for model in model_names:
        for metric in metrics_to_plot:
            # Handle cases where AUPRC might be None (e.g., rules)
             value = eval_data[model].get(metric)
             plot_data[metric].append(value if value is not None else 0)


    df_plot = pd.DataFrame(plot_data, index=model_names)

    # Bar chart comparing key metrics
    plt.figure(figsize=(14, 8))
    df_plot.plot(kind='bar', rot=0)
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xlabel('Model / Method')
    plt.legend(title='Metric')
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(output_dir / f"performance_comparison.{format}")
    plt.close()

    # Confusion Matrices (Example for RF if available)
    if 'random_forest' in eval_data and 'confusion_matrix' in eval_data['random_forest']:
         cm_data = eval_data['random_forest']['confusion_matrix']
         cm_array = np.array([[cm_data['tn'], cm_data['fp']], [cm_data['fn'], cm_data['tp']]])
         plt.figure(figsize=(8, 6))
         sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                     xticklabels=['Predicted Normal', 'Predicted Fraud'],
                     yticklabels=['Actual Normal', 'Actual Fraud'])
         plt.title('Confusion Matrix - Random Forest')
         plt.tight_layout()
         plt.savefig(output_dir / f"confusion_matrix_rf.{format}")
         plt.close()

    # PR Curves are plotted directly in evaluate_performance.py if --plot flag is used

    print("Finished evaluation visualizations.")


def visualize_integrated_results(data, output_dir, format="png"):
    """Create visualizations for the integrated risk score."""
    integrated_df = data.get('integrated')
    if integrated_df is None:
        print("Skipping integrated results visualizations: No integrated data loaded.")
        return

    if 'risk_score' not in integrated_df.columns or 'risk_level' not in integrated_df.columns:
        print("Warning: 'risk_score' or 'risk_level' missing from integrated data.")
        return

    print("Creating integrated risk score visualizations...")

    # Risk score distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(integrated_df['risk_score'], bins=30, kde=True)
    plt.axvline(x=config.RISK_THRESHOLD_MEDIUM, color='orange', linestyle='--', label=f'Medium Risk ({config.RISK_THRESHOLD_MEDIUM})')
    plt.axvline(x=config.RISK_THRESHOLD_HIGH, color='red', linestyle='--', label=f'High Risk ({config.RISK_THRESHOLD_HIGH})')
    plt.title('Integrated Risk Score Distribution')
    plt.xlabel('Risk Score')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"integrated_risk_score_distribution.{format}")
    plt.close()

    # Risk Level distribution (Pie Chart)
    level_counts = integrated_df['risk_level'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(level_counts, labels=level_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ffcc99','#ff9999']) # Blue, Orange, Red
    plt.title('Distribution of Risk Levels')
    plt.savefig(output_dir / f"integrated_risk_level_pie.{format}")
    plt.close()

    # Risk score vs Amount (if available)
    if 'amt' in integrated_df.columns:
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='risk_score', y='amt', hue='risk_level',
                        data=integrated_df, palette={'Low': 'blue', 'Medium': 'orange', 'High': 'red'},
                        alpha=0.7)
        plt.title('Transaction Amount vs Integrated Risk Score')
        plt.xlabel('Integrated Risk Score')
        plt.ylabel('Amount ($)')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Risk Level')
        plt.tight_layout()
        plt.savefig(output_dir / f"integrated_score_vs_amount.{format}")
        plt.close()

    print("Finished integrated risk score visualizations.")


def create_summary_report(output_dir, format="png"):
    """Create a summary HTML report linking all generated visualizations."""
    report_path = output_dir.parent / "visualization_report.html" # Place report outside viz dir

    # Collect list of all generated visualizations inside the viz directory
    viz_files = sorted(list(output_dir.glob(f"*.{format}")))

    if not viz_files:
        print("No visualization files found to create report.")
        return

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fraud Detection Visualization Report</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; background-color: #f4f4f4; }}
            h1 {{ color: #333; border-bottom: 2px solid #3498db; padding-bottom: 10px;}}
            h2 {{ color: #555; margin-top: 30px; }}
            .viz-container {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
            .viz-item {{ background-color: #fff; border: 1px solid #ddd; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); overflow: hidden;}}
            .viz-item img {{ max-width: 100%; height: auto; display: block; }}
            .viz-item p {{ text-align: center; padding: 10px; margin: 0; background: #eee; font-weight: bold; color: #333;}}
            a {{ color: #3498db; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <h1>Fraud Detection Visualization Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Visualizations are stored in the <a href="{output_dir.name}/">'{output_dir.name}/'</a> directory.</p>

        <h2>Visualizations</h2>
        <div class="viz-container">
    """

    # Add each visualization to the report
    for viz_file in viz_files:
        file_name = viz_file.name
        relative_path = f"{output_dir.name}/{file_name}" # Path relative to HTML file
        friendly_name = ' '.join(file_name.replace(f'.{format}', '').split('_')).title()

        html_content += f"""
            <div class="viz-item">
                <a href="{relative_path}" target="_blank">
                    <img src="{relative_path}" alt="{friendly_name}">
                    <p>{friendly_name}</p>
                </a>
            </div>
        """

    html_content += """
        </div>
    </body>
    </html>
    """

    # Write HTML report
    try:
        with open(report_path, 'w') as f:
            f.write(html_content)
        print(f"\nCreated HTML visualization report at: {report_path}")
    except Exception as e:
        print(f"Error writing HTML report: {e}")


def main():
    """Main execution function."""
    args = parse_arguments()

    # Load data files
    data = load_data_files(args)

    if not data:
        print("No data files were loaded. Cannot generate visualizations.")
        return 1

    # Set up visualization environment
    output_dir = setup_visualizations(args.output)

    # --- Generate Visualizations ---
    # Call *all* visualization functions. They will internally check if needed data exists.
    print("\n--- Generating Visualizations ---")
    visualize_transaction_amounts(data, output_dir, format=args.format)
    visualize_rule_violations(data, output_dir, format=args.format)
    visualize_time_patterns(data, output_dir, format=args.format)
    visualize_distance_patterns(data, output_dir, format=args.format)
    visualize_ml_results(data, output_dir, format=args.format)
    visualize_evaluation_results(data, output_dir, format=args.format)
    visualize_integrated_results(data, output_dir, format=args.format)

    # Create summary HTML report
    create_summary_report(output_dir, format=args.format)

    print(f"\n--- Visualization Generation Complete ---")
    print(f"Output saved to: {output_dir}")
    return 0

if __name__ == "__main__":
    # Make sure working directory is project root if running directly
    # This helps config.py find paths correctly
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    print(f"Changed working directory to: {os.getcwd()}")
    sys.exit(main())