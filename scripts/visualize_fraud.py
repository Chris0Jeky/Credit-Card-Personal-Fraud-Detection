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

# --- Paste these functions into visualize_fraud.py ---
# --- Place them after setup_visualizations() and before visualize_ml_results() ---

def visualize_transaction_amounts(data, output_dir, format="png"):
    """Create visualizations related to transaction amounts."""
    df = data.get('transactions')
    if df is None or 'amt' not in df.columns:
        print("Skipping amount visualizations: Transaction data or 'amt' column missing.")
        return

    print("Creating transaction amount visualizations...")
    # Transaction amount distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df['amt'], bins=30, kde=True)
    plt.title('Transaction Amount Distribution')
    plt.xlabel('Amount ($)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(output_dir / f"transaction_amount_distribution.{format}")
    plt.close()

    # Transaction amount boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['amt'])
    plt.title('Transaction Amount Boxplot')
    plt.xlabel('Amount ($)')
    plt.tight_layout()
    plt.savefig(output_dir / f"transaction_amount_boxplot.{format}")
    plt.close()

    # Log-scaled amount distribution (better for skewed data)
    # Check for non-positive values before log scaling
    if (df['amt'] > 0).any():
        plt.figure(figsize=(12, 6))
        sns.histplot(np.log1p(df.loc[df['amt'] > 0, 'amt']), bins=30, kde=True) # Use log1p and filter > 0
        plt.title('Transaction Amount Distribution (Log-scaled)')
        plt.xlabel('Log(Amount + 1)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(output_dir / f"transaction_amount_log_distribution.{format}")
        plt.close()
    else:
        print("   Skipping log amount distribution: No positive amounts found.")

    print("Finished transaction amount visualizations.")

def visualize_rule_violations(data, output_dir, format="png"):
    """Create visualizations related to rule violations."""
    df = data.get('flagged')
    if df is None or 'rule_flags' not in df.columns:
        print("Skipping rule violation visualizations: Flagged data or 'rule_flags' column missing.")
        return

    print("Creating rule violation visualizations...")

    # Extract rule violation types
    def extract_violations(flags_str):
        if not isinstance(flags_str, str) or not flags_str:
            return []
        violations = []
        # Improved regex to capture only the type after RULE_VIOLATION:
        rule_pattern = r'RULE_VIOLATION:([A-Z_]+)'
        matches = re.findall(rule_pattern, flags_str)
        violations.extend([match.strip() for match in matches])
        return violations

    all_violations = []
    for flags in df['rule_flags'].dropna():
        all_violations.extend(extract_violations(flags))

    if not all_violations:
         print("   No specific RULE_VIOLATIONs found in flags to visualize.")
         return

    violation_counts = pd.Series(all_violations).value_counts()

    # Rule violation bar chart
    plt.figure(figsize=(14, 8))
    violation_counts.plot(kind='bar', color='crimson')
    plt.title('Rule Violation Counts')
    plt.xlabel('Rule Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / f"rule_violation_counts.{format}")
    plt.close()

    # Amount analysis by rule type (if amount data available)
    if 'amt' in df.columns:
        rule_amt_data = []
        for _, row in df.iterrows():
            violations = extract_violations(row['rule_flags'])
            if violations and pd.notna(row['amt']):
                for violation in violations:
                    rule_amt_data.append({
                        'rule_type': violation,
                        'amount': row['amt']
                    })

        if rule_amt_data:
            rule_amt_df = pd.DataFrame(rule_amt_data)
            # Box plot of amount by rule type
            plt.figure(figsize=(16, 10))
            sns.boxplot(x='rule_type', y='amount', data=rule_amt_df)
            plt.title('Transaction Amount by Rule Violation Type')
            plt.xlabel('Rule Type')
            plt.ylabel('Amount ($)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_dir / f"amount_by_rule_type.{format}")
            plt.close()

    print("Finished rule violation visualizations.")


def visualize_time_patterns(data, output_dir, format="png"):
    """Create visualizations related to time patterns."""
    df = data.get('transactions')
    if df is None or 'trans_date_trans_time' not in df.columns:
        print("Skipping time pattern visualizations: Transaction data or timestamp column missing.")
        return

    print("Creating time pattern visualizations...")
    df = df.copy() # Avoid SettingWithCopyWarning

    # Convert timestamp to datetime robustly
    try:
        df['trans_datetime'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
        df.dropna(subset=['trans_datetime'], inplace=True) # Drop rows where conversion failed
        if df.empty:
            print("   No valid timestamps found after conversion.")
            return
        df['hour'] = df['trans_datetime'].dt.hour
        df['day_of_week'] = df['trans_datetime'].dt.dayofweek # Monday=0, Sunday=6
        df['date'] = df['trans_datetime'].dt.date
    except Exception as e:
        print(f"Error processing timestamps: {e}")
        return

    # Transactions by hour of day
    plt.figure(figsize=(12, 6))
    sns.countplot(x='hour', data=df, color='steelblue', order=range(24)) # Ensure all hours shown
    plt.title('Transactions by Hour of Day')
    plt.xlabel('Hour (0-23)')
    plt.ylabel('Number of Transactions')
    plt.xticks(range(0, 24, 2))
    plt.tight_layout()
    plt.savefig(output_dir / f"transactions_by_hour.{format}")
    plt.close()

    # Transactions by day of week
    plt.figure(figsize=(10, 6))
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    sns.countplot(x='day_of_week', data=df, color='lightseagreen', order=range(7))
    plt.title('Transactions by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Transactions')
    plt.xticks(range(7), day_names)
    plt.tight_layout()
    plt.savefig(output_dir / f"transactions_by_day.{format}")
    plt.close()

    # Transaction volume over time (if multiple dates exist)
    if df['date'].nunique() > 1:
        daily_counts = df.groupby('date').size()
        plt.figure(figsize=(14, 6))
        daily_counts.plot(kind='line', marker='.', linestyle='-')
        plt.title('Daily Transaction Volume')
        plt.xlabel('Date')
        plt.ylabel('Number of Transactions')
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        # Format x-axis for readability if range is large
        if (df['date'].max() - df['date'].min()).days > 30:
             plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10)) # Limit number of date ticks
             plt.gcf().autofmt_xdate() # Auto-rotate dates
        plt.savefig(output_dir / f"daily_transaction_volume.{format}")
        plt.close()
    else:
        print("   Skipping daily volume plot: Only one transaction date found.")

    # Time patterns of flagged transactions (requires flagged data)
    flagged_df = data.get('flagged')
    if flagged_df is not None and 'rule_triggered' in flagged_df.columns and 'trans_date_trans_time' in flagged_df.columns:
        flagged_df = flagged_df.copy()
        try:
            flagged_df['trans_datetime'] = pd.to_datetime(flagged_df['trans_date_trans_time'], errors='coerce')
            flagged_df.dropna(subset=['trans_datetime'], inplace=True)
            if not flagged_df.empty:
                flagged_df['hour'] = flagged_df['trans_datetime'].dt.hour

                # Ensure rule_triggered is numeric (0 or 1)
                flagged_df['rule_triggered'] = pd.to_numeric(flagged_df['rule_triggered'], errors='coerce').fillna(0).astype(int)

                # Flagged transactions by hour (Rate)
                hourly_flags = flagged_df.groupby('hour')['rule_triggered'].agg(['sum', 'count'])
                hourly_flags['flag_rate'] = (hourly_flags['sum'] / hourly_flags['count']) * 100

                plt.figure(figsize=(14, 7))
                hourly_flags['flag_rate'].plot(kind='bar', color='darkorange')
                plt.title('Percentage of Rule-Triggered Transactions by Hour')
                plt.xlabel('Hour of Day')
                plt.ylabel('Percentage Triggered (%)')
                plt.ylim(0, 100)
                plt.grid(axis='y', linestyle='--')
                plt.xticks(range(0, 24, 2))
                plt.tight_layout()
                plt.savefig(output_dir / f"flagged_by_hour_rate.{format}")
                plt.close()
        except Exception as e:
            print(f"   Warning: Could not plot flagged time patterns: {e}")

    print("Finished time pattern visualizations.")


def visualize_distance_patterns(data, output_dir, format="png"):
    """Create visualizations related to transaction distances from home."""
    # Requires merged/consistent data with coords and potentially flags
    df = data.get('flagged', data.get('transactions')) # Prefer flagged, fallback to raw
    if df is None:
        print("Skipping distance visualizations: No transaction data found.")
        return

    # Check if we have location data needed for distance calc
    required_cols = ['merch_lat', 'merch_long', 'lat', 'long']
    if not all(col in df.columns for col in required_cols):
        print("Skipping distance visualizations: Coordinate columns missing.")
        return

    print("Creating distance pattern visualizations...")
    df = df.copy()

    # Calculate distances from home
    # Use the same robust function from check_rules.py
    from geopy.distance import geodesic # Need geopy here too
    import re # Need re here too

    def calculate_distance_viz(row):
        # Avoid NameError by defining the helper inside or importing globally
        if pd.isna(row.get('merch_lat')) or pd.isna(row.get('merch_long')) or pd.isna(row.get('lat')) or pd.isna(row.get('long')):
            return np.nan
        try:
            return geodesic((float(row['lat']), float(row['long'])), (float(row['merch_lat']), float(row['merch_long']))).km
        except ValueError:
            return np.nan # Handle potential errors during conversion or calc

    df['distance_km'] = df.apply(calculate_distance_viz, axis=1)

    # Drop rows where distance couldn't be calculated
    df_dist = df.dropna(subset=['distance_km']).copy()
    if df_dist.empty:
        print("   No valid distances calculated to visualize.")
        return

    # Distance distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df_dist['distance_km'], bins=30, kde=True)
    plt.title('Transaction Distance from Home Distribution')
    plt.xlabel('Distance (km)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(output_dir / f"distance_distribution.{format}")
    plt.close()

    # Distance vs Amount scatter plot
    if 'amt' in df_dist.columns:
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='distance_km', y='amt', data=df_dist, alpha=0.7)
        plt.title('Transaction Amount vs Distance from Home')
        plt.xlabel('Distance (km)')
        plt.ylabel('Amount ($)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"amount_vs_distance.{format}")
        plt.close()

    # Distance patterns by rule trigger status (if available)
    if 'rule_triggered' in df_dist.columns:
        df_dist['rule_triggered'] = pd.to_numeric(df_dist['rule_triggered'], errors='coerce').fillna(0).astype(int)
        flagged = df_dist[df_dist['rule_triggered'] == 1]['distance_km']
        unflagged = df_dist[df_dist['rule_triggered'] == 0]['distance_km']

        plt.figure(figsize=(14, 7))
        sns.histplot(unflagged, color="blue", label="Not Triggered", kde=True, alpha=0.5, bins=20)
        sns.histplot(flagged, color="red", label="Rule Triggered", kde=True, alpha=0.5, bins=20)
        plt.title('Transaction Distance by Rule Trigger Status')
        plt.xlabel('Distance (km)')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"distance_by_rule_status.{format}")
        plt.close()

    print("Finished distance pattern visualizations.")

# --- End of missing functions ---

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