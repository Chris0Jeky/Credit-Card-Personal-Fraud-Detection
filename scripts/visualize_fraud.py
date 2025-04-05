"""
Generates visualizations from transaction data to help identify fraud patterns.
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

# --- Configuration ---
DATA_DIR = Path(__file__).parent.parent / "data" / "simulation_output"
TRANSACTIONS_FILE = DATA_DIR / "simulated_account_transactions.csv"
FLAGGED_FILE = DATA_DIR / "simulated_transactions_with_flags.csv"
PCA_RESULTS_FILE = DATA_DIR / "pca_fraud_predictions.csv"
SIMULATED_RESULTS_FILE = DATA_DIR / "simulated_fraud_predictions.csv"
OUTPUT_DIR = DATA_DIR / "visualizations"

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate fraud detection visualizations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--transactions", 
        type=str,
        default=str(TRANSACTIONS_FILE),
        help="Path to the transactions CSV file"
    )
    parser.add_argument(
        "--flagged", 
        type=str,
        default=str(FLAGGED_FILE),
        help="Path to the flagged transactions CSV file"
    )
    parser.add_argument(
        "--pca", 
        type=str,
        default=str(PCA_RESULTS_FILE),
        help="Path to the PCA model results file"
    )
    parser.add_argument(
        "--simulated", 
        type=str,
        default=str(SIMULATED_RESULTS_FILE),
        help="Path to the simulated features model results file"
    )
    parser.add_argument(
        "--output", 
        type=str,
        default=str(OUTPUT_DIR),
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--format", 
        type=str,
        default="png",
        choices=["png", "svg", "pdf", "jpg"],
        help="Output file format"
    )
    return parser.parse_args()

def load_data_files(args):
    """Load all available data files."""
    data = {}
    
    try:
        if Path(args.transactions).exists():
            data['transactions'] = pd.read_csv(args.transactions)
            print(f"Loaded {len(data['transactions'])} transactions.")
        else:
            print(f"Warning: Transactions file not found at {args.transactions}")
    except Exception as e:
        print(f"Error loading transactions: {e}")
    
    try:
        if Path(args.flagged).exists():
            data['flagged'] = pd.read_csv(args.flagged)
            print(f"Loaded {len(data['flagged'])} flagged transactions.")
        else:
            print(f"Warning: Flagged transactions file not found at {args.flagged}")
    except Exception as e:
        print(f"Error loading flagged transactions: {e}")
    
    try:
        if Path(args.pca).exists():
            data['pca'] = pd.read_csv(args.pca)
            print(f"Loaded PCA model results for {len(data['pca'])} transactions.")
        else:
            print(f"Warning: PCA results file not found at {args.pca}")
    except Exception as e:
        print(f"Error loading PCA results: {e}")
    
    try:
        if Path(args.simulated).exists():
            data['simulated'] = pd.read_csv(args.simulated)
            print(f"Loaded simulated features model results for {len(data['simulated'])} transactions.")
        else:
            print(f"Warning: Simulated features results file not found at {args.simulated}")
    except Exception as e:
        print(f"Error loading simulated features results: {e}")
    
    return data

def setup_visualizations(output_dir, style="whitegrid"):
    """Set up the visualization environment."""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up seaborn style
    sns.set_style(style)
    sns.set_context("talk")
    
    # Set default figure size
    plt.rcParams["figure.figsize"] = (12, 8)
    
    return output_dir

def visualize_transaction_amounts(data, output_dir, format="png"):
    """Create visualizations related to transaction amounts."""
    if 'transactions' not in data:
        return
    
    df = data['transactions']
    
    if 'amt' not in df.columns:
        print("Warning: 'amt' column not found in transactions.")
        return
    
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
    plt.figure(figsize=(12, 6))
    sns.histplot(np.log1p(df['amt']), bins=30, kde=True)
    plt.title('Transaction Amount Distribution (Log-scaled)')
    plt.xlabel('Log(Amount + 1)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(output_dir / f"transaction_amount_log_distribution.{format}")
    plt.close()
    
    print("Created transaction amount visualizations.")

def visualize_rule_violations(data, output_dir, format="png"):
    """Create visualizations related to rule violations."""
    if 'flagged' not in data or 'rule_flags' not in data['flagged'].columns:
        return
    
    df = data['flagged']
    
    # Extract rule violation types
    def extract_violations(flags_str):
        if not isinstance(flags_str, str) or not flags_str:
            return []
        violations = []
        for flag in flags_str.split(';'):
            if 'RULE_VIOLATION:' in flag:
                rule_type = flag.split('RULE_VIOLATION:')[1].split('(')[0].strip()
                violations.append(rule_type)
        return violations
    
    all_violations = []
    for flags in df['rule_flags'].dropna():
        all_violations.extend(extract_violations(flags))
    
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
    
    # Add amount analysis by rule type if amount data is available
    if 'amt' in df.columns:
        # Create a new DataFrame to analyze amount by rule type
        rule_amt_data = []
        
        for _, row in df.iterrows():
            if isinstance(row['rule_flags'], str) and row['rule_flags']:
                violations = extract_violations(row['rule_flags'])
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
    
    print("Created rule violation visualizations.")

def visualize_time_patterns(data, output_dir, format="png"):
    """Create visualizations related to time patterns."""
    if 'transactions' not in data or 'trans_date_trans_time' not in data['transactions'].columns:
        return
    
    df = data['transactions'].copy()
    
    # Convert timestamp to datetime
    df['trans_datetime'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_datetime'].dt.hour
    df['day_of_week'] = df['trans_datetime'].dt.dayofweek
    df['date'] = df['trans_datetime'].dt.date
    
    # Transactions by hour of day
    plt.figure(figsize=(12, 6))
    sns.countplot(x='hour', data=df, color='steelblue')
    plt.title('Transactions by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Number of Transactions')
    plt.xticks(range(0, 24, 2))
    plt.tight_layout()
    plt.savefig(output_dir / f"transactions_by_hour.{format}")
    plt.close()
    
    # Transactions by day of week
    plt.figure(figsize=(10, 6))
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    ax = sns.countplot(x='day_of_week', data=df, color='lightseagreen')
    plt.title('Transactions by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Transactions')
    plt.xticks(range(7), day_names)
    plt.tight_layout()
    plt.savefig(output_dir / f"transactions_by_day.{format}")
    plt.close()
    
    # Transaction volume over time
    daily_counts = df.groupby('date').size()
    plt.figure(figsize=(14, 6))
    daily_counts.plot(kind='line', marker='o')
    plt.title('Daily Transaction Volume')
    plt.xlabel('Date')
    plt.ylabel('Number of Transactions')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"daily_transaction_volume.{format}")
    plt.close()
    
    # If we have flagged data, add time patterns of flagged transactions
    if 'flagged' in data and 'rule_flags' in data['flagged'].columns:
        flagged_df = data['flagged'].copy()
        
        if 'trans_date_trans_time' in flagged_df.columns:
            flagged_df['trans_datetime'] = pd.to_datetime(flagged_df['trans_date_trans_time'])
            flagged_df['hour'] = flagged_df['trans_datetime'].dt.hour
            flagged_df['has_flag'] = flagged_df['rule_flags'].notna() & (flagged_df['rule_flags'] != '')
            
            # Flagged transactions by hour
            plt.figure(figsize=(14, 7))
            hourly_flag_rate = flagged_df.groupby('hour')['has_flag'].mean() * 100
            hourly_flag_rate.plot(kind='bar', color='darkorange')
            plt.title('Percentage of Flagged Transactions by Hour')
            plt.xlabel('Hour of Day')
            plt.ylabel('Percentage of Transactions Flagged')
            plt.ylim(0, 100)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / f"flagged_by_hour.{format}")
            plt.close()
    
    print("Created time pattern visualizations.")

def visualize_distance_patterns(data, output_dir, format="png"):
    """Create visualizations related to transaction distances."""
    # We need both account information and transaction data
    if 'transactions' not in data or 'flagged' not in data:
        return
    
    trans_df = data['transactions'].copy()
    flagged_df = data['flagged'].copy()
    
    # Check if we have location data
    required_cols = ['merch_lat', 'merch_long', 'lat', 'long']
    if not all(col in trans_df.columns for col in required_cols):
        return
    
    # Calculate distances from home for each transaction
    def calculate_distance(row):
        from geopy.distance import geodesic
        
        if pd.isna(row['merch_lat']) or pd.isna(row['merch_long']) or pd.isna(row['lat']) or pd.isna(row['long']):
            return np.nan
        
        try:
            return geodesic(
                (row['lat'], row['long']),
                (row['merch_lat'], row['merch_long'])
            ).km
        except:
            return np.nan
    
    trans_df['distance_km'] = trans_df.apply(calculate_distance, axis=1)
    
    # Distance distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(trans_df['distance_km'].dropna(), bins=30, kde=True)
    plt.title('Transaction Distance from Home Distribution')
    plt.xlabel('Distance (km)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(output_dir / f"distance_distribution.{format}")
    plt.close()
    
    # Distance vs Amount scatter plot
    if 'amt' in trans_df.columns:
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='distance_km', y='amt', data=trans_df, alpha=0.7)
        plt.title('Transaction Amount vs Distance from Home')
        plt.xlabel('Distance (km)')
        plt.ylabel('Amount ($)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"amount_vs_distance.{format}")
        plt.close()
    
    # If we have flagged data, add distance patterns of flagged transactions
    if 'rule_flags' in flagged_df.columns:
        flagged_df['has_flag'] = flagged_df['rule_flags'].notna() & (flagged_df['rule_flags'] != '')
        flagged_df['distance_km'] = flagged_df.apply(calculate_distance, axis=1)
        
        # Split into flagged and unflagged
        flagged = flagged_df[flagged_df['has_flag']]['distance_km'].dropna()
        unflagged = flagged_df[~flagged_df['has_flag']]['distance_km'].dropna()
        
        # Distance distribution by flag status
        plt.figure(figsize=(14, 7))
        sns.histplot(unflagged, color="blue", label="Unflagged", kde=True, alpha=0.5)
        sns.histplot(flagged, color="red", label="Flagged", kde=True, alpha=0.5)
        plt.title('Transaction Distance by Flag Status')
        plt.xlabel('Distance (km)')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"distance_by_flag_status.{format}")
        plt.close()
    
    print("Created distance pattern visualizations.")

def visualize_ml_results(data, output_dir, format="png"):
    """Create visualizations from ML model results."""
    # Check if we have ML results from either model
    has_pca = 'pca' in data and 'anomaly_score' in data['pca'].columns
    has_simulated = 'simulated' in data and 'fraud_probability' in data['simulated'].columns
    
    if not (has_pca or has_simulated):
        return
    
    # PCA anomaly detection results
    if has_pca:
        pca_df = data['pca'].copy()
        
        # Anomaly score distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(pca_df['anomaly_score'], bins=30, kde=True)
        plt.axvline(x=0, color='red', linestyle='--', label='Threshold')
        plt.title('PCA Anomaly Score Distribution')
        plt.xlabel('Anomaly Score')
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
                            palette={0: 'blue', 1: 'red'})
            plt.title('Transaction Amount vs PCA Anomaly Score')
            plt.xlabel('Anomaly Score')
            plt.ylabel('Amount ($)')
            plt.grid(True, alpha=0.3)
            plt.legend(title='Anomaly Flag')
            plt.tight_layout()
            plt.savefig(output_dir / f"pca_score_vs_amount.{format}")
            plt.close()
    
    # Simulated features model results
    if has_simulated:
        sim_df = data['simulated'].copy()
        
        # Fraud probability distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(sim_df['fraud_probability'], bins=30, kde=True)
        plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold (0.5)')
        plt.title('Fraud Probability Distribution')
        plt.xlabel('Fraud Probability')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"fraud_probability_distribution.{format}")
        plt.close()
        
        # Fraud probability vs Amount (if available)
        if 'amt' in sim_df.columns:
            plt.figure(figsize=(12, 8))
            sns.scatterplot(x='fraud_probability', y='amt', 
                            hue='predicted_fraud', data=sim_df, 
                            palette={0: 'blue', 1: 'red'})
            plt.title('Transaction Amount vs Fraud Probability')
            plt.xlabel('Fraud Probability')
            plt.ylabel('Amount ($)')
            plt.grid(True, alpha=0.3)
            plt.legend(title='Predicted Fraud')
            plt.tight_layout()
            plt.savefig(output_dir / f"fraud_probability_vs_amount.{format}")
            plt.close()
    
    # Compare models if both are available
    if has_pca and has_simulated:
        # First, ensure we can match transactions between the two datasets
        if 'trans_num' in pca_df.columns and 'trans_num' in sim_df.columns:
            # Merge the results
            merged_df = pd.merge(
                pca_df[['trans_num', 'anomaly', 'anomaly_score']], 
                sim_df[['trans_num', 'predicted_fraud', 'fraud_probability']], 
                on='trans_num', 
                how='inner',
                suffixes=('_pca', '_sim')
            )
            
            if not merged_df.empty:
                # Create a confusion matrix-like plot
                confusion = pd.crosstab(
                    merged_df['anomaly'], 
                    merged_df['predicted_fraud'],
                    rownames=['PCA Anomaly'], 
                    colnames=['Simulated Fraud'],
                    normalize='all'
                ) * 100
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(confusion, annot=True, fmt='.1f', cmap='Blues', cbar=True)
                plt.title('Model Agreement (% of Transactions)')
                plt.tight_layout()
                plt.savefig(output_dir / f"model_agreement.{format}")
                plt.close()
                
                # Scatter plot comparing scores
                plt.figure(figsize=(12, 8))
                sns.scatterplot(
                    x='anomaly_score', 
                    y='fraud_probability',
                    hue=(merged_df['anomaly'] != merged_df['predicted_fraud']),
                    data=merged_df,
                    palette={False: 'blue', True: 'red'}
                )
                plt.axhline(y=0.5, color='grey', linestyle='--')
                plt.axvline(x=0, color='grey', linestyle='--')
                plt.title('PCA Anomaly Score vs Fraud Probability')
                plt.xlabel('PCA Anomaly Score')
                plt.ylabel('Fraud Probability')
                plt.legend(title='Models Disagree')
                plt.tight_layout()
                plt.savefig(output_dir / f"model_score_comparison.{format}")
                plt.close()
    
    print("Created ML model result visualizations.")

def create_summary_report(data, output_dir, format="png"):
    """Create a summary report with all visualizations."""
    report_path = output_dir / "visualization_report.html"
    
    # Collect list of all generated visualizations
    viz_files = list(output_dir.glob(f"*.{format}"))
    
    if not viz_files:
        return
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fraud Detection Visualization Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            .viz-container {{ display: flex; flex-wrap: wrap; }}
            .viz-item {{ margin: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
            .viz-item img {{ max-width: 100%; height: auto; }}
            .viz-item p {{ padding: 10px; margin: 0; background: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>Fraud Detection Visualization Report</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="viz-container">
    """
    
    # Add each visualization to the report
    for viz_file in sorted(viz_files):
        file_name = viz_file.name
        friendly_name = ' '.join(file_name.replace(f'.{format}', '').split('_')).title()
        
        html_content += f"""
            <div class="viz-item">
                <img src="{file_name}" alt="{friendly_name}">
                <p>{friendly_name}</p>
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Write HTML report
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"Created visualization report at {report_path}")

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
    
    # Generate visualizations
    visualize_transaction_amounts(data, output_dir, format=args.format)
    visualize_rule_violations(data, output_dir, format=args.format)
    visualize_time_patterns(data, output_dir, format=args.format)
    visualize_distance_patterns(data, output_dir, format=args.format)
    visualize_ml_results(data, output_dir, format=args.format)
    
    # Create summary report
    create_summary_report(data, output_dir, format=args.format)
    
    print(f"\nAll visualizations have been saved to {output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
