# scripts/generate_transaction_report.py
"""
Generate a transaction-by-transaction fraud detection report in a user-friendly format.
This script combines the results from all detection methods and produces a detailed
report on each transaction.
"""

import pandas as pd
import sys
import argparse
from pathlib import Path
import json
from datetime import datetime

# Import configuration
try:
    import config
except ModuleNotFoundError:
    print("Error: config.py not found. Make sure it's in the project root.")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate Transaction Fraud Report")
    parser.add_argument(
        "--input", type=str, default=str(config.INTEGRATED_ASSESSMENT_FILE),
        help="Path to integrated results CSV file"
    )
    parser.add_argument(
        "--output", type=str, default=str(config.SIMULATION_OUTPUT_DIR / "transaction_report.txt"),
        help="Path to save the text report"
    )
    parser.add_argument(
        "--json-output", type=str, default=str(config.SIMULATION_OUTPUT_DIR / "transaction_report.json"),
        help="Path to save JSON version of the report"
    )
    parser.add_argument(
        "--threshold", type=float, default=config.RISK_THRESHOLD_HIGH,
        help="Risk score threshold to mark as suspicious"
    )
    return parser.parse_args()

def load_data(input_file):
    """Load and validate input data."""
    try:
        print(f"Loading data from {input_file}...")
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} transactions.")
        return df
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

def format_transaction(row, threshold):
    """Format a single transaction for the report."""
    # Determine transaction status based on risk score or risk level
    status = "🚨 SUSPICIOUS" if row.get('risk_score', 0) >= threshold or row.get('risk_level') == 'High' else "✅ LEGITIMATE"
    
    # Format timestamp
    timestamp = row.get('trans_date_trans_time', 'Unknown Date/Time')
    try:
        dt = pd.to_datetime(timestamp)
        timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        pass  # Keep original if parsing fails
    
    # Build transaction details
    details = {
        "Transaction ID": row.get('trans_num', 'Unknown'),
        "Date/Time": timestamp,
        "Merchant": row.get('merchant', 'Unknown'),
        "Amount": f"${row.get('amt', 0):.2f}",
        "Category": row.get('category', 'Unknown')
    }
    
    # Build detection signals
    signals = {
        "Rule Violations": row.get('rule_flags', 'None') if row.get('rule_flags') else 'None',
        "Rules Triggered": "Yes" if row.get('rule_triggered', 0) == 1 else "No",
        "Anomaly": "Yes" if row.get('anomaly', 0) == 1 else "No",
        "Anomaly Score": f"{row.get('anomaly_score', 0):.4f}",
        "ML Fraud Prediction": "Yes" if row.get('predicted_fraud', 0) == 1 else "No",
        "Fraud Probability": f"{row.get('fraud_probability', 0):.2%}",
        "Overall Risk Level": row.get('risk_level', 'Unknown'),
        "Risk Score": f"{row.get('risk_score', 0):.4f}"
    }
    
    # Additional metrics if available
    additional = {}
    if 'distance_km' in row:
        additional['Distance from Home'] = f"{row['distance_km']:.1f} km"
    if 'is_fraud' in row:
        additional['Actual Fraud'] = "Yes" if row['is_fraud'] == 1 else "No"

    return {
        "status": status,
        "details": details,
        "signals": signals,
        "additional": additional
    }

def generate_text_report(transactions, threshold):
    """Generate a formatted text report for all transactions."""
    report = []
    
    # Report header
    report.append("=" * 80)
    report.append(f"TRANSACTION FRAUD DETECTION REPORT")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    report.append("")
    
    # Count suspicious transactions
    suspicious_count = sum(1 for data in transactions if data['status'].startswith("🚨"))
    report.append(f"Summary: {suspicious_count} suspicious transactions out of {len(transactions)} total")
    report.append("")
    
    # Detailed transaction reports
    for i, data in enumerate(transactions, 1):
        report.append("-" * 80)
        report.append(f"TRANSACTION #{i}: {data['status']}")
        report.append("-" * 80)
        
        # Transaction details
        report.append("Transaction Details:")
        for key, value in data['details'].items():
            report.append(f"  {key}: {value}")
        
        # Detection signals
        report.append("\nDetection Signals:")
        for key, value in data['signals'].items():
            report.append(f"  {key}: {value}")
        
        # Additional information
        if data['additional']:
            report.append("\nAdditional Information:")
            for key, value in data['additional'].items():
                report.append(f"  {key}: {value}")
        
        report.append("")
    
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    return "\n".join(report)

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Load and process data
    df = load_data(args.input)
    
    # Sort transactions by risk score (descending) for better reporting
    if 'risk_score' in df.columns:
        df = df.sort_values('risk_score', ascending=False)
    
    # Format each transaction
    formatted_transactions = [format_transaction(row, args.threshold) for _, row in df.iterrows()]
    
    # Generate text report
    text_report = generate_text_report(formatted_transactions, args.threshold)
    
    # Save text report
    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(text_report)
        print(f"Text report saved to {output_path}")
    except Exception as e:
        print(f"Error saving text report: {e}", file=sys.stderr)
    
    # Save JSON report
    try:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        json_report = {
            "report_date": datetime.now().isoformat(),
            "total_transactions": len(formatted_transactions),
            "suspicious_count": sum(1 for data in formatted_transactions if data['status'].startswith("🚨")),
            "transactions": formatted_transactions
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=2)
        print(f"JSON report saved to {json_path}")
    except Exception as e:
        print(f"Error saving JSON report: {e}", file=sys.stderr)
    
    print("Report generation complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
