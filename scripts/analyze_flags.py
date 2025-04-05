"""
Analyzes flagged transactions to provide insights and statistics.
This script loads transaction data with rule flags and generates summary statistics.
"""

import pandas as pd
from pathlib import Path
import sys
import re
import argparse
import json
from collections import Counter
import os

# --- Configuration ---
DATA_DIR = Path(__file__).parent.parent / "data" / "simulation_output"
FLAGGED_TRANSACTIONS_FILE = DATA_DIR / "simulated_transactions_with_flags.csv"
OUTPUT_DIR = DATA_DIR

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze transaction flags and provide statistics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input", 
        type=str,
        default=str(FLAGGED_TRANSACTIONS_FILE),
        help="Path to the flagged transactions CSV file"
    )
    parser.add_argument(
        "--output", 
        type=str,
        default=str(OUTPUT_DIR / "flag_analysis.json"),
        help="Path to save the JSON analysis results"
    )
    parser.add_argument(
        "--detailed", 
        action="store_true",
        help="Include more detailed statistics"
    )
    return parser.parse_args()

def extract_rule_violations(flags_str):
    """
    Extract individual rule violations from the rule_flags column.
    Returns a list of violations.
    """
    if not isinstance(flags_str, str):
        return []
    
    violations = []
    rule_pattern = r'RULE_VIOLATION:([^(;]+)'
    matches = re.findall(rule_pattern, flags_str)
    
    for match in matches:
        violations.append(match.strip())
    
    return violations

def analyze_flags(df, detailed=False):
    """Analyze the flagged transactions and generate statistics."""
    results = {}
    
    # Basic statistics
    total_transactions = len(df)
    flagged_transactions = df[df['rule_flags'].notna() & (df['rule_flags'] != '')].shape[0]
    
    results['summary'] = {
        'total_transactions': total_transactions,
        'flagged_transactions': flagged_transactions,
        'percentage_flagged': round((flagged_transactions / total_transactions) * 100, 2) if total_transactions > 0 else 0
    }
    
    # Extract and count individual rule violations
    all_violations = []
    
    for flags in df['rule_flags'].dropna():
        violations = extract_rule_violations(flags)
        all_violations.extend(violations)
    
    violation_counts = Counter(all_violations)
    
    results['rule_violations'] = {
        'total_violations': len(all_violations),
        'unique_violation_types': len(violation_counts),
        'violation_counts': dict(violation_counts.most_common())
    }
    
    # Time-based analysis
    if 'trans_date_trans_time' in df.columns:
        df['trans_date'] = pd.to_datetime(df['trans_date_trans_time']).dt.date
        daily_counts = df.groupby('trans_date').apply(
            lambda x: sum(x['rule_flags'].notna() & (x['rule_flags'] != ''))
        ).to_dict()
        
        results['time_analysis'] = {
            'days_with_transactions': len(daily_counts),
            'daily_flag_counts': {str(k): v for k, v in daily_counts.items()}
        }
    
    # Amount-based analysis
    if 'amt' in df.columns:
        flagged_amounts = df[df['rule_flags'].notna() & (df['rule_flags'] != '')]['amt']
        unflagged_amounts = df[~(df['rule_flags'].notna() & (df['rule_flags'] != ''))]['amt']
        
        results['amount_analysis'] = {
            'flagged_transactions': {
                'mean': round(flagged_amounts.mean(), 2) if not flagged_amounts.empty else 0,
                'median': round(flagged_amounts.median(), 2) if not flagged_amounts.empty else 0,
                'min': round(flagged_amounts.min(), 2) if not flagged_amounts.empty else 0,
                'max': round(flagged_amounts.max(), 2) if not flagged_amounts.empty else 0
            },
            'unflagged_transactions': {
                'mean': round(unflagged_amounts.mean(), 2) if not unflagged_amounts.empty else 0,
                'median': round(unflagged_amounts.median(), 2) if not unflagged_amounts.empty else 0,
                'min': round(unflagged_amounts.min(), 2) if not unflagged_amounts.empty else 0,
                'max': round(unflagged_amounts.max(), 2) if not unflagged_amounts.empty else 0
            }
        }
    
    # Merchant analysis
    if 'merchant' in df.columns:
        merchant_flags = df.groupby('merchant').apply(
            lambda x: sum(x['rule_flags'].notna() & (x['rule_flags'] != ''))
        ).sort_values(ascending=False)
        
        results['merchant_analysis'] = {
            'unique_merchants': df['merchant'].nunique(),
            'top_flagged_merchants': dict(merchant_flags.head(5).items())
        }
    
    # Detailed statistics
    if detailed:
        # Fraud label analysis if available
        if 'is_fraud' in df.columns:
            fraud_flag_match = df.groupby('is_fraud').apply(
                lambda x: sum(x['rule_flags'].notna() & (x['rule_flags'] != ''))
            ).to_dict()
            
            results['fraud_label_analysis'] = {
                'fraud_transactions': int(df['is_fraud'].sum()),
                'fraud_flag_match': {str(k): int(v) for k, v in fraud_flag_match.items()}
            }
        
        # Co-occurring rules analysis
        co_occurrences = {}
        
        for flags in df['rule_flags'].dropna():
            violations = extract_rule_violations(flags)
            if len(violations) > 1:
                for i in range(len(violations)):
                    for j in range(i + 1, len(violations)):
                        pair = tuple(sorted([violations[i], violations[j]]))
                        co_occurrences[pair] = co_occurrences.get(pair, 0) + 1
        
        sorted_co_occurrences = sorted(co_occurrences.items(), key=lambda x: x[1], reverse=True)
        
        results['co_occurring_rules'] = {
            'count': len(sorted_co_occurrences),
            'top_pairs': {f"{pair[0]} & {pair[1]}": count for (pair, count) in sorted_co_occurrences[:5]}
        }
        
        # Transaction velocity analysis
        if 'trans_date_trans_time' in df.columns:
            df['trans_datetime'] = pd.to_datetime(df['trans_date_trans_time'])
            df = df.sort_values('trans_datetime')
            
            # Calculate time differences between consecutive transactions
            df['time_diff'] = df['trans_datetime'].diff().dt.total_seconds() / 60  # in minutes
            
            results['velocity_analysis'] = {
                'mean_time_between_trans_mins': round(df['time_diff'].mean(), 2),
                'median_time_between_trans_mins': round(df['time_diff'].median(), 2),
                'transactions_under_5_mins_apart': int(sum(df['time_diff'] < 5)),
                'transactions_under_60_mins_apart': int(sum(df['time_diff'] < 60)),
            }
            
            # Check if there are multiple transactions in quick succession
            if sum(df['time_diff'] < 5) > 0:
                quick_trans = df[df['time_diff'] < 5].copy()
                quick_trans['quick_sequence'] = quick_trans['trans_datetime'].shift(-1) - quick_trans['trans_datetime']
                results['velocity_analysis']['quick_succession_details'] = {
                    'count': len(quick_trans),
                    'percentage': round((len(quick_trans) / total_transactions) * 100, 2)
                }
    
    return results

def print_analysis(analysis):
    """Print the analysis results in a readable format."""
    print("\n=== TRANSACTION FLAG ANALYSIS ===\n")
    
    # Print summary
    print("SUMMARY:")
    print(f"  Total Transactions: {analysis['summary']['total_transactions']}")
    print(f"  Flagged Transactions: {analysis['summary']['flagged_transactions']} ({analysis['summary']['percentage_flagged']}%)")
    
    # Print rule violations
    print("\nRULE VIOLATIONS:")
    violations = analysis['rule_violations']['violation_counts']
    total = analysis['rule_violations']['total_violations']
    for rule, count in violations.items():
        print(f"  {rule}: {count} ({round((count/total)*100, 1)}%)")
    
    # Print merchant analysis if available
    if 'merchant_analysis' in analysis:
        print("\nTOP FLAGGED MERCHANTS:")
        for merchant, flags in analysis['merchant_analysis']['top_flagged_merchants'].items():
            print(f"  {merchant}: {flags} flags")
    
    # Print amount analysis if available
    if 'amount_analysis' in analysis:
        print("\nAMOUNT ANALYSIS:")
        flagged = analysis['amount_analysis']['flagged_transactions']
        print(f"  Flagged Transactions - Mean: ${flagged['mean']}, Max: ${flagged['max']}")
    
    # Print co-occurring rules if available
    if 'co_occurring_rules' in analysis:
        print("\nCO-OCCURRING RULES:")
        for pair, count in analysis['co_occurring_rules']['top_pairs'].items():
            print(f"  {pair}: {count} occurrences")
    
    # Print velocity analysis if available
    if 'velocity_analysis' in analysis:
        print("\nVELOCITY ANALYSIS:")
        velocity = analysis['velocity_analysis']
        print(f"  Mean time between transactions: {velocity['mean_time_between_trans_mins']} minutes")
        print(f"  Transactions under 60 minutes apart: {velocity['transactions_under_60_mins_apart']}")
    
    print("\n=== END OF ANALYSIS ===\n")

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Load flagged transactions
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Error: Input file not found at {input_file}", file=sys.stderr)
        return 1
    
    try:
        print(f"Loading flagged transactions from {input_file}...")
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} transactions.")
    except Exception as e:
        print(f"Error loading transactions: {e}", file=sys.stderr)
        return 1
    
    # Check if rule_flags column exists
    if 'rule_flags' not in df.columns:
        print("Error: The 'rule_flags' column is missing from the input file.", file=sys.stderr)
        return 1
    
    # Analyze flags
    analysis = analyze_flags(df, detailed=args.detailed)
    
    # Print analysis
    print_analysis(analysis)
    
    # Save analysis to JSON
    output_file = Path(args.output)
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Analysis saved to {output_file}")
    except Exception as e:
        print(f"Error saving analysis: {e}", file=sys.stderr)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
