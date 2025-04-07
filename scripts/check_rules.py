# scripts/check_rules.py

import pandas as pd
import json
from pathlib import Path
from geopy.distance import geodesic
from datetime import datetime, timedelta
import sys

# Import configuration
try:
    import config
except ModuleNotFoundError:
    print("Error: config.py not found. Make sure it's in the project root.")
    sys.exit(1)

# --- Configuration --- (Now loaded from config.py)
ACCOUNT_DETAILS_FILE = config.ACCOUNT_DETAILS_FILE
TRANSACTIONS_FILE = config.TRANSACTIONS_FILE
RESULTS_OUTPUT_FILE = config.FLAGGED_TRANSACTIONS_FILE # Renamed for clarity

# Rule Thresholds from config
LOCATION_THRESHOLD_KM = config.LOCATION_THRESHOLD_KM
HIGH_AMOUNT_THRESHOLD = config.HIGH_AMOUNT_THRESHOLD
RECENT_ACCOUNT_DAYS = config.RECENT_ACCOUNT_DAYS
RECENT_ACCOUNT_HIGH_VELOCITY_THRESHOLD = config.RECENT_ACCOUNT_HIGH_VELOCITY_THRESHOLD
# ---

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in kilometers between two lat/lon points."""
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return float('inf') # Indicate missing data
    try:
        # Ensure coordinates are valid floats before calculating
        return geodesic((float(lat1), float(lon1)), (float(lat2), float(lon2))).km
    except ValueError as e:
        print(f"Warning: Invalid coordinates for distance calc: ({lat1},{lon1}) to ({lat2},{lon2}). Error: {e}")
        return float('inf') # Indicate calculation error

def check_transaction_rules(transaction_row, account_details, recent_transactions):
    """
    Applies rules to a single transaction.
    `recent_transactions` is a list of timestamps from the last 24h for velocity checks.
    Returns a tuple: (list_of_flags, rule_triggered_flag)
    """
    flags = []
    rule_triggered_flag = 0 # 0 = No violation, 1 = Violation detected
    account_lat = account_details.get('lat')
    account_lon = account_details.get('long')
    account_creation_str = account_details.get('account_creation_date')

    # --- Date Checks ---
    trans_dt = None
    try:
        trans_dt_str = transaction_row.get('trans_date_trans_time', '')
        trans_dt = pd.to_datetime(trans_dt_str) if trans_dt_str else None
    except (ValueError, TypeError) as e:
        flags.append(f"INVALID_TIMESTAMP ({e})")

    acc_creation_dt = None
    if account_creation_str:
        try:
            acc_creation_dt = datetime.fromisoformat(account_creation_str)
        except (ValueError, TypeError) as e:
            flags.append(f"INVALID_ACCOUNT_DATE_FORMAT ({e})")

    # Rule 1: Transaction before account creation
    if trans_dt and acc_creation_dt and trans_dt < acc_creation_dt:
        flags.append(
            f"RULE_VIOLATION:TRANSACTION_BEFORE_ACCOUNT_CREATION ({trans_dt.date()} vs {acc_creation_dt.date()})")
        rule_triggered_flag = 1

    # --- Location Checks ---
    # Rule 2: Transaction far from account home location
    merch_lat = transaction_row.get('merch_lat')
    merch_lon = transaction_row.get('merch_long')
    if merch_lat is not None and merch_lon is not None and account_lat is not None and account_lon is not None:
        distance = calculate_distance(merch_lat, merch_lon, account_lat, account_lon)
        if distance > LOCATION_THRESHOLD_KM:
            flags.append(f"RULE_VIOLATION:LOCATION_FAR_FROM_HOME ({distance:.0f}km > {LOCATION_THRESHOLD_KM}km)")
            rule_triggered_flag = 1
    elif not (merch_lat is not None and merch_lon is not None):
         flags.append("INFO:Missing_Merchant_Coordinates_For_Location_Check")
    elif not (account_lat is not None and account_lon is not None):
         flags.append("INFO:Missing_Account_Coordinates_For_Location_Check")


    # --- Amount Checks ---
    # Rule 3: High transaction amount
    if 'amt' in transaction_row:
        try:
            amount = pd.to_numeric(transaction_row['amt'], errors='coerce')
            if pd.notna(amount) and amount > HIGH_AMOUNT_THRESHOLD:
                flags.append(f"RULE_VIOLATION:HIGH_AMOUNT (${amount:.2f} > ${HIGH_AMOUNT_THRESHOLD})")
                rule_triggered_flag = 1
            elif pd.isna(amount):
                flags.append("INFO:Missing_Amount_For_Check")
        except Exception as e:
             flags.append(f"INFO:Error_Processing_Amount ({e})")


    # --- Velocity Checks ---
    # Rule 4: High velocity for recently created account
    if acc_creation_dt and trans_dt:
        account_age = trans_dt - acc_creation_dt
        if account_age.days <= RECENT_ACCOUNT_DAYS:
            # Count transactions within the last 24 hours of the current transaction
            one_day_before = trans_dt - timedelta(days=1)
            # Ensure timestamps are valid datetimes before comparing
            count_last_24h = sum(
                1 for ts in recent_transactions if isinstance(ts, datetime) and ts >= one_day_before and ts < trans_dt
            )
            # Add 1 for the current transaction itself when comparing to threshold
            if (count_last_24h + 1) > RECENT_ACCOUNT_HIGH_VELOCITY_THRESHOLD:
                flags.append(
                    f"RULE_VIOLATION:HIGH_VELOCITY_RECENT_ACCOUNT ({count_last_24h + 1} trans in 24h > {RECENT_ACCOUNT_HIGH_VELOCITY_THRESHOLD})")
                rule_triggered_flag = 1

    # Add more rules here...

    return flags, rule_triggered_flag

def main():
    """Main rule checking logic."""
    print("--- Starting Rule-Based Checks ---")

    # 1. Load simulated account details
    print(f"Loading account details from {ACCOUNT_DETAILS_FILE}...")
    try:
        with open(ACCOUNT_DETAILS_FILE, 'r') as f:
            account_details = json.load(f)
        print(f"   Loaded details for account {account_details.get('account_id', 'N/A')}")
    except FileNotFoundError:
        print(f"Error: Account details file not found at {ACCOUNT_DETAILS_FILE}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading account details: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Load simulated transactions
    print(f"Loading transactions from {TRANSACTIONS_FILE}...")
    try:
        transactions_df = pd.read_csv(TRANSACTIONS_FILE)
        # Ensure 'trans_num' exists for linking results
        if 'trans_num' not in transactions_df.columns:
            print("Warning: 'trans_num' column missing in input transactions. Results might be harder to link.", file=sys.stderr)
            # Generate one if missing? Might be risky if order changes.
            # transactions_df['trans_num'] = [f'rule_{i}' for i in range(len(transactions_df))]

        # Convert timestamp column for easier handling
        transactions_df['trans_ts'] = pd.to_datetime(transactions_df['trans_date_trans_time'], errors='coerce')
        # Sort by time for velocity checks
        transactions_df = transactions_df.sort_values(by='trans_ts').reset_index(drop=True)
        print(f"   Loaded {len(transactions_df)} transactions.")
    except FileNotFoundError:
        print(f"Error: Transactions file not found at {TRANSACTIONS_FILE}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading or processing transactions: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Apply rules transaction by transaction
    print("\nApplying rules to transactions...")
    flagged_count = 0
    all_timestamps = transactions_df['trans_ts'].dropna().tolist() # List of all valid timestamps

    results_data = [] # Store results dictionaries

    for index, transaction in transactions_df.iterrows():
        # Get timestamps relevant for velocity check (up to current transaction)
        current_ts = transaction['trans_ts']
        relevant_timestamps = [ts for ts in all_timestamps if isinstance(current_ts, datetime) and isinstance(ts, datetime) and ts < current_ts]

        flags, rule_triggered = check_transaction_rules(transaction.to_dict(), account_details, relevant_timestamps)

        result_row = transaction.to_dict()
        result_row['rule_flags'] = '; '.join(flags) if flags else '' # Join flags into a string
        result_row['rule_triggered'] = rule_triggered # Add the binary flag

        results_data.append(result_row)

        if rule_triggered:
            flagged_count += 1
            # Optional: Print details for triggered transactions
            # print(f"\n--- Transaction {index} ({transaction.get('trans_num', 'N/A')}) ---")
            # print(f"  Timestamp: {transaction.get('trans_date_trans_time', 'N/A')}")
            # print(f"  Amount: {transaction.get('amt', 'N/A'):.2f}, Merchant: {transaction.get('merchant', 'N/A')}")
            # print(f"  !! Flags Triggered: {result_row['rule_flags']}")

    if flagged_count == 0:
        print("\nNo transactions triggered rule violations.")
    else:
        print(
            f"\n--- Summary: {flagged_count} out of {len(transactions_df)} transactions triggered one or more rule violations. ---")

    # Save results with flags added
    results_df = pd.DataFrame(results_data)
    # Drop the temporary 'trans_ts' column if it exists
    if 'trans_ts' in results_df.columns:
        results_df = results_df.drop(columns=['trans_ts'])

    results_df.to_csv(RESULTS_OUTPUT_FILE, index=False)
    print(f"Results with flags saved to {RESULTS_OUTPUT_FILE}")

    print("--- Rule Checking Complete ---")

if __name__ == "__main__":
    main()