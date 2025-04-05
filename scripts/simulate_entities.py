import pandas as pd
from faker import Faker
from pathlib import Path
import random
import json
from datetime import datetime, timedelta

# --- Configuration ---
# Path to the dataset we'll sample transaction patterns from
SOURCE_DATASET_PATH = Path("./Datasets/Credit Card Transactions Fraud Detection Dataset/fraudTrain.csv")
# Directory to save the simulated data
OUTPUT_DIR = Path("../data/simulation_output")

# Files to save
ACCOUNT_DETAILS_FILE = OUTPUT_DIR / "simulated_account_details.json"
TRANSACTIONS_FILE = OUTPUT_DIR / "simulated_account_transactions.csv"

NUM_TRANSACTIONS_TO_SIMULATE = 75 # Generate a decent number of transactions
MIN_ACCOUNT_AGE_DAYS = 60       # Account created between 60 days...
MAX_ACCOUNT_AGE_DAYS = 365 * 2  # ...and 2 years ago
MIN_DAYS_BEFORE_FIRST_TRANS = 1 # First transaction happens at least 1 day...
MAX_DAYS_BEFORE_FIRST_TRANS = 30 # ...and up to 30 days after account creation
# ---

fake = Faker()

def create_simulated_account():
    """Generates details for a single simulated account."""
    profile = fake.profile()
    # Safer address parsing
    address_parts = profile.get('address', '').split('\n')
    street = address_parts[0] if len(address_parts) > 0 else fake.street_address()
    city_state_zip = address_parts[1] if len(address_parts) > 1 else f"{fake.city()}, {fake.state_abbr()} {fake.zipcode()}"

    city, state, zip_code = fake.city(), fake.state_abbr(), fake.zipcode()  # Defaults
    try:
        parts = city_state_zip.split(',')
        city = parts[0].strip()
        state_zip = parts[1].strip().split(' ')
        if len(state_zip) >= 2:
            state = state_zip[0]
            zip_code = state_zip[1]
    except Exception:
        pass  # Keep defaults if parsing fails

    creation_delta_days = random.randint(MIN_ACCOUNT_AGE_DAYS, MAX_ACCOUNT_AGE_DAYS)
    account_creation_date = datetime.now() - timedelta(days=creation_delta_days)

    account = {
        "account_id": fake.uuid4(),  # Unique ID for this simulated account
        "cc_num": fake.credit_card_number(card_type='visa'),  # Generate a plausible CC number
        "first": profile.get('name', 'Unknown Unknown').split(' ')[0],
        "last": profile.get('name', 'Unknown Unknown').split(' ')[-1],
        "gender": "M" if profile.get('sex') == 'M' else "F",
        "street": street,
        "city": city,
        "state": state,
        "zip": zip_code,
        "lat": float(profile.get('current_location', (fake.latitude(), 0))[0]),  # Get lat/lon
        "long": float(profile.get('current_location', (0, fake.longitude()))[1]),
        "city_pop": random.randint(5000, 2000000),  # Add city_pop if needed by dataset columns
        "job": profile.get('job', fake.job()),
        "dob": profile.get('birthdate', fake.date_of_birth(minimum_age=18, maximum_age=90)).strftime("%Y-%m-%d"),
        "account_creation_date": account_creation_date.isoformat(),  # Store as ISO string
        "email": profile.get('mail', fake.email()),
        # Add any other fields you might want to simulate
    }
    print(f"   Simulated Account Details:")
    print(f"     ID: {account['account_id']}")
    print(f"     Name: {account['first']} {account['last']}")
    print(f"     Location: {account['city']}, {account['state']} ({account['lat']:.4f}, {account['long']:.4f})")
    print(f"     Created: {account['account_creation_date']}")
    return account

def simulate_transactions(account_details, source_df, num_transactions):
    """Generates transactions for the account by sampling from source_df."""
    simulated_data = []
    # Ensure we have enough unique samples if possible, otherwise allow replacement
    replace_sampling = num_transactions > len(source_df)
    sampled_transactions = source_df.sample(n=num_transactions, replace=replace_sampling)

    # Start transactions sometime after account creation
    account_creation_dt = datetime.fromisoformat(account_details['account_creation_date'])
    first_trans_delta = timedelta(days=random.randint(MIN_DAYS_BEFORE_FIRST_TRANS, MAX_DAYS_BEFORE_FIRST_TRANS))
    current_trans_time = account_creation_dt + first_trans_delta

    # Select columns from the source_df that we want to keep directly
    # Keep merchant details, amount, category, and potentially the original fraud flag
    cols_to_keep = ['merchant', 'category', 'amt', 'merch_lat', 'merch_long', 'is_fraud']
    # Ensure all expected columns exist in source_df
    cols_to_keep = [col for col in cols_to_keep if col in source_df.columns]

    for _, source_row in sampled_transactions.iterrows():
        new_trans = {}

        # Copy relevant fields from the source transaction
        for col in cols_to_keep:
            new_trans[col] = source_row[col]

            # Overwrite/Add customer and transaction-specific details
            new_trans['account_id'] = account_details['account_id']  # Link to account
            new_trans['cc_num'] = account_details['cc_num']
            new_trans['first'] = account_details['first']  # Add customer name for context if needed
            new_trans['last'] = account_details['last']
            # Add customer location at time of transaction (same as home for simplicity now)
            new_trans['lat'] = account_details['lat']
            new_trans['long'] = account_details['long']

            # Add other account fields if they exist as columns in the original dataset schema
            for key in ['gender', 'city', 'state', 'zip', 'job', 'dob', 'city_pop']:
                if key in source_df.columns:  # Only add if it's an expected column
                    new_trans[key] = account_details[key]

            # Generate new transaction time & ID
            # Add variable time delta: seconds to hours
            time_delta_seconds = random.randint(30, 6 * 60 * 60)  # 30 seconds to 6 hours
            current_trans_time += timedelta(seconds=time_delta_seconds)
            new_trans['trans_date_trans_time'] = current_trans_time.strftime("%Y-%m-%d %H:%M:%S")
            new_trans['unix_time'] = int(current_trans_time.timestamp())
            new_trans['trans_num'] = f"sim_{fake.uuid4()}"  # Mark as simulated

            # NOTE: We are keeping the 'is_fraud' flag from the sampled source transaction.
            # This means our simulated user might perform 'fraudulent' actions based on the source data's labels.

            simulated_data.append(new_trans)

        # Create DataFrame with columns in a sensible order (match original if possible)
        # Get column order from original dataset if available
        if 'account_id' not in source_df.columns:
            # Define a sensible default order if source_df doesn't match perfectly
            col_order = ['account_id', 'trans_date_trans_time', 'cc_num', 'merchant', 'category', 'amt', 'first',
                         'last', 'gender', 'street', 'city', 'state', 'zip', 'lat', 'long', 'city_pop', 'job',
                         'dob', 'trans_num', 'unix_time', 'merch_lat', 'merch_long', 'is_fraud']
            # Filter order to only include columns actually generated
            col_order = [col for col in col_order if col in simulated_data[0]]
        else:
            col_order = list(source_df.columns)  # Try to match source

        return pd.DataFrame(simulated_data, columns=col_order)

    def main():
        """Main simulation logic."""
        print("--- Starting Entity Simulation ---")

        # 1. Ensure output directory exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {OUTPUT_DIR.resolve()}")

        # 2. Create simulated account
        print("Creating simulated account...")
        account = create_simulated_account()

        # Save account details
        try:
            with open(ACCOUNT_DETAILS_FILE, 'w') as f:
                json.dump(account, f, indent=4)
            print(f"   Account details saved to {ACCOUNT_DETAILS_FILE.name}")
        except Exception as e:
            print(f"Error saving account details: {e}")
            return  # Stop if we can't save account

        # 3. Load source transaction data
        print(f"Loading source transaction data from {SOURCE_DATASET_PATH}...")
        try:
            # Try to infer index column if it exists and is unnamed
            temp_df = pd.read_csv(SOURCE_DATASET_PATH, nrows=1)
            if temp_df.columns[0].startswith('Unnamed:'):
                source_df = pd.read_csv(SOURCE_DATASET_PATH, index_col=0)
            else:
                source_df = pd.read_csv(SOURCE_DATASET_PATH)
            print(f"   Loaded {len(source_df)} source transactions.")
        except FileNotFoundError:
            print(f"Error: Source dataset not found at {SOURCE_DATASET_PATH}")
            return
        except Exception as e:
            print(f"Error loading source dataset: {e}")
            return

        # 4. Simulate transactions
        print(f"Simulating {NUM_TRANSACTIONS_TO_SIMULATE} transactions...")
        simulated_df = simulate_transactions(account, source_df, NUM_TRANSACTIONS_TO_SIMULATE)

        # 5. Save simulated transactions
        try:
            simulated_df.to_csv(TRANSACTIONS_FILE, index=False)
            print(f"   Simulated transactions saved to {TRANSACTIONS_FILE.name} ({len(simulated_df)} rows)")
        except Exception as e:
            print(f"Error saving simulated transactions: {e}")

        print("--- Simulation Complete ---")

    if __name__ == "__main__":
        main()
