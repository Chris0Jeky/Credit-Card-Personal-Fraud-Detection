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
