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

