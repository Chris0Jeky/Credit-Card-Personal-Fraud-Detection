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