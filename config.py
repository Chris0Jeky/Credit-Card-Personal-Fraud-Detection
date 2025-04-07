# config.py
"""Central configuration file for the Fraud Detection System."""

from pathlib import Path
import os

# --- Project Structure ---
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
SIMULATION_OUTPUT_DIR = DATA_DIR / "simulation_output"
MODELS_DIR = ROOT_DIR / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained_models"
SCRIPTS_DIR = ROOT_DIR / "scripts"
VISUALIZATIONS_DIR = SIMULATION_OUTPUT_DIR / "visualizations"
SOURCE_DATA_DIR = ROOT_DIR / "Datasets" / "Credit Card Transactions Fraud Detection Dataset"

# --- Output Directories (Ensure they exist) ---
SIMULATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

# --- Input Files ---
# Source dataset for sampling transaction patterns
SOURCE_DATASET_PATH = SOURCE_DATA_DIR / "fraudTrain.csv"
# List of legitimate companies for merchant check
LEGIT_COMPANIES_FILE = DATA_DIR / "legit_companies.txt"

# --- Simulation Parameters ---
# Number of transactions to simulate per run
# Can be overridden by command line args in runner.py
DEFAULT_NUM_TRANSACTIONS = 75
MIN_ACCOUNT_AGE_DAYS = 60
MAX_ACCOUNT_AGE_DAYS = 365 * 2
MIN_DAYS_BEFORE_FIRST_TRANS = 1
MAX_DAYS_BEFORE_FIRST_TRANS = 30

# --- Simulation Output Files ---
ACCOUNT_DETAILS_FILE = SIMULATION_OUTPUT_DIR / "simulated_account_details.json"
TRANSACTIONS_FILE = SIMULATION_OUTPUT_DIR / "simulated_account_transactions.csv" # Raw simulation
FLAGGED_TRANSACTIONS_FILE = SIMULATION_OUTPUT_DIR / "simulated_transactions_with_flags.csv" # After rules
PCA_PREDICTIONS_FILE = SIMULATION_OUTPUT_DIR / "pca_fraud_predictions.csv" # PCA model output
RF_PREDICTIONS_FILE = SIMULATION_OUTPUT_DIR / "simulated_fraud_predictions.csv" # RF model output
INTEGRATED_ASSESSMENT_FILE = SIMULATION_OUTPUT_DIR / "integrated_fraud_assessment.csv" # Combined results
FLAG_ANALYSIS_FILE = SIMULATION_OUTPUT_DIR / "flag_analysis.json" # Rule analysis output
EVALUATION_REPORT_FILE = SIMULATION_OUTPUT_DIR / "evaluation_report.json" # Performance metrics

# --- Rule Engine Parameters ---
LOCATION_THRESHOLD_KM = 500
HIGH_AMOUNT_THRESHOLD = 1000
RECENT_ACCOUNT_DAYS = 30
RECENT_ACCOUNT_HIGH_VELOCITY_THRESHOLD = 5

# --- ML Model Parameters ---
# PCA / Isolation Forest
PCA_N_COMPONENTS = 10
PCA_CONTAMINATION = 0.05 # Expected anomaly fraction in simulated data
PCA_MODEL_FILE = TRAINED_MODELS_DIR / "isolation_forest_model.pkl"
PCA_TRANSFORMER_FILE = TRAINED_MODELS_DIR / "pca_transformer.pkl"
PCA_SCALER_FILE = TRAINED_MODELS_DIR / "standard_scaler.pkl"
# Features for PCA model (must be numeric)
PCA_NUMERIC_FEATURES = ['amt', 'lat', 'long', 'merch_lat', 'merch_long', 'city_pop']

# Random Forest
RF_MODEL_FILE = TRAINED_MODELS_DIR / "random_forest_model.pkl"
RF_FEATURES = ['amt', 'hour', 'day_of_week', 'category_encoded', 'distance_km'] # Example features
RF_TARGET = 'is_fraud'
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
RF_PREDICTION_THRESHOLD = 0.5 # Probability threshold to classify as fraud

# --- Integration Parameters ---
# Weights for combining different fraud signals into a final risk score
# Adjust these based on perceived importance or tuning
WEIGHT_RULE = 0.30
WEIGHT_PCA_SCORE = 0.30 # Note: PCA score is inverted (lower = more anomalous)
WEIGHT_RF_PROB = 0.40
# Risk Score Thresholds for categorization
RISK_THRESHOLD_MEDIUM = 0.4
RISK_THRESHOLD_HIGH = 0.7

# --- Visualization ---
DEFAULT_VIZ_FORMAT = "png"

# --- Script Paths ---
# These are now relative to ROOT_DIR defined above
SIMULATE_SCRIPT = SCRIPTS_DIR / "simulate_entities.py"
CHECK_RULES_SCRIPT = SCRIPTS_DIR / "check_rules.py"
MERCHANT_CHECKER_SCRIPT = SCRIPTS_DIR / "merchant_checker.py"
ANALYZE_FLAGS_SCRIPT = SCRIPTS_DIR / "analyze_flags.py"
VISUALIZE_SCRIPT = SCRIPTS_DIR / "visualize_fraud.py"
EVALUATE_SCRIPT = SCRIPTS_DIR / "evaluate_performance.py" # New
INTEGRATE_SCRIPT = SCRIPTS_DIR / "integrate_results.py"   # New

MODEL_PCA_PREDICT_SCRIPT = MODELS_DIR / "predict_pca_fraud.py"
MODEL_RF_PREDICT_SCRIPT = MODELS_DIR / "predict_simulated_fraud.py" # Renamed for clarity