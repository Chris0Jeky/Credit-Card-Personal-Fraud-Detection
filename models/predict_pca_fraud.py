"""
PCA-based Fraud Detection Model.
This script loads simulated transaction data, applies PCA transformation
and predicts fraud using a simple anomaly detection approach.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import pickle
import joblib
import os

# --- Configuration ---
MODEL_DIR = Path(__file__).parent
DATA_DIR = MODEL_DIR.parent / "data" / "simulation_output"
MODELS_OUTPUT_DIR = MODEL_DIR / "trained_models"
RESULTS_DIR = DATA_DIR

# Ensure directories exist
MODELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Input/output files
DEFAULT_INPUT_FILE = DATA_DIR / "simulated_account_transactions.csv"
MODEL_FILE = MODELS_OUTPUT_DIR / "isolation_forest_model.pkl"
PCA_FILE = MODELS_OUTPUT_DIR / "pca_transformer.pkl"
SCALER_FILE = MODELS_OUTPUT_DIR / "standard_scaler.pkl"
RESULTS_FILE = RESULTS_DIR / "pca_fraud_predictions.csv"

# Model parameters
N_COMPONENTS = 10  # Number of PCA components to use
CONTAMINATION = 0.05  # Expected fraction of anomalies

# Features to use for modeling (numeric only)
NUMERIC_FEATURES = ['amt', 'lat', 'long', 'merch_lat', 'merch_long']
# ---

def load_data(input_file):
    """Load and preprocess transaction data."""
    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} transactions.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

def preprocess_data(df):
    """Prepare data for PCA and anomaly detection."""
    print("Preprocessing data...")
    
    # Select only numeric features that exist in the dataset
    features_to_use = [f for f in NUMERIC_FEATURES if f in df.columns]
    if not features_to_use:
        print("Error: No usable numeric features found in the dataset!", file=sys.stderr)
        sys.exit(1)
    
    print(f"Using features: {features_to_use}")
    X = df[features_to_use].copy()
    
    # Handle missing values
    for col in X.columns:
        # Calculate column median, ignoring NaN values
        col_median = X[col].median()
        # Fill NaN values with median (avoiding inplace=True)
        X[col] = X[col].fillna(col_median)
    
    # Some columns might be object types with numeric values, convert to float
    for col in X.columns:
        try:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].fillna(X[col].median())
        except Exception as e:
            print(f"Warning: Couldn't convert column {col} to numeric: {e}")
    
    return X

def fit_or_load_transformers(X, force_train=False):
    """Fit or load PCA and StandardScaler."""
    if not force_train and PCA_FILE.exists() and SCALER_FILE.exists():
        print("Loading existing transformers...")
        try:
            scaler = joblib.load(SCALER_FILE)
            pca = joblib.load(PCA_FILE)
            return scaler, pca
        except Exception as e:
            print(f"Error loading transformers: {e}. Will train new ones.")
    
    print("Training new transformers...")
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    min_components = min(N_COMPONENTS, X.shape[1], X.shape[0])
    pca = PCA(n_components=min_components)
    pca.fit(X_scaled)
    
    # Save transformers
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(pca, PCA_FILE)
    
    # Print variance explained
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"PCA with {min_components} components explains {explained_var:.2%} of variance")
    
    return scaler, pca

def fit_or_load_model(X_pca, force_train=False):
    """Fit or load the anomaly detection model."""
    if not force_train and MODEL_FILE.exists():
        print("Loading existing model...")
        try:
            model = joblib.load(MODEL_FILE)
            return model
        except Exception as e:
            print(f"Error loading model: {e}. Will train a new one.")
    
    print("Training new isolation forest model...")
    # Initialize and train the model
    model = IsolationForest(
        contamination=CONTAMINATION,
        random_state=42,
        n_estimators=100,
        max_samples='auto'
    )
    model.fit(X_pca)
    
    # Save the model
    joblib.dump(model, MODEL_FILE)
    
    return model

def predict_anomalies(model, X_pca, transactions_df):
    """Use the model to predict anomalies and return results."""
    print("Predicting anomalies...")
    
    # Get anomaly scores (-1 for anomalies, 1 for normal)
    y_pred = model.predict(X_pca)
    
    # Get decision function scores (negative = more anomalous)
    anomaly_scores = model.decision_function(X_pca)
    
    # Create results dataframe
    results_df = transactions_df.copy()
    results_df['anomaly'] = np.where(y_pred == -1, 1, 0)
    results_df['anomaly_score'] = anomaly_scores
    
    # Count anomalies
    anomaly_count = (y_pred == -1).sum()
    print(f"Detected {anomaly_count} anomalies ({anomaly_count/len(y_pred):.2%})")
    
    return results_df

def main():
    """Main execution function."""
    print("\n=== Starting PCA-based Fraud Detection ===\n")
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])
    else:
        input_file = DEFAULT_INPUT_FILE
    
    # Load and preprocess data
    transactions_df = load_data(input_file)
    X = preprocess_data(transactions_df)
    
    # Fit/load transformers and transform data
    scaler, pca = fit_or_load_transformers(X)
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    
    # Fit/load and apply model
    model = fit_or_load_model(X_pca)
    results_df = predict_anomalies(model, X_pca, transactions_df)
    
    # Save results
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"\nResults saved to {RESULTS_FILE}\n")
    
    # Print top anomalies with transaction info
    print("\nTop 5 most anomalous transactions:")
    top_anomalies = results_df.sort_values('anomaly_score').head(5)
    for idx, row in top_anomalies.iterrows():
        print(f"  Transaction {row.get('trans_num', idx)}")
        print(f"    Date: {row.get('trans_date_trans_time', 'N/A')}")
        print(f"    Amount: ${row.get('amt', 'N/A')}")
        print(f"    Merchant: {row.get('merchant', 'N/A')}")
        print(f"    Anomaly Score: {row['anomaly_score']:.4f}")
        print(f"    Original Fraud Label: {row.get('is_fraud', 'N/A')}")
        print()
    
    print("=== PCA-based Fraud Detection Complete ===")

if __name__ == "__main__":
    main()
